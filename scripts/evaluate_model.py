"""Run one self-contained RT-DETR deployment ONNX on the fixed COCO evaluation images."""
import argparse
import csv
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image, ImageDraw

from common import ROOT, sha256
from metrics import evaluate_predictions
from model_audit import inspect_model

sys.path.insert(0, str(ROOT))
import prepare_data


class MemorySampler:
    """Sample this process during session creation and image inference."""

    def __init__(self, gpu=False):
        self.process = psutil.Process()
        self.rss_baseline = self.process.memory_info().rss
        self.rss_peak = self.rss_baseline
        self.gpu_peak = None
        self.gpu_baseline = None
        self.gpu_error = None
        self.handle = None
        self.nvml = None
        self.stop_event = threading.Event()
        self.thread = None
        if gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_baseline = self._gpu_bytes()
                self.gpu_peak = self.gpu_baseline
            except Exception as error:
                self.gpu_error = str(error)

    def _gpu_bytes(self):
        processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        allocations = [int(item.usedGpuMemory) for item in processes
                       if item.pid == os.getpid() and int(item.usedGpuMemory) < (1 << 60)]
        return sum(allocations)

    def _poll(self):
        while not self.stop_event.wait(0.02):
            try:
                self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)
                if self.handle is not None:
                    value = self._gpu_bytes()
                    self.gpu_peak = max(self.gpu_peak, value)
            except Exception as error:
                self.gpu_error = str(error)
                self.handle = None

    def start(self):
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)
        if self.handle is not None:
            try:
                self.gpu_peak = max(self.gpu_peak, self._gpu_bytes())
            except Exception as error:
                self.gpu_error = str(error)
        if self.nvml is not None:
            self.nvml.nvmlShutdown()

    def result(self):
        return {
            'rss_baseline_bytes': self.rss_baseline,
            'rss_peak_bytes': self.rss_peak,
            'rss_increase_bytes': self.rss_peak - self.rss_baseline,
            'gpu_process_baseline_bytes': self.gpu_baseline,
            'gpu_process_peak_bytes': self.gpu_peak,
            'gpu_memory_error': self.gpu_error,
            'memory_note': 'Sampled process RSS includes model, runtime, inputs and predictions; GPU value is sampled process allocation when NVML is available.',
        }


def check_interface(session):
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}
    if set(inputs) != {'images', 'orig_target_sizes'} or not {'labels', 'boxes', 'scores'} <= set(outputs):
        raise ValueError('Expected RT-DETR deploy ONNX inputs images/orig_target_sizes and outputs labels/boxes/scores. Export with the official deploy wrapper.')
    if inputs['images'].type not in ('tensor(float)', 'tensor(float16)'):
        raise ValueError('images input must be float32 or float16')
    if inputs['orig_target_sizes'].type not in ('tensor(int64)', 'tensor(int32)'):
        raise ValueError('orig_target_sizes must be int64 or int32')
    return (np.float16 if inputs['images'].type == 'tensor(float16)' else np.float32,
            np.int32 if inputs['orig_target_sizes'].type == 'tensor(int32)' else np.int64)


def load_image(item, size, image_dtype, size_dtype):
    path = ROOT / 'data/coco/val2017' / item['file_name']
    with Image.open(path) as source:
        image = source.convert('RGB')
        width, height = image.size
        if (width, height) != (item['width'], item['height']):
            raise ValueError(f'Image dimensions differ from COCO annotation: {path}')
        resized = image.resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    batch = np.ascontiguousarray(array.transpose(2, 0, 1)[None].astype(image_dtype))
    original_size = np.array([[width, height]], dtype=size_dtype)
    return batch, original_size


def decode(outputs, item, category_ids):
    labels, boxes, scores = [np.asarray(value) for value in outputs]
    if labels.ndim != 2 or scores.shape != labels.shape or boxes.shape != (*labels.shape, 4) or labels.shape[0] != 1:
        raise ValueError(f'Unexpected RT-DETR output shapes: {labels.shape}, {boxes.shape}, {scores.shape}')
    if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(scores)):
        raise ValueError('Model output has NaN or infinity')
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('Model scores must be probabilities in [0, 1]')
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= len(category_ids)):
        raise ValueError('Model labels must be contiguous COCO class indices 0..79')
    records = []
    for label, box, score in zip(labels[0], boxes[0], scores[0]):
        if score <= 0:
            continue
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            continue
        records.append({'image_id': item['id'], 'category_id': category_ids[int(label)],
                        'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(score)})
    return records


def save_preview(image_item, records, destination, threshold=0.5):
    with Image.open(ROOT / 'data/coco/val2017' / image_item['file_name']) as source:
        image = source.convert('RGB')
    draw = ImageDraw.Draw(image)
    for item in sorted((row for row in records if row['score'] >= threshold),
                       key=lambda row: row['score'], reverse=True)[:30]:
        x, y, w, h = item['bbox']
        draw.rectangle((x, y, x + w, y + h), outline='red', width=3)
        draw.text((x, max(0, y - 12)), f"{item['category_id']} {item['score']:.2f}", fill='red')
    image.save(destination, quality=90)


def percentile(values, rank):
    return float(np.percentile(values, rank)) if values else None


def evaluate(model_path, output_dir, provider='cpu', preview_count=3):
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not model_path.is_file() or model_path.suffix.lower() != '.onnx':
        raise ValueError('Upload a self-contained .onnx model file')
    if output_dir.exists():
        raise FileExistsError(f'Choose a new result directory: {output_dir}')
    if provider not in ('cpu', 'cuda'):
        raise ValueError('provider must be cpu or cuda')
    prepare_data.verify()
    protocol_path = ROOT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    ann_path = ROOT / 'data/coco/annotations/evaluation.json'
    ann = json.loads(ann_path.read_text(encoding='utf-8'))
    images = sorted(ann['images'], key=lambda item: item['file_name'])
    if len(images) != protocol['evaluation_images']:
        raise ValueError('Unexpected evaluation image count')
    category_ids = [item['id'] for item in sorted(ann['categories'], key=lambda item: item['id'])]
    if len(category_ids) != protocol['category_count']:
        raise ValueError('Unexpected category count')
    audit = inspect_model(model_path)
    available = ort.get_available_providers()
    primary = 'CUDAExecutionProvider' if provider == 'cuda' else 'CPUExecutionProvider'
    if primary not in available:
        raise RuntimeError(f'{primary} unavailable in this ONNX Runtime installation: {available}')
    # The requested provider must initialize; node-level CPU placement is still possible.
    options = ort.SessionOptions()
    options.log_severity_level = 2
    memory = MemorySampler(gpu=provider == 'cuda')
    memory.start()
    try:
        load_start = time.perf_counter()
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=[primary])
        session.disable_fallback()
        if session.get_providers()[0] != primary:
            raise RuntimeError(f'Requested provider {primary}, got {session.get_providers()}')
        load_s = time.perf_counter() - load_start
        image_dtype, size_dtype = check_interface(session)
        predictions = []
        inference_ms, wall_ms = [], []
        preview_items = []
        warmups = min(protocol['runtime_warmup_images'], len(images))
        for index, item in enumerate(images):
            started = time.perf_counter()
            batch, orig_size = load_image(item, protocol['input_size'], image_dtype, size_dtype)
            feed = {'images': batch, 'orig_target_sizes': orig_size}
            if index < warmups:
                session.run(['labels', 'boxes', 'scores'], feed)
            infer_start = time.perf_counter()
            outputs = session.run(['labels', 'boxes', 'scores'], feed)
            infer_end = time.perf_counter()
            rows = decode(outputs, item, category_ids)
            predictions.extend(rows)
            if len(preview_items) < preview_count:
                preview_items.append((item, rows))
            # Exclude warmup images from latency but keep their measured predictions for AP.
            if index >= warmups:
                inference_ms.append((infer_end - infer_start) * 1000)
                wall_ms.append((time.perf_counter() - started) * 1000)
            if (index + 1) % 100 == 0 or index + 1 == len(images):
                print(f'Inference: {index + 1}/{len(images)}', flush=True)
    finally:
        memory.finish()
    scores = evaluate_predictions(ann_path, predictions,
                                  threshold=protocol['score_threshold_for_precision_recall'])
    output_dir.mkdir(parents=True)
    (output_dir / 'predictions.json').write_text(json.dumps(predictions, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'quantization_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    with (output_dir / 'class_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores['class_wise'][0]))
        writer.writeheader()
        writer.writerows(scores['class_wise'])
    for index, (item, rows) in enumerate(preview_items, 1):
        save_preview(item, rows, output_dir / f'preview_{index}.jpg')
    resource = {
        'model_file_bytes': model_path.stat().st_size,
        'model_file_mib': model_path.stat().st_size / 1048576,
        'model_load_seconds': load_s,
        'inference_images_timed': len(inference_ms),
        'warmup_images': warmups,
        'inference_ms_mean': statistics.mean(inference_ms),
        'inference_ms_p50': percentile(inference_ms, 50),
        'inference_ms_p95': percentile(inference_ms, 95),
        'inference_fps': 1000 / statistics.mean(inference_ms),
        'end_to_end_ms_mean': statistics.mean(wall_ms),
        'end_to_end_ms_p50': percentile(wall_ms, 50),
        'end_to_end_ms_p95': percentile(wall_ms, 95),
        'end_to_end_fps': 1000 / statistics.mean(wall_ms),
        **memory.result(),
    }
    metrics = {key: value for key, value in scores.items() if key != 'class_wise'}
    metrics['resource'] = resource
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    provenance = {
        'model_name': model_path.name,
        'model_sha256': sha256(model_path),
        'model_format': 'self-contained ONNX',
        'evaluation_annotations_sha256': sha256(ann_path),
        'data_manifest_sha256': sha256(ROOT / 'data/coco/manifest.json'),
        'protocol_sha256': sha256(protocol_path),
        'onnxruntime_version': ort.__version__,
        'provider': primary,
        'available_providers': available,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'prediction_count': len(predictions),
        'image_count': len(images),
        'measurement_note': 'Latency excludes first 10 warmup images and measures session.run only; end-to-end includes image decode, resize and output conversion. Compare on the same Colab runtime and provider.',
    }
    (output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f"mAP50:95={metrics['mAP50_95']:.4f}; AP50={metrics['mAP50']:.4f}; size={resource['model_file_mib']:.2f} MiB; inference={resource['inference_ms_p50']:.1f} ms p50")
    print(f'Results: {output_dir}')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Self-contained RT-DETR deploy .onnx')
    parser.add_argument('--output', required=True, help='New result directory')
    parser.add_argument('--provider', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--preview-count', type=int, default=3)
    args = parser.parse_args()
    if args.preview_count < 0:
        parser.error('--preview-count must be nonnegative')
    evaluate(args.model, args.output, args.provider, args.preview_count)


if __name__ == '__main__':
    main()
