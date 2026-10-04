"""Evaluate a real RT-DETR ONNX file on the shared COCO images."""
import argparse
import csv
import gc
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image, ImageDraw

from common import ROOT, sha256
from metrics import evaluate_predictions

sys.path.insert(0, str(ROOT))
import prepare_data


class MemorySampler:
    """Measure session creation/inference; graph audit and profiling are separate."""

    def __init__(self, gpu=False):
        self.process = psutil.Process()
        self.rss_baseline = self.process.memory_info().rss
        self.rss_peak = self.rss_baseline
        self.gpu_peak = None
        self.gpu_baseline = None
        self.gpu_error = None
        self.gpu_observed = False
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
                self.handle = None

    def _gpu_bytes(self):
        entries = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        matching = [item for item in entries if item.pid == os.getpid()]
        if not matching:
            return None
        allocations = [int(item.usedGpuMemory) for item in matching]
        if any(value < 0 or value >= (1 << 60) for value in allocations):
            raise RuntimeError('NVML process allocation is unavailable')
        self.gpu_observed = True
        return sum(allocations)

    def sample(self):
        self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)
        if self.handle is not None:
            try:
                value = self._gpu_bytes()
                if value is not None:
                    self.gpu_peak = value if self.gpu_peak is None else max(self.gpu_peak, value)
            except Exception as error:
                self.gpu_error = str(error)
                self.gpu_peak = None
                self.handle = None

    def _poll(self):
        while not self.stop_event.wait(0.02):
            self.sample()

    def start(self):
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.sample()
        if self.nvml is not None:
            if not self.gpu_observed and self.gpu_error is None:
                self.gpu_error = 'NVML could not attribute a GPU allocation to this process; reported as null.'
            self.nvml.nvmlShutdown()

    def result(self):
        return {
            'rss_baseline_bytes': self.rss_baseline,
            'rss_peak_bytes': self.rss_peak,
            'rss_increase_bytes': self.rss_peak - self.rss_baseline,
            'gpu_process_baseline_bytes': self.gpu_baseline,
            'gpu_process_peak_bytes': self.gpu_peak,
            'gpu_memory_error': self.gpu_error,
            'memory_note': '20ms sampled process RSS during session creation, warmup and measured inference. Includes runtime, inputs and accumulated predictions; graph audit and placement profiling are excluded. GPU allocation is null when NVML attribution is unavailable.',
        }


def check_interface(session, input_size=640):
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}
    if set(inputs) != {'images', 'orig_target_sizes'} or not {'labels', 'boxes', 'scores'} <= set(outputs):
        raise ValueError('Expected deploy inputs images/orig_target_sizes and outputs labels/boxes/scores.')
    if inputs['images'].type not in ('tensor(float)', 'tensor(float16)'):
        raise ValueError('images input must be float32 or float16')
    if inputs['orig_target_sizes'].type not in ('tensor(int64)', 'tensor(int32)'):
        raise ValueError('orig_target_sizes must be int64 or int32')
    for name, expected in [('images', [1, 3, input_size, input_size]), ('orig_target_sizes', [1, 2])]:
        actual = inputs[name].shape
        if len(actual) != len(expected) or any(isinstance(a, int) and a != b for a, b in zip(actual, expected)):
            raise ValueError(f'{name} shape {actual} is incompatible with {expected}')
    return (np.float16 if inputs['images'].type == 'tensor(float16)' else np.float32,
            np.int32 if inputs['orig_target_sizes'].type == 'tensor(int32)' else np.int64)


def load_image(item, size, image_dtype, size_dtype, data_dir=None):
    data = Path(data_dir) if data_dir is not None else ROOT / 'data/coco'
    path = data / 'val2017' / item['file_name']
    with Image.open(path) as source:
        image = source.convert('RGB')
        width, height = image.size
        if (width, height) != (item['width'], item['height']):
            raise ValueError(f'Image dimensions differ from COCO annotation: {path}')
        array = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
    batch = np.ascontiguousarray(array.transpose(2, 0, 1)[None].astype(image_dtype))
    return batch, np.array([[width, height]], dtype=size_dtype)


def decode(outputs, item, category_ids):
    labels, boxes, scores = [np.asarray(value) for value in outputs]
    if labels.ndim != 2 or scores.shape != labels.shape or boxes.shape != (*labels.shape, 4) or labels.shape[0] != 1:
        raise ValueError(f'Unexpected output shapes: {labels.shape}, {boxes.shape}, {scores.shape}')
    if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(scores)):
        raise ValueError('Model output has NaN or infinity')
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('Scores must be probabilities in [0, 1]')
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= len(category_ids)):
        raise ValueError('Labels must be contiguous COCO class indices 0..79')
    records = []
    for label, box, score in zip(labels[0], boxes[0], scores[0]):
        x1, y1, x2, y2 = map(float, box)
        if score > 0 and x2 > x1 and y2 > y1:
            records.append({'image_id': item['id'], 'category_id': category_ids[int(label)],
                            'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(score)})
    return records


def save_preview(item, records, destination, threshold=0.5, data_dir=None):
    data = Path(data_dir) if data_dir is not None else ROOT / 'data/coco'
    with Image.open(data / 'val2017' / item['file_name']) as source:
        image = source.convert('RGB')
    draw = ImageDraw.Draw(image)
    for record in sorted((row for row in records if row['score'] >= threshold),
                         key=lambda row: row['score'], reverse=True)[:30]:
        x, y, w, h = record['bbox']
        draw.rectangle((x, y, x + w, y + h), outline='red', width=3)
        draw.text((x, max(0, y - 12)), f"{record['category_id']} {record['score']:.2f}", fill='red')
    image.save(destination, quality=90)


def percentile(values, rank):
    return float(np.percentile(values, rank)) if values else None


def session_options(threads, profiling=False, profile_prefix=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 2
    options.enable_profiling = profiling
    if profile_prefix is not None:
        options.profile_file_prefix = str(profile_prefix)
    return options


def providers_for(primary):
    if primary == 'CUDAExecutionProvider':
        # Preserve FP32 arithmetic for the baseline on Ampere and newer GPUs.
        return [(primary, {'device_id': '0', 'use_tf32': '0'}), 'CPUExecutionProvider']
    return [primary]


def isolated_audit(model_path):
    # A separate process prevents parsing the raw ONNX from inflating RSS baseline.
    completed = subprocess.run([sys.executable, str(Path(__file__).with_name('model_audit.py')),
                                '--model', str(model_path)], capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
    if completed.returncode:
        raise ValueError('ONNX graph audit failed:\n' + completed.stderr[-4000:])
    return json.loads(completed.stdout)


def provider_placement(model_path, feed, primary, threads, output_dir):
    try:
        with tempfile.TemporaryDirectory(prefix='rtdetr_profile_') as folder:
            options = session_options(threads, profiling=True, profile_prefix=Path(folder) / 'ort')
            session = ort.InferenceSession(str(model_path), options, providers=providers_for(primary))
            session.disable_fallback()
            if session.get_providers()[0] != primary:
                raise RuntimeError('Requested provider did not initialize in profiling session')
            session.run(['labels', 'boxes', 'scores'], feed)
            trace = Path(session.end_profiling())
            events = json.loads(trace.read_text(encoding='utf-8'))
            del session
            shutil.copyfile(trace, output_dir / 'runtime_profile.json')
        counts, operators = Counter(), {}
        for event in events:
            args = event.get('args', {})
            provider = args.get('provider')
            if event.get('cat') == 'Node' and event.get('name', '').endswith('_kernel_time') and provider:
                counts[provider] += 1
                operators.setdefault(provider, Counter())[args.get('op_name', 'unknown')] += 1
        return {'status': 'recorded' if counts else 'no_kernel_events',
                'requested_provider': primary, 'kernel_event_counts': dict(counts),
                'operator_counts_by_provider': {key: dict(value) for key, value in operators.items()},
                'cuda_uses_cpu_nodes': bool(counts['CPUExecutionProvider']) if primary == 'CUDAExecutionProvider' and counts else None,
                'note': 'Separate one-image profiling session, excluded from latency/RSS measurements. Counts are kernel events, not percentages of runtime or proof of numeric precision.'}
    except Exception as error:
        return {'status': 'unavailable', 'requested_provider': primary, 'kernel_event_counts': {},
                'cuda_uses_cpu_nodes': None, 'error': str(error)}


def evaluate(model_path, output_dir, provider='cpu', preview_count=3, data_dir=None,
             max_images=None, repeats=1, threads=4):
    model_path, output_dir = Path(model_path).resolve(), Path(output_dir).resolve()
    data = Path(data_dir).resolve() if data_dir is not None else ROOT / 'data/coco'
    if not model_path.is_file() or model_path.suffix.lower() != '.onnx':
        raise ValueError('Upload a self-contained .onnx model file')
    if output_dir.exists():
        raise FileExistsError(f'Choose a new result directory: {output_dir}')
    if provider not in ('cpu', 'cuda') or repeats < 1 or threads < 1 or preview_count < 0:
        raise ValueError('Invalid provider/repeats/threads/preview_count')
    prepare_data.verify(data_dir=data)
    protocol_path = ROOT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    ann_path = data / 'annotations/evaluation.json'
    ann = json.loads(ann_path.read_text(encoding='utf-8'))
    images = sorted(ann['images'], key=lambda item: item['file_name'])
    if len(images) != protocol['evaluation_images']:
        raise ValueError('Unexpected evaluation image count')
    if max_images is not None:
        if not 1 <= max_images <= len(images):
            raise ValueError('max_images must be between 1 and the full evaluation count')
        images = images[:max_images]
    category_ids = [item['id'] for item in sorted(ann['categories'], key=lambda item: item['id'])]
    if len(category_ids) != protocol['category_count']:
        raise ValueError('Unexpected category count')
    audit = isolated_audit(model_path)
    primary = 'CUDAExecutionProvider' if provider == 'cuda' else 'CPUExecutionProvider'
    if primary not in ort.get_available_providers():
        raise RuntimeError(f'{primary} unavailable: {ort.get_available_providers()}')
    if provider == 'cuda':
        ort.preload_dlls(directory='')
    output_dir.mkdir(parents=True)
    memory = MemorySampler(gpu=provider == 'cuda')
    memory.start()
    try:
        started = time.perf_counter()
        session = ort.InferenceSession(str(model_path), session_options(threads), providers=providers_for(primary))
        session.disable_fallback()
        if session.get_providers()[0] != primary:
            raise RuntimeError(f'Requested provider {primary}, got {session.get_providers()}')
        load_s = time.perf_counter() - started
        effective_options = session.get_provider_options()
        image_dtype, size_dtype = check_interface(session, protocol['input_size'])
        warmups = min(protocol['runtime_warmup_images'], len(images))
        for item in images[:warmups]:
            batch, size = load_image(item, protocol['input_size'], image_dtype, size_dtype, data)
            session.run(['labels', 'boxes', 'scores'], {'images': batch, 'orig_target_sizes': size})
        predictions, inference_ms, wall_ms, preview_items = [], [], [], []
        for index, item in enumerate(images):
            wall_start = time.perf_counter()
            batch, size = load_image(item, protocol['input_size'], image_dtype, size_dtype, data)
            feed = {'images': batch, 'orig_target_sizes': size}
            for repeat in range(repeats):
                infer_start = time.perf_counter()
                outputs = session.run(['labels', 'boxes', 'scores'], feed)
                infer_end = time.perf_counter()
                inference_ms.append((infer_end - infer_start) * 1000)
                if repeat == 0:
                    rows = decode(outputs, item, category_ids)
                    predictions.extend(rows)
                    # E2E is one inference, not the sum of latency repetitions.
                    wall_ms.append((time.perf_counter() - wall_start) * 1000)
                    if len(preview_items) < preview_count:
                        preview_items.append((item, rows))
            if (index + 1) % 100 == 0 or index + 1 == len(images):
                print(f'Inference: {index + 1}/{len(images)}', flush=True)
        memory.sample()
    finally:
        memory.finish()
    resource = {
        'model_file_bytes': model_path.stat().st_size,
        'model_file_mib': model_path.stat().st_size / 1048576,
        'model_load_seconds': load_s, 'inference_images_timed': len(images),
        'inference_runs_timed': len(inference_ms), 'latency_repeats': repeats,
        'warmup_images': warmups,
        'inference_ms_mean': statistics.mean(inference_ms),
        'inference_ms_p50': percentile(inference_ms, 50), 'inference_ms_p95': percentile(inference_ms, 95),
        'inference_fps': 1000 / statistics.mean(inference_ms),
        'end_to_end_ms_mean': statistics.mean(wall_ms),
        'end_to_end_ms_p50': percentile(wall_ms, 50), 'end_to_end_ms_p95': percentile(wall_ms, 95),
        'end_to_end_fps': 1000 / statistics.mean(wall_ms), **memory.result(),
    }
    del session
    gc.collect()
    placement = provider_placement(model_path, feed, primary, threads, output_dir)
    (output_dir / 'provider_placement.json').write_text(json.dumps(placement, indent=2), encoding='utf-8')
    scores = evaluate_predictions(ann_path, predictions, threshold=protocol['score_threshold_for_precision_recall'],
                                  image_ids=[item['id'] for item in images])
    (output_dir / 'predictions.json').write_text(json.dumps(predictions, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'quantization_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    with (output_dir / 'class_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores['class_wise'][0]))
        writer.writeheader()
        writer.writerows(scores['class_wise'])
    for index, (item, rows) in enumerate(preview_items, 1):
        save_preview(item, rows, output_dir / f'preview_{index}.jpg', data_dir=data)
    metrics = {key: value for key, value in scores.items() if key != 'class_wise'}
    metrics.update(resource=resource, image_count=len(images), full_evaluation=len(images) == protocol['evaluation_images'])
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    provenance = {
        'model_name': model_path.name, 'model_sha256': sha256(model_path),
        'model_format': 'self-contained ONNX', 'evaluation_annotations_sha256': sha256(ann_path),
        'data_manifest_sha256': sha256(data / 'manifest.json'), 'protocol_sha256': sha256(protocol_path),
        'onnxruntime_version': ort.__version__, 'provider': primary,
        'provider_options': effective_options, 'available_providers': ort.get_available_providers(),
        'provider_placement_status': placement['status'], 'cpu_threads': threads,
        'cuda_tf32': False if provider == 'cuda' else None,
        'python': platform.python_version(), 'platform': platform.platform(),
        'processor': platform.processor(), 'prediction_count': len(predictions),
        'image_count': len(images), 'evaluated_image_ids': [item['id'] for item in images],
        'full_evaluation': metrics['full_evaluation'], 'latency_repeats': repeats,
        'evaluator_sha256': sha256(Path(__file__)),
        'measurement_note': 'Separate warmup pass, then time every selected image. AP uses only first repetition predictions; session.run latency includes CPU/GPU transfers, E2E includes decode/resize/output conversion. Three-image checks are not final accuracy/performance results.',
    }
    (output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f"mAP50:95={metrics['mAP50_95']:.4f}; AP50={metrics['mAP50']:.4f}; size={resource['model_file_mib']:.2f} MiB; inference={resource['inference_ms_p50']:.1f} ms p50")
    print(f'Results: {output_dir}')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--provider', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--preview-count', type=int, default=3)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--max-images', type=int, help='Partial execution check; omit for the fixed 1000-image evaluation')
    parser.add_argument('--repeats', type=int, default=1, help='Latency repetitions per image; AP uses one prediction set')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    evaluate(args.model, args.output, args.provider, args.preview_count, args.data_dir,
             args.max_images, args.repeats, args.threads)


if __name__ == '__main__':
    main()
