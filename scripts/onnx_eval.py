"""Measure real ONNX PTQ models on the fixed COCO subset using ONNX Runtime CPU."""
import argparse
import csv
import gc
import math
import platform
import statistics
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch
from tqdm import tqdm

from coco_protocol import COCO_CATEGORY_IDS, loader, settings, verified_weight, verify_source
from common import ROOT, dump, path, sha256
from metrics import evaluate_predictions
from onnx_graph import audit_model, model_properties

import sys
sys.path.insert(0, str(ROOT))
import prepare_data


INT8_RUNTIME_OPS = {'QLinearConv', 'QLinearMatMul', 'MatMulInteger', 'ConvInteger',
                    'QGemm', 'MatMulIntegerToFloat', 'DynamicQuantizeMatMul'}


def cpu_model():
    source = Path('/proc/cpuinfo')
    if source.is_file():
        for line in source.read_text(encoding='utf-8').splitlines():
            if line.startswith('model name'):
                return line.partition(':')[2].strip()
    return platform.processor() or platform.machine()


class MemorySampler:
    def __init__(self):
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.peak = 0
        self.worker = threading.Thread(target=self._watch, daemon=True)

    def _watch(self):
        while not self.stop.wait(.02):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.peak = self.process.memory_info().rss
        self.worker.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.worker.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def timing(values):
    values = sorted(values)
    return {'mean': statistics.mean(values), 'p50': statistics.median(values),
            'p95': values[math.ceil(.95 * len(values)) - 1]}


def new_session(filename, optimized_file):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(optimized_file)
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(filename), sess_options=options, providers=['CPUExecutionProvider'])
    load_seconds = time.perf_counter() - t0
    if session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('Expected CPU-only ONNX Runtime execution')
    optimized = onnx.load(str(optimized_file))
    counts = Counter(node.op_type for node in optimized.graph.node)
    return session, load_seconds, dict(counts)


def detections(labels, boxes, scores, image_id):
    result = []
    for label, box, score in zip(labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()):
        if not 0 <= int(label) < 80 or not np.isfinite(box).all() or not math.isfinite(score):
            raise ValueError('Model returned invalid label, box or score')
        x1, y1, x2, y2 = box
        result.append({'image_id': image_id, 'category_id': COCO_CATEGORY_IDS[int(label)],
                       'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': score})
    return result


def evaluate(model_path, run_id, warmup):
    cfg = settings()
    prepare_data.verify()
    verify_source()
    _, checkpoint_hash = verified_weight(cfg)
    model_path = Path(model_path).resolve()
    properties = model_properties(model_path)
    kind = properties.get('ptq_kind')
    if kind not in {'onnx_fp32', 'onnx_int8', 'onnx_int4'}:
        raise ValueError('Use an ONNX model from onnx_real_ptq.py')
    expected = {'checkpoint_sha256': checkpoint_hash,
                'data_manifest_sha256': sha256(path('data/coco/manifest.json')),
                'config_sha256': sha256(ROOT / 'configs/coco_ptq.yaml')}
    if any(properties.get(key) != value for key, value in expected.items()):
        raise ValueError('Uploaded model does not match this COCO checkpoint/data/protocol')
    output = ROOT / 'runs/onnx_eval' / run_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            session, load_seconds, optimized_ops = new_session(model_path, Path(temporary) / 'optimized.onnx')
        if kind == 'onnx_int8' and not any(optimized_ops.get(op, 0) for op in INT8_RUNTIME_OPS):
            raise RuntimeError('INT8 QDQ graph did not fuse into an integer ONNX Runtime operator; cannot claim real INT8 execution')
        if kind == 'onnx_int4' and optimized_ops.get('MatMulNBits', 0) == 0:
            raise RuntimeError('Packed INT4 MatMulNBits did not survive ONNX Runtime optimization')
        rss_after_load = psutil.Process().memory_info().rss
        batches = loader(cfg['evaluation_images'], cfg['evaluation_annotations'])
        iterator = iter(batches)
        for _ in range(warmup):
            images, _, sizes = next(iterator)
            session.run(None, {'images': images.numpy(), 'orig_target_sizes': torch.stack(sizes).T.numpy()})
        del iterator
        gc.collect()
        samples = {key: [] for key in ('decode_ms', 'runtime_ms', 'total_ms')}
        predictions = []
        started = time.perf_counter()
        with MemorySampler() as sampler:
            iterator = iter(batches)
            for _ in tqdm(range(len(batches)), desc=f'ONNX {kind}'):
                t0 = time.perf_counter()
                images, image_ids, sizes = next(iterator)
                t1 = time.perf_counter()
                labels, boxes, scores = session.run(None, {'images': images.numpy(),
                    'orig_target_sizes': torch.stack(sizes).T.numpy()})
                t2 = time.perf_counter()
                predictions.extend(detections(labels, boxes, scores, int(image_ids[0])))
                t3 = time.perf_counter()
                samples['decode_ms'].append((t1 - t0) * 1000)
                samples['runtime_ms'].append((t2 - t1) * 1000)
                samples['total_ms'].append((t3 - t0) * 1000)
        wall_seconds = time.perf_counter() - started
        metrics = evaluate_predictions(path(cfg['evaluation_annotations']), predictions)
        profile = {'images': len(batches), 'warmup_images': warmup,
                   'timing_ms_per_image': {key: timing(value) for key, value in samples.items()},
                   'observed_fps_including_python': len(batches) / wall_seconds,
                   'evaluation_wall_seconds': wall_seconds, 'model_load_seconds': load_seconds,
                   'peak_process_rss_bytes': sampler.peak, 'rss_after_load_bytes': rss_after_load,
                   'system_ram_total_bytes': psutil.virtual_memory().total,
                   'onnx_model_bytes': model_path.stat().st_size,
                   'cuda_peak_allocated_bytes': None, 'cuda_peak_reserved_bytes': None}
        dump(output / 'metrics.json', {key: value for key, value in metrics.items() if key != 'class_wise'})
        with (output / 'class_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics['class_wise'][0]))
            writer.writeheader()
            writer.writerows(metrics['class_wise'])
        dump(output / 'profile.json', profile)
        dump(output / 'provenance.json', {'run_id': run_id, 'kind': kind,
             'model_sha256': sha256(model_path), 'model_properties': properties,
             'checkpoint_sha256': checkpoint_hash, 'data_manifest_sha256': expected['data_manifest_sha256'],
             'config_sha256': expected['config_sha256'], 'evaluator_sha256': sha256(Path(__file__)),
             'onnxruntime': ort.__version__, 'onnx': onnx.__version__, 'python': platform.python_version(),
             'platform': platform.platform(), 'cpu': cpu_model(), 'logical_cpu_count': psutil.cpu_count(),
             'providers': session.get_providers(), 'optimized_runtime_op_counts': optimized_ops,
             'artifact_audit': audit_model(model_path),
             'interpretation': 'INT8: static QDQ with CPU integer kernels; INT4: packed weight-only MatMulNBits, remaining operators FP32'})
        print(f'{run_id}: AP={metrics["mAP50_95"]:.4f}; p50={profile["timing_ms_per_image"]["total_ms"]["p50"]:.1f} ms; model={model_path.stat().st_size / 2**20:.1f} MiB')
    except BaseException:
        if not any(output.iterdir()):
            output.rmdir()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum() or not 0 <= args.warmup < 1000:
        raise ValueError('Use a simple run-id and 0 <= warmup < 1000')
    evaluate(args.model, args.run_id, args.warmup)


if __name__ == '__main__':
    main()
