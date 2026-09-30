"""Evaluate COCO AP and Colab runtime resources for official or exported fake-PTQ RT-DETR."""
import argparse
import csv
import gc
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
import torchvision
from tqdm import tqdm

from coco_artifact import unpack
from coco_run import COCO_CATEGORY_IDS, loader, settings, verified_weight, verify_source
from common import ROOT, dump, path, seed_all, sha256, sync
from metrics import evaluate_predictions
from official import build_coco, load_coco_pretrained
from phase1_quant import install, target_modules

sys.path.insert(0, str(ROOT))
import prepare_data


def percentiles(values):
    ordered = sorted(values)
    return {'mean': statistics.mean(ordered), 'p50': statistics.median(ordered),
            'p95': ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)]}


class MemorySampler:
    def __init__(self):
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.peak = 0
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.wait(0.02):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.peak = self.process.memory_info().rss
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


@torch.inference_mode()
def measure(model, postprocessor, batches, dev, warmup):
    iterator = iter(batches)
    for _ in range(warmup):
        images, _, sizes = next(iterator)
        original = torch.stack(sizes).T.to(dev)
        postprocessor(model(images.to(dev)), original)
        sync(dev)
    del iterator
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)
    samples = {key: [] for key in ('decode_ms', 'transfer_ms', 'model_ms', 'postprocess_ms', 'total_ms')}
    predictions = []
    started = time.perf_counter()
    with MemorySampler() as memory:
        iterator = iter(batches)
        for _ in tqdm(range(len(batches)), desc='COCO evaluation'):
            t0 = time.perf_counter()
            images, image_ids, sizes = next(iterator)
            t1 = time.perf_counter()
            images = images.to(dev)
            original = torch.stack(sizes).T.to(dev)
            sync(dev)
            t2 = time.perf_counter()
            outputs = model(images)
            sync(dev)
            t3 = time.perf_counter()
            detections = postprocessor(outputs, original)
            sync(dev)
            t4 = time.perf_counter()
            for image_id, detected in zip(image_ids.tolist(), detections):
                boxes = detected['boxes'].cpu().clone()
                boxes[:, 2:] -= boxes[:, :2]
                for label, box, score in zip(detected['labels'].cpu().tolist(), boxes.tolist(), detected['scores'].cpu().tolist()):
                    predictions.append({'image_id': image_id, 'category_id': COCO_CATEGORY_IDS[label],
                                        'bbox': box, 'score': score})
            t5 = time.perf_counter()
            for key, a, b in [('decode_ms', t0, t1), ('transfer_ms', t1, t2),
                              ('model_ms', t2, t3), ('postprocess_ms', t3, t4), ('total_ms', t0, t5)]:
                samples[key].append((b - a) * 1000)
    elapsed = time.perf_counter() - started
    profile = {'images': len(batches), 'warmup_images': warmup,
               'timing_ms_per_image': {key: percentiles(values) for key, values in samples.items()},
               'observed_fps_including_python': len(batches) / elapsed,
               'evaluation_wall_seconds': elapsed,
               'peak_process_rss_bytes': memory.peak,
               'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(dev) if dev.type == 'cuda' else None,
               'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(dev) if dev.type == 'cuda' else None,
               'system_ram_total_bytes': psutil.virtual_memory().total}
    return predictions, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='official or path to exported .pth artifact')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()
    if not args.run_id.replace('-', '').replace('_', '').isalnum() or args.warmup < 0 or args.warmup >= 1000:
        raise ValueError('Use a simple run-id and 0 <= warmup < 1000')
    cfg = settings()
    prepare_data.verify()
    verify_source()
    checkpoint, checkpoint_hash = verified_weight(cfg)
    manifest_hash = sha256(path('data/coco/manifest.json'))
    config_hash = sha256(ROOT / 'configs/coco_phase1.yaml')
    output = ROOT / 'runs/colab_eval' / args.run_id
    if output.exists():
        raise FileExistsError(output)
    seed_all(cfg['seed'])
    dev = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                       'cpu' if args.device == 'auto' else args.device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    t0 = time.perf_counter()
    model, postprocessor = build_coco()
    coverage = None
    if args.model == 'official':
        load_coco_pretrained(model, checkpoint)
        identity = {'kind': 'official_fp32', 'artifact_sha256': None, 'artifact_bytes': checkpoint.stat().st_size}
    else:
        artifact_file = Path(args.model).resolve()
        artifact = torch.load(artifact_file, map_location='cpu', weights_only=True)
        group, bits, ranges = unpack(model, artifact, checkpoint_hash, manifest_hash, config_hash)
        coverage = {'group': group, 'bits': bits, 'observed_modules': len(ranges)}
        identity = {'kind': 'fake_ptq', 'artifact_sha256': sha256(artifact_file),
                    'artifact_bytes': artifact_file.stat().st_size}
    model.to(dev).eval()
    postprocessor.to(dev).eval()
    if coverage:
        modules = target_modules(model, coverage['group'])
        install(model, modules, ranges, coverage['bits'], weight_already_quantized=True)
    sync(dev)
    load_seconds = time.perf_counter() - t0
    rss_after_load = psutil.Process().memory_info().rss
    batches = loader(cfg['evaluation_images'], cfg['evaluation_annotations'])
    predictions, profile = measure(model, postprocessor, batches, dev, args.warmup)
    metrics = evaluate_predictions(path(cfg['evaluation_annotations']), predictions)
    profile['model_load_seconds'] = load_seconds
    profile['rss_after_load_bytes'] = rss_after_load
    profile['artifact_bytes'] = identity['artifact_bytes']
    output.mkdir(parents=True)
    dump(output / 'metrics.json', {k: v for k, v in metrics.items() if k != 'class_wise'})
    with (output / 'class_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics['class_wise'][0]))
        writer.writeheader()
        writer.writerows(metrics['class_wise'])
    dump(output / 'profile.json', profile)
    dump(output / 'provenance.json', {'run_id': args.run_id, **identity, 'coverage': coverage,
         'checkpoint_sha256': checkpoint_hash, 'data_manifest_sha256': manifest_hash,
         'config_sha256': config_hash, 'device': str(dev), 'gpu': torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None,
         'evaluator_sha256': sha256(ROOT / 'scripts/colab_eval.py'),
         'torch': torch.__version__, 'torchvision': torchvision.__version__, 'python': platform.python_version(),
         'platform': platform.platform(), 'cpu': platform.processor(),
         'method': 'fake quantization, Conv2d/Linear only; integer storage is not packed INT4/INT6; FP32 kernels',
         'latency_scope': 'Colab runtime, batch=1, 640x640, 1000 COCO val2017 subset images'})
    print(f'{args.run_id}: mAP50:95={metrics["mAP50_95"]:.6f}; total p50={profile["timing_ms_per_image"]["total_ms"]["p50"]:.1f}ms; peak RSS={profile["peak_process_rss_bytes"] / 2**20:.1f} MiB')


if __name__ == '__main__':
    main()
