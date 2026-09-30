"""Evaluate the unchanged 80-class COCO-pretrained RT-DETR and fake-PTQ variants."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import ROOT, dump, path, seed_all, sha256, device
from metrics import evaluate_predictions
from official import build_coco, load_coco_pretrained
from phase1_quant import observe, install, target_modules

sys.path.insert(0, str(ROOT))
import prepare_data


# COCO's official 80 category IDs have gaps. Model output indices are contiguous.
COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90,
)


def settings():
    config_path = ROOT / 'configs/coco_phase1.yaml'
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if (cfg['seed'], cfg['image_size'], cfg['calibration_size'], cfg['evaluation_size']) != (42, 640, 512, 1000):
        raise ValueError('COCO phase-one data protocol changed')
    if cfg['weight_bits'] != [8, 6, 4] or cfg['activation_bits'] != [8, 6, 4]:
        raise ValueError('COCO phase-one bit widths changed')
    if cfg['postprocess_top_queries'] != 300 or cfg['coco_max_dets'] != 100:
        raise ValueError('COCO evaluation protocol changed')
    expected = {
        'weight_scheme': 'symmetric_per_output_channel_minmax',
        'activation_scheme': 'asymmetric_per_tensor_minmax',
        'quantizable_modules': ['Conv2d', 'Linear'],
        'calibration_images': 'data/coco/val2017',
        'calibration_annotations': 'data/coco/annotations/calibration.json',
        'evaluation_images': 'data/coco/val2017',
        'evaluation_annotations': 'data/coco/annotations/evaluation.json',
        'checkpoint': 'weights/rtdetr_r50vd_6x_coco_from_paddle.pth',
    }
    if any(cfg[key] != value for key, value in expected.items()):
        raise ValueError('COCO phase-one paths or quantization protocol changed')
    return cfg


class CocoSubset(Dataset):
    def __init__(self, image_root, annotation_file):
        self.root = path(image_root)
        self.data = json.loads(path(annotation_file).read_text(encoding='utf-8'))
        self.images = self.data['images']
        if tuple(sorted(category['id'] for category in self.data['categories'])) != COCO_CATEGORY_IDS:
            raise ValueError('COCO category IDs differ from the official 80-class mapping')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        with Image.open(self.root / info['file_name']) as image:
            rgb = image.convert('RGB')
        width, height = rgb.size
        if (width, height) != (info['width'], info['height']):
            raise ValueError(f'Image dimensions changed: {info["file_name"]}')
        resized = rgb.resize((640, 640), resample=Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float().div_(255)
        return tensor, int(info['id']), (width, height)


def loader(image_root, annotation_file):
    return DataLoader(CocoSubset(image_root, annotation_file), batch_size=1,
                      shuffle=False, num_workers=0)


def verified_weight(cfg):
    filename = path(cfg['checkpoint'])
    if not filename.is_file():
        raise FileNotFoundError(f'Run setup.ps1 to download the COCO pretrained weight: {filename}')
    expected = json.loads((ROOT / 'upstream-lock.json').read_text(encoding='utf-8'))['checkpoint_sha256']
    actual = sha256(filename)
    if actual != expected:
        raise ValueError('COCO pretrained weight differs from upstream-lock.json')
    return filename, actual


def verify_source():
    expected = json.loads((ROOT / 'upstream-lock.json').read_text(encoding='utf-8'))['commit']
    vendor = ROOT / 'vendor/RT-DETR'
    try:
        actual = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
        changed = subprocess.check_output(['git', '-C', str(vendor), 'status', '--porcelain'], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError('Official RT-DETR source is missing; run setup.ps1') from error
    if actual != expected or changed.strip():
        raise ValueError('Official RT-DETR source differs from upstream-lock.json')


@torch.inference_mode()
def predictions_for(model, postprocessor, batches, dev):
    result = []
    for images, image_ids, sizes in tqdm(batches, desc='COCO evaluation'):
        outputs = model(images.to(dev))
        original_sizes = torch.stack(sizes).T.to(dev)
        detections = postprocessor(outputs, original_sizes)
        for image_id, detected in zip(image_ids.tolist(), detections):
            boxes = detected['boxes'].cpu().clone()
            boxes[:, 2:] -= boxes[:, :2]
            for label, box, score in zip(detected['labels'].cpu().tolist(), boxes.tolist(),
                                         detected['scores'].cpu().tolist()):
                result.append({'image_id': image_id,
                               'category_id': COCO_CATEGORY_IDS[label],
                               'bbox': box, 'score': score})
    return result


def evaluate(cfg, group, bits, device_name, save_predictions=False):
    if group == 'baseline' and bits is not None:
        raise ValueError('FP32 baseline has no --bits')
    if group != 'baseline' and bits not in (8, 6, 4):
        raise ValueError('Select --bits 8, 6 or 4 for fake PTQ')
    prepare_data.verify()
    verify_source()
    checkpoint, checkpoint_hash = verified_weight(cfg)
    name = 'fp32' if group == 'baseline' else f'{group}_w{bits}a{bits}'
    output = path(cfg['run_dir']) / name
    if output.exists():
        raise FileExistsError(f'Existing result is preserved: {output}')
    seed_all(cfg['seed'])
    dev = device(device_name)
    model, postprocessor = build_coco()
    load_report = load_coco_pretrained(model, checkpoint)
    model.to(dev).eval()
    postprocessor.to(dev).eval()
    coverage, ranges = None, {}
    if group != 'baseline':
        modules = target_modules(model, group)
        calibration = loader(cfg['calibration_images'], cfg['calibration_annotations'])
        ranges = observe(model, modules, (images.to(dev) for images, _, _ in tqdm(calibration, desc='Calibration')))
        coverage = install(model, modules, ranges, bits)
    evaluation = loader(cfg['evaluation_images'], cfg['evaluation_annotations'])
    predictions = predictions_for(model, postprocessor, evaluation, dev)
    metrics = evaluate_predictions(path(cfg['evaluation_annotations']), predictions)
    output.mkdir(parents=True)
    dump(output / 'metrics.json', metrics)
    with (output / 'class_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics['class_wise'][0]))
        writer.writeheader()
        writer.writerows(metrics['class_wise'])
    if save_predictions:
        dump(output / 'predictions.json', predictions)
    manifest = path('data/coco/manifest.json')
    dump(output / 'provenance.json', {
        'model': 'RT-DETR v1 R50-vd, official COCO pretrained, 80 classes, unchanged',
        'load_report': load_report, 'run_id': name, 'checkpoint_sha256': checkpoint_hash,
        'data_manifest_sha256': sha256(manifest),
        'config_sha256': sha256(ROOT / 'configs/coco_phase1.yaml'),
        'calibration_ranges': ranges, 'coverage': coverage,
        'device': str(dev), 'torch': torch.__version__, 'python': sys.version,
        'interpretation': 'Conv2d/Linear fake quantization accuracy only; no integer deployment timing',
    })
    print(f'{name}: mAP50:95={metrics["mAP50_95"]:.6f}, AP50={metrics["mAP50"]:.6f}, AP75={metrics["mAP75"]:.6f}')


def report(cfg):
    root = path(cfg['run_dir'])
    baseline_file = root / 'fp32/metrics.json'
    if not baseline_file.exists():
        raise FileNotFoundError('Run coco.cmd run --group baseline first')
    baseline = json.loads(baseline_file.read_text(encoding='utf-8'))['mAP50_95']
    baseline_provenance = json.loads((root / 'fp32/provenance.json').read_text(encoding='utf-8'))
    identity = ('checkpoint_sha256', 'data_manifest_sha256', 'config_sha256')
    rows = []
    for group in ('fp32', 'all', 'backbone', 'encoder', 'decoder', 'head'):
        for bits in ((None,) if group == 'fp32' else (8, 6, 4)):
            name = 'fp32' if bits is None else f'{group}_w{bits}a{bits}'
            filename = root / name / 'metrics.json'
            if filename.exists():
                provenance = json.loads((root / name / 'provenance.json').read_text(encoding='utf-8'))
                if any(provenance[key] != baseline_provenance[key] for key in identity):
                    raise ValueError(f'Cannot combine runs from different model/data/config: {name}')
                result = json.loads(filename.read_text(encoding='utf-8'))
                rows.append({'run_id': name, 'status': 'complete',
                             'mAP50_95': result['mAP50_95'],
                             'delta_ap_points': 100 * (baseline - result['mAP50_95']),
                             'AP50': result['mAP50'], 'AP75': result['mAP75']})
            else:
                rows.append({'run_id': name, 'status': 'missing',
                             'mAP50_95': '', 'delta_ap_points': '', 'AP50': '', 'AP75': ''})
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(root / 'summary.csv')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run', 'report'])
    parser.add_argument('--group', choices=['baseline', 'all', 'backbone', 'encoder', 'decoder', 'head'], default='baseline')
    parser.add_argument('--bits', type=int, choices=[8, 6, 4])
    parser.add_argument('--device', choices=['cpu', 'cuda', 'xpu'], default='cpu')
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args(argv)
    cfg = settings()
    if args.command == 'check':
        prepare_data.verify()
        verify_source()
        filename, digest = verified_weight(cfg)
        print(f'COCO data and pretrained model ready: {filename} SHA256={digest}')
    elif args.command == 'report':
        report(cfg)
    else:
        evaluate(cfg, args.group, args.bits, args.device, args.save_predictions)


if __name__ == '__main__':
    main()
