"""Run the fixed phase-one FP32 / one-component fake-PTQ accuracy experiments."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import ROOT, config, path, dump, seed_all, sha256, device
from data import DetectionDataset, collate, verify_prepared
from metrics import evaluate_predictions
from official import build, load_trained
from phase1_data import verify as verify_eval
from phase1_quant import observe, install, target_modules
from run import validate_checkpoint

LOCK = ROOT / 'configs/phase1_checkpoint.json'


def phase_config():
    value = yaml.safe_load((ROOT / 'configs/phase1.yaml').read_text(encoding='utf-8'))
    if value['weight_bits'] != value['activation_bits'] or value['weight_bits'] != [8, 6, 4]:
        raise ValueError('First-stage protocol requires matching W/A widths 8, 6, 4')
    if value['evaluation_size'] != 1000 or value['calibration_size'] != 512 or value['image_size'] != 640:
        raise ValueError('First-stage evaluation, calibration and input sizes changed')
    expected = dict(weight_scheme='symmetric_per_output_channel_minmax',
                    activation_scheme='asymmetric_per_tensor_minmax',
                    activation_observation='fp32_inputs',
                    quantizable_modules=['Conv2d','Linear'],
                    calibration_batch_size=1, evaluation_batch_size=1,
                    postprocess_top_queries=300, coco_max_dets=100)
    for key, setting in expected.items():
        if value[key] != setting:
            raise ValueError(f'Unsupported first-stage protocol setting: {key}')
    return value


def checkpoint(c, filename, require_lock=True):
    ckpt = path(filename or str(path(c['output']) / 'best.pth'))
    validate_checkpoint(ckpt)
    digest = sha256(ckpt)
    if require_lock:
        lock = json.loads(LOCK.read_text(encoding='utf-8'))
        if lock['sha256'] != digest:
            raise ValueError('Checkpoint differs from the team lock file')
    return ckpt, digest


def make_lock(c, filename):
    ckpt, digest = checkpoint(c, filename, False)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    if 'model' not in state or 'epoch' not in state:
        raise ValueError('Expected a BDD100K fine-tuned RT-DETR checkpoint, not COCO pretrained weights')
    if LOCK.exists():
        current = json.loads(LOCK.read_text(encoding='utf-8'))
        if current['sha256'] != digest:
            raise FileExistsError('Team checkpoint lock already exists with a different SHA256')
    else:
        dump(LOCK, {'sha256': digest, 'model': 'RT-DETR v1 R50-vd',
                    'checkpoint_file': ckpt.name, 'epoch': int(state['epoch']),
                    'instruction': 'Commit this lock; share checkpoint and its .sha256.json sidecar separately.'})
    print('Team checkpoint SHA256:', digest)


def data_loader(images, annotations):
    ds = DetectionDataset(images, annotations, flip=0.)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)


@torch.inference_mode()
def predictions_for(model, post, dl, dev):
    records = []
    model.eval()
    for images, targets in tqdm(dl, desc='Phase one evaluation'):
        output = model(images.to(dev))
        sizes = torch.stack([t['orig_size'].to(dev) for t in targets])
        detections = post(output, sizes)
        for target, det in zip(targets, detections):
            boxes = det['boxes'].cpu().clone()
            boxes[:, 2:] -= boxes[:, :2]
            for label, box, score in zip(det['labels'].cpu().tolist(), boxes.tolist(),
                                          det['scores'].cpu().tolist()):
                records.append({'image_id': int(target['image_id'].item()),
                                'category_id': label + 1, 'bbox': box, 'score': score})
    return records


def run(c, pc, group, bits, filename, device_name=None, save_predictions=False):
    if group != 'baseline' and (group not in ('all','backbone','encoder','decoder','head') or bits not in (8,6,4)):
        raise ValueError('Specify baseline or a valid group with --bits 8, 6 or 4')
    if group == 'baseline' and bits is not None:
        raise ValueError('Baseline has no --bits')
    if group != 'baseline' and bits is None:
        raise ValueError('Quantized runs require --bits 8, 6 or 4')
    verify_prepared(c)
    shared_eval = verify_eval()
    if pc['evaluation_size'] != shared_eval['count']:
        raise ValueError('Evaluation manifest does not match phase-one protocol')
    calibration = json.loads(path('data/calibration_manifest.json').read_text(encoding='utf-8'))
    if calibration['count'] != pc['calibration_size']:
        raise ValueError('Calibration manifest does not match phase-one protocol')
    ckpt, digest = checkpoint(c, filename)
    run_id = 'fp32' if group == 'baseline' else f'{group}_w{bits}a{bits}'
    out = path(pc['run_dir']) / run_id
    if out.exists():
        raise FileExistsError(f'Result exists; preserve it: {out}')
    seed_all(pc['seed'])
    dev = device(device_name or c['device'])
    model, _, post = build()
    load_trained(model, ckpt)
    model.to(dev).eval()
    coverage = None
    ranges = {}
    if group != 'baseline':
        modules = target_modules(model, group)
        if not modules:
            raise RuntimeError(f'No quantizable Conv2d/Linear module in {group}')
        calib_dl = data_loader(path(c['train_images']), path('data/annotations/calibration.json'))
        ranges = observe(model, modules, (images.to(dev) for images, _ in tqdm(calib_dl, desc='FP32 calibration')))
        coverage = install(model, modules, ranges, bits)
    eval_dl = data_loader(path(c['val_images']), path('data/annotations/phase1_eval.json'))
    predictions = predictions_for(model, post, eval_dl, dev)
    metrics = evaluate_predictions(path('data/annotations/phase1_eval.json'), predictions, c['score_threshold'])
    out.mkdir(parents=True)
    dump(out / 'metrics.json', metrics)
    with (out / 'class_metrics.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics['class_wise'][0]))
        writer.writeheader(); writer.writerows(metrics['class_wise'])
    if save_predictions:
        dump(out / 'predictions.json', predictions)
    dump(out / 'provenance.json', {
        'run_id': run_id, 'checkpoint_sha256': digest, 'checkpoint_file': str(ckpt),
        'phase1_config_sha256': sha256(ROOT / 'configs/phase1.yaml'),
        'finetune_config_sha256': sha256(ROOT / 'configs/finetune.yaml'),
        'eval_canonical_sha256': shared_eval['canonical_annotations_sha256'],
        'calibration_files_sha256': shared_eval['calibration_files_sha256'],
        'calibration_ranges': ranges, 'coverage': coverage, 'device': str(dev),
        'torch': torch.__version__, 'python': sys.version,
        'interpretation': 'fake quantization accuracy only; no INT4/INT6 hardware timing',
    })
    print(f'{run_id}: mAP50:95={metrics["mAP50_95"]:.6f}; AP50={metrics["mAP50"]:.6f}; AP75={metrics["mAP75"]:.6f}')
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['lock-checkpoint', 'run'])
    parser.add_argument('--group', default='baseline', choices=['baseline','all','backbone','encoder','decoder','head'])
    parser.add_argument('--bits', type=int, choices=[8,6,4])
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', choices=['cpu','cuda','xpu'])
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args()
    c, pc = config('configs/finetune.yaml'), phase_config()
    if args.command == 'lock-checkpoint':
        make_lock(c, args.checkpoint)
    else:
        run(c, pc, args.group, args.bits, args.checkpoint, args.device, args.save_predictions)


if __name__ == '__main__':
    main()
