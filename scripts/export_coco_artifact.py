"""Calibrate and export a single portable COCO RT-DETR fake-PTQ model file."""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from coco_artifact import pack
from coco_run import loader, settings, verified_weight, verify_source
from common import ROOT, device, path, seed_all, sha256
from official import build_coco, load_coco_pretrained
from phase1_quant import observe, target_modules

import sys
sys.path.insert(0, str(ROOT))
import prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, choices=['all', 'backbone', 'encoder', 'decoder', 'head'])
    parser.add_argument('--bits', required=True, type=int, choices=[4, 6, 8])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'xpu'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cfg = settings()
    prepare_data.verify()
    verify_source()
    checkpoint, checkpoint_hash = verified_weight(cfg)
    seed_all(cfg['seed'])
    dev = device(args.device)
    model, _ = build_coco()
    load_coco_pretrained(model, checkpoint)
    model.to(dev).eval()
    modules = target_modules(model, args.group)
    calibration = loader(cfg['calibration_images'], cfg['calibration_annotations'])
    ranges = observe(model, modules, (images.to(dev) for images, _, _ in tqdm(calibration, desc='Calibration')))
    artifact = pack(model, args.group, args.bits, ranges, checkpoint_hash,
                    sha256(path('data/coco/manifest.json')), sha256(ROOT / 'configs/coco_phase1.yaml'))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(f'Exported {args.group} W{args.bits}A{args.bits}: {output} ({output.stat().st_size} bytes)')
    print('Integer weights use int8 storage; inference dequantizes to FP32. This is not packed INT4 or integer acceleration.')


if __name__ == '__main__':
    main()
