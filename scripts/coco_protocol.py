"""Shared fixed COCO data and official model identity for the real ONNX PTQ study."""
import json
import subprocess

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import ROOT, path, sha256

COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90,
)


def settings():
    cfg = yaml.safe_load((ROOT / 'configs/coco_ptq.yaml').read_text(encoding='utf-8'))
    expected = {'seed': 42, 'image_size': 640, 'calibration_size': 512, 'evaluation_size': 1000,
                'calibration_images': 'data/coco/val2017',
                'calibration_annotations': 'data/coco/annotations/calibration.json',
                'evaluation_images': 'data/coco/val2017',
                'evaluation_annotations': 'data/coco/annotations/evaluation.json',
                'checkpoint': 'weights/rtdetr_r50vd_6x_coco_from_paddle.pth'}
    if cfg != expected:
        raise ValueError('The fixed COCO PTQ protocol has changed')
    return cfg


class CocoSubset(Dataset):
    def __init__(self, image_root, annotation_file):
        self.root = path(image_root)
        data = json.loads(path(annotation_file).read_text(encoding='utf-8'))
        self.images = data['images']
        categories = data['categories']
        if tuple(sorted(category['id'] for category in categories)) != COCO_CATEGORY_IDS:
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
        raise FileNotFoundError(f'Run scripts/colab_setup.py to download the COCO weight: {filename}')
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
        raise RuntimeError('Official RT-DETR source is missing; run scripts/colab_setup.py') from error
    if actual != expected or changed.strip():
        raise ValueError('Official RT-DETR source differs from upstream-lock.json')
