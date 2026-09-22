import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '29320b6fd828f8e0987a71426cf2d961b09dfed7'
WEIGHT_URL = 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth'
CLASSES = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']

def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    temp.replace(p)

def config(filename):
    import yaml
    c = yaml.safe_load(path(filename).read_text(encoding='utf-8'))
    if c['input_size'] != 640 or c['precision'] != 'fp32':
        raise ValueError('This baseline requires 640x640, FP32.')
    for key in ['batch_size', 'accumulation_steps', 'epochs', 'calibration_size']:
        if c[key] < 1:
            raise ValueError(f'{key} must be positive')
    if c['warmup_epochs'] >= c['epochs']:
        raise ValueError('warmup_epochs must be smaller than epochs')
    return c

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Grid sampling backward may be nondeterministic on accelerators.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

def device(name):
    import torch
    if name == 'xpu' and not torch.xpu.is_available():
        raise RuntimeError('Intel XPU unavailable. Check driver, or set device: cpu explicitly.')
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install the CUDA PyTorch build first.')
    return torch.device(name)

def sync(dev):
    import torch
    if dev.type in ('cuda', 'xpu'):
        getattr(torch, dev.type).synchronize()

def environment(c):
    import torch
    import torchvision
    repo = ROOT / 'vendor/RT-DETR'
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != COMMIT:
        raise RuntimeError(f'Unexpected upstream commit: {commit}')
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True)
    if dirty.strip():
        raise RuntimeError('Upstream source changed; use a clean pinned repository.')
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                torchvision=torchvision.__version__, repo_commit=commit,
                config=c, weight_url=WEIGHT_URL, classes=CLASSES,
                xpu=torch.xpu.get_device_name(0) if torch.xpu.is_available() else None,
                cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                deterministic='warn_only; bitwise reproducibility is not guaranteed',
                code_hashes={str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'scripts').glob('*.py'))},
                packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines())
