"""Small shared file and reproducibility helpers for the COCO ONNX PTQ workflow."""
import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def path(value):
    target = Path(value)
    return target if target.is_absolute() else ROOT / target


def sha256(filename):
    digest = hashlib.sha256()
    with open(filename, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(filename):
    """Hash code/config identically across LF and CRLF Git checkouts."""
    return hashlib.sha256(Path(filename).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def dump(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(filename)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
