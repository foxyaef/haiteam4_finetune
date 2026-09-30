"""Shared paths and file hashing for the fixed COCO protocol."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(filename):
    digest = hashlib.sha256()
    with open(filename, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
