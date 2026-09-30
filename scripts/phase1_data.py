"""Make a fixed, verifiable BDD100K evaluation subset for the team."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from common import ROOT, config, dump, path, sha256
from data import CATEGORIES

MANIFEST = path('data/phase1_eval_manifest.json')
ANNOTATIONS = path('data/annotations/phase1_eval.json')


def canonical_digest(images, annotations):
    """Hash annotation content without machine-specific source paths or JSON layout."""
    names = {image['id']: image['file_name'] for image in images}
    content = {
        'images': sorted((i['file_name'], i['width'], i['height']) for i in images),
        'boxes': sorted((names[a['image_id']], a['category_id'], a['bbox'], a['iscrowd'])
                        for a in annotations),
    }
    encoded = json.dumps(content, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def build(config_file='configs/finetune.yaml', count=1000, seed=42):
    c = config(config_file)
    if count < 1:
        raise ValueError('Evaluation count must be positive')
    source = json.loads(path(c['val_annotations']).read_text(encoding='utf-8'))
    if source['categories'] != CATEGORIES:
        raise ValueError('Expected the shared BDD100K 10-class mapping')
    images_by_name = {i['file_name']: i for i in source['images']}
    if len(images_by_name) != len(source['images']):
        raise ValueError('Duplicate evaluation image names')
    if count > len(images_by_name):
        raise ValueError('Evaluation count exceeds validation size')
    selected = sorted(random.Random(seed).sample(sorted(images_by_name), count))
    selected_images = [images_by_name[name] for name in selected]
    ids = {i['id'] for i in selected_images}
    selected_annotations = [a for a in source['annotations'] if a['image_id'] in ids]
    calibration = json.loads(path('data/calibration_manifest.json').read_text(encoding='utf-8'))
    if set(selected) & set(calibration['files']):
        raise ValueError('Calibration and evaluation images overlap')
    for image in selected_images:
        if not (path(c['val_images']) / image['file_name']).is_file():
            raise FileNotFoundError(image['file_name'])
    counts = Counter(a['category_id'] for a in selected_annotations if not a.get('iscrowd'))
    result = dict(info={'source_split': 'official validation', 'selection_seed': seed,
                        'note': 'Shared phase-one dev subset, not a held-out test set'},
                  images=selected_images, annotations=selected_annotations,
                  categories=source['categories'])
    digest = canonical_digest(selected_images, selected_annotations)
    manifest = dict(seed=seed, count=count, files=selected,
                    files_sha256=hashlib.sha256('\n'.join(selected).encode()).hexdigest(),
                    canonical_annotations_sha256=digest,
                    calibration_files_sha256=hashlib.sha256(
                        '\n'.join(sorted(calibration['files'])).encode()).hexdigest(),
                    category_counts={str(k): counts[k] for k in range(1, 11)},
                    source_val_annotations_sha256=sha256(path(c['val_annotations'])))
    if MANIFEST.exists() or ANNOTATIONS.exists():
        if not (MANIFEST.exists() and ANNOTATIONS.exists()):
            raise FileExistsError('Only one phase-one output exists; inspect before rebuilding')
        current = json.loads(MANIFEST.read_text(encoding='utf-8'))
        saved = json.loads(ANNOTATIONS.read_text(encoding='utf-8'))
        if current['canonical_annotations_sha256'] != digest or canonical_digest(
                saved['images'], saved['annotations']) != digest:
            raise ValueError('Existing phase-one evaluation selection differs')
        print('Verified existing evaluation subset:', digest)
        return current
    dump(ANNOTATIONS, result)
    dump(MANIFEST, manifest)
    print('Created evaluation subset:', count, 'images; content SHA256:', digest)
    return manifest


def verify(config_file='configs/finetune.yaml'):
    c = config(config_file)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    subset = json.loads(ANNOTATIONS.read_text(encoding='utf-8'))
    calibration = json.loads(path('data/calibration_manifest.json').read_text(encoding='utf-8'))
    if len(subset['images']) != manifest['count']:
        raise ValueError('Evaluation image count changed')
    if sorted(i['file_name'] for i in subset['images']) != manifest['files']:
        raise ValueError('Evaluation image list changed')
    if canonical_digest(subset['images'], subset['annotations']) != manifest['canonical_annotations_sha256']:
        raise ValueError('Evaluation annotation content changed')
    if manifest['calibration_files_sha256'] != hashlib.sha256(
            '\n'.join(sorted(calibration['files'])).encode()).hexdigest():
        raise ValueError('Calibration image list changed')
    if set(manifest['files']) & set(calibration['files']):
        raise ValueError('Calibration and evaluation overlap')
    if sha256(path(c['val_annotations'])) != manifest['source_val_annotations_sha256']:
        raise ValueError('Source validation annotations changed')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build', 'verify'])
    parser.add_argument('--config', default='configs/finetune.yaml')
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.config, args.count, args.seed)
    else:
        print(json.dumps(verify(args.config), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
