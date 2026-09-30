#!/usr/bin/env python3
"""Download and organize the COCO 2017 data used by the pretrained-model study.

Standard-library only. Run from any working directory: python prepare_data.py
Downloads the official val images and train/val annotations archives, but extracts
only 512 calibration and 1,000 evaluation val images. No model is trained here.
"""

import argparse
import hashlib
import json
import random
import shutil
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/coco'
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
SEED = PROTOCOL['seed']
CALIBRATION_COUNT = PROTOCOL['calibration_images']
EVALUATION_COUNT = PROTOCOL['evaluation_images']
ARCHIVES = {
    'annotations_trainval2017.zip': {
        'url': 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'bytes': 252907541,
        'sha256': '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268',
    },
    'val2017.zip': {
        'url': 'https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip',
        'bytes': 815585330,
        'sha256': '4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05',
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value, migrate_annotation_hashes=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8')
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing != value:
            if not migrate_annotation_hashes:
                raise ValueError(f'Existing file has different contents: {path}')
            adjusted = json.loads(json.dumps(existing))
            for name in ('calibration', 'evaluation'):
                adjusted['subsets'][name]['annotation_sha256'] = value['subsets'][name]['annotation_sha256']
            if adjusted != value:
                raise ValueError(f'Existing file has different contents: {path}')
        if path.read_bytes() == canonical:
            return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(canonical)
    temporary.replace(path)


def download(name):
    spec = ARCHIVES[name]
    target = DATA / 'downloads' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != spec['bytes'] or sha256(target) != spec['sha256']:
            raise ValueError(f'Existing archive failed the pinned SHA256: {target}')
        print('Verified existing archive:', target, flush=True)
        return target

    partial = target.with_suffix('.zip.part')
    if partial.exists() and partial.stat().st_size == spec['bytes']:
        if sha256(partial) != spec['sha256']:
            raise ValueError(f'Completed partial archive failed SHA256: {partial}')
        partial.replace(target)
        print('Verified completed partial archive:', target, flush=True)
        return target
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > spec['bytes']:
            raise ValueError(f'Partial archive is too large: {partial}')
        headers = {'User-Agent': 'RT-DETR-COCO-research/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            request = urllib.request.Request(spec['url'], headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset:
                    expected_range = f'bytes {offset}-{spec["bytes"] - 1}/{spec["bytes"]}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                        raise ValueError('Server did not honor the requested byte range; partial file preserved')
                elif response.status != 200:
                    raise ValueError(f'Unexpected HTTP status {response.status}')
                with partial.open('ab' if offset else 'wb') as output:
                    while block := response.read(4 * 1024 * 1024):
                        output.write(block)
            if partial.stat().st_size != spec['bytes']:
                raise IOError(f'Incomplete download: {partial.stat().st_size}/{spec["bytes"]} bytes')
            if sha256(partial) != spec['sha256']:
                raise ValueError(f'Downloaded archive failed the pinned SHA256: {partial}')
            partial.replace(target)
            print('Downloaded and verified:', target, flush=True)
            return target
        except (OSError, IOError) as error:
            if attempt == 4:
                raise RuntimeError(f'Download failed; rerun to resume {partial}') from error
            print(f'Download interrupted ({error}); retry {attempt + 1}/5', flush=True)
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f'Download failed: {name}')


def selected_subsets(source):
    images = source['images']
    categories = source['categories']
    if len(images) != 5000 or len(categories) != 80:
        raise ValueError('Expected COCO 2017 val: 5,000 images and 80 categories')
    ordered = sorted(images, key=lambda image: image['file_name'])
    if len({image['file_name'] for image in ordered}) != len(ordered):
        raise ValueError('Duplicate COCO image filenames')
    if any(Path(image['file_name']).name != image['file_name'] for image in ordered):
        raise ValueError('Unexpected image path in COCO annotations')
    chosen = random.Random(SEED).sample(ordered, CALIBRATION_COUNT + EVALUATION_COUNT)
    calibration = sorted(chosen[:CALIBRATION_COUNT], key=lambda image: image['file_name'])
    evaluation = sorted(chosen[CALIBRATION_COUNT:], key=lambda image: image['file_name'])
    anns = source['annotations']
    subsets = {}
    for name, subset_images in [('calibration', calibration), ('evaluation', evaluation)]:
        ids = {image['id'] for image in subset_images}
        subsets[name] = {
            'info': {'source': 'COCO 2017 val', 'selection_seed': SEED,
                     'purpose': name, 'note': 'Disjoint research subset; not the full official val result'},
            'licenses': source.get('licenses', []),
            'images': subset_images,
            'annotations': [annotation for annotation in anns if annotation['image_id'] in ids],
            'categories': categories,
        }
    return subsets


def extract_selected(archive, names):
    destination = DATA / 'val2017'
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for index, name in enumerate(sorted(names), 1):
            if Path(name).name != name or not name.endswith('.jpg'):
                raise ValueError(f'Unsafe COCO image name: {name}')
            member = zipped.getinfo(f'val2017/{name}')
            target = destination / name
            if target.exists():
                crc = 0
                with target.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        crc = zlib.crc32(block, crc)
                if target.stat().st_size != member.file_size or crc != member.CRC:
                    raise ValueError(f'Existing COCO image differs from pinned archive: {target}')
            else:
                temporary = target.with_suffix('.jpg.part')
                with zipped.open(member) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                temporary.replace(target)
            if index % 200 == 0 or index == len(names):
                print(f'Images checked: {index}/{len(names)}', flush=True)


def prepare():
    annotation_zip = download('annotations_trainval2017.zip')
    with zipfile.ZipFile(annotation_zip) as zipped:
        source_bytes = zipped.read('annotations/instances_val2017.json')
    source = json.loads(source_bytes)
    subsets = selected_subsets(source)
    names = {image['file_name'] for subset in subsets.values() for image in subset['images']}
    if len(names) != CALIBRATION_COUNT + EVALUATION_COUNT:
        raise ValueError('Calibration and evaluation images overlap')
    image_zip = download('val2017.zip')
    extract_selected(image_zip, names)
    for name, subset in subsets.items():
        save_json(DATA / 'annotations' / f'{name}.json', subset)
    manifest = {
        'model': PROTOCOL['model'],
        'source_split': PROTOCOL['source_split'], 'seed': SEED,
        'calibration_images': CALIBRATION_COUNT, 'evaluation_images': EVALUATION_COUNT,
        'archive_sha256': {name: spec['sha256'] for name, spec in ARCHIVES.items()},
        'annotation_source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'subsets': {name: {
            'files_sha256': hashlib.sha256('\n'.join(image['file_name'] for image in subset['images']).encode()).hexdigest(),
            'annotation_sha256': sha256(DATA / 'annotations' / f'{name}.json'),
        } for name, subset in subsets.items()},
    }
    save_json(DATA / 'manifest.json', manifest, migrate_annotation_hashes=True)
    verify()


def verify():
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    if any(manifest.get(key) != PROTOCOL[key] for key in
           ('model', 'source_split', 'seed', 'calibration_images', 'evaluation_images')):
        raise ValueError('COCO subset differs from protocol.json')
    if manifest['archive_sha256'] != {name: spec['sha256'] for name, spec in ARCHIVES.items()}:
        raise ValueError('COCO archive lock differs from the project')
    for name, expected_count in [('calibration', CALIBRATION_COUNT), ('evaluation', EVALUATION_COUNT)]:
        annotation_path = DATA / 'annotations' / f'{name}.json'
        subset = json.loads(annotation_path.read_text(encoding='utf-8'))
        images = subset['images']
        if len(images) != expected_count or len(subset['categories']) != 80:
            raise ValueError(f'Unexpected {name} subset size or categories')
        expected = manifest['subsets'][name]
        if sha256(annotation_path) != expected['annotation_sha256']:
            raise ValueError(f'{name} annotation hash changed')
        if hashlib.sha256('\n'.join(image['file_name'] for image in images).encode()).hexdigest() != expected['files_sha256']:
            raise ValueError(f'{name} image list changed')
        for image in images:
            if not (DATA / 'val2017' / image['file_name']).is_file():
                raise FileNotFoundError(image['file_name'])
    calibration = {image['id'] for image in json.loads((DATA / 'annotations/calibration.json').read_text())['images']}
    evaluation = {image['id'] for image in json.loads((DATA / 'annotations/evaluation.json').read_text())['images']}
    if calibration & evaluation:
        raise ValueError('Calibration/evaluation overlap')
    print('COCO data ready: val 512 calibration + 1,000 evaluation images')
    print('  data/coco/downloads/      verified source archives')
    print('  data/coco/val2017/        selected JPEG images only')
    print('  data/coco/annotations/    calibration.json, evaluation.json')
    print('  data/coco/manifest.json   selection and archive hashes')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Check an existing prepared COCO subset')
    args = parser.parse_args(argv)
    verify() if args.verify_only else prepare()


if __name__ == '__main__':
    main()
