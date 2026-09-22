import json
import math
import random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm
from common import CLASSES, dump, path, sha256

ALIASES = {'person': 'pedestrian', 'motor': 'motorcycle', 'bike': 'bicycle'}
CATEGORIES = [{'id': i + 1, 'name': n} for i, n in enumerate(CLASSES)]

def convert(labels, images, destination, box_convention='scalabel_inclusive'):
    if box_convention not in ('scalabel_inclusive', 'continuous_xyxy'):
        raise ValueError('Unknown box coordinate convention')
    source = json.loads(Path(labels).read_text(encoding='utf-8'))
    frames = source if isinstance(source, list) else source.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('Expected BDD100K/Scalabel frames, not COCO or tracking data.')
    result = dict(info={'source': str(labels), 'sha256': sha256(labels), 'box_convention': box_convention}, images=[], annotations=[], categories=CATEGORIES)
    counts, ignored, seen = Counter(), Counter(), set()
    filtered_boxes = []
    for frame in tqdm(sorted(frames, key=lambda f: f['name']), desc='Convert + check images'):
        name = frame['name']
        if name in seen or Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError(f'Duplicate or unsafe image name: {name}')
        seen.add(name)
        with Image.open(Path(images) / name) as im:
            width, height = im.size
            im.verify()
        image_id = len(result['images']) + 1
        result['images'].append(dict(id=image_id, file_name=name, width=width, height=height))
        for label in frame.get('labels') or []:
            category = ALIASES.get(label['category'], label['category'])
            box = label.get('box2d')
            if box is None:
                ignored['non_box_labels'] += 1
                continue
            if category not in CLASSES:
                raise ValueError(f'Unknown box category {category!r}; check the Detection label release.')
            x1, y1, x2, y2 = [float(box[k]) for k in ('x1', 'y1', 'x2', 'y2')]
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                raise ValueError(f'Nonfinite box: {name}')
            if x2 < x1 or y2 < y1:
                raise ValueError(f'Inverted box: {name}')
            # Official Scalabel Box2D includes the last pixel: width=x2-x1+1.
            # Convert endpoints to continuous XYXY before clipping and XYWH.
            if box_convention == 'scalabel_inclusive':
                x2 += 1.
                y2 += 1.
            x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
            y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
            if x2 <= x1 or y2 <= y1:
                filtered_boxes.append(dict(image=name, label_id=label.get('id'), original_box=box,
                                           reason='No positive area remains inside image after clipping'))
                ignored['boxes_without_area_inside_image'] += 1
                continue
            attrs = label.get('attributes') or {}
            crowd = int(bool(attrs.get('crowd', False) or attrs.get('ignored', False)))
            result['annotations'].append(dict(id=len(result['annotations']) + 1, image_id=image_id,
                category_id=CLASSES.index(category)+1, bbox=[x1, y1, x2-x1, y2-y1],
                area=(x2-x1)*(y2-y1), iscrowd=crowd))
            counts[category] += 1
    dump(destination, result)
    filtered_path = str(destination) + '.filtered_boxes.json'
    dump(filtered_path, filtered_boxes)
    return result, dict(images=len(frames), objects=len(result['annotations']), counts=dict(counts), skipped=dict(ignored),
                        labels_sha256=sha256(labels), annotations_sha256=sha256(destination),
                        filtered_boxes_sha256=sha256(filtered_path))

def prepare(c):
    if path('data/dataset_manifest.json').exists():
        raise FileExistsError('Dataset manifest already exists. Preserve it for reproducibility; use a new project for another dataset release.')
    convention = c['bbox_coordinate_convention']
    train, tr = convert(path(c['train_labels']), path(c['train_images']), path(c['train_annotations']), convention)
    val, va = convert(path(c['val_labels']), path(c['val_images']), path(c['val_annotations']), convention)
    tn = {x['file_name'] for x in train['images']}
    vn = {x['file_name'] for x in val['images']}
    if tn & vn:
        raise ValueError('Train/validation leakage: overlapping filenames')
    # Both published detection label releases occur in common distributions.
    if tr['images'] not in (69863, 70000) or va['images'] != 10000:
        raise ValueError(f'Unexpected official split sizes: {tr["images"]}/{va["images"]}. No random resplitting is allowed.')
    if c['calibration_size'] > len(tn):
        raise ValueError('Calibration subset larger than train')
    selection = sorted(random.Random(c['seed']).sample(sorted(tn), c['calibration_size']))
    selection_set = set(selection)
    selected_ids = {x['id'] for x in train['images'] if x['file_name'] in selection_set}
    calibration = dict(info={'source': 'official train only', 'seed': c['seed']}, categories=CATEGORIES,
        images=[x for x in train['images'] if x['id'] in selected_ids],
        annotations=[x for x in train['annotations'] if x['image_id'] in selected_ids])
    dump(path('data/annotations/calibration.json'), calibration)
    dump(path('data/calibration_manifest.json'), dict(seed=c['seed'], count=len(selection), files=selection,
        train_annotations_sha256=tr['annotations_sha256'], calibration_sha256=sha256(path('data/annotations/calibration.json')),
        policy='Fixed train subset for PTQ; retained in full FP32 training split; never sample validation.'))
    dump(path('data/dataset_manifest.json'), dict(train=tr, val=va, classes=CLASSES,
        calibration_manifest_sha256=sha256(path('data/calibration_manifest.json')),
        source='User-provided official BDD100K Detection splits; no random split',
        box_convention=convention,
        conversion='Configured Box2D convention -> continuous xyxy -> clip -> xywh; aliases normalized; crowd/ignored -> iscrowd',
        image_integrity='All referenced images opened and PIL-verified; download extraction additionally records data/image_sha256.jsonl when available.',
        image_manifest_sha256=sha256(path('data/image_sha256.jsonl')) if path('data/image_sha256.jsonl').exists() else None,
        download_provenance_sha256=sha256(path('reports/dataset_download_provenance.json')) if path('reports/dataset_download_provenance.json').exists() else None))

def verify_prepared(c):
    manifest = json.loads(path('data/dataset_manifest.json').read_text(encoding='utf-8'))
    for split in ('train', 'val'):
        if sha256(path(c[f'{split}_annotations'])) != manifest[split]['annotations_sha256']:
            raise RuntimeError(f'{split} annotation hash changed')
    calibration_path = path('data/calibration_manifest.json')
    if sha256(calibration_path) != manifest['calibration_manifest_sha256']:
        raise RuntimeError('Calibration manifest hash changed')
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    if sha256(path('data/annotations/calibration.json')) != calibration['calibration_sha256']:
        raise RuntimeError('Calibration annotation hash changed')
    return manifest

class DetectionDataset(Dataset):
    def __init__(self, images, annotations, flip=0.):
        self.root = Path(images)
        self.data = json.loads(Path(annotations).read_text(encoding='utf-8'))
        if self.data['categories'] != CATEGORIES:
            raise ValueError('Class mapping must be exactly the documented 10 classes, IDs 1..10.')
        self.images = self.data['images']
        self.annotations = {im['id']: [] for im in self.images}
        for ann in self.data['annotations']:
            if not ann.get('iscrowd', 0):
                self.annotations[ann['image_id']].append(ann)
        self.flip = flip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        with Image.open(self.root / info['file_name']) as img:
            img = img.convert('RGB')
        w, h = img.size
        if (w, h) != (info['width'], info['height']):
            raise ValueError('Image dimensions changed after preparation')
        boxes, labels = [], []
        for ann in self.annotations[info['id']]:
            x, y, bw, bh = ann['bbox']
            boxes.append([(x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h])
            labels.append(ann['category_id'] - 1)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if random.random() < self.flip:
            img = ImageOps.mirror(img)
            boxes[:, 0] = 1 - boxes[:, 0]
        img = img.resize((640, 640), resample=Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div_(255)
        return tensor, dict(boxes=boxes, labels=torch.tensor(labels, dtype=torch.int64),
                            image_id=torch.tensor(info['id']), orig_size=torch.tensor([w, h]))

def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)

def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
