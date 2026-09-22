#!/usr/bin/env python3
"""BDD100K original Detection downloader, Python 3.9+, standard library only.

Run from your project: python download_bdd100k.py --root .
Downloads pinned original Berkeley images/Labels (NOT Detection 2020), resumes
interrupted transfers, verifies SHA256 and ZIP CRC, extracts train/val only,
and creates labels/official_raw/det_{train,val}.json. It never starts training.
Also creates COCO train/val, fixed seed-42 calibration (512 images), and manifests.
No project code or third-party Python packages required. Allow roughly 20 GB free disk space.
Do not run concurrent copies against the same project. Originals are retained.
"""
ARCHIVES = {
    'bdd100k_labels.zip': (189638612, '7f1f9043c70a6ff0788a323cfb914aeced109a37b7150740d670a347c881394a'),
    'bdd100k_images_100k.zip': (5669071832, 'afd149eadb49657ef5ad7bb6e9d77cc8b7927a576ee2b475b720028274e02784'),
}
SOURCE = 'http://128.32.162.150/bdd100k/'
import argparse,concurrent.futures,hashlib,json,math,os,threading,time,urllib.request,tempfile,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'data/downloads'
CACHE=Path(os.environ.get('BDD100K_DOWNLOAD_CACHE',str(Path(tempfile.gettempdir())/'rtdetr_bdd100k_download_cache')))
BLOCK=64*1024*1024

def status(value):
    value['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    temp=DEST/'download-status.tmp'
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(DEST/'download-status.json')
    print(json.dumps(value),flush=True)

def sha256(p):
    p=Path(p)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def download(url):
    DEST.mkdir(parents=True,exist_ok=True)
    name=url.rsplit('/',1)[1]
    target=DEST/name
    expected_size, expected_hash = ARCHIVES[name]
    if target.exists():
        if target.stat().st_size != expected_size or sha256(target) != expected_hash:
            raise RuntimeError('Existing archive differs from pinned dataset: '+str(target))
        write_json(target.with_suffix('.download.json'), {'url':url,'bytes':expected_size,'sha256':expected_hash})
        print('Already verified:',name,flush=True)
        return
    for attempt in range(30):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as r:
                total=int(r.headers['Content-Length'])
                if total != expected_size: raise RuntimeError('Official archive size changed')
                signature={'url':url,'bytes':total,'etag':r.headers.get('ETag'),'last_modified':r.headers.get('Last-Modified')}
            break
        except Exception as e:
            status({'phase':'waiting_for_network','file':name,'attempt':attempt+1,'error':str(e)})
            if attempt==29:raise
            time.sleep(20)
    record=target.with_suffix('.download.json')
    if target.exists() and record.exists():
        old=json.loads(record.read_text())
        if target.stat().st_size==total and old['sha256']==sha256(target):
            print('Already verified:',name,flush=True)
            return
        raise RuntimeError('Existing archive hash mismatch: '+name)
    CACHE.mkdir(parents=True,exist_ok=True)
    parts=CACHE/(name+'.parts')
    parts.mkdir(exist_ok=True)
    lock=parts/'source.json'
    if lock.exists() and json.loads(lock.read_text())!=signature:
        raise RuntimeError('Remote source changed; keep old parts separately.')
    lock.write_text(json.dumps(signature,indent=2))
    count=math.ceil(total/BLOCK)
    def task(i):
        start=i*BLOCK; end=min(total,(i+1)*BLOCK)-1
        piece=parts/f'{i:04}.part'
        if piece.exists() and piece.stat().st_size==end-start+1:return
        for attempt in range(30):
            try:
                have=piece.stat().st_size if piece.exists() else 0
                if have>end-start+1:raise RuntimeError('Chunk larger than expected')
                if have==end-start+1:return
                resume=start+have
                req=urllib.request.Request(url,headers={'Range':f'bytes={resume}-{end}','User-Agent':'BDD100K-research-downloader/1.0'})
                with urllib.request.urlopen(req,timeout=90) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {resume}-{end}/{total}':
                        raise RuntimeError('Unexpected range response')
                    with piece.open('ab') as f:
                        while b:=r.read(1024*1024):f.write(b)
                if piece.stat().st_size!=end-start+1:raise IOError('Incomplete chunk')
                return
            except Exception:
                if attempt==29:raise
                time.sleep(min(20,2**attempt))
    began=time.monotonic()
    def downloaded_bytes():
        return sum((parts/f'{i:04}.part').stat().st_size for i in range(count) if (parts/f'{i:04}.part').exists())
    initial=downloaded_bytes()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(task,i) for i in range(count)]
        while any(not f.done() for f in futures):
            done=downloaded_bytes()
            elapsed=time.monotonic()-began
            status({'phase':'downloading','file':name,'downloaded_gb':round(done/1e9,3),'total_gb':round(total/1e9,3),'percent':round(100*done/total,1),'elapsed_seconds':round(elapsed),'mb_per_second':round((done-initial)/max(1,elapsed)/1e6,2)})
            time.sleep(15)
        for f in futures:f.result()
    status({'phase':'assembling_and_hashing','file':name,'percent':100})
    temp=CACHE/(name+'.assembling')
    h=hashlib.sha256()
    with temp.open('wb') as out:
        for i in range(count):
            with (parts/f'{i:04}.part').open('rb') as f:
                for b in iter(lambda:f.read(4*1024*1024),b''):out.write(b);h.update(b)
    if temp.stat().st_size!=total:raise IOError('Assembled size mismatch')
    if h.hexdigest()!=expected_hash:raise RuntimeError('SHA256 mismatch: archive was NOT accepted. Preserve cache for inspection.')
    # Only finalized archives are published into the OneDrive project.
    publishing=target.with_suffix('.zip.publishing')
    shutil.copyfile(temp,publishing)
    if sha256(publishing)!=expected_hash:raise RuntimeError('Copied archive hash mismatch')
    publishing.replace(target)
    signature.update(sha256=h.hexdigest(),retrieved_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    record.write_text(json.dumps(signature,indent=2))
    temp.unlink()
    # Delete only our exact numbered chunk files after a verified assembly.
    for i in range(count):(parts/f'{i:04}.part').unlink()
    status({'phase':'complete','file':name,**signature})


"""Normalize official per-image raw labels and safely extract official train/val images."""
import argparse,collections,hashlib,json,os,time,zipfile,zlib
from pathlib import Path,PurePosixPath

DOWNLOADS=ROOT/'data/downloads'
REPORTS=ROOT/'reports'

def write_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
    temp.replace(p)

def archive_provenance(filename):
    p=DOWNLOADS/filename
    record=json.loads(p.with_suffix('.download.json').read_text())
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    if p.stat().st_size!=record['bytes'] or h.hexdigest()!=record['sha256']:
        raise ValueError('Downloaded archive integrity mismatch')
    return record

def labels():
    provenance=archive_provenance('bdd100k_labels.zip')
    aliases={'person':'pedestrian','bike':'bicycle','motor':'motorcycle'}
    classes=['pedestrian','rider','car','truck','bus','train','motorcycle','bicycle','traffic light','traffic sign']
    summary={}
    with zipfile.ZipFile(DOWNLOADS/'bdd100k_labels.zip') as z:
        for split,expected in [('train',70000),('val',10000)]:
            members=sorted(n for n in z.namelist() if n.startswith(f'100k/{split}/') and n.endswith('.json'))
            if len(members)!=expected:raise ValueError(f'{split} labels count {len(members)} != {expected}')
            output=ROOT/f'data/bdd100k/labels/official_raw/det_{split}.json'
            output.parent.mkdir(parents=True,exist_ok=True)
            temp=output.with_suffix('.tmp')
            counts=collections.Counter();empty=0;invalid=[];seen=set();nonbox=0
            with temp.open('w',encoding='utf-8') as f:
                f.write('[')
                for i,member in enumerate(members):
                    raw=json.loads(z.read(member)) # ZIP CRC checked by read
                    stem=PurePosixPath(member).stem
                    if raw['name']!=stem or stem in seen:raise ValueError('Name mismatch or duplicate')
                    seen.add(stem)
                    frames=raw.get('frames',[])
                    if len(frames)!=1:raise ValueError(f'Expected exactly one labeled frame: {member}')
                    objects=frames[0].get('objects') or []
                    boxes=[obj for obj in objects if obj.get('box2d') is not None]
                    nonbox+=len(objects)-len(boxes)
                    for obj in boxes:
                        category=aliases.get(obj['category'],obj['category'])
                        if category not in classes:raise ValueError('Unknown box category '+category)
                        counts[category]+=1
                        b=obj['box2d']
                        if b['x2']<=b['x1'] or b['y2']<=b['y1']:
                            invalid.append({'file':member,'label_id':obj.get('id'),'box':b})
                    if not boxes:empty+=1
                    frame={'name':stem+'.jpg','attributes':raw.get('attributes',{}),'labels':boxes}
                    if i:f.write(',\n')
                    json.dump(frame,f,ensure_ascii=False,separators=(',',':'))
                    if (i+1)%10000==0:print(f'LABELS {split} {i+1}/{expected}',flush=True)
                f.write(']')
            temp.replace(output)
            h=hashlib.sha256(output.read_bytes()).hexdigest()
            summary[split]={'frames':len(members),'empty_images':empty,'box_counts':dict(counts),'non_box_objects_excluded':nonbox,
                            'single_pixel_dimension_boxes':invalid,'normalized_file':str(output.relative_to(ROOT)),'sha256':h}
    write_json(REPORTS/'dataset_download_provenance.json',{'source_page':'http://bdd-data.berkeley.edu/download.html',
        'label_release':'Official Berkeley raw per-image Labels archive; NOT Detection 2020 release',
        'reason':'Detection 2020 button currently serves unrelated JPEG images; use official original 2D box annotations.',
        'transformation':'frames[0].objects with box2d -> labels; preserve box coordinates and label attributes; retain all official train/val images; append .jpg to image name',
        'box_coordinate_convention':'Scalabel inclusive Box2D; conversion width=x2-x1+1, height=y2-y1+1 before clipping',
        'coordinate_reference':'https://github.com/scalabel/scalabel/blob/master/scalabel/label/transforms.py',
        'archive':provenance,'normalized':summary})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def images():
    provenance=archive_provenance('bdd100k_images_100k.zip')
    base=(ROOT/'data/bdd100k/images').resolve()
    base.mkdir(parents=True,exist_ok=True)
    report=ROOT/'data/image_sha256.jsonl'
    temp=report.with_suffix('.tmp')
    counts=collections.Counter();written=0;kept=0
    began=time.monotonic()
    with zipfile.ZipFile(DOWNLOADS/'bdd100k_images_100k.zip') as z,temp.open('w',encoding='utf-8') as manifest:
        members=[i for i in z.infolist() if not i.is_dir() and (i.filename.startswith('100k/train/') or i.filename.startswith('100k/val/'))]
        for i,info in enumerate(members):
            pieces=PurePosixPath(info.filename).parts
            if len(pieces)!=3 or pieces[0]!='100k' or pieces[1] not in ('train','val') or not pieces[2].endswith('.jpg'):
                raise ValueError('Unexpected archive member '+info.filename)
            target=base.joinpath(*pieces).resolve()
            if not target.is_relative_to(base):raise ValueError('Unsafe ZIP path')
            target.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256()
            if target.exists():
                crc=0
                with target.open('rb') as f:
                    for b in iter(lambda:f.read(1024*1024),b''):digest.update(b);crc=zlib.crc32(b,crc)
                if target.stat().st_size!=info.file_size or crc!=info.CRC:
                    raise ValueError('Existing image differs; preserve for inspection: '+str(target))
                kept+=1
            else:
                partial=target.with_suffix('.jpg.part')
                with z.open(info) as src,partial.open('wb') as dst:
                    for b in iter(lambda:src.read(1024*1024),b''):dst.write(b);digest.update(b)
                partial.replace(target)
                written+=1
            counts[pieces[1]]+=1
            manifest.write(json.dumps({'path':str(target.relative_to(ROOT)),'sha256':digest.hexdigest(),'bytes':info.file_size})+'\n')
            if (i+1)%2000==0:
                progress={'phase':'extracting','images':i+1,'total':len(members),'elapsed_seconds':round(time.monotonic()-began)}
                write_json(REPORTS/'dataset-extraction-status.json',progress)
                print(json.dumps(progress),flush=True)
    if counts!={'train':70000,'val':10000}:raise ValueError('Official split count mismatch')
    temp.replace(report)
    write_json(REPORTS/'dataset_images_provenance.json',{'archive':provenance,'extracted':dict(counts),'new_files':written,'verified_existing':kept,
               'test_images':'20,000 retained only in original ZIP; not extracted or used for training/evaluation',
               'image_manifest_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'crc_verified':True})
    write_json(REPORTS/'dataset-extraction-status.json',{'phase':'complete','images':sum(counts.values()),
               'total':len(members),'elapsed_seconds':round(time.monotonic()-began),'crc_verified':True})
    print('EXTRACTION COMPLETE',dict(counts),flush=True)


import random,struct
from collections import Counter
CLASSES=['pedestrian','rider','car','truck','bus','train','motorcycle','bicycle','traffic light','traffic sign']
ALIASES={'person':'pedestrian','motor':'motorcycle','bike':'bicycle'}
CATEGORIES=[{'id':i+1,'name':name} for i,name in enumerate(CLASSES)]
def path(p):return ROOT/Path(p)
def dump(p,value):write_json(Path(p),value)
def tqdm(values,desc=''):
    for i,value in enumerate(values):
        if i%10000==0:print(desc,i,'/',len(values),flush=True)
        yield value
def jpeg_size(p):
    # Read JPEG SOF dimensions without loading decoded pixels or requiring Pillow.
    with p.open('rb') as f:
        if f.read(2)!=b'\xff\xd8':raise ValueError('Not a JPEG: '+str(p))
        while True:
            if f.read(1)!=b'\xff':raise ValueError('Invalid JPEG marker: '+str(p))
            marker=f.read(1)
            while marker==b'\xff':marker=f.read(1)
            if not marker:raise ValueError('Truncated JPEG')
            code=marker[0]
            if code in (0xd9,0xda):raise ValueError('JPEG has no dimensions')
            if code==0x01 or 0xd0<=code<=0xd7:continue
            raw=f.read(2)
            if len(raw)!=2:raise ValueError('Truncated JPEG segment')
            size=struct.unpack('>H',raw)[0]
            if size<2:raise ValueError('Invalid JPEG segment size')
            payload=f.read(size-2)
            if len(payload)!=size-2:raise ValueError('Truncated JPEG segment')
            if code in (0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf):
                if len(payload)<5:raise ValueError('Truncated JPEG dimensions')
                height,width=struct.unpack('>HH',payload[1:5])
                if not width or not height:raise ValueError('Invalid dimensions')
                return width,height
def dataset_config():
    return dict(seed=42,calibration_size=512,bbox_coordinate_convention='scalabel_inclusive',
        train_labels='data/bdd100k/labels/official_raw/det_train.json',
        val_labels='data/bdd100k/labels/official_raw/det_val.json',
        train_images='data/bdd100k/images/100k/train',val_images='data/bdd100k/images/100k/val',
        train_annotations='data/annotations/train.json',val_annotations='data/annotations/val.json')
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
        width, height = jpeg_size(Path(images) / name)
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
        source='Pinned Berkeley original BDD100K Detection splits; no random split',
        box_convention=convention,
        conversion='Configured Box2D convention -> continuous xyxy -> clip -> xywh; aliases normalized; crowd/ignored -> iscrowd',
        image_integrity='Pinned archive SHA256 and extracted image CRC/SHA256 verified; JPEG SOF dimensions parsed without pixel decoding.',
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


def main():
    global ROOT, DEST, CACHE, DOWNLOADS, REPORTS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                        help='Project directory; defaults to the directory containing this script')
    parser.add_argument('--cache-dir', type=Path, help='Persistent resume cache, outside synced folders')
    args=parser.parse_args()
    ROOT=args.root.expanduser().resolve()
    DEST=DOWNLOADS=ROOT/'data/downloads'
    REPORTS=ROOT/'reports'
    CACHE=(args.cache_dir.expanduser().resolve() if args.cache_dir else
           Path(tempfile.gettempdir())/('bdd100k_cache_'+hashlib.sha256(str(ROOT).encode()).hexdigest()[:12]))
    for folder in (DEST, REPORTS, CACHE):folder.mkdir(parents=True,exist_ok=True)
    # OS lock is released even if interrupted; lock file itself may remain.
    with (ROOT/'data/.bdd100k-download.lock').open('a+b') as lock:
        if os.name=='nt':
            import msvcrt
            if lock.tell()==0:lock.write(b'0');lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ARCHIVES:download(SOURCE+name)
        labels()
        images()
        c=dataset_config()
        if (ROOT/'data/dataset_manifest.json').exists():
            existing=verify_prepared(c)
            for split in ('train','val'):
                if sha256(path(c[split+'_labels']))!=existing[split]['labels_sha256']:
                    raise RuntimeError('Existing manifest belongs to different labels; use a new root')
            if existing['box_convention']!=c['bbox_coordinate_convention']:
                raise RuntimeError('Existing manifest uses another coordinate convention')
            print('Existing COCO annotations and calibration hashes verified; preserved.')
        else:
            prepare(c)
        verify_prepared(c)
        write_json(REPORTS/'server-download-complete.json',
                   {'status':'complete','train_images':70000,'val_images':10000,
                    'label_release':'official original Labels, NOT Detection 2020',
                    'calibration_images':512,'seed':42,
                    'training_started':False,'root':str(ROOT)})
        print('DONE: images, normalized labels, COCO train/val, calibration (512), manifests ready.',flush=True)

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted. Run the same command again to resume.',flush=True)
        raise SystemExit(130)
