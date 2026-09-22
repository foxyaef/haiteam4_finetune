"""Normalize official per-image raw labels and safely extract official train/val images."""
import argparse,collections,hashlib,json,os,time,zipfile,zlib
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
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

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['labels','images']);args=parser.parse_args()
    labels() if args.stage=='labels' else images()
