import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import torch
from common import config,path,dump,seed_all,device,sha256
from data import DetectionDataset,verify_prepared,collate
from official import build,load_initial

c=config('configs/finetune.yaml')
seed_all(c['seed'])
manifest=verify_prepared(c)
calibration=json.loads(path('data/calibration_manifest.json').read_text(encoding='utf-8'))
train=json.loads(path(c['train_annotations']).read_text(encoding='utf-8'))
train_names={im['file_name'] for im in train['images']}
train_count=len(train_names)
del train
val=DetectionDataset(path(c['val_images']),path(c['val_annotations']))
val_names={im['file_name'] for im in val.images}
assert train_count==70000 and len(val_names)==10000
assert not train_names & val_names
assert len(calibration['files'])==512
assert set(calibration['files'])<=train_names
assert not set(calibration['files']) & val_names
image,target=val[0]
assert image.shape==(3,640,640)
assert target['labels'].min()>=0 and target['labels'].max()<10
assert torch.isfinite(target['boxes']).all()
assert (target['boxes'][:,2:]>0).all()
dev=device(c['device'])
model,criterion,post=build()
load_initial(model,path(c['initial_checkpoint']))
model.eval().to(dev)
with torch.inference_mode():
    output=model(image.unsqueeze(0).to(dev))
    result=post(output,target['orig_size'].unsqueeze(0).to(dev))[0]
assert tuple(output['pred_logits'].shape)==(1,300,10)
assert torch.isfinite(output['pred_logits']).all()
report=dict(status='ready',train_images=train_count,val_images=len(val_names),
    train_val_overlap=0,calibration_images=len(calibration['files']),calibration_val_overlap=0,
    real_image_forward_pass='passed',device=str(dev),input=[1,3,640,640],
    tested_image=val.images[0]['file_name'],detections=len(result['scores']),
    trained_on_bdd100k=False,source_release='Berkeley official original Labels, not Detection 2020',
    box_coordinate_convention=c['bbox_coordinate_convention'],
    dataset_manifest_sha256=sha256(path('data/dataset_manifest.json')),
    class_counts={'train':manifest['train']['counts'],'val':manifest['val']['counts']},
    filtered_boxes={'train':manifest['train']['skipped'],'val':manifest['val']['skipped']},
    completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
dump(path('reports/dataset_ready.json'),report)
print(json.dumps(report,ensure_ascii=False,indent=2))
