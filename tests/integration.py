"""Real 640x640 model on synthetic images: epoch checkpoint -> resume -> evaluation.

This uses a synthetic manifest in memory, never a real BDD100K manifest. Artifacts
are kept under runs/integration_test and labeled synthetic.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run
from common import config, dump, path, sha256
from data import CATEGORIES

class StopAfterEpochOne(Exception):
    pass

def main():
    c = config('configs/finetune.yaml')
    out = path('runs/integration_test')
    if (out/'result.json').exists():
        raise FileExistsError('Integration result exists; preserve it instead of overwriting.')
    data = path('runs/integration_fixture')
    data.mkdir(parents=True,exist_ok=True)
    coco=dict(info={'synthetic':True},categories=CATEGORIES,images=[],annotations=[])
    for i in range(2):
        im=Image.new('RGB',(1280,720),(60+i*20,80,100))
        ImageDraw.Draw(im).rectangle((100,100,300,500),fill=(180,90,60))
        im.save(data/f'{i}.jpg')
        coco['images'].append(dict(id=i+1,file_name=f'{i}.jpg',width=1280,height=720))
        coco['annotations'].append(dict(id=i+1,image_id=i+1,category_id=1+i*9,bbox=[100,100,200,400],area=80000,iscrowd=0))
    dump(data/'annotations.json',coco)
    c.update(epochs=2,warmup_epochs=0,batch_size=1,accumulation_steps=2,early_stopping_patience=0,
             output=str(out),train_images=str(data),val_images=str(data),
             train_annotations=str(data/'annotations.json'),val_annotations=str(data/'annotations.json'))
    manifest={'synthetic':True,'annotations_sha256':sha256(data/'annotations.json')}
    run.verify_prepared=lambda _:manifest
    original=run.save_checkpoint
    def stop_after_first(filename,payload):
        original(filename,payload)
        if Path(filename).name=='last.pth' and payload['epoch']==0:
            raise StopAfterEpochOne()
    run.save_checkpoint=stop_after_first
    try:
        run.train(c,SimpleNamespace(resume=None))
    except StopAfterEpochOne:
        pass
    finally:
        run.save_checkpoint=original
    run.train(c,SimpleNamespace(resume=str(out/'last.pth')))
    rows=(out/'history.csv').read_text().strip().splitlines()
    assert len(rows)==3, rows
    last=run.torch.load(out/'last.pth',map_location='cpu',weights_only=True)
    assert last['epoch']==1 and last['global_step']==2
    best=run.torch.load(out/'best.pth',map_location='cpu',weights_only=True)
    scores=[json.loads((out/f'epoch_{i:03d}/metrics.json').read_text())['mAP50_95'] for i in (1,2)]
    assert best['best_mAP50_95']==max(scores)
    run.validate_checkpoint(out/'best.pth')
    run.validate_checkpoint(out/'last.pth')
    dump(out/'result.json',dict(status='passed',synthetic=True,device=c['device'],epochs=2,
        resumed_from_epoch=1,optimizer_steps=last['global_step'],mAP_scores=scores,
        best_selection_verified=True,checkpoint_roundtrip_verified=True,
        note='Tiny synthetic integration only. These scores have no BDD100K interpretation.'))
    print('INTEGRATION PASSED')

if __name__=='__main__':
    main()
