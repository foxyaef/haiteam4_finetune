import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
import psutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from common import ROOT, config, path, sha256, dump, seed_all, device, sync, environment
from data import DetectionDataset, collate, worker_seed, prepare, verify_prepared
from official import build, load_initial, load_trained, component
from metrics import evaluate_predictions

def loader(c, split):
    ds = DetectionDataset(path(c[f'{split}_images']), path(c[f'{split}_annotations']),
                          c['horizontal_flip'] if split == 'train' else 0.)
    return DataLoader(ds, batch_size=c['batch_size'] if split=='train' else 1,
        shuffle=split=='train', num_workers=c['workers'], collate_fn=collate,
        worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(c['seed']),
        pin_memory=False, drop_last=False)

def move(targets, dev):
    return [{k: v.to(dev) for k, v in t.items()} for t in targets]

def save_checkpoint(filename, value):
    filename = Path(filename)
    temp = filename.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(filename)
    dump(str(filename)+'.sha256.json', {'sha256': sha256(filename)})

def cpu_weights(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}

def validate_checkpoint(filename):
    sidecar = Path(str(filename)+'.sha256.json')
    if not sidecar.is_file():
        raise FileNotFoundError(f'Missing checkpoint hash: {sidecar}')
    expected = json.loads(sidecar.read_text(encoding='utf-8-sig'))['sha256']
    if sha256(filename) != expected:
        raise ValueError('Checkpoint hash mismatch')

@torch.inference_mode()
def evaluate(model, criterion, post, dl, dev, c, output, save_predictions=False):
    model.eval()
    criterion.eval()
    predictions, loss_sum, count = [], 0., 0
    for images, targets in tqdm(dl, desc='Validation FP32'):
        targets = move(targets, dev)
        result = model(images.to(dev))
        losses = criterion(result, targets)
        loss_sum += float(sum(losses.values()).item()) * len(targets)
        count += len(targets)
        sizes = torch.stack([t['orig_size'] for t in targets])  # official post uses [width,height]
        detections = post(result, sizes)
        for target, det in zip(targets, detections):
            boxes = det['boxes'].cpu()
            boxes[:, 2:] -= boxes[:, :2]
            for label, box, score in zip(det['labels'].cpu().tolist(), boxes.tolist(), det['scores'].cpu().tolist()):
                predictions.append(dict(image_id=int(target['image_id'].item()), category_id=label+1, bbox=box, score=score))
    metrics = evaluate_predictions(path(c['val_annotations']), predictions, c['score_threshold'])
    metrics['validation_loss_main'] = loss_sum / count
    metrics['validation_images'] = count
    metrics['validation_annotations_sha256'] = sha256(path(c['val_annotations']))
    dump(output / 'metrics.json', metrics)
    with (output / 'class_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics['class_wise'][0]))
        writer.writeheader()
        writer.writerows(metrics['class_wise'])
    if save_predictions:
        # COCO JSON for sharing with other implementations; can be large on full val.
        dump(output / 'predictions.json', predictions)
    return metrics

def train(c, args):
    manifest = verify_prepared(c)
    out = path(c['output'])
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(out.iterdir()):
        raise FileExistsError('Run directory already contains results. Set a new output or use --resume.')
    seed_all(c['seed'])
    dev = device(c['device'])
    model, criterion, post = build()
    if args.resume:
        validate_checkpoint(path(args.resume))
        state = load_trained(model, path(args.resume))
        if state['config'] != c or state['dataset_manifest'] != manifest:
            raise ValueError('Resume config/dataset differs from original run')
    else:
        validate_checkpoint(path(c['initial_checkpoint']))
        report = load_initial(model, path(c['initial_checkpoint']))
        dump(out / 'initial_load.json', report)
    model, criterion = model.to(dev), criterion.to(dev)
    groups = []
    for prefix, lr in [('backbone.', c['backbone_lr']), ('', c['lr'])]:
        params = [p for n,p in model.named_parameters() if p.requires_grad and
                  (n.startswith('backbone.') if prefix else not n.startswith('backbone.'))]
        groups.append({'params': params, 'lr': lr, 'base_lr': lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=c['weight_decay'])
    start, best, stale, global_step = 0, -1., 0, 0
    if args.resume:
        optimizer.load_state_dict(state['optimizer'])
        start, best, stale, global_step = state['epoch']+1, state['best_mAP50_95'], state['stale'], state['global_step']
        del state
    env = environment(c)
    env['initial_checkpoint_sha256'] = sha256(path(c['initial_checkpoint']))
    env['dataset_manifest'] = manifest
    dump(out / ('resume_environment.json' if args.resume else 'environment.json'), env)
    import yaml
    (out / 'resolved_config.yaml').write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf-8')
    train_dl, val_dl = loader(c, 'train'), loader(c, 'val')
    updates = math.ceil(len(train_dl) / c['accumulation_steps'])
    total_steps = updates * c['epochs']
    warmup = updates * c['warmup_epochs']
    for epoch in range(start, c['epochs']):
        # Epoch-boundary resume reproduces sampler, augmentation and denoising seeds.
        seed_all(c['seed'] + epoch)
        train_dl.generator.manual_seed(c['seed'] + epoch)
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, main_loss, samples = 0., 0., 0
        begin = time.perf_counter()
        progress = tqdm(train_dl, desc=f'Epoch {epoch+1}/{c["epochs"]}')
        for i, (images, targets) in enumerate(progress):
            targets = move(targets, dev)
            output = model(images.to(dev), targets)
            losses = criterion(output, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch={epoch}, batch={i}')
            group_start = (i // c['accumulation_steps']) * c['accumulation_steps']
            # Normalize by images, including the final incomplete microbatch/group.
            group_images = min(c['accumulation_steps']*c['batch_size'], len(train_dl.dataset)-group_start*c['batch_size'])
            (loss * len(targets) / group_images).backward()
            total_loss += loss.item() * len(targets)
            main_loss += sum(v.item() for k,v in losses.items() if '_aux_' not in k and '_dn_' not in k)*len(targets)
            samples += len(targets)
            if (i+1) % c['accumulation_steps'] == 0 or i+1 == len(train_dl):
                if global_step < warmup:
                    factor = (global_step + 1) / max(1, warmup)
                else:
                    fraction = (global_step-warmup)/max(1, total_steps-warmup-1)
                    factor = c['min_lr_ratio'] + (1-c['min_lr_ratio'])*(1+math.cos(math.pi*fraction))/2
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr'] * factor
                torch.nn.utils.clip_grad_norm_(model.parameters(), c['clip_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            progress.set_postfix(loss=f'{loss.item():.3f}')
        epoch_dir = out / f'epoch_{epoch+1:03d}'
        epoch_dir.mkdir(exist_ok=True)
        metrics = evaluate(model, criterion, post, val_dl, dev, c, epoch_dir)
        row = dict(epoch=epoch+1, train_loss_total=total_loss/samples, train_loss_main=main_loss/samples,
                   validation_loss_main=metrics['validation_loss_main'], mAP50_95=metrics['mAP50_95'],
                   mAP50=metrics['mAP50'], precision=metrics['precision'], recall=metrics['recall'],
                   lr=optimizer.param_groups[-1]['lr'], seconds=time.perf_counter()-begin)
        with (out/'history.csv').open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        improved = metrics['mAP50_95'] > best
        stale = 0 if improved else stale+1
        best = max(best, metrics['mAP50_95'])
        weights = cpu_weights(model)
        payload = dict(model=weights, epoch=epoch, config=c, best_mAP50_95=best,
                       selection='full official validation mAP50_95; raw FP32 model, no EMA',
                       dataset_manifest=manifest)
        if improved:
            save_checkpoint(out/'best.pth', payload)
            dump(out/'best_metrics.json', metrics)
        payload.update(optimizer=optimizer.state_dict(), global_step=global_step, stale=stale)
        save_checkpoint(out/'last.pth', payload)
        print(json.dumps(row), flush=True)
        del payload, weights
        if c['early_stopping_patience'] > 0 and stale >= c['early_stopping_patience']:
            print('Early stopping: validation mAP did not improve.', flush=True)
            break

def smoke(c, args):
    seed_all(c['seed'])
    dev = device(c['device'])
    out = path('runs/smoke')
    out.mkdir(parents=True, exist_ok=True)
    model, criterion, post = build()
    validate_checkpoint(path(c['initial_checkpoint']))
    load_report = load_initial(model, path(c['initial_checkpoint']))
    model, criterion = model.to(dev), criterion.to(dev)
    model.train()
    # Synthetic data exercises the full 640x640 model, denoising, matcher and backward.
    image = torch.rand(1, 3, 640, 640, device=dev)
    target = [dict(labels=torch.tensor([0,9], device=dev),
                   boxes=torch.tensor([[.25,.4,.15,.2],[.7,.3,.1,.1]], device=dev))]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    begin = time.perf_counter()
    output = model(image, target)
    loss = sum(criterion(output, target).values())
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite smoke loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sync(dev)
    train_seconds = time.perf_counter()-begin
    del output, optimizer
    model.eval()
    with torch.no_grad():
        prediction = model(image)
        result = post(prediction, torch.tensor([[1280,720]], device=dev))[0]
    report = dict(status='passed', device=str(dev), input=[1,3,640,640], precision='fp32',
        synthetic=True, loss=float(loss.item()), train_step_seconds=train_seconds,
        logits_shape=list(prediction['pred_logits'].shape), boxes_shape=list(prediction['pred_boxes'].shape),
        detections=len(result['scores']), initial_load=load_report,
        environment=environment(c), note='Synthetic smoke only, not BDD100K training or accuracy.')
    dump(out/'smoke_result.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('environment','initial_load')}, indent=2))

def benchmark(c, args):
    seed_all(c['seed'])
    dev = device(c['device'])
    model, _, post = build()
    checkpoint = path(args.checkpoint or str(path(c['output'])/'best.pth'))
    validate_checkpoint(checkpoint)
    load_trained(model, checkpoint)
    model.eval().to(dev)
    image = torch.rand(1,3,640,640,device=dev)
    size = torch.tensor([[1280,720]],device=dev)
    def infer():
        post(model(image), size)
    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.01):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                infer()
            sync(dev)
            if dev.type in ('cuda','xpu'):
                getattr(torch, dev.type).reset_peak_memory_stats()
            timings=[]
            for _ in range(args.iterations):
                sync(dev)
                begin=time.perf_counter()
                infer()
                sync(dev)
                timings.append((time.perf_counter()-begin)*1000)
    finally:
        stop.set()
        watcher.join()
    import numpy as np
    out=path(c['output'])/'benchmark'
    out.mkdir(parents=True,exist_ok=True)
    report=dict(device=str(dev), batch_size=1, input=[640,640], precision='fp32',
        scope='preloaded random input; model forward + postprocessor; excludes disk/decode/resize/host transfer',
        warmup=args.warmup, iterations=args.iterations, latency_ms_mean=float(np.mean(timings)),
        latency_ms_p50=float(np.median(timings)), latency_ms_p95=float(np.percentile(timings,95)),
        fps=1000/float(np.mean(timings)), checkpoint_bytes=checkpoint.stat().st_size,
        tensor_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values()),
        peak_process_rss_bytes_sampled=peak_rss[0],
        peak_device_allocated_bytes=getattr(torch,dev.type).max_memory_allocated() if dev.type in ('cuda','xpu') else None,
        checkpoint_sha256=sha256(checkpoint), environment=environment(c))
    dump(out/'efficiency.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='environment'},indent=2))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['check','prepare','smoke','train','evaluate','benchmark','module-map'])
    parser.add_argument('--config',default='configs/finetune.yaml')
    parser.add_argument('--device',choices=['cpu','xpu','cuda'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--save-predictions',action='store_true')
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--iterations',type=int,default=50)
    args=parser.parse_args()
    c=config(args.config)
    if args.device:
        c['device']=args.device
    if c['score_threshold'] < 0 or c['score_threshold'] > 1 or c['pr_iou'] != .5 or c['coco_max_dets'] != 100:
        raise ValueError('Evaluator fixed to IoU=.5, maxDets=100; threshold must be in [0,1]')
    if args.command=='check':
        report=environment(c)
        report['device_available']=str(device(c['device']))
        report['dataset_prepared']=path('data/dataset_manifest.json').is_file()
        report['ram_gib']=psutil.virtual_memory().total/2**30
        report['initial_checkpoint_exists']=path(c['initial_checkpoint']).is_file()
        dump(path('reports/environment.json'),report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    elif args.command=='prepare':
        prepare(c)
    elif args.command=='smoke':
        smoke(c,args)
    elif args.command=='train':
        train(c,args)
    elif args.command=='benchmark':
        if args.warmup < 1 or args.iterations < 1:
            raise ValueError('warmup and iterations must be positive')
        benchmark(c,args)
    elif args.command=='module-map':
        model,_,_=build()
        dump(path('reports/module_map.json'),[dict(name=n,group=component(n),shape=list(p.shape),parameters=p.numel()) for n,p in model.named_parameters()])
    else:
        verify_prepared(c)
        seed_all(c['seed'])
        dev=device(c['device'])
        model,criterion,post=build()
        ckpt=path(args.checkpoint or str(path(c['output'])/'best.pth'))
        validate_checkpoint(ckpt)
        state=load_trained(model,ckpt)
        if state['dataset_manifest'] != verify_prepared(c):
            raise ValueError('Checkpoint was trained on a different dataset manifest')
        del state
        out=path(c['output'])/'evaluation'
        out.mkdir(parents=True,exist_ok=True)
        metrics=evaluate(model.to(dev),criterion.to(dev),post,loader(c,'val'),dev,c,out,args.save_predictions)
        dump(out/'provenance.json',dict(checkpoint_sha256=sha256(ckpt),environment=environment(c)))
        print(f'mAP50:95={metrics["mAP50_95"]:.6f}')

if __name__=='__main__':
    main()
