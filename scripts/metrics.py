"""One evaluator shared by FP32 and future PTQ predictions. All metrics in [0,1]."""
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from common import CLASSES

def valid_mean(values):
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else None

def evaluate_predictions(annotation_file, predictions, threshold=0.5):
    gt = COCO(str(annotation_file))
    if predictions:
        dt = gt.loadRes(predictions)
    else:
        dt = COCO()
        dt.dataset = dict(images=gt.dataset['images'], categories=gt.dataset['categories'], annotations=[])
        dt.createIndex()
    ev = COCOeval(gt, dt, 'bbox')
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    # COCO's score-ordered one-to-one matching, IoU=.50, all areas, maxDets=100.
    totals = {i: [0, 0, 0] for i in range(1, 11)}  # TP FP eligible GT
    for item in ev.evalImgs:
        if item is None or list(item['aRng']) != list(ev.params.areaRng[0]):
            continue
        count = totals[item['category_id']]
        keep = np.asarray(item['dtScores']) >= threshold
        ignore = np.asarray(item['dtIgnore'][0], dtype=bool)
        matched = np.asarray(item['dtMatches'][0]) > 0
        count[0] += int(np.sum(keep & ~ignore & matched))
        count[1] += int(np.sum(keep & ~ignore & ~matched))
        count[2] += int(np.sum(~np.asarray(item['gtIgnore'], dtype=bool)))
    rows = []
    for index, cid in enumerate(ev.params.catIds):
        tp, fp, total = totals[cid]
        rows.append(dict(id=int(cid), name=CLASSES[cid-1],
            AP=valid_mean(ev.eval['precision'][:, :, index, 0, 2]),
            AP50=valid_mean(ev.eval['precision'][0, :, index, 0, 2]),
            AR100=valid_mean(ev.eval['recall'][:, index, 0, 2]),
            precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/total if total else None, tp=tp, fp=fp, fn=total-tp, gt=total))
    tp, fp, total = np.sum(list(totals.values()), axis=0).tolist()
    return dict(mAP50_95=float(ev.stats[0]), mAP50=float(ev.stats[1]), AR100=float(ev.stats[8]),
        precision=tp/(tp+fp) if tp+fp else None, recall=tp/total if total else None,
        class_wise=rows, protocol=dict(evaluator='pycocotools bbox', IoU='0.50:0.05:0.95',
        maxDets=[1,10,100], score_threshold=threshold, precision_recall_IoU=0.5,
        class_absent_policy='AP/Recall null; COCO mean excludes categories with no eligible GT',
        aggregation='Precision/Recall micro; AP macro; AP uses all scores, not thresholded'))
