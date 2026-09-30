"""Score model-independent COCO bbox predictions on the fixed evaluation subset."""
import argparse
import csv
import json
import math
from pathlib import Path

from common import ROOT, sha256
from metrics import evaluate_predictions

import sys
sys.path.insert(0, str(ROOT))
import prepare_data


def evaluate(prediction_file, output):
    prepare_data.verify()
    annotation_file = ROOT / 'data/coco/annotations/evaluation.json'
    protocol_file = ROOT / 'protocol.json'
    annotations = json.loads(annotation_file.read_text(encoding='utf-8'))
    protocol = json.loads(protocol_file.read_text(encoding='utf-8'))
    image_ids = {image['id'] for image in annotations['images']}
    category_ids = {item['id'] for item in annotations['categories']}
    predictions = json.loads(Path(prediction_file).read_text(encoding='utf-8'))
    if not isinstance(predictions, list):
        raise ValueError('Predictions must be a JSON list of COCO bbox detections')
    for index, item in enumerate(predictions):
        if not isinstance(item, dict) or not {'image_id', 'category_id', 'bbox', 'score'} <= item.keys():
            raise ValueError(f'Prediction {index} needs image_id, category_id, bbox and score')
        if item['image_id'] not in image_ids or item['category_id'] not in category_ids:
            raise ValueError(f'Prediction {index} has an unknown image/category ID')
        box, score = item['bbox'], item['score']
        if (not isinstance(box, list) or len(box) != 4 or
                any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in box) or
                box[2] < 0 or box[3] < 0 or
                not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError(f'Prediction {index} has an invalid COCO xywh box or score')
    scores = evaluate_predictions(annotation_file, predictions,
                                  threshold=protocol['score_threshold_for_precision_recall'])
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new result directory: {output}')
    output.mkdir(parents=True)
    (output / 'metrics.json').write_text(json.dumps(
        {key: value for key, value in scores.items() if key != 'class_wise'},
        ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'class_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores['class_wise'][0]))
        writer.writeheader()
        writer.writerows(scores['class_wise'])
    (output / 'provenance.json').write_text(json.dumps({
        'predictions_sha256': sha256(prediction_file),
        'evaluation_annotations_sha256': sha256(annotation_file),
        'data_manifest_sha256': sha256(ROOT / 'data/coco/manifest.json'),
        'protocol_sha256': sha256(protocol_file),
        'prediction_count': len(predictions),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'mAP50:95={scores["mAP50_95"]:.4f}; AP50={scores["mAP50"]:.4f}; AP75={scores["mAP75"]:.4f}')
    print(f'Results: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, help='COCO bbox predictions JSON')
    parser.add_argument('--output', required=True, help='New results directory')
    args = parser.parse_args()
    evaluate(args.predictions, args.output)


if __name__ == '__main__':
    main()
