"""Common metric CLI accepts only detections for the fixed evaluation images."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_predictions as shared


class SharedEvaluationTests(unittest.TestCase):
    def fixture(self, root, image_id=1):
        annotation = root / 'data/coco/annotations/evaluation.json'
        annotation.parent.mkdir(parents=True)
        annotation.write_text(json.dumps({
            'info': {}, 'licenses': [],
            'images': [{'id': 1, 'file_name': 'image.jpg', 'width': 100, 'height': 100}],
            'categories': [{'id': 1, 'name': 'object'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                             'bbox': [10, 10, 20, 20], 'area': 400, 'iscrowd': 0}],
        }), encoding='utf-8')
        (root / 'data/coco/manifest.json').write_text('{}', encoding='utf-8')
        (root / 'protocol.json').write_text(
            json.dumps({'score_threshold_for_precision_recall': 0.5}), encoding='utf-8')
        predictions = root / 'predictions.json'
        predictions.write_text(json.dumps([{'image_id': image_id, 'category_id': 1,
                                            'bbox': [10, 10, 20, 20], 'score': 0.9}]), encoding='utf-8')
        return predictions

    def test_perfect_prediction_produces_coco_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions = self.fixture(root)
            with patch.object(shared, 'ROOT', root), patch.object(shared.prepare_data, 'verify'):
                shared.evaluate(predictions, root / 'results')
            metrics = json.loads((root / 'results/metrics.json').read_text(encoding='utf-8'))
            self.assertAlmostEqual(metrics['mAP50_95'], 1.0)
            self.assertAlmostEqual(metrics['mAP50'], 1.0)
            self.assertTrue((root / 'results/class_metrics.csv').is_file())
            self.assertTrue((root / 'results/provenance.json').is_file())

    def test_rejects_predictions_outside_evaluation_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions = self.fixture(root, image_id=2)
            with patch.object(shared, 'ROOT', root), patch.object(shared.prepare_data, 'verify'):
                with self.assertRaisesRegex(ValueError, 'unknown image/category ID'):
                    shared.evaluate(predictions, root / 'results')


if __name__ == '__main__':
    unittest.main()
