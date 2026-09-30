"""Integration check: an uploaded ONNX is executed on images and measured."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_model


def make_model(path):
    tensors = [
        ('labels', [0], TensorProto.INT64, [1, 1]),
        ('boxes', [10., 10., 30., 30.], TensorProto.FLOAT, [1, 1, 4]),
        ('scores', [.99], TensorProto.FLOAT, [1, 1]),
    ]
    nodes = [helper.make_node('Constant', [], [name], value=helper.make_tensor(name + '_value', dtype, shape, values))
             for name, values, dtype, shape in tensors]
    graph = helper.make_graph(nodes, 'constant detector', [
        helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 640, 640]),
        helper.make_tensor_value_info('orig_target_sizes', TensorProto.INT64, [1, 2]),
    ], [helper.make_tensor_value_info(name, dtype, shape) for name, _, dtype, shape in tensors])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 16)],
                                ir_version=8), path)


class ModelEvaluationTests(unittest.TestCase):
    def test_actual_model_inference_and_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images_dir = root / 'data/coco/val2017'
            images_dir.mkdir(parents=True)
            images = []
            annotations = []
            for i in (1, 2):
                name = f'{i}.jpg'
                Image.new('RGB', (50, 50), color=(20, 50, 80)).save(images_dir / name)
                images.append({'id': i, 'file_name': name, 'width': 50, 'height': 50})
                annotations.append({'id': i, 'image_id': i, 'category_id': 1,
                                    'bbox': [10, 10, 20, 20], 'area': 400, 'iscrowd': 0})
            ann_dir = root / 'data/coco/annotations'
            ann_dir.mkdir(parents=True)
            (ann_dir / 'evaluation.json').write_text(json.dumps({
                'info': {}, 'licenses': [], 'images': images, 'annotations': annotations,
                'categories': [{'id': 1, 'name': 'object'}],
            }), encoding='utf-8')
            (root / 'data/coco/manifest.json').write_text('{}', encoding='utf-8')
            (root / 'protocol.json').write_text(json.dumps({
                'evaluation_images': 2, 'category_count': 1, 'input_size': 640,
                'runtime_warmup_images': 1, 'score_threshold_for_precision_recall': 0.5,
            }), encoding='utf-8')
            model = root / 'fp32.onnx'
            make_model(model)
            with patch.object(evaluate_model, 'ROOT', root), patch.object(evaluate_model.prepare_data, 'verify'):
                metrics = evaluate_model.evaluate(model, root / 'result', preview_count=1)
            self.assertAlmostEqual(metrics['mAP50_95'], 1.0)
            self.assertEqual(metrics['resource']['inference_images_timed'], 1)
            self.assertEqual(metrics['resource']['model_file_bytes'], model.stat().st_size)
            self.assertGreater(metrics['resource']['rss_peak_bytes'], 0)
            self.assertEqual(len(json.loads((root / 'result/predictions.json').read_text())), 2)
            self.assertTrue((root / 'result/preview_1.jpg').is_file())
            self.assertFalse(json.loads((root / 'result/quantization_audit.json').read_text())['has_int4_evidence'])


if __name__ == '__main__':
    unittest.main()
