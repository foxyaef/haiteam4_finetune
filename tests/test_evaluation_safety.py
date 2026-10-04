"""Regression checks for INT4 detection, failed models, ZIP output and GPU attribution."""
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_model
import evaluate_suite
from model_audit import inspect_model


class EvaluationSafetyTests(unittest.TestCase):
    def test_int4_dequantization_is_not_int8_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'int4.onnx'
            weight = TensorProto(name='weight', data_type=TensorProto.INT4, dims=[2], raw_data=b'\x21')
            graph = helper.make_graph([helper.make_node('DequantizeLinear', ['weight', 'scale'], ['out'])],
                'int4', [], [helper.make_tensor_value_info('out', TensorProto.FLOAT16, [2])],
                [weight, numpy_helper.from_array(np.asarray(1., dtype=np.float16), 'scale')])
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21)], ir_version=10), path)
            result = inspect_model(path)
            self.assertTrue(result['has_int4_evidence'])
            self.assertFalse(result['has_int8_evidence'])

    def test_unattributed_gpu_memory_is_null(self):
        fake = types.SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
            nvmlDeviceGetHandleByIndex=lambda _: object(), nvmlDeviceGetComputeRunningProcesses=lambda _: [])
        with patch.dict(sys.modules, {'pynvml': fake}):
            sampler = evaluate_model.MemorySampler(gpu=True)
            sampler.start()
            sampler.finish()
        self.assertIsNone(sampler.result()['gpu_process_peak_bytes'])
        self.assertIn('attribute', sampler.result()['gpu_memory_error'])

    def test_failed_model_does_not_prevent_remaining_results_and_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'protocol.json').write_text('{}', encoding='utf-8')
            models = [root / name for name in ('fp32.onnx', 'encoder_a_bad.onnx', 'encoder_z_good.onnx')]
            for path in models:
                path.write_bytes(b'test')

            def simulate(command, log_path):
                model = Path(command[command.index('--model') + 1])
                log_path.write_text('model run\n', encoding='utf-8')
                if model.stem == 'encoder_a_bad':
                    return 1
                destination = Path(command[command.index('--output') + 1])
                destination.mkdir()
                resource = {'rss_peak_bytes': 100, 'rss_increase_bytes': 20, 'gpu_process_peak_bytes': None,
                    'inference_ms_p50': 1., 'inference_ms_p95': 2., 'inference_fps': 1000.,
                    'end_to_end_ms_p50': 2., 'end_to_end_fps': 500.}
                (destination / 'metrics.json').write_text(json.dumps({'mAP50_95': .5, 'mAP50': .6, 'mAP75': .4,
                    'image_count': 3, 'full_evaluation': False, 'resource': resource}), encoding='utf-8')
                (destination / 'provider_placement.json').write_text(json.dumps({'status': 'recorded',
                    'kernel_event_counts': {'CPUExecutionProvider': 3}, 'cuda_uses_cpu_nodes': None}), encoding='utf-8')
                (destination / 'quantization_audit.json').write_text(json.dumps({'has_int8_evidence': False,
                    'has_int4_evidence': False}), encoding='utf-8')
                return 0

            with patch.object(evaluate_suite, 'ROOT', root), patch.object(evaluate_suite, 'run_model', side_effect=simulate):
                rows, archive = evaluate_suite.evaluate_suite(models, root / 'run', max_images=3)
            self.assertEqual([row['status'] for row in rows], ['passed', 'failed', 'passed'])
            self.assertIsNone(rows[1]['mAP50_95'])
            self.assertEqual(rows[2]['delta_mAP_pp'], 0.)
            with zipfile.ZipFile(archive) as bundle:
                self.assertIn('summary.csv', bundle.namelist())
                self.assertEqual(len(json.loads(bundle.read('errors.json'))), 1)


if __name__ == '__main__':
    unittest.main()
