"""Downloaded result bundles must compare only matching actual-PTQ models."""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_real_ptq import analyze


class RealPtqAnalysisTests(unittest.TestCase):
    def bundle(self, filename, different_source=False, different_exporter=False, external=False):
        with zipfile.ZipFile(filename, 'w') as bundle:
            models = [('onnx_fp32', .4, 100), ('onnx_int8', .39, 60), ('onnx_int4', .38, 80)]
            if external:
                models.append(('onnx_experiment', .37, 90))
            for kind, ap, size in models:
                source_hash = 'bad' if kind == 'onnx_int4' and different_source else 'fp32hash'
                props = {'exporter_sha256': 'other' if kind == 'onnx_int4' and different_exporter else 'exporter'}
                if kind != 'onnx_fp32':
                    props['fp32_model_sha256'] = source_hash
                provenance = {'kind': kind, 'run_id': kind, 'model_sha256': 'fp32hash' if kind == 'onnx_fp32' else kind,
                              'model_properties': props,
                              'method_label': 'team_method' if kind == 'onnx_experiment' else kind,
                              'identity_status': 'team_declared' if kind == 'onnx_experiment' else 'project_generated',
                              'checkpoint_sha256': 'weight', 'data_manifest_sha256': 'data',
                              'config_sha256': 'config', 'evaluator_sha256': 'code',
                              'onnxruntime': '1.22.1', 'providers': ['CPUExecutionProvider'],
                              'optimized_runtime_op_counts': {'QLinearConv': 2 if kind == 'onnx_int8' else 0},
                              'artifact_audit': {'matmul_nbits_nodes': 1 if kind == 'onnx_int4' else 0,
                                                 'int4_weight_elements': 512 if kind == 'onnx_int4' else 0}}
                profile = {'onnx_model_bytes': size, 'timing_ms_per_image': {'runtime_ms': {'p50': 10, 'p95': 15}},
                           'observed_fps_including_python': 70, 'peak_process_rss_bytes': 2**30}
                metrics = {'mAP50_95': ap, 'mAP50': .6, 'mAP75': .42}
                for name, value in [('provenance', provenance), ('profile', profile), ('metrics', metrics)]:
                    bundle.writestr(f'{kind}/{name}.json', json.dumps(value))

    def test_three_models_one_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'results.zip'
            self.bundle(source)
            analyze([source], Path(directory) / 'report')
            text = (Path(directory) / 'report/analysis.md').read_text(encoding='utf-8')
            self.assertIn('onnx_int4', text)
            self.assertIn('2.00', text)

    def test_allow_other_fp32_serialization_with_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'results.zip'
            self.bundle(source, different_source=True)
            analyze([source], Path(directory) / 'report')
            self.assertIn('해시가 이번 기준 파일과 다릅니다',
                          (Path(directory) / 'report/analysis.md').read_text(encoding='utf-8'))

    def test_reject_different_exporter(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'results.zip'
            self.bundle(source, different_exporter=True)
            with self.assertRaisesRegex(ValueError, 'exporter'):
                analyze([source], Path(directory) / 'report')

    def test_team_model_is_labeled_as_declared(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'results.zip'
            self.bundle(source, external=True)
            analyze([source], Path(directory) / 'report')
            report = (Path(directory) / 'report/analysis.md').read_text(encoding='utf-8')
            self.assertIn('team_method', report)
            self.assertIn('등록자가 선언한 정보', report)


if __name__ == '__main__':
    unittest.main()
