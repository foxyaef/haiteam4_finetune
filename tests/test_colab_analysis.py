"""Local analysis must accept the notebook's two-result ZIP and reject mismatched data."""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_colab import analyze


class ColabAnalysisTests(unittest.TestCase):
    def bundle(self, filename, mismatch=False):
        with zipfile.ZipFile(filename, 'w') as zipped:
            for name, kind, ap in [('fp32', 'official_fp32', .40), ('backbone_w4', 'fake_ptq', .38)]:
                provenance = {'run_id': name, 'kind': kind, 'coverage': None if kind == 'official_fp32' else {'group': 'backbone', 'bits': 4},
                              'checkpoint_sha256': 'weight', 'data_manifest_sha256': 'other' if mismatch and kind == 'fake_ptq' else 'data',
                              'config_sha256': 'config', 'evaluator_sha256': 'code', 'device': 'cuda', 'gpu': 'T4',
                              'torch': '2.x', 'torchvision': '0.x'}
                profile = {'timing_ms_per_image': {'model_ms': {'p50': 10}, 'total_ms': {'p50': 12, 'p95': 15}},
                           'observed_fps_including_python': 70, 'peak_process_rss_bytes': 2**30,
                           'cuda_peak_allocated_bytes': 2**29, 'artifact_bytes': 2**26}
                for file, value in [('provenance', provenance), ('profile', profile),
                                    ('metrics', {'mAP50_95': ap, 'mAP50': .6, 'mAP75': .42})]:
                    zipped.writestr(f'{name}/{file}.json', json.dumps(value))

    def test_two_result_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'results.zip'
            self.bundle(bundle)
            analyze([bundle], Path(directory) / 'out')
            report = (Path(directory) / 'out/analysis.md').read_text(encoding='utf-8')
            self.assertIn('2.00', report)
            self.assertIn('backbone_w4', report)

    def test_reject_mismatched_data(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'results.zip'
            self.bundle(bundle, mismatch=True)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                analyze([bundle], Path(directory) / 'out')


if __name__ == '__main__':
    unittest.main()
