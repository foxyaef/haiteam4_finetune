import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import phase1_data
from data import CATEGORIES
from phase1_quant import target_modules, observe, install


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 4, 1)
        self.encoder = nn.Conv2d(4, 4, 1)
        self.decoder = nn.Module()
        self.decoder.dec_score_head = nn.Linear(4, 10)

    def forward(self, x):
        x = self.encoder(self.backbone(x))
        return self.decoder.dec_score_head(x.mean((-2, -1)))


class PhaseOneTests(unittest.TestCase):
    def test_component_fake_quant_changes_only_selected_path(self):
        torch.manual_seed(7)
        model = TinyModel().eval()
        sample = torch.rand(2, 3, 4, 4)
        original = model(sample)
        modules = target_modules(model, 'backbone')
        self.assertEqual(set(modules), {'backbone'})
        self.assertEqual(set(target_modules(model, 'head')), {'decoder.dec_score_head'})
        ranges = observe(model, modules, [sample])
        report = install(model, modules, ranges, 4)
        self.assertEqual(report['quantized_modules_by_group']['backbone'], 1)
        self.assertNotEqual(float((model(sample) - original).detach().abs().max()), 0.0)
        self.assertEqual(report['unobserved_modules'], [])
        self.assertEqual(set(target_modules(model, 'encoder')), {'encoder'})

    def test_fixed_subset_and_canonical_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = []
            for index in range(12):
                name = f'{index:02}.jpg'
                (root / name).touch()
                images.append(dict(id=index+1, file_name=name, width=100, height=80))
            anns = [dict(id=i+1, image_id=i+1, category_id=1,
                         bbox=[1., 2., 3., 4.], iscrowd=0) for i in range(12)]
            source = dict(info={'source': str(root)}, images=images,
                          annotations=anns, categories=CATEGORIES)
            (root/'val.json').write_text(json.dumps(source), encoding='utf-8')
            (root/'cal.json').write_text(json.dumps({'count': 1, 'files': ['train.jpg']}), encoding='utf-8')
            cfg = {'val_annotations': str(root/'val.json'), 'val_images': str(root)}
            with patch.object(phase1_data, 'config', return_value=cfg), \
                 patch.object(phase1_data, 'MANIFEST', root/'manifest.json'), \
                 patch.object(phase1_data, 'ANNOTATIONS', root/'subset.json'), \
                 patch.object(phase1_data, 'path', side_effect=lambda p: root/'cal.json' if p == 'data/calibration_manifest.json' else Path(p)):
                created = phase1_data.build(count=5, seed=42)
                self.assertEqual(phase1_data.verify()['files'], created['files'])
                self.assertEqual(phase1_data.build(count=5, seed=42)['files'], created['files'])
                source['info']['source'] = 'another_computer'
                self.assertEqual(phase1_data.canonical_digest(images, anns),
                                 phase1_data.canonical_digest(source['images'], source['annotations']))


if __name__ == '__main__':
    unittest.main()
