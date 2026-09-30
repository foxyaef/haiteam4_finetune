import json
import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_data


class CocoPreparationTests(unittest.TestCase):
    def test_complete_partial_archive_is_verified_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            folder = data_root / 'downloads'
            folder.mkdir()
            payload = b'completed download'
            partial = folder / 'sample.zip.part'
            partial.write_bytes(payload)
            spec = {'sample.zip': {'url': 'https://example.invalid/sample.zip',
                                   'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}
            with patch.object(prepare_data, 'DATA', data_root), patch.object(prepare_data, 'ARCHIVES', spec):
                self.assertEqual(prepare_data.download('sample.zip'), folder / 'sample.zip')
                self.assertFalse(partial.exists())
                self.assertEqual(prepare_data.download('sample.zip').read_bytes(), payload)

    def test_one_command_builds_disjoint_reproducible_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'coco'
            images = [{'id': index + 1, 'file_name': f'{index + 1:012}.jpg',
                       'width': 10, 'height': 10} for index in range(5000)]
            source = {'images': images, 'categories': [{'id': i, 'name': str(i)} for i in range(1, 81)],
                      'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                                       'bbox': [1, 1, 2, 2], 'area': 4, 'iscrowd': 0}]}
            subsets = prepare_data.selected_subsets(source)
            names = {image['file_name'] for subset in subsets.values() for image in subset['images']}
            self.assertEqual(len(subsets['calibration']['images']), 512)
            self.assertEqual(len(subsets['evaluation']['images']), 1000)
            self.assertEqual(len(names), 1512)
            self.assertEqual(subsets, prepare_data.selected_subsets(source))

            annotations_zip = root / 'annotations.zip'
            images_zip = root / 'images.zip'
            with zipfile.ZipFile(annotations_zip, 'w') as zipped:
                zipped.writestr('annotations/instances_val2017.json', json.dumps(source))
            with zipfile.ZipFile(images_zip, 'w') as zipped:
                for name in names:
                    zipped.writestr(f'val2017/{name}', b'synthetic-image')

            def fake_download(name):
                return annotations_zip if name.startswith('annotations') else images_zip

            with patch.object(prepare_data, 'DATA', data_root), patch.object(
                    prepare_data, 'download', side_effect=fake_download):
                prepare_data.prepare()
                prepare_data.verify()
                prepare_data.prepare()  # Rerunning preserves the same outputs.
                self.assertEqual(len(list((data_root / 'val2017').glob('*.jpg'))), 1512)


if __name__ == '__main__':
    unittest.main()
