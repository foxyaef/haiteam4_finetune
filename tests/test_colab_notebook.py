"""Verify the downloadable notebook runs the current source snapshot."""
import ast
import base64
import hashlib
import io
import json
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ColabNotebookTests(unittest.TestCase):
    def test_code_cells_compile_and_embedded_files_match_repository(self):
        notebook = json.loads((ROOT / 'colab/COCO_Eval.ipynb').read_text(encoding='utf-8'))
        assignments = {}
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            parsed = ast.parse(source)
            for node in parsed.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in ('payload_encoded', 'payload_sha256'):
                        assignments[name] = ast.literal_eval(node.value)
        payload = base64.b64decode(assignments['payload_encoded'])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), assignments['payload_sha256'])
        self.assertEqual(notebook['metadata']['evaluation_bundle_sha256'], assignments['payload_sha256'])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            manifest = json.loads(archive.read('embedded_manifest.json'))
            for name, expected_hash in manifest['files_sha256'].items():
                contents = archive.read(name)
                self.assertEqual(hashlib.sha256(contents).hexdigest(), expected_hash)
                self.assertEqual(contents, (ROOT / name).read_bytes().replace(b'\r\n', b'\n'))


if __name__ == '__main__':
    unittest.main()
