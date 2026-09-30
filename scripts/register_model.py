"""Register a team-produced ONNX model for the shared COCO evaluator.

Registration checks the deploy interface and records a declared FP32 source.
It does not prove the quantization method or the model's training history.
"""
import argparse
from pathlib import Path

import onnx

from common import sha256
from onnx_graph import audit_model, model_properties, set_model_properties
from model_contract import check_session


def register(model_path, baseline_path, output_path, method):
    model_path, baseline_path, output_path = map(Path, (model_path, baseline_path, output_path))
    if output_path.exists():
        raise FileExistsError(output_path)
    if model_path.resolve() == baseline_path.resolve() or output_path.resolve() in (model_path.resolve(), baseline_path.resolve()):
        raise ValueError('Input, FP32 baseline and output must be different files')
    if not method.strip() or len(method) > 100:
        raise ValueError('Provide a short, non-empty method name')
    baseline = model_properties(baseline_path)
    if baseline.get('ptq_kind') != 'onnx_fp32':
        raise ValueError('Baseline must be the project FP32 ONNX export')
    onnx.checker.check_model(str(model_path))
    audit_model(model_path)  # Reject external weight files; Colab uploads one .onnx only.
    check_session(model_path)
    properties = model_properties(model_path)
    if properties.get('ptq_kind') in {'onnx_fp32', 'onnx_int8', 'onnx_int4', 'onnx_experiment'}:
        raise ValueError('Model already has project metadata; upload it directly')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(model_path))
    onnx.save_model(model, str(output_path))
    set_model_properties(output_path, {
        'ptq_kind': 'onnx_experiment', 'method_label': method.strip(),
        'identity_status': 'team_declared', 'source_model_sha256': sha256(model_path),
        'fp32_model_sha256': sha256(baseline_path),
        **{key: baseline[key] for key in ('checkpoint_sha256', 'data_manifest_sha256',
                                          'config_sha256', 'exporter_sha256')},
    })
    onnx.checker.check_model(str(output_path))
    check_session(output_path)
    print(f'Registered: {output_path} ({method.strip()}); model origin is team-declared')
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Self-contained ONNX produced locally')
    parser.add_argument('--baseline', required=True, help='Project FP32 ONNX export')
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', required=True, help='Experiment label, e.g. BRECQ-backbone')
    args = parser.parse_args()
    register(args.model, args.baseline, args.output, args.method)


if __name__ == '__main__':
    main()
