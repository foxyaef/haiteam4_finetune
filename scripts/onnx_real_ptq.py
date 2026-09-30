"""Export RT-DETR to ONNX and create actual INT8 and packed INT4 PTQ models.

INT8: static QDQ for Conv/MatMul with COCO calibration.
INT4: packed weight-only MatMulNBits for constant-weight MatMul; other ops stay FP32.
"""
import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torchvision
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils

from coco_protocol import loader, settings, verified_weight, verify_source
from common import ROOT, dump, path, seed_all, sha256, sha256_text
from model_contract import check_session
from official import build_coco, load_coco_pretrained
from onnx_graph import audit_model, model_properties, normalize_gemm, set_model_properties

import sys
sys.path.insert(0, str(ROOT))
import prepare_data


def checked_environment():
    cfg = settings()
    prepare_data.verify()
    verify_source()
    checkpoint, checkpoint_hash = verified_weight(cfg)
    return cfg, checkpoint, checkpoint_hash


class Deploy(torch.nn.Module):
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()

    def forward(self, images, original_sizes):
        return self.postprocessor(self.model(images), original_sizes)


def export_fp32(output):
    cfg, checkpoint, checkpoint_hash = checked_environment()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seed_all(cfg['seed'])
    model, postprocessor = build_coco()
    load_coco_pretrained(model, checkpoint)
    wrapped = Deploy(model.eval(), postprocessor.eval()).eval()
    dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    sizes = torch.tensor([[640, 640]], dtype=torch.int64)
    with torch.inference_mode():
        torch.onnx.export(wrapped, (dummy, sizes), str(output),
                          input_names=['images', 'orig_target_sizes'],
                          output_names=['labels', 'boxes', 'scores'],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    converted = normalize_gemm(output)
    set_model_properties(output, {'ptq_kind': 'onnx_fp32', 'checkpoint_sha256': checkpoint_hash,
                                 'data_manifest_sha256': sha256(path('data/coco/manifest.json')),
                                 'config_sha256': sha256_text(ROOT / 'configs/coco_ptq.yaml'),
                                 'exporter_sha256': sha256_text(Path(__file__)),
                                 'torch_version': torch.__version__,
                                 'torchvision_version': torchvision.__version__})
    check_session(output)
    audit = audit_model(output)
    metadata = {'kind': 'onnx_fp32', 'checkpoint_sha256': checkpoint_hash,
                'data_manifest_sha256': sha256(path('data/coco/manifest.json')),
                'config_sha256': sha256_text(ROOT / 'configs/coco_ptq.yaml'),
                'model_sha256': sha256(output), 'normalized_gemm_nodes': len(converted),
                'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'audit': audit}
    dump(sidecar(output), metadata)
    print(f'FP32 ONNX: {output} ({output.stat().st_size / 2**20:.1f} MiB); normalized Gemm={len(converted)}')
    return output


def sidecar(model_path):
    return Path(str(model_path) + '.manifest.json')


class CocoCalibration(CalibrationDataReader):
    def __init__(self, cfg):
        self.iterator = iter(loader(cfg['calibration_images'], cfg['calibration_annotations']))

    def get_next(self):
        try:
            images, _, sizes = next(self.iterator)
        except StopIteration:
            return None
        return {'images': images.numpy(), 'orig_target_sizes': torch.stack(sizes).T.numpy()}

    def rewind(self):
        cfg = settings()
        self.iterator = iter(loader(cfg['calibration_images'], cfg['calibration_annotations']))


def quantize(source, output, precision):
    cfg, _, checkpoint_hash = checked_environment()
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not source.is_file():
        raise FileNotFoundError('Export the FP32 ONNX model first')
    origin = model_properties(source)
    if (origin.get('ptq_kind') != 'onnx_fp32' or origin.get('checkpoint_sha256') != checkpoint_hash
            or origin.get('exporter_sha256') != sha256_text(Path(__file__))):
        raise ValueError('FP32 ONNX model or source checkpoint changed')
    if origin.get('data_manifest_sha256') != sha256(path('data/coco/manifest.json')) or origin.get('config_sha256') != sha256_text(ROOT / 'configs/coco_ptq.yaml'):
        raise ValueError('FP32 ONNX data or protocol differs from current environment')
    output.parent.mkdir(parents=True, exist_ok=True)
    if precision == 'int8':
        quantize_static(str(source), str(output), CocoCalibration(cfg),
                        quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
                        op_types_to_quantize=['Conv', 'MatMul'], per_channel=True,
                        use_external_data_format=False,
                        extra_options={'ActivationSymmetric': True, 'WeightSymmetric': True})
        method = 'static QDQ INT8 activation+weight, Conv/MatMul, 512 COCO calibration images'
    elif precision == 'int4':
        config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=32, is_symmetric=True, accuracy_level=4,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=('MatMul',), quant_axes=(('MatMul', 0),))
        graph = quant_utils.load_model_with_shape_infer(source)
        converter = matmul_nbits_quantizer.MatMulNBitsQuantizer(graph, algo_config=config)
        converter.process()
        converter.model.save_model_to_file(str(output), False)
        method = 'packed signed INT4 weight-only MatMulNBits, constant-weight MatMul, block=32; other ops FP32'
    else:
        raise ValueError('precision must be int8 or int4')
    set_model_properties(output, {'ptq_kind': f'onnx_{precision}', 'checkpoint_sha256': checkpoint_hash,
                                  'data_manifest_sha256': origin['data_manifest_sha256'],
                                  'config_sha256': origin['config_sha256'],
                                  'fp32_model_sha256': sha256(source), 'ptq_method': method,
                                  'exporter_sha256': origin['exporter_sha256'],
                                  'torch_version': origin['torch_version'],
                                  'torchvision_version': origin['torchvision_version']})
    onnx.checker.check_model(str(output))
    audit = audit_model(output)
    if precision == 'int8' and audit['int8_qdq_nodes'] == 0:
        raise RuntimeError('INT8 conversion created no QDQ nodes')
    if precision == 'int4' and audit['matmul_nbits_nodes'] == 0:
        raise RuntimeError('INT4 conversion created no MatMulNBits nodes; the graph is not actually INT4')
    check_session(output)
    metadata = {'kind': f'onnx_{precision}', 'method': method,
                'checkpoint_sha256': checkpoint_hash, 'data_manifest_sha256': origin['data_manifest_sha256'],
                'config_sha256': origin['config_sha256'], 'fp32_model_sha256': sha256(source),
                'model_sha256': sha256(output), 'onnx': onnx.__version__, 'onnxruntime': ort.__version__,
                'audit': audit}
    dump(sidecar(output), metadata)
    print(f'{precision.upper()} ONNX: {output} ({output.stat().st_size / 2**20:.1f} MiB); {method}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--output', default='artifacts/rtdetr_fp32.onnx')
    quant = commands.add_parser('quantize')
    quant.add_argument('--source', default='artifacts/rtdetr_fp32.onnx')
    quant.add_argument('--output', required=True)
    quant.add_argument('--precision', required=True, choices=['int8', 'int4'])
    args = parser.parse_args()
    export_fp32(args.output) if args.command == 'export' else quantize(args.source, args.output, args.precision)


if __name__ == '__main__':
    main()
