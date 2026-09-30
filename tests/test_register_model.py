"""A team ONNX can be registered without claiming its technique was verified."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from onnx_graph import model_properties, set_model_properties
from register_model import register


def make_deploy_model(filename, count=300):
    inputs = [
        helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 640, 640]),
        helper.make_tensor_value_info('orig_target_sizes', TensorProto.INT64, [1, 2]),
    ]
    outputs = [
        helper.make_tensor_value_info('labels', TensorProto.INT64, [1, count]),
        helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, count, 4]),
        helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, count]),
    ]
    nodes = [
        helper.make_node('ReduceSum', ['orig_target_sizes'], ['size_sum'], keepdims=0),
        helper.make_node('Mul', ['size_sum', 'zero_int'], ['int_zero']),
        helper.make_node('Add', ['base_labels', 'int_zero'], ['labels']),
        helper.make_node('ReduceSum', ['images'], ['image_sum'], keepdims=0),
        helper.make_node('Mul', ['image_sum', 'zero_float'], ['float_zero']),
        helper.make_node('Add', ['base_boxes', 'float_zero'], ['boxes']),
        helper.make_node('ReduceMean', ['boxes'], ['scores'], axes=[2], keepdims=0),
    ]
    initializers = [
        numpy_helper.from_array(np.zeros((1, count), dtype=np.int64), 'base_labels'),
        numpy_helper.from_array(np.zeros((1, count, 4), dtype=np.float32), 'base_boxes'),
        numpy_helper.from_array(np.array(0, dtype=np.int64), 'zero_int'),
        numpy_helper.from_array(np.array(0, dtype=np.float32), 'zero_float'),
    ]
    model = helper.make_model(helper.make_graph(nodes, 'deploy_contract', inputs, outputs, initializers),
                              opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 10
    onnx.save(model, filename)


class RegisterModelTests(unittest.TestCase):
    def test_registers_with_declared_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, candidate, output = [root / name for name in ('base.onnx', 'candidate.onnx', 'registered.onnx')]
            make_deploy_model(baseline)
            set_model_properties(baseline, {'ptq_kind': 'onnx_fp32', 'checkpoint_sha256': 'weights',
                                            'data_manifest_sha256': 'data', 'config_sha256': 'config',
                                            'exporter_sha256': 'exporter'})
            make_deploy_model(candidate)
            register(candidate, baseline, output, 'team_method')
            props = model_properties(output)
            self.assertEqual(props['ptq_kind'], 'onnx_experiment')
            self.assertEqual(props['identity_status'], 'team_declared')
            self.assertEqual(props['method_label'], 'team_method')
            self.assertEqual(props['checkpoint_sha256'], 'weights')

    def test_rejects_wrong_detection_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, candidate, output = [root / name for name in ('base.onnx', 'candidate.onnx', 'registered.onnx')]
            make_deploy_model(baseline)
            set_model_properties(baseline, {'ptq_kind': 'onnx_fp32', 'checkpoint_sha256': 'weights',
                                            'data_manifest_sha256': 'data', 'config_sha256': 'config',
                                            'exporter_sha256': 'exporter'})
            make_deploy_model(candidate, count=299)
            with self.assertRaisesRegex(ValueError, 'detection shapes'):
                register(candidate, baseline, output, 'invalid')


if __name__ == '__main__':
    unittest.main()
