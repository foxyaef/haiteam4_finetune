"""Small ONNX graph test for Gemm normalization and real packed INT4 execution."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

try:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
        QuantFormat, QuantType, matmul_nbits_quantizer, quant_utils, quantize_static)
    from onnx_graph import audit_model, normalize_gemm, model_properties, set_model_properties
except ImportError:
    onnx = None


@unittest.skipIf(onnx is None, 'Install requirements-onnx.txt to test ONNX quantization')
class OnnxGraphTests(unittest.TestCase):
    def tiny_model(self, filename):
        weight = np.random.default_rng(42).standard_normal((16, 32)).astype(np.float32)
        bias = np.random.default_rng(7).standard_normal(16).astype(np.float32)
        graph = helper.make_graph(
            [helper.make_node('Gemm', ['X', 'W', 'B'], ['Y'], transB=1, name='linear')], 'linear',
            [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 32])],
            [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 16])],
            [numpy_helper.from_array(weight, 'W'), numpy_helper.from_array(bias, 'B')])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
        model.ir_version = 10
        onnx.save(model, filename)

    def test_gemm_to_matmul_and_packed_int4_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fp32, int4 = root / 'fp32.onnx', root / 'int4.onnx'
            self.tiny_model(fp32)
            x = np.random.default_rng(19).standard_normal((1, 32)).astype(np.float32)
            expected = ort.InferenceSession(str(fp32), providers=['CPUExecutionProvider']).run(None, {'X': x})[0]
            self.assertEqual(normalize_gemm(fp32), ['linear'])
            set_model_properties(fp32, {'ptq_kind': 'onnx_fp32'})
            self.assertEqual(model_properties(fp32)['ptq_kind'], 'onnx_fp32')
            actual = ort.InferenceSession(str(fp32), providers=['CPUExecutionProvider']).run(None, {'X': x})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            self.assertEqual(audit_model(fp32)['constant_weight_matmul_nodes'], 1)
            config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=32, is_symmetric=True, accuracy_level=4,
                quant_format=QuantFormat.QOperator,
                op_types_to_quantize=('MatMul',), quant_axes=(('MatMul', 0),))
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                quant_utils.load_model_with_shape_infer(fp32), algo_config=config)
            quant.process()
            quant.model.save_model_to_file(str(int4), False)
            self.assertEqual(audit_model(int4)['matmul_nbits_nodes'], 1)
            options = ort.SessionOptions()
            optimized = root / 'optimized_int4.onnx'
            options.optimized_model_filepath = str(optimized)
            session = ort.InferenceSession(str(int4), sess_options=options, providers=['CPUExecutionProvider'])
            self.assertIn('MatMulNBits', {node.op_type for node in onnx.load(optimized).graph.node})
            result = session.run(None, {'X': x})[0]
            self.assertTrue(np.isfinite(result).all())
            self.assertGreater(float(np.max(np.abs(result - actual))), 0.0)

    def test_static_int8_runtime_uses_integer_operator(self):
        class Reader(CalibrationDataReader):
            def __init__(self):
                self.values = iter(np.random.default_rng(5).standard_normal((8, 1, 32)).astype(np.float32))

            def get_next(self):
                try:
                    return {'X': next(self.values)}
                except StopIteration:
                    return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fp32, int8, optimized = root / 'fp32.onnx', root / 'int8.onnx', root / 'optimized.onnx'
            self.tiny_model(fp32)
            normalize_gemm(fp32)
            quantize_static(str(fp32), str(int8), Reader(), quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                            calibrate_method=CalibrationMethod.MinMax, op_types_to_quantize=['MatMul'],
                            per_channel=True, use_external_data_format=False,
                            extra_options={'ActivationSymmetric': True, 'WeightSymmetric': True})
            self.assertGreater(audit_model(int8)['int8_qdq_nodes'], 0)
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.optimized_model_filepath = str(optimized)
            session = ort.InferenceSession(str(int8), sess_options=options, providers=['CPUExecutionProvider'])
            operators = {node.op_type for node in onnx.load(optimized).graph.node}
            self.assertTrue(operators & {'QLinearMatMul', 'MatMulInteger', 'QGemm', 'MatMulIntegerToFloat'}, operators)
            prediction = session.run(None, {'X': np.ones((1, 32), dtype=np.float32)})[0]
            self.assertTrue(np.isfinite(prediction).all())


if __name__ == '__main__':
    unittest.main()
