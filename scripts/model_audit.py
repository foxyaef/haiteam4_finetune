"""Inspect ONNX quantization evidence; names and declared bit widths are not proof."""
from collections import Counter

import onnx


def inspect_model(path):
    model = onnx.load(str(path), load_external_data=False)
    external = [tensor.name for tensor in model.graph.initializer
                if tensor.data_location == onnx.TensorProto.EXTERNAL]
    if external:
        raise ValueError('External tensor data found. Upload a self-contained .onnx model.')
    onnx.checker.check_model(model)
    ops = Counter(node.op_type for node in model.graph.node)
    dtypes = Counter(onnx.TensorProto.DataType.Name(tensor.data_type)
                     for tensor in model.graph.initializer)
    bits = []
    for node in model.graph.node:
        if node.op_type == 'MatMulNBits':
            for attr in node.attribute:
                if attr.name == 'bits':
                    bits.append(int(attr.i))
    int8_ops = sum(count for name, count in ops.items()
                   if name.startswith('QLinear') or name in ('MatMulInteger', 'ConvInteger'))
    qdq = ops['QuantizeLinear'] + ops['DequantizeLinear']
    int4_types = dtypes['INT4'] + dtypes['UINT4']
    return {
        'op_counts': dict(sorted(ops.items())),
        'initializer_dtypes': dict(sorted(dtypes.items())),
        'matmul_nbits_bits': bits,
        'int8_integer_op_count': int8_ops,
        'qdq_node_count': qdq,
        'int4_initializer_count': int4_types,
        'has_int8_evidence': bool(int8_ops or qdq or dtypes['INT8'] or dtypes['UINT8']),
        'has_int4_evidence': bool(4 in bits or int4_types),
        'interpretation': 'Graph evidence only; does not prove every component or activation uses the claimed precision.',
    }
