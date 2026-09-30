"""ONNX graph normalization, provenance and quantization coverage audits."""
from collections import Counter
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


def normalize_gemm(filename):
    """Expose constant-weight Linear as MatMul+Add for the INT4 MatMul quantizer."""
    graph = onnx.load(str(filename))
    initializers = {item.name: item for item in graph.graph.initializer}
    names = {item.name for item in graph.graph.initializer}
    converted = []
    nodes = []
    for index, node in enumerate(graph.graph.node):
        attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        eligible = (node.op_type == 'Gemm' and len(node.input) >= 2 and node.input[1] in initializers
                    and attrs.get('transA', 0) == 0 and attrs.get('transB', 0) == 1
                    and attrs.get('alpha', 1.0) == 1.0 and attrs.get('beta', 1.0) == 1.0)
        if not eligible:
            nodes.append(node)
            continue
        weight_name = f'ptq_matmul_weight_{index}'
        if weight_name in names:
            raise ValueError('Unexpected ONNX initializer name collision')
        weight = numpy_helper.to_array(initializers[node.input[1]]).T.copy()
        graph.graph.initializer.append(numpy_helper.from_array(weight, weight_name))
        middle = f'ptq_matmul_output_{index}'
        nodes.append(helper.make_node('MatMul', [node.input[0], weight_name],
                                      [middle if len(node.input) > 2 else node.output[0]],
                                      name=f'ptq_matmul_{index}'))
        if len(node.input) > 2:
            nodes.append(helper.make_node('Add', [middle, node.input[2]], list(node.output),
                                          name=f'ptq_bias_{index}'))
        converted.append(node.name or str(index))
    graph.graph.ClearField('node')
    graph.graph.node.extend(nodes)
    used = set()
    def record_inputs(subgraph):
        for subnode in subgraph.node:
            used.update(subnode.input)
            for attribute in subnode.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    record_inputs(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        record_inputs(nested)
    record_inputs(graph.graph)
    kept = [item for item in graph.graph.initializer if item.name in used]
    graph.graph.ClearField('initializer')
    graph.graph.initializer.extend(kept)
    onnx.checker.check_model(graph)
    onnx.save_model(graph, str(filename))
    return converted


def model_properties(filename):
    model = onnx.load(str(filename))
    return {item.key: item.value for item in model.metadata_props}


def set_model_properties(filename, properties):
    model = onnx.load(str(filename))
    helper.set_model_props(model, {key: str(value) for key, value in properties.items()})
    onnx.save_model(model, str(filename))


def audit_model(filename):
    graph = onnx.load(str(filename))
    nodes = Counter(node.op_type for node in graph.graph.node)
    initializers = list(graph.graph.initializer)
    if any(item.data_location == onnx.TensorProto.EXTERNAL for item in initializers):
        raise ValueError('Expected one self-contained .onnx file, found external weight data')
    initializer_names = {item.name for item in initializers}
    constant_matmuls = sum(node.op_type == 'MatMul' and len(node.input) > 1 and node.input[1] in initializer_names
                           for node in graph.graph.node)
    int4_weight_elements = 0
    for node in graph.graph.node:
        if node.op_type == 'MatMulNBits':
            attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
            int4_weight_elements += int(attributes.get('K', 0)) * int(attributes.get('N', 0))
    return {'op_counts': dict(nodes),
            'initializers_by_dtype': dict(Counter(onnx.TensorProto.DataType.Name(x.data_type) for x in initializers)),
            'onnx_bytes': Path(filename).stat().st_size,
            'constant_weight_matmul_nodes': constant_matmuls,
            'matmul_nbits_nodes': nodes.get('MatMulNBits', 0),
            'int4_weight_elements': int4_weight_elements,
            'int8_qdq_nodes': nodes.get('QuantizeLinear', 0) + nodes.get('DequantizeLinear', 0)}
