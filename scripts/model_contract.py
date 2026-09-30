"""The common RT-DETR deploy ONNX input/output contract."""
import numpy as np
import onnxruntime as ort


def check_session(filename):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(filename), sess_options=options, providers=['CPUExecutionProvider'])
    inputs = {item.name: item for item in session.get_inputs()}
    if set(inputs) != {'images', 'orig_target_sizes'} or [item.name for item in session.get_outputs()] != ['labels', 'boxes', 'scores']:
        raise ValueError('Unexpected RT-DETR ONNX input/output interface')
    if inputs['images'].type != 'tensor(float)' or inputs['orig_target_sizes'].type != 'tensor(int64)':
        raise ValueError('Expected float32 images and int64 orig_target_sizes')
    labels, boxes, scores = session.run(None, {'images': np.zeros((1, 3, 640, 640), dtype=np.float32),
                                               'orig_target_sizes': np.array([[640, 640]], dtype=np.int64)})
    if labels.shape != (1, 300) or boxes.shape != (1, 300, 4) or scores.shape != (1, 300):
        raise ValueError('RT-DETR ONNX produced unexpected detection shapes')
    if labels.dtype.kind not in 'iu' or boxes.dtype.kind != 'f' or scores.dtype.kind != 'f':
        raise ValueError('Expected integer labels and floating-point boxes/scores')
    if not np.isfinite(labels).all() or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError('RT-DETR ONNX smoke inference produced non-finite detections')
    return session
