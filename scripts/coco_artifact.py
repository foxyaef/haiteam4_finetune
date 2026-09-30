"""Portable phase-one fake-PTQ artifact; integer weight storage, FP32 execution."""
import math

import torch

from common import ROOT, sha256
from phase1_quant import target_modules

FORMAT = 'rtdetr-coco-fake-ptq-v1'


def pack(model, group, bits, ranges, checkpoint_hash, manifest_hash, config_hash):
    modules = target_modules(model, group)
    if not ranges or not set(ranges).issubset(modules):
        raise ValueError('Calibration ranges do not match selected modules')
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    scales = {}
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    for name in ranges:
        weight = state[name + '.weight'].float()
        scale = weight.abs().flatten(1).amax(1).div(qmax).clamp_min(1e-8)
        state[name + '.weight'] = torch.round(weight / scale.reshape(-1, *([1] * (weight.ndim - 1)))).clamp(qmin, qmax).to(torch.int8)
        scales[name] = scale
    return {'format': FORMAT, 'group': group, 'bits': bits, 'state': state,
            'scales': scales, 'ranges': {name: list(map(float, pair)) for name, pair in ranges.items()},
            'checkpoint_sha256': checkpoint_hash, 'data_manifest_sha256': manifest_hash,
            'config_sha256': config_hash}


def unpack(model, artifact, checkpoint_hash, manifest_hash, config_hash):
    if type(artifact) is not dict or artifact.get('format') != FORMAT:
        raise ValueError('Unsupported artifact format; use export_coco_artifact.py')
    for key, expected in [('checkpoint_sha256', checkpoint_hash),
                          ('data_manifest_sha256', manifest_hash), ('config_sha256', config_hash)]:
        if artifact.get(key) != expected:
            raise ValueError(f'Artifact {key} differs from this evaluation environment')
    group, bits = artifact.get('group'), artifact.get('bits')
    if group not in ('all', 'backbone', 'encoder', 'decoder', 'head') or bits not in (4, 6, 8):
        raise ValueError('Invalid group or bits in artifact')
    expected_state = model.state_dict()
    state = artifact.get('state')
    ranges, scales = artifact.get('ranges'), artifact.get('scales')
    modules = target_modules(model, group)
    if not isinstance(state, dict) or set(state) != set(expected_state):
        raise ValueError('Artifact state keys differ from RT-DETR model')
    if not isinstance(ranges, dict) or not isinstance(scales, dict) or not ranges or not set(ranges).issubset(modules) or set(scales) != set(ranges):
        raise ValueError('Artifact calibration coverage is invalid')
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    restored = {}
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected_state[key].shape:
            raise ValueError(f'Invalid tensor shape: {key}')
        name = key[:-7] if key.endswith('.weight') else None
        if name in ranges:
            scale = scales[name]
            if tensor.dtype != torch.int8 or tensor.numel() == 0 or tensor.min() < qmin or tensor.max() > qmax:
                raise ValueError(f'Invalid integer weight: {key}')
            if not isinstance(scale, torch.Tensor) or scale.shape != (tensor.shape[0],) or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                raise ValueError(f'Invalid weight scale: {key}')
            restored[key] = tensor.float() * scale.reshape(-1, *([1] * (tensor.ndim - 1)))
        else:
            if tensor.dtype != expected_state[key].dtype:
                raise ValueError(f'Invalid tensor dtype: {key}')
            restored[key] = tensor
    for name, pair in ranges.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in pair) or pair[0] > pair[1]:
            raise ValueError(f'Invalid activation range: {name}')
    model.load_state_dict(restored, strict=True)
    return group, bits, ranges
