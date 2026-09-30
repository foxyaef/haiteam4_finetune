"""Explicit fake WnAn PTQ for the phase-one accuracy sensitivity study.

Only called torch.nn.Conv2d / torch.nn.Linear modules are wrapped. Functional
matmuls, normalization, sampling, bias and accumulators remain floating point.
This module makes no claim about integer deployment speed or packed file size.
"""
import math
import types

import torch
from torch import nn

from official import component

GROUPS = ('backbone', 'encoder', 'decoder', 'head')


def target_modules(model, group):
    if group not in ('all', *GROUPS):
        raise ValueError('Group must be all, backbone, encoder, decoder or head')
    return {name: module for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
            and not isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear)
            and (group == 'all' or component(name + '.weight') == group)}


@torch.inference_mode()
def observe(model, modules, images):
    """Run FP32 calibration images and collect min/max of module inputs."""
    ranges = {}
    handles = []
    for name, module in modules.items():
        def hook(_module, inputs, _name=name):
            x = inputs[0]
            low, high = x.amin(), x.amax()
            old = ranges.get(_name)
            ranges[_name] = (torch.minimum(old[0], low), torch.maximum(old[1], high)) if old else (low, high)
        handles.append(module.register_forward_pre_hook(hook))
    try:
        model.eval()
        for batch in images:
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    if not ranges:
        raise RuntimeError('No selected Conv2d/Linear module ran during calibration')
    result = {name: (float(lo.item()), float(hi.item())) for name, (lo, hi) in ranges.items()}
    if any(not all(math.isfinite(x) for x in pair) for pair in result.values()):
        raise FloatingPointError('Nonfinite activation seen during calibration')
    return result


def activation_qparams(bounds, bits):
    if bits not in (4, 6, 8):
        raise ValueError('Supported activation widths are 4, 6 and 8')
    lo, hi = min(bounds[0], 0.0), max(bounds[1], 0.0)
    qmax = (1 << bits) - 1
    scale = max((hi - lo) / qmax, 1e-8)
    zero = max(0, min(qmax, round(-lo / scale)))
    return scale, zero, 0, qmax


def quantize_weight(weight, bits):
    if bits not in (4, 6, 8):
        raise ValueError('Supported weight widths are 4, 6 and 8')
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    amax = weight.detach().abs().flatten(1).amax(dim=1)
    scales = torch.clamp(amax / qmax, min=1e-8).to(dtype=torch.float32)
    zero = torch.zeros(scales.numel(), dtype=torch.int32, device=weight.device)
    return torch.fake_quantize_per_channel_affine(weight, scales, zero, 0, qmin, qmax)


def install(model, modules, ranges, bits, weight_already_quantized=False):
    """Install fake quantization on observed modules; return a coverage report."""
    missing = set(modules) - set(ranges)
    report = {'selected_modules': len(modules), 'observed_modules': len(ranges),
              'unobserved_modules': sorted(missing), 'quantized_parameters': 0,
              'quantized_modules_by_group': {k: 0 for k in GROUPS}}
    if not set(ranges).issubset(modules):
        raise ValueError('Calibration contains a module outside the requested group')
    for name, (lo, hi) in ranges.items():
        module = modules[name]
        scale, zero, qmin, qmax = activation_qparams((lo, hi), bits)
        # Precompute the quantized floating tensor; inference still computes in FP32.
        weight = module.weight.detach() if weight_already_quantized else quantize_weight(module.weight.detach(), bits)
        if isinstance(module, nn.Conv2d):
            def forward(self, x, _weight=weight, _scale=scale, _zero=zero,
                        _qmin=qmin, _qmax=qmax):
                x = torch.fake_quantize_per_tensor_affine(x, _scale, _zero, _qmin, _qmax)
                return self._conv_forward(x, _weight, self.bias)
        else:
            def forward(self, x, _weight=weight, _scale=scale, _zero=zero,
                        _qmin=qmin, _qmax=qmax):
                x = torch.fake_quantize_per_tensor_affine(x, _scale, _zero, _qmin, _qmax)
                return nn.functional.linear(x, _weight, self.bias)
        module.forward = types.MethodType(forward, module)
        group = component(name + '.weight')
        report['quantized_modules_by_group'][group] += 1
        report['quantized_parameters'] += module.weight.numel()
    report['bits'] = bits
    report['mode'] = 'fake_quant_accuracy_only'
    return report
