"""Load unchanged official RT-DETR v1 COCO components for ONNX export.

Namespace packages bypass upstream eager imports of obsolete torchvision
datapoints and optional dependencies. Model classes remain in pinned vendor/.
"""
import sys
import types
import torch
from common import ROOT

def _config(num_classes, remap):
    base = ROOT / 'vendor/RT-DETR/rtdetr_pytorch'
    for name in ['src', 'src.nn', 'src.nn.backbone', 'src.zoo', 'src.zoo.rtdetr', 'src.misc']:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(base.joinpath(*name.split('.')))]
            sys.modules[name] = package
    from src.nn.backbone.presnet import PResNet
    from src.zoo.rtdetr.rtdetr import RTDETR
    from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
    from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
    from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
    from src.zoo.rtdetr.rtdetr_criterion import SetCriterion
    from src.zoo.rtdetr.matcher import HungarianMatcher
    from src.core import YAMLConfig
    return YAMLConfig(str(base / 'configs/rtdetr/include/rtdetr_r50vd.yml'),
                     num_classes=num_classes, remap_mscoco_category=remap,
                     RTDETR={'multi_scale': None}, PResNet={'pretrained': False})


def build_coco():
    """Build the unchanged 80-class COCO model, without BDD head replacement."""
    cfg = _config(80, False)
    return cfg.model.float(), cfg.postprocessor


def load_coco_pretrained(model, filename):
    """Strictly load every tensor, including the original 80-class heads."""
    state = torch.load(filename, map_location='cpu', weights_only=True)
    weights = state['ema']['module'] if 'ema' in state else state.get('model', state)
    model.load_state_dict(weights, strict=True)
    return {'loaded_tensors': len(weights), 'reinitialized': []}
