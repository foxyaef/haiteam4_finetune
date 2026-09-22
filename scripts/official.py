"""Load unchanged official v1 components without importing obsolete, unused data transforms.

Namespace packages bypass upstream eager __init__ imports (old torchvision 0.15
datapoints and optional RegNet/transformers). All model and loss classes remain
the original upstream code; our dataset and training loop live outside vendor/.
"""
import sys
import types
import torch
from common import ROOT

def build():
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
    cfg = YAMLConfig(str(base / 'configs/rtdetr/include/rtdetr_r50vd.yml'),
                     num_classes=10, remap_mscoco_category=False,
                     RTDETR={'multi_scale': None}, PResNet={'pretrained': False})
    return cfg.model.float(), cfg.criterion, cfg.postprocessor

def load_initial(model, filename):
    # Only this known official download is allowed as an initial checkpoint.
    state = torch.load(filename, map_location='cpu', weights_only=True)
    state = state['ema']['module'] if 'ema' in state else state.get('model', state)
    own = model.state_dict()
    allowed = ('decoder.enc_score_head.', 'decoder.dec_score_head.', 'decoder.denoising_class_embed.')
    loaded, skipped = {}, []
    for k, v in state.items():
        if k not in own or v.shape != own[k].shape:
            if not k.startswith(allowed):
                raise RuntimeError(f'Unexpected checkpoint mismatch: {k}')
            skipped.append(k)
        else:
            loaded[k] = v
    missing = sorted(set(own) - set(loaded))
    if any(not k.startswith(allowed) for k in missing):
        raise RuntimeError(f'Unexpected missing weights: {missing}')
    model.load_state_dict(loaded, strict=False)
    return dict(loaded_tensors=len(loaded), reinitialized=missing, skipped=skipped,
                reason='80 COCO classes -> 10 BDD100K classes; classifier and denoising embedding reset')

def load_trained(model, filename):
    state = torch.load(filename, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    return state

def component(name):
    if name.startswith('backbone.'):
        return 'backbone'
    if name.startswith('encoder.'):
        return 'encoder'
    if name.startswith(('decoder.enc_score_head.', 'decoder.enc_bbox_head.', 'decoder.dec_score_head.', 'decoder.dec_bbox_head.')):
        return 'head'
    return 'decoder'
