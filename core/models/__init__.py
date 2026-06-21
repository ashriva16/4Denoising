"""
Model and UNet backend registries.
With this approach it's possible to modify the complete model architecture, or just the UNet type used  (e.g. BlurPool instead of MaxPool)
Usage:
    from core.models import build_model, get_unet, MODEL_REGISTRY, UNET_REGISTRY
"""

import os
import importlib
import torch.nn as nn

MODEL_REGISTRY = {}
UNET_REGISTRY = {}


def register_model(name):
    def register_fn(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model: {name}")
        if not issubclass(cls, nn.Module):
            raise ValueError(f"Model {name} must extend nn.Module")
        MODEL_REGISTRY[name] = cls
        return cls
    return register_fn


def register_unet(name):
    def register_fn(cls):
        if name in UNET_REGISTRY:
            raise ValueError(f"Cannot register duplicate UNet: {name}")
        UNET_REGISTRY[name] = cls
        return cls
    return register_fn


def get_unet(name):
    if name not in UNET_REGISTRY:
        available = list(UNET_REGISTRY.keys())
        raise KeyError(f"Unknown UNet: '{name}'. Available: {available}")
    return UNET_REGISTRY[name]


def build_model(args):
    cls = MODEL_REGISTRY[args.model]
    unet_cls = getattr(args, 'unet_cls', None)
    if hasattr(cls, 'build_model'):
        # New-style models accept unet_cls
        import inspect
        sig = inspect.signature(cls.build_model)
        if 'unet_cls' in sig.parameters:
            return cls.build_model(args, unet_cls=unet_cls)
    return cls.build_model(args)


# Auto-import all .py files in this directory
_dir = os.path.dirname(__file__)
for _file in sorted(os.listdir(_dir)):
    if _file.endswith('.py') and _file[0].isalpha() and _file != '__init__.py':
        _module = _file[:_file.find('.py')]
        importlib.import_module('core.models.' + _module)
