# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    构建整个模型架构，由 ``cfg.MODEL.META_ARCHITECTURE`` 定义。
    根据配置函数里面的内容，找到对应的函数，然后调用创建模型。
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    # import pdb;pdb.set_trace()
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model
