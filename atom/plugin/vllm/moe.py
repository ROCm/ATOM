# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Type, TypeVar
import logging
from atom.plugin.prepare import is_vllm

logger = logging.getLogger("atom")

T = TypeVar("T")


def _apply_moe_decoration(original_cls: Type[T]) -> Type[T]:
    """
    Apply the actual decoration to the MoE class.
    This is called lazily during instantiation.
    """
    is_vllm_mode = is_vllm()
    if is_vllm_mode:
        # Rename this class because vLLM will call the modular
        # kernel init method for all modules of the model, whose name is FusedMoE,
        # to init the inside kernel, while for plugin mode, the atom maintains
        # the kernel lifecycle by itself, so there is no need to call init on
        # vllm side
        original_cls.__name__ = "ATOMFusedMoE"
        original_cls.__qualname__ = "ATOMFusedMoE"

    return original_cls


def FusedMoEDecoratorForPluginMode(cls: Type[T]) -> Type[T]:
    """
    Lazy decorator that defers class modification until first instantiation
    """
    original_cls = cls
    decorated_cls_cache = {"value": None}

    def get_decorated_class():
        if decorated_cls_cache["value"] is not None:
            return decorated_cls_cache["value"]

        decorated = _apply_moe_decoration(original_cls)
        decorated_cls_cache["value"] = decorated
        return decorated

    class LazyMoEWrapper(original_cls):
        def __new__(cls, *args, **kwargs):
            decorated_cls = get_decorated_class()
            return decorated_cls(*args, **kwargs)

    # Preserve the original class name and module for the wrapper
    LazyMoEWrapper.__name__ = original_cls.__name__
    LazyMoEWrapper.__qualname__ = original_cls.__qualname__
    LazyMoEWrapper.__module__ = original_cls.__module__

    logger.info("Create lazy wrapper for FusedMoE to change the naming")
    return LazyMoEWrapper
