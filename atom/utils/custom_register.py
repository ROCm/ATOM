# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch.library import Library
from typing import Callable, Optional, Union, get_args, get_origin

aiter_lib = Library("aiter", "FRAGMENT")

_SCHEMA_ENUM_TYPES = {"ActivationType", "QuantType"}


def _schema_safe_annotation(annotation):
    if getattr(annotation, "__name__", None) in _SCHEMA_ENUM_TYPES:
        return int

    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if (
            len(non_none_args) == 1
            and getattr(non_none_args[0], "__name__", None) in _SCHEMA_ENUM_TYPES
        ):
            return Optional[int]
    return annotation


def _schema_safe_default(default):
    if getattr(type(default), "__name__", None) in _SCHEMA_ENUM_TYPES and hasattr(
        default, "value"
    ):
        return default.value
    return default


def _build_schema_safe_callable(op_func: Callable) -> Callable:
    import inspect

    sig = inspect.signature(op_func)
    parameters = []
    changed = False
    for param in sig.parameters.values():
        new_annotation = _schema_safe_annotation(param.annotation)
        new_default = _schema_safe_default(param.default)
        changed = changed or new_annotation is not param.annotation
        changed = changed or new_default is not param.default
        parameters.append(param.replace(annotation=new_annotation, default=new_default))

    return_annotation = _schema_safe_annotation(sig.return_annotation)
    changed = changed or return_annotation is not sig.return_annotation
    if not changed:
        return op_func

    def schema_func(*args, **kwargs):
        return op_func(*args, **kwargs)

    schema_func.__name__ = op_func.__name__
    schema_func.__qualname__ = op_func.__qualname__
    schema_func.__module__ = op_func.__module__
    schema_func.__defaults__ = getattr(op_func, "__defaults__", None)
    schema_func.__kwdefaults__ = getattr(op_func, "__kwdefaults__", None)
    schema_func.__signature__ = sig.replace(
        parameters=parameters, return_annotation=return_annotation
    )
    schema_func.__annotations__ = {
        param.name: param.annotation
        for param in parameters
        if param.annotation is not inspect.Signature.empty
    }
    if return_annotation is not inspect.Signature.empty:
        schema_func.__annotations__["return"] = return_annotation
    return schema_func


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
    dispatch_key: str = "CUDA",
    tags: tuple[torch.Tag, ...] = (),
):

    import torch.library
    schema_func = _build_schema_safe_callable(op_func)

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(schema_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(schema_func, mutates_args)
    my_lib = target_lib or aiter_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
