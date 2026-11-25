# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import torch
from torch.library import Library

aiter_lib = Library("aiter", "FRAGMENT")

_P = ParamSpec("_P")
_R = TypeVar("_R")


def direct_register_custom_op(
    op_name: str,
    op_func: Callable[_P, _R],
    mutates_args: list[str],
    fake_impl: Callable | None = None,
    target_lib: Library | None = None,
    dispatch_key: str = "CUDA",
    tags: tuple[torch.Tag, ...] = (),
) -> Callable[_P, _R]:
    """Register ``op_func`` as ``torch.ops.<ns>.<op_name>`` and return a caller
    with ``op_func``'s signature. The caller runs ``op_func`` directly when
    ``eager_direct_call`` holds for its first argument and goes through
    ``torch.ops.<ns>.<op_name>`` otherwise; see ``eager_direct_call``."""

    import torch.library

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)
    my_lib = target_lib or aiter_lib
    # Guard against double registration. torch's op registry (torch.ops) is a
    # C++ global that outlives Python module state, so re-importing the module
    # that runs this (e.g. after a sys.modules wipe in tests, or a second
    # import path) would call define() twice for the same op and raise. Skip
    # if the op already exists — production single-import flows are unaffected.
    op_namespace = getattr(torch.ops, my_lib.ns)
    if not hasattr(op_namespace, op_name):
        my_lib.define(op_name + schema_str, tags=tags)
        my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
        if fake_impl is not None:
            my_lib._register_fake(op_name, fake_impl)
    return _eager_direct_caller(
        op_func, getattr(op_namespace, op_name), f"{my_lib.ns}::{op_name}"
    )


def _eager_direct_caller(
    op_func: Callable[_P, _R], op: Callable, name: str
) -> Callable[_P, _R]:
    @functools.wraps(op_func)
    def call(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if args and eager_direct_call(args[0]):
            # Under the profiler, the direct call is recorded under the op's
            # name so traces count it like the dispatched op.
            if torch.autograd._profiler_enabled():
                with torch.profiler.record_function(name):
                    return op_func(*args, **kwargs)
            return op_func(*args, **kwargs)
        return op(*args, **kwargs)

    return call


def eager_direct_call(x: object) -> bool:
    """Whether a custom op registered here may run its Python implementation
    directly instead of through ``torch.ops``: plain eager execution on real
    tensors. torch.compile, an active TorchDispatchMode (FakeTensorMode,
    make_fx, ...) and tensor subclasses keep the dispatch, since they handle
    the op as one opaque call. Relies on the private
    ``torch._C._len_torch_dispatch_stack()``."""
    return (
        not torch.compiler.is_compiling()
        and type(x) is torch.Tensor
        and not torch._C._len_torch_dispatch_stack()
    )
