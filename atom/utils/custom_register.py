# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch.library import Library
from typing import Callable, Optional

aiter_lib = Library("aiter", "FRAGMENT")


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
    if hasattr(op_namespace, op_name):
        return
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
