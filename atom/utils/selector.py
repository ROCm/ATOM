# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import cache
from typing import Type

from atom.model_ops.attentions.backends import AttentionBackend
from atom.utils import resolve_obj_by_qualname


def get_attn_backend(
    block_size: int,
    use_mla: bool = False,
) -> Type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""
    # Accessing envs.* behind an @lru_cache decorator can cause the wrong
    # value to be returned from the cache if the value changes between calls.
    # To avoid this, we read envs.VLLM_USE_V1 here and pass it explicitly to the
    # private function.
    return _cached_get_attn_backend(
        block_size=block_size,
        use_mla=use_mla,
    )


@cache
def _cached_get_attn_backend(
    block_size: int,
    use_mla: bool = False,
) -> Type[AttentionBackend]:

    # get device-specific attn_backend
    attention_cls = get_attn_backend_cls(block_size, use_mla)
    if not attention_cls:
        raise ValueError(f"Invalid attention backend for {attention_cls}")
    return resolve_obj_by_qualname(attention_cls)


def get_attn_backend_cls(block_size, use_mla) -> str:
    if use_mla:
        # if block_size == 1:
        return "atom.model_ops.attentions.aiter_mla.AiterMLABackend"  # noqa: E501
        # else:
        #     raise ValueError(
        #         f" The selected backend"
        #         f"does not support block size {block_size}."
        #         "(currently only supports block size 1)")

    return "atom.model_ops.attentions.aiter_attention.AiterBackend"  # noqa: E501
