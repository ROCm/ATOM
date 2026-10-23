# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-step GDN state views consumed by the Qwen layer kernel."""

from dataclasses import dataclass

import torch

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.shapes import (
    ALIGN,
    GDN_CONV_WIDTH,
    GDN_HEAD_DIM,
    QSA_COMPRESS_RATIO,
    QSA_HEAD_DIM,
    QSA_INDEX_HEAD_DIM,
)
from atom.mono.runtime.consensus import MonoUnsupported


@dataclass(frozen=True)
class GdnStepView:
    """Stable graph pointers used by one fixed-width GDN layer body."""

    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    state_read: torch.Tensor
    state_write: torch.Tensor


@dataclass(frozen=True)
class QsaStepView:
    """Graph-stable QSA addressing and the four persistent cache planes."""

    metadata: object
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    compressed_slot_mapping: torch.Tensor
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    raw_key_cache: torch.Tensor
    compressed_key_cache: torch.Tensor
    rope_position_cache: torch.Tensor | None


def _tensor(
    value,
    name: str,
    suffix: tuple[int, ...],
    dtype: torch.dtype,
    alignment: int,
    stride_suffix: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise MonoUnsupported(f"Qwen mono GDN has no {name}")
    if tuple(value.shape[-len(suffix) :]) != suffix:
        raise MonoUnsupported(
            f"Qwen mono GDN {name} is {tuple(value.shape)}, expected *{suffix}"
        )
    layout_ok = (
        value.is_contiguous()
        if stride_suffix is None
        else tuple(value.stride()[-len(stride_suffix) :]) == stride_suffix
    )
    if value.dtype != dtype or not layout_ok:
        raise MonoUnsupported(
            f"Qwen mono GDN {name} is {value.dtype} stride={value.stride()}"
        )
    if value.data_ptr() % alignment:
        raise MonoUnsupported(f"Qwen mono GDN {name} is not {alignment}-byte aligned")
    return value


def gdn_step_view(
    forward_context, layer_id: int, tokens: int, tp: int = 1
) -> GdnStepView:
    """Bind one non-speculative decode step without copying either state plane."""

    metadata = getattr(forward_context.attn_metadata, "gdn_metadata", None)
    if metadata is None:
        raise MonoUnsupported("Qwen mono GDN metadata is absent")
    if (
        metadata.num_prefills,
        metadata.num_decodes,
        metadata.num_actual_tokens,
        metadata.num_spec_decodes,
    ) != (0, tokens, tokens, 0):
        raise MonoUnsupported(
            f"Qwen mono GDN requires {tokens} non-speculative decodes"
        )
    if getattr(metadata, "replayssm", False):
        raise MonoUnsupported("Qwen mono GDN does not use ReplaySSM")

    read = metadata.non_spec_state_indices_in_tensor
    write = metadata.non_spec_state_indices_tensor
    if read is None:
        read = write
    read = _tensor(read, "state_read", (tokens,), torch.int32, 4)
    write = _tensor(write, "state_write", (tokens,), torch.int32, 4)

    caches = getattr(forward_context, "kv_cache_data", None)
    cache = None if caches is None else caches.get(f"layer_{layer_id}")
    if cache is None:
        raise MonoUnsupported(f"Qwen mono GDN layer {layer_id} has no state cache")
    dims = Dims(tp)
    conv = _tensor(
        cache.k_cache,
        "conv_state",
        (GDN_CONV_WIDTH - 1, dims.gdn_conv_dim),
        torch.bfloat16,
        ALIGN,
    )
    recurrent = _tensor(
        cache.v_cache,
        "recurrent_state",
        (dims.gdn_value_heads, GDN_HEAD_DIM, GDN_HEAD_DIM),
        torch.float32,
        ALIGN,
        (GDN_HEAD_DIM * GDN_HEAD_DIM, 1, GDN_HEAD_DIM),
    )
    return GdnStepView(conv, recurrent, read, write)


def _dynamic_tensor(
    value,
    name: str,
    dtype: torch.dtype,
    ndim: int,
    alignment: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise MonoUnsupported(f"Qwen mono QSA has no {name}")
    if value.dtype != dtype or value.dim() != ndim or not value.is_contiguous():
        raise MonoUnsupported(
            f"Qwen mono QSA {name} is {tuple(value.shape)} {value.dtype} "
            f"stride={value.stride()}"
        )
    if value.data_ptr() % alignment:
        raise MonoUnsupported(f"Qwen mono QSA {name} is not {alignment}-byte aligned")
    return value


def qsa_step_view(
    forward_context, attention, tokens: int, tp: int = 1, max_model_len: int = 8192
) -> QsaStepView:
    """Bind one decode QSA K1/K2 seam without allocating or copying metadata."""

    qsa = getattr(forward_context.attn_metadata, "qsa_metadata", None)
    if qsa is None:
        raise MonoUnsupported("Qwen mono QSA metadata is absent")
    qsa = qsa.for_tokens(tokens)

    block_tables = _dynamic_tensor(qsa.block_tables, "block_tables", torch.int32, 2, 4)
    pointers = {
        name: _tensor(getattr(qsa, name), name, (tokens,), dtype, alignment)
        for name, dtype, alignment in (
            ("slot_mapping", torch.int64, 8),
            ("compressed_slot_mapping", torch.int64, 8),
            ("token_to_req", torch.int32, 4),
            ("logical_positions", torch.int64, 8),
        )
    }
    seq_lens = _dynamic_tensor(qsa.seq_lens, "seq_lens", torch.int32, 1, 4)
    if block_tables.shape[0] < tokens or seq_lens.numel() < tokens:
        raise MonoUnsupported(
            "Qwen mono QSA metadata has fewer request rows than decode tokens"
        )
    max_seq_len = int(qsa.max_seq_len)
    if max_seq_len <= 0:
        raise MonoUnsupported(f"Qwen mono QSA max_seq_len is {max_seq_len}")
    if max_seq_len > max_model_len:
        raise MonoUnsupported(
            f"Qwen mono QSA max_seq_len {max_seq_len} exceeds deployment {max_model_len}"
        )

    k_cache = _dynamic_tensor(attention.k_cache, "k_cache", torch.bfloat16, 4, ALIGN)
    v_cache = _dynamic_tensor(attention.v_cache, "v_cache", torch.bfloat16, 4, ALIGN)
    if tuple(k_cache.shape) != tuple(v_cache.shape):
        raise MonoUnsupported("Qwen mono QSA K/V cache shapes differ")
    dims = Dims(tp)
    if tuple(k_cache.shape[-2:]) != (dims.qsa_kv_heads, QSA_HEAD_DIM):
        raise MonoUnsupported(
            f"Qwen mono TP{tp} QSA K/V cache tail is "
            f"{tuple(k_cache.shape[-2:])}, expected "
            f"{(dims.qsa_kv_heads, QSA_HEAD_DIM)}"
        )
    pages, page_size = k_cache.shape[:2]

    cache_shapes = {
        "raw_key_cache": (pages, page_size, 1, QSA_INDEX_HEAD_DIM),
        "compressed_key_cache": (
            pages,
            page_size // QSA_COMPRESS_RATIO,
            1,
            QSA_INDEX_HEAD_DIM,
        ),
    }
    caches = {}
    for name, shape in cache_shapes.items():
        value = _dynamic_tensor(
            getattr(attention, name), name, torch.bfloat16, 4, ALIGN
        )
        if tuple(value.shape) != shape:
            raise MonoUnsupported(
                f"Qwen mono QSA {name.removesuffix('_key_cache')} cache is {tuple(value.shape)}"
            )
        caches[name] = value
    position_cache = attention.rope_position_cache
    if position_cache is not None:
        position_cache = _dynamic_tensor(
            position_cache, "rope_position_cache", torch.int64, 4, 8
        )
        if tuple(position_cache.shape) != (pages, page_size, 1, 3):
            raise MonoUnsupported(
                f"Qwen mono QSA position cache is {tuple(position_cache.shape)}"
            )

    return QsaStepView(
        metadata=qsa,
        block_tables=block_tables,
        **pointers,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        k_cache=k_cache,
        v_cache=v_cache,
        **caches,
        rope_position_cache=position_cache,
    )
