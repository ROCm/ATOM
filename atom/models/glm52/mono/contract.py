# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The per-step input contract: whether one decode step's tensors are shaped
the way the GLM-5.2 mono kernel requires, and whether the KV caches it reads
are the fused dtype/layout the kernel was built for.
"""

from __future__ import annotations

import torch

from atom.models.glm52.mono.weights import attention_impl, need
from atom.mono.runtime.consensus import MonoUnsupported


def mono_layers(model):
    """The layers mono owns.

    This is the single definition of "the layers mono owns": the MoE suffix
    of the model's layers, identified by ``layer.mlp`` carrying ``experts``.
    Every per-step check and kernel dispatch must agree on exactly this set —
    widening it to every layer reaches dense layers that have no mono-owned
    fused KV cache, and refuses (or dispatches) wrongly.
    """
    return [
        layer
        for layer in model.layers[model.start_layer : model.end_layer]
        if hasattr(layer.mlp, "experts")
    ]


def shared_sparse_buffer(layers) -> torch.Tensor:
    shared = None
    for layer in layers:
        impl = attention_impl(layer)
        current = getattr(impl, "sparse_kv_indices_buffer", None)
        need(
            current is not None
            and current.dtype is torch.int32
            and current.is_contiguous(),
            "sparse index buffer",
        )
        if shared is None:
            shared = current
        else:
            need(
                current.data_ptr() == shared.data_ptr(),
                "IndexShare layers do not share physical sparse indices",
            )
        indexer = getattr(layer.self_attn, "indexer", None)
        if indexer is not None:
            need(
                indexer.sparse_kv_indices_buffer.data_ptr() == shared.data_ptr(),
                "full IndexShare layer is not bound to the shared sparse indices",
            )
    need(shared is not None, "no sparse index buffer")
    return shared


def step_refusal(
    *, fwd, context, metadata, samples, positions, inputs_embeds, hidden_size
) -> str | None:
    """Why this step's inputs are outside the kernel's contract, or None.

    Every read here must be TP-uniform: the kernels wait in-kernel on every
    peer rank, so a rank that refused alone leaves the others spinning on the
    GPU for ever. Never gate on a per-rank flag such as ``is_dummy_run``.
    """
    if (
        positions.dtype is not torch.int64
        or positions.numel() != samples
        or not positions.is_contiguous()
    ):
        return "positions layout"
    if inputs_embeds is not None and (
        inputs_embeds.shape != (samples, hidden_size)
        or inputs_embeds.dtype is not torch.bfloat16
        or not inputs_embeds.is_contiguous()
    ):
        return "inputs_embeds layout"
    if fwd.ubatch_slices is not None:
        return "ubatch slices"
    if getattr(context, "running_tokens", samples) != samples:
        return "running tokens"
    if (
        metadata.slot_mapping.dtype is not torch.int64
        or metadata.slot_mapping.numel() < samples
    ):
        return "slot mapping"
    if (
        metadata.sparse_kv_indptr.dtype is not torch.int32
        or metadata.sparse_kv_indptr.numel() < samples + 1
    ):
        return "sparse kv indptr"
    return None


def kv_caches_ok(
    fwd, layers, kv_cache_dtype: str, shard, shared: torch.Tensor, samples: int
) -> bool:
    """Whether ``shared``'s capacity and every mono layer's fused KV cache in
    ``fwd`` satisfy the kernel's contract.

    ``layers`` must be the layers mono owns (see ``mono_layers``): checking a
    dense layer's cache here would refuse wrongly, since dense layers carry no
    mono-owned fused cache.
    """
    try:
        need(shared.numel() >= samples, "sparse index capacity")
        fp8_dtypes = {
            dtype
            for dtype in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e4m3fnuz", None),
            )
            if dtype is not None
        }
        for layer in layers:
            cache = fwd.kv_cache_data[f"layer_{layer.layer_idx}"].k_cache
            expected_dtype = (
                cache.dtype in fp8_dtypes
                if kv_cache_dtype == "fp8"
                else cache.dtype is torch.bfloat16
            )
            need(
                expected_dtype
                and cache.is_contiguous()
                and cache.shape[-1] == shard.kv_lora + shard.pe_dim,
                f"layer {layer.layer_idx} fused {kv_cache_dtype} cache",
            )
    except (KeyError, MonoUnsupported):
        return False
    return True
