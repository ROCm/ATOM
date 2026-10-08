# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor contracts for MiniMax-M3 sparse decode, independent of serving engines."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TPContext:
    """An existing TP group; construction and close are collective on cpu_group.

    device must be the current CUDA device on this rank."""

    cpu_group: object
    rank: int
    size: int
    device: torch.device


@dataclass(frozen=True)
class LayerSpec:
    """One TP4 sparse layer. Projection weights are original BF16 matrices;
    expert weights are AITER-shuffled MXFP4 with shuffled E8M0 scales.
    Norm weights use Gemma's (1 + weight) convention and partial NeoX RoPE.
    ATOM retains these tensors and owns any conversion copies.
    """

    layer_id: int
    g_in: torch.Tensor
    w_qkv: torch.Tensor
    g_q: torch.Tensor
    g_k: torch.Tensor
    g_iq: torch.Tensor
    g_ik: torch.Tensor
    cos_sin: torch.Tensor
    w_o: torch.Tensor
    g_post: torch.Tensor
    gate: torch.Tensor
    bias: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    eps: float
    route_scale: float
    swiglu_limit: float
    shared_weight: float = 1.0
    sm_scale: float = 128**-0.5
    init_blocks: int = 0
    local_blocks: int = 1


@dataclass(frozen=True)
class CacheSpec:
    """Native FP8 main/index storage with independent allocation.

    main is [blocks, 2, 128, 128]. k/v are the native page16 shuffle views:
    K starts at main, V starts 8 pages later, and each main block spans 16
    physical pages. Scales are positive FP32 scalars. Index storage is
    [index_blocks, 128, 128] E4M3. All storage and scales must remain stable
    until captured graphs are destroyed and the runtime has closed.
    """

    layer_id: int
    main: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    index: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    max_context: int = 16384
    layout: str = "packed_page16_scalar_fp8"


@dataclass(frozen=True)
class StepMetadata:
    """A uniform decode/verify batch, padded to token_count rows.

    Tables have one row per request. main_table entries are physical main
    block IDs * 2 (each expands to eight page16 IDs); index_table entries
    are physical 128-token index block IDs. seq_lens includes all query tokens.
    Slots have one entry per padded token: main_slots are page16 token
    addresses, index_slots independently address index storage; -1 is padding.
    query_len is the uniform tokens per request, including padded requests.
    """

    main_table: torch.Tensor
    index_table: torch.Tensor
    seq_lens: torch.Tensor
    main_slots: torch.Tensor
    index_slots: torch.Tensor
    token_count: int
    query_len: int
