# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``weights.SparseMoeLayer``'s shared-expert gate: the shared expert fused into the
MXFP4 routed tables (129 experts, ``num_fused_shared_experts == 1``: an MXFP4 shared
expert, or ``ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4``) and an MXFP8 shared MLP of its own
beside 128 routed experts are both accepted, each with its own layout; a table the
wrong size for its layout, or an unfused shared expert that is not the checkpoint's
MXFP8, is refused (CPU, no kernel, fake modules at the served shapes)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
from aiter import QuantType, dtypes

from atom.models.minimax_m3.mono import weights
from atom.models.minimax_m3.mono.config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_Q_HEADS,
    N_ROUTED,
    O_K,
    ONE_INDEX_HEAD,
    ROTARY_DIM,
    TOP_K,
    TOPK_BLOCKS,
)
from atom.mono.runtime.consensus import MonoUnsupported


class _Sized:
    """An expert table by size alone (the MXFP4 tables are GBs): what the gate reads."""

    def __init__(self, numel, shuffled=True):
        self._numel = numel
        self.is_shuffled = shuffled
        self.data = self

    def numel(self):
        return self._numel

    def element_size(self):
        return 1


def _fp8(rows, cols):
    return torch.empty(rows, cols, dtype=torch.uint8).view(dtypes.fp8)


def _bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


def _ptpc(rows, cols):
    return SimpleNamespace(
        quant_type=QuantType.per_Token,
        weight=_fp8(rows, cols),
        weight_scale=torch.ones(rows, 1),
        params_dtype=dtypes.fp8,
        native_a8_group_rows=None,
        is_output_padded=False,
    )


def _mxfp8(rows, cols, native=True):
    return SimpleNamespace(
        quant_type=QuantType.per_1x32,
        weight=_fp8(rows, cols),
        weight_scale=torch.zeros(rows, cols // 32, dtype=torch.uint8),
        params_dtype=dtypes.fp8,
        native_a8_group_rows=1 if native else None,
        is_output_padded=False,
    )


def _layer(n_fused=0, shared="mxfp8", table_experts=None):
    """A sparse-attention MoE layer as ``SparseMoeLayer`` reads it: ``n_fused`` fused
    shared experts, the unfused shared expert ``shared`` ("mxfp8", "fp4" for one
    that is not the checkpoint's MXFP8, None for none) and routed tables of
    ``table_experts`` experts (default: what the layout holds)."""
    if table_experts is None:
        table_experts = N_ROUTED + n_fused
    heads = ONE_INDEX_HEAD
    impl = SimpleNamespace(kv_cache_dtype="fp8", index_cache=_fp8(4, HEAD_DIM))
    attn = SimpleNamespace(
        is_indexed_sparse_attention=True,
        num_heads=LOCAL_Q_HEADS,
        num_kv_heads=1,
        num_idx_heads=heads.count,
        head_dim=HEAD_DIM,
        rotary_emb=SimpleNamespace(
            rotary_dim=ROTARY_DIM, _minimax_m3_cos_sin_cache=_bf16(16, ROTARY_DIM)
        ),
        topk_blocks=TOPK_BLOCKS,
        init_blocks=0,
        local_blocks=1,
        attn=SimpleNamespace(impl=impl),
        qkv_proj=_ptpc(heads.rows, HIDDEN),
        o_proj=_ptpc(HIDDEN, O_K),
        skip_index_topk=False,
        **{
            n: SimpleNamespace(weight=_bf16(HEAD_DIM))
            for n in ("q_norm", "k_norm", "index_q_norm", "index_k_norm")
        },
    )
    shared_mlp = None
    if n_fused == 0 and shared is not None:
        shared_mlp = SimpleNamespace(
            gate_up_proj=_mxfp8(2 * INTER, HIDDEN, native=shared == "mxfp8"),
            down_proj=_mxfp8(HIDDEN, INTER, native=shared == "mxfp8"),
        )
    experts = SimpleNamespace(
        num_fused_shared_experts=n_fused,
        top_k=TOP_K,
        quant_method=SimpleNamespace(
            is_guinterleave=False, intermediate_pad=0, hidden_pad=0
        ),
        w13_weight=_Sized(table_experts * 2 * INTER * HIDDEN // 2),
        w2_weight=_Sized(table_experts * HIDDEN * INTER // 2),
        w13_weight_scale=_Sized(table_experts * 2 * INTER * HIDDEN // 32),
        w2_weight_scale=_Sized(table_experts * HIDDEN * INTER // 32),
    )
    moe = SimpleNamespace(
        experts=experts,
        fuse_shared_experts=n_fused > 0,
        shared_experts=shared_mlp,
        gate=SimpleNamespace(weight=_bf16(N_ROUTED, HIDDEN)),
        e_score_correction_bias=torch.zeros(N_ROUTED),
    )
    return SimpleNamespace(
        self_attn=attn,
        is_moe_layer=True,
        block_sparse_moe=moe,
        input_layernorm=SimpleNamespace(weight=_bf16(HIDDEN)),
        post_attention_layernorm=SimpleNamespace(weight=_bf16(HIDDEN)),
    )


@pytest.fixture(autouse=True)
def _preshuffled(monkeypatch):
    monkeypatch.setattr(weights, "weight_is_stored_preshuffled", lambda *a, **k: True)


def _bind(layer):
    return weights.SparseMoeLayer.from_layer(layer, 3, ONE_INDEX_HEAD)


def test_a_fused_shared_expert_is_the_tables_last_expert():
    lw = _bind(_layer(n_fused=1))
    assert lw.fused_shared
    assert lw.w13s is None and lw.s13s is None and lw.w2s is None and lw.s2s is None


def test_an_mxfp8_shared_expert_keeps_its_own_weights():
    layer = _layer()
    lw = _bind(layer)
    assert not lw.fused_shared
    sh = layer.block_sparse_moe.shared_experts
    assert lw.w13s.data_ptr() == sh.gate_up_proj.weight.data_ptr()
    assert lw.s13s.shape == (2 * INTER, HIDDEN // 32)
    assert lw.w2s.data_ptr() == sh.down_proj.weight.data_ptr()
    assert lw.s2s.shape == (HIDDEN, INTER // 32)


@pytest.mark.parametrize(
    "layout, reason",
    [
        ({"n_fused": 1, "table_experts": N_ROUTED}, "expert weight bytes"),
        ({"table_experts": N_ROUTED + 1}, "expert weight bytes"),
        ({"n_fused": 2}, "2 fused shared experts"),
        ({"shared": None}, "no shared expert"),
        ({"shared": "fp4"}, "shared gate_up: not the checkpoint's native 1x32"),
    ],
    ids=[
        "fused-128-table",
        "unfused-129-table",
        "two-fused",
        "no-shared-expert",
        "unfused-not-native-mxfp8",
    ],
)
def test_a_layout_the_kernels_cannot_read_is_refused(layout, reason):
    with pytest.raises(MonoUnsupported, match=reason):
        _bind(_layer(**layout))
