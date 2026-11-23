# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts for ATOM_MINIMAX_M3_PA_MXFP8_OUT: the NVFP4 sparse attention
writes o_proj's MXFP8 input (E4M3 + E8M0 per 32) instead of its BF16 output."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from atom.model_ops import base_attention
from atom.models.minimax_m3 import model as m3_model
from atom.models.minimax_m3.model import MiniMaxM3SparseAttention


class _QKV:
    def __call__(self, hidden_states, x_scale=None):
        del x_scale
        return hidden_states


class _OProj:
    def __init__(self, native_a8):
        self.native_a8_group_rows = native_a8
        self.calls = []

    def __call__(self, values, x_scale=None):
        self.calls.append((values, x_scale))
        return values


class _Attention:
    def __init__(self, kv_format="nvfp4"):
        # Before the KV cache is bound -- when torch.compile traces the forward
        # in the warmup run -- `kv_nvfp4` is still False.
        self.impl = SimpleNamespace(kv_format=kv_format, kv_nvfp4=False)
        self.mxfp8_calls = []
        self.plain_calls = []

    def forward_mxfp8(self, q, k, v, positions, qkv=None):
        self.mxfp8_calls.append(q)
        return torch.zeros_like(q, dtype=torch.uint8), torch.ones(
            (q.shape[0], q.shape[1] // 32), dtype=torch.uint8
        )

    def __call__(self, *args, **kwargs):
        self.plain_calls.append(args)
        return args[0]


def _layer(*, enabled=True):
    return SimpleNamespace(
        qkv_proj=_QKV(),
        o_proj=_OProj(1),
        attn=_Attention(),
        q_size=64,
        kv_size=2,
        index_q_size=2,
        idx_head_dim=2,
        sparse_pa_mxfp8_out=enabled,
    )


def test_mxfp8_out_hands_the_attention_fp8_and_scales_to_o_proj():
    layer = _layer()
    MiniMaxM3SparseAttention.forward(layer, torch.arange(3), torch.randn(3, 72))
    assert len(layer.attn.mxfp8_calls) == 1 and not layer.attn.plain_calls
    values, scale = layer.o_proj.calls[0]
    assert values.dtype == torch.uint8 and scale.shape == (3, 2)


def test_mxfp8_out_keeps_the_bf16_path_otherwise():
    layer = _layer(enabled=False)
    MiniMaxM3SparseAttention.forward(layer, torch.arange(3), torch.randn(3, 72))
    assert not layer.attn.mxfp8_calls and len(layer.attn.plain_calls) == 1
    assert layer.o_proj.calls[0][1] is None


def _mx_conditions(monkeypatch, *, flag=True, kv_format="nvfp4", native_a8=1):
    monkeypatch.setattr(m3_model.envs, "ATOM_MINIMAX_M3_PA_MXFP8_OUT", flag)
    monkeypatch.setattr(m3_model, "nvfp4_reads_page_table", lambda: True)
    return m3_model._sparse_pa_mxfp8_conditions(
        _Attention(kv_format), _OProj(native_a8)
    )


def test_mxfp8_out_follows_the_declared_kv_format_not_the_binding(monkeypatch):
    """Decided at construction: torch.compile traces the forward in the warmup
    run, before the KV cache is bound (``impl.kv_nvfp4`` still False), and the
    compiled graph keeps that branch for every later step."""
    assert all(_mx_conditions(monkeypatch).values())


@pytest.mark.parametrize(
    "kwargs",
    [{"flag": False}, {"kv_format": ""}, {"native_a8": None}],
    ids=["flag-off", "fp8-kv", "o_proj-not-native-a8"],
)
def test_mxfp8_out_conditions_refuse(monkeypatch, kwargs):
    assert not all(_mx_conditions(monkeypatch, **kwargs).values())


@pytest.mark.parametrize("kv_format", ["nvfp4", ""])
def test_the_pool_takes_the_kv_format_the_impl_declared(kv_format):
    from atom.model_ops.attentions.aiter_attention import AiterAttentionMetadataBuilder

    module = SimpleNamespace(
        impl=SimpleNamespace(is_indexed_sparse_attention=True, kv_format=kv_format),
        base_attention=True,
        use_mla=False,
        num_kv_heads=1,
        head_dim=128,
    )
    kinds = AiterAttentionMetadataBuilder._module_kinds(None, module)
    assert [k.kv_format for k in kinds if hasattr(k, "kv_format")] == [kv_format]


def test_nvfp4_attention_passes_the_mxfp8_output_and_keeps_the_split(monkeypatch):
    """The MXFP8 output takes any output mode, so the plan is the BF16 call's
    (and so are its values, which the split path quantized)."""
    from atom.model_ops.minimax_m3 import sparse_attn_nvfp4

    captured = {}
    monkeypatch.setattr(
        importlib.import_module("aiter.ops.flydsl.pa_decode"),
        "pa_decode",
        lambda *a, **kw: captured.update(kw),
    )
    monkeypatch.setattr(
        "aiter.ops.triton.gluon.pa_decode_gluon.get_recommended_splits",
        lambda *_: 3,
    )
    monkeypatch.setattr(
        base_attention,
        "_flydsl_prepare_explicit_plan",
        lambda **kw: captured.setdefault("plan", kw),
    )
    monkeypatch.setattr(
        base_attention, "_flydsl_plan_scratch", lambda *a, **kw: (None, None, None)
    )
    rows = 5
    q = torch.randn(rows, 16, 128, dtype=torch.bfloat16)
    payload = torch.zeros(4, 1, 128, 64, dtype=torch.uint8)
    scales = torch.zeros(4, 1, 128, 8, dtype=torch.uint8)
    fp8 = torch.empty(rows, 16 * 128, dtype=torch.uint8)
    e8m0 = torch.empty(rows, 64, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    sparse_attn_nvfp4._flydsl_attention(
        q, payload, scales, payload, scales,
        torch.zeros(rows, 1, dtype=torch.int32), torch.ones(rows, dtype=torch.int32),
        128**-0.5, None, None, quantized_output=fp8, output_scale=e8m0,
    )  # fmt: skip
    assert captured["plan"]["plan_partition_cap"] == 3
    assert captured["store_output"] is False
    assert captured["quantized_output"].shape == (rows, 16, 128)
    assert captured["quantized_output"].data_ptr() == fp8.data_ptr()
    assert captured["output_scale"] is e8m0


def test_fill_unit_scale_writes_e8m0_one_and_fp32_one():
    e8m0 = torch.zeros(2, 4, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    base_attention.fill_unit_scale(e8m0)
    assert (e8m0.view(torch.uint8) == 127).all()
    fp32 = torch.zeros(2, 1)
    base_attention.fill_unit_scale(fp32)
    assert (fp32 == 1).all()
