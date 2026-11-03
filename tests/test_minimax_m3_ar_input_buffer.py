# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only F5 call-chain, alias, phase, and fallback contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.model_ops import linear as linear_mod
from atom.model_ops import moe as moe_mod
from atom.models.minimax_m3 import model as m3


class _Producer(nn.Module):
    def __init__(self, supported=True):
        super().__init__()
        self.supported = supported
        self.outs = []
        self.qkv_proj = object()
        self.gate_up_proj = object()

    def supports_out(self):
        return self.supported

    def forward(self, hidden_states, *args, out=None, **kwargs):
        self.outs.append(out)
        return torch.empty_like(hidden_states) if out is None else out


class _IdentityNorm(nn.Module):
    weight = None
    variance_epsilon = 1e-6


def _decoder(monkeypatch, *, attention=True, ffn=True):
    layer = m3.MiniMaxM3DecoderLayer.__new__(m3.MiniMaxM3DecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = _Producer(attention)
    layer.mlp = _Producer(ffn)
    layer.is_moe_layer = False
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    monkeypatch.setattr(m3, "_linear_consumes_per_token_fp8", lambda _linear: False)
    monkeypatch.setattr(
        m3,
        "fused_allreduce_gemma_rms_norm",
        lambda hidden, residual, _norm: (hidden, residual),
    )
    return layer


def test_decoder_reuses_one_registered_alias_for_o_proj_and_dense_down(monkeypatch):
    layer = _decoder(monkeypatch)
    hidden = torch.randn(5, 16)
    residual = torch.randn_like(hidden)
    arena = torch.empty(4096, dtype=torch.uint8)

    output, _ = layer(
        torch.arange(5),
        hidden,
        residual,
        registered_ar_input=arena,
    )

    assert layer.self_attn.outs[0].data_ptr() == arena.data_ptr()
    assert layer.mlp.outs[0].data_ptr() == arena.data_ptr()
    assert output.data_ptr() == arena.data_ptr()


@pytest.mark.parametrize(
    "attention,ffn,expected",
    [(False, True, (False, True)), (True, False, (True, False))],
)
def test_unsupported_producer_keeps_its_ordinary_allocation(
    monkeypatch, attention, ffn, expected
):
    layer = _decoder(monkeypatch, attention=attention, ffn=ffn)
    hidden = torch.randn(3, 8)
    layer(
        torch.arange(3),
        hidden,
        torch.randn_like(hidden),
        registered_ar_input=torch.empty(1024, dtype=torch.uint8),
    )
    actual = (layer.self_attn.outs[0] is not None, layer.mlp.outs[0] is not None)
    assert actual == expected


def test_too_small_arena_falls_back_without_partial_alias(monkeypatch):
    layer = _decoder(monkeypatch)
    hidden = torch.randn(4, 32)
    layer(
        torch.arange(4),
        hidden,
        torch.randn_like(hidden),
        registered_ar_input=torch.empty(8, dtype=torch.uint8),
    )
    assert layer.self_attn.outs == [None]
    assert layer.mlp.outs == [None]


def test_ptpc_linear_advertises_out_only_for_its_preshuffle_path(monkeypatch):
    linear = linear_mod.LinearBase.__new__(linear_mod.LinearBase)
    linear.native_a8_group_rows = None
    linear.quant_type = linear_mod.QuantType.per_Token
    linear.params_dtype = linear_mod.dtypes.fp8
    linear.bias = None
    linear.is_output_padded = False
    monkeypatch.setattr(linear_mod, "use_triton_gemm", lambda: False)
    assert linear.supports_out()
    monkeypatch.setattr(linear_mod, "use_triton_gemm", lambda: True)
    assert not linear.supports_out()


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, positions, hidden, residual, **kwargs):
        self.seen.append(kwargs["registered_ar_input"])
        return hidden, hidden if residual is None else residual


class _Embed(nn.Module):
    def forward(self, ids):
        return torch.zeros(ids.shape[0], 8)


@pytest.mark.parametrize("prefill", [False, True])
def test_model_offers_the_arena_only_to_prefill(monkeypatch, prefill):
    model = m3.MiniMaxM3Model.__new__(m3.MiniMaxM3Model)
    nn.Module.__init__(model)
    # Match the eager-mode state installed by @support_torch_compile's
    # production constructor while keeping this call on the real __call__ and
    # forward path.
    model.do_not_compile = True
    layer = _Layer()
    model.embed_tokens = _Embed()
    model.layers = nn.ModuleList([layer])
    model.start_layer, model.end_layer = 0, 1
    model.aux_hidden_state_layers = ()
    model.norm = _IdentityNorm()
    model._registered_ar_input = torch.empty(1024, dtype=torch.uint8)
    model._registered_ar_input_lease = object()
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(m3, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        m3,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(is_prefill=prefill)),
    )
    monkeypatch.setattr(
        m3,
        "fused_allreduce_gemma_rms_norm",
        lambda hidden, residual, _norm: (hidden, residual),
    )

    model(torch.arange(2), torch.arange(2))
    assert len(layer.seen) == 1
    assert (layer.seen[0] is model._registered_ar_input) is prefill
    if not prefill:
        assert layer.seen[0] is None


def test_moe_into_output_copies_only_if_backend_declines(monkeypatch):
    output = torch.empty(2, 4)
    produced = torch.arange(8, dtype=output.dtype).view_as(output)
    backend = SimpleNamespace(
        forward_impl=lambda hidden, logits, output=None: produced
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"moe": backend})
    )
    monkeypatch.setattr(moe_mod, "get_current_atom_config", lambda: config)
    moe_mod.moe_forward_into_output(
        torch.empty_like(output), torch.empty(2, 3), output, "moe"
    )
    assert torch.equal(output, produced)


def test_mxfp4_moe_advertises_out_only_for_direct_gfx1250_path(monkeypatch):
    method = moe_mod.Mxfp4MoEMethod.__new__(moe_mod.Mxfp4MoEMethod)
    method.fused_experts = None
    method.is_gfx1250 = True
    method.is_guinterleave = True
    method.quant_type = moe_mod.QuantType.per_1x32
    method.use_triton = False
    method.use_triton_ep = False
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            use_ep=False,
            dp_size=1,
            dp_logical_ratio=1,
            pcp_merge=False,
        )
    )
    layer = SimpleNamespace(reduce_results=False)
    monkeypatch.delenv("AITER_DISABLE_GROUPED_A8W4", raising=False)
    assert method.supports_out(layer)
    monkeypatch.setenv("AITER_DISABLE_GROUPED_A8W4", "1")
    assert not method.supports_out(layer)


def test_minimax_moe_passes_flat_registered_output_to_experts():
    module = m3.MiniMaxM3MoE.__new__(m3.MiniMaxM3MoE)
    nn.Module.__init__(module)
    module.gate = _Producer()
    module.experts = _Producer()
    module.shared_experts = None
    module.fuse_shared_experts = True
    module.routed_scaling_factor = 1.0
    hidden = torch.randn(2, 3, 8)
    out = torch.empty_like(hidden)

    result = module(hidden, out=out)
    assert len(module.experts.outs) == 1
    assert module.experts.outs[0].data_ptr() == out.data_ptr()
    assert result.data_ptr() == out.data_ptr()
