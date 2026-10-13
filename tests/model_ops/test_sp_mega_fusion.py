# SPDX-License-Identifier: MIT
"""Decoder call-chain regression for the complete SP MegaMoE fusion region."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("aiter")

from atom.config import use_custom_atom_config
from atom.model_ops import sp_mega_moe as fusion
from atom.models.minimax_m3 import model


def test_full_fusion_has_a_distinct_compilation_cache_key():
    from conftest import atom_config_double

    from atom.config import Config

    config = atom_config_double()
    config.sp_mega_moe_enabled = False
    native = Config.compute_hash(config)
    config.sp_mega_moe_enabled = True
    assert Config.compute_hash(config) != native


def test_model_consumes_tail_without_repeating_norm_gate_or_topk(monkeypatch):
    class Decoder(SimpleNamespace):
        __call__ = model.MiniMaxM3DecoderLayer.forward

    def norm(x, residual, weight):
        raise AssertionError("separate post-attention/final RMSNorm")

    monkeypatch.setattr(model, "fused_allreduce_gemma_rms_norm", norm)
    monkeypatch.setattr(model, "_linear_consumes_per_token_fp8", lambda _: False)
    monkeypatch.setattr(
        model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    layers = []
    for index in range(2):
        layer = Decoder(
            _tp_replicated_o_proj=False,
            _m3_fused_gemma_fp8=False,
            _sp_mega_layer_name=f"expert{index}",
            is_moe_layer=True,
            input_layernorm=Mock(side_effect=lambda x: x * 2),
            post_attention_layernorm=SimpleNamespace(weight=3.0, variance_epsilon=1e-6),
            self_attn=Mock(side_effect=lambda **kw: kw["hidden_states"] + 3),
            block_sparse_moe=Mock(side_effect=AssertionError("separate gate/topk")),
        )
        layer.input_layernorm.weight = 2.0
        layer.block_sparse_moe.gate.weight = 5.0
        layer.block_sparse_moe.e_score_correction_bias = torch.zeros(128)
        layers.append(layer)
    calls = []

    def fused(attention, residual, post, gate, bias, next_weight, name, eps, fp8):
        calls.append(name)
        # Stand in for the device boundary: residual+post-norm+MoE+tail norm.
        post_res = attention + residual
        summed = post_res * post * gate + post_res
        return summed * next_weight, torch.ones(attention.shape[0], 1), summed

    monkeypatch.setattr(
        torch.ops.aiter, "minimax_m3_sp_mega_block", fused, raising=False
    )
    final_norm = SimpleNamespace(weight=7.0, variance_epsilon=1e-6)
    instance = SimpleNamespace(
        _tp_replicated_o_proj=False,
        start_layer=0,
        end_layer=2,
        layers=layers,
        aux_hidden_state_layers=(1,),
        norm=final_norm,
    )
    x = torch.ones(16, 8, dtype=torch.bfloat16)
    result, aux = model.MiniMaxM3Model.forward(
        instance, None, torch.arange(16), inputs_embeds=x
    )
    # Layer 0: attn=5, post_res=6, tail_res=96, next_input=192.
    # Continue with the same BF16 arithmetic, including each residual rounding.
    expected = (x * 2 + 3) + x
    first_res = expected * 3 * 5 + expected
    expected = first_res * 2 + 3 + first_res
    expected = (expected * 3 * 5 + expected) * 7
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(aux[0], torch.full_like(x, 96))
    assert calls == ["expert0", "expert1"]
    layers[0].input_layernorm.assert_called_once()
    layers[1].input_layernorm.assert_not_called()
    for layer in layers:
        layer.block_sparse_moe.assert_not_called()


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("rows", [0, 16, 32, 65, 128])
def test_full_op_preserves_rank_rows_residual_and_output_ownership(
    monkeypatch, fp8, rows
):
    from aiter import dtypes

    # Real custom-op implementation and backend orchestration; only the device
    # boundaries are replaced. Shared scratch is overwritten on every launch.
    x = (torch.arange(rows) % 8).bfloat16()[:, None].expand(rows, 8).contiguous()
    residual = torch.ones_like(x)
    expected_residual = (x + residual) * 2 + (x + residual)
    scratch = torch.empty((4 * 32, 8), dtype=torch.bfloat16)
    tail_calls, plain_calls = [], []

    def norm(x, residual, weight, eps, fp8):
        r = x + residual
        return r.to(dtypes.fp8 if fp8 else r.dtype), torch.ones(r.shape[0], 1), r

    monkeypatch.setattr(fusion, "_norm_input", norm)

    def route(layer, hidden, router):
        return (
            torch.ones(hidden.shape[0], 5),
            hidden[:, :1].int().expand(-1, 5).contiguous(),
        )

    def prefix_forward(x, r, w, *, out, res_out, router):
        _gate, _bias, ids, weights = router
        out.copy_(x + r)
        res_out.copy_(out)
        tw, ti = route(None, out, None)
        ids.copy_(ti)
        weights.copy_(tw)

    def launch(hidden, weights, ids, *, tail=None, bf16=False):
        torch.testing.assert_close(ids[:, :1].bfloat16(), hidden[:, :1])
        y = hidden * 2
        if tail is None:
            plain_calls.append(hidden.shape[0])
            return y
        tail_calls.append(hidden.shape[0])
        res_in, res_out, _nw = tail
        res_out.copy_(y + res_in)
        m = hidden.shape[0]
        # Non-owning rows contain poison; selecting global row 0 would fail.
        scratch.fill_(-42)
        scratch[2 * m : 3 * m].copy_(res_out)
        q = scratch.to(dtypes.fp8).view(torch.uint8)
        result = (y, q, torch.ones(m * 4))
        return (*result, scratch) if bf16 else result

    backend = fusion.SPMegaMoE(
        SimpleNamespace(
            cfg=SimpleNamespace(rank=2, world_size=4, tail_eps=1e-6),
            max_local_tokens=32,
            set_weights=Mock(),
            forward=launch,
            tail_ok=lambda m: m >= 16,
            error_flag=lambda: torch.zeros(1, dtype=torch.int32),
        ),
        allow_activation_change=True,
    )
    backend.prefix = SimpleNamespace(
        mmax=64,
        eps=1e-6,
        routes=lambda m: m % 16 == 0,
        forward=prefix_forward,
        error_flag=lambda: torch.zeros(1, dtype=torch.int32),
    )
    backend._route = route
    layer = SimpleNamespace(
        _sp_mega_moe=backend,
        w13_weight=object(),
        w13_weight_scale=object(),
        w2_weight=object(),
        w2_weight_scale=object(),
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"e": layer})
    )
    weight = torch.ones(8, dtype=torch.bfloat16)
    gate = torch.zeros(128, 8, dtype=torch.bfloat16)
    bias = torch.zeros(128)
    before_x, before_res = x.clone(), residual.clone()
    with use_custom_atom_config(config):
        first = fusion.minimax_m3_sp_mega_block(
            x, residual, weight, gate, bias, weight, "e", 1e-6, fp8
        )
        second = fusion.minimax_m3_sp_mega_block(
            x, residual, weight, gate, bias, weight, "e", 1e-6, fp8
        )
    second[0].zero_()
    second[2].zero_()
    expected_normalized = expected_residual.to(dtypes.fp8) if fp8 else expected_residual
    torch.testing.assert_close(first[0].float(), expected_normalized.float())
    torch.testing.assert_close(first[2], expected_residual)
    torch.testing.assert_close(x, before_x)
    torch.testing.assert_close(residual, before_res)
    assert first[1].shape == (rows, 1)
    assert first[0].dtype == (dtypes.fp8 if fp8 else torch.bfloat16)
    if rows == 65:
        assert plain_calls == [1, 1]
        assert tail_calls == [32, 32, 32, 32]


@pytest.mark.parametrize("fp8", [False, True])
def test_full_op_symbolic_schema(fp8):
    from aiter import dtypes
    from torch.fx.experimental.proxy_tensor import make_fx

    def run(x, w, g, b):
        return torch.ops.aiter.minimax_m3_sp_mega_block(
            x, x, w, g, b, w, "unloaded", 1e-6, fp8
        )

    args = (
        torch.empty(65, 6144, dtype=torch.bfloat16),
        torch.empty(6144, dtype=torch.bfloat16),
        torch.empty(128, 6144, dtype=torch.bfloat16),
        torch.empty(128),
    )
    graph = make_fx(run, tracing_mode="symbolic")(*args)
    calls = [n for n in graph.graph.nodes if n.op == "call_function"]
    assert any(
        n.target == torch.ops.aiter.minimax_m3_sp_mega_block.default for n in calls
    )
    fake = run(*(x.to("meta") for x in args))
    assert [tuple(x.shape) for x in fake] == [(65, 6144), (65, 1), (65, 6144)]
    assert fake[0].dtype == (dtypes.fp8 if fp8 else torch.bfloat16)
