# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``dsa_indexer.bind``: the hook that routes a model's DSA indexers through
the mono launch on the steps a runner serves with it, and a layer's refusal
read from the Indexer alone (its k projected outside, its q unscaled bf16)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("flydsl", reason="the indexer kernel module imports FlyDSL")
pytest.importorskip("aiter", reason="the indexer binding imports aiter")

import torch

from atom.mono.dsa_indexer import bind, dispatch


class _Indexer:
    def __init__(self):
        self.calls = []

    def forward(self, hidden_states, qr, qr_scale, positions, rotary_emb=None, *,
                projected_k_weights=None):  # fmt: skip
        self.calls.append(qr)
        return "own"


def _hook(monkeypatch, modules):
    launched = []
    monkeypatch.setattr(
        bind, "launch_args", lambda module, qr, k, w, *rest: [module, rest[-1]]
    )
    monkeypatch.setattr(
        bind, "_run_compiled", lambda launcher, *args: launched.append((launcher, args))
    )
    hook = bind.IndexerHook(modules)
    hook.attach(
        SimpleNamespace(peer_offset=4096, replicated=False),
        "scr", "pl", None, "peers", 3, 1000,
    )  # fmt: skip
    return hook, launched


def test_the_hook_serves_a_step_with_the_launcher_and_otherwise_the_own_forward(
    monkeypatch,
):
    modules = {4: _Indexer(), 9: _Indexer()}
    hook, launched = _hook(monkeypatch, modules)
    # outside a served step: the indexer's own (class) forward
    hook.dispatch(4, modules[4], "h", "q", None, "p", "k", "w")
    assert modules[4].calls == ["q"] and not launched
    with hook.serving("launcher"):
        hook.dispatch(9, modules[9], "h", "q9", None, "p", "k", "w")
    assert len(launched) == 1 and modules[9].calls == []
    launcher, args = launched[0]
    assert launcher == "launcher" and args[0] is modules[9] and args[1] == 1009
    # and its own again once the step is over
    hook.dispatch(9, modules[9], "h", "q", None, "p", "k", "w")
    assert modules[9].calls == ["q"]


def test_removing_the_hook_gives_back_the_class_forward(monkeypatch):
    modules = {4: _Indexer()}
    hook, _ = _hook(monkeypatch, modules)
    assert modules[4].__dict__[bind._HOOKED] == (hook, 4)
    assert "forward" in modules[4].__dict__
    hook.remove()
    assert "forward" not in modules[4].__dict__
    assert bind._HOOKED not in modules[4].__dict__
    assert modules[4].forward.__func__ is _Indexer.forward


_SHAPE = bind.IndexerShape(
    heads=32, head_dim=128, topk=2048, q_lora=1536, wq_dtype="bf16",
    kv_dtype="fp8", block_rows=128, rope_dim=0,
)  # fmt: skip


def _indexer(**fields):
    """An Indexer as the refusal reads it: its own flags, wq_b, index cache."""
    return SimpleNamespace(
        **{
            "external_k_weights": True,
            "quantized_q": False,
            "rope_dim": 0,
            "use_qk_rope_cache_fusion": False,
            **fields,
        },
        wq_b=SimpleNamespace(weight=torch.empty(_SHAPE.qrows, _SHAPE.q_lora)),
        k_cache=SimpleNamespace(kv_cache=[torch.empty(4 * _SHAPE.block_bytes)]),
    )


def _module(**fields):
    """An Indexer as the routing refusal reads it before its cache exists."""
    return SimpleNamespace(**{
        "n_head": _SHAPE.heads, "head_dim": _SHAPE.head_dim,
        "topk_tokens": _SHAPE.topk, "q_lora_rank": _SHAPE.q_lora,
        "_indexer_fp4": False, "rope_dim": 0, "external_k_weights": True,
        "quantized_q": False, "use_qk_rope_cache_fusion": False,
        "wq_b": SimpleNamespace(
            weight=torch.empty(_SHAPE.qrows, _SHAPE.q_lora, dtype=torch.bfloat16),
            quant_type=SimpleNamespace(value=0, name="No"),
        ),
        "k_cache": None,  # allocated after the model is wrapped
        **fields,
    })  # fmt: skip


def _cfg(dcp=1):
    return SimpleNamespace(
        kv_cache_block_size=_SHAPE.block_rows,
        decode_context_parallel_size=dcp,
        tensor_parallel_size=4,
    )


def test_an_indexer_projecting_its_own_k_is_never_routed(monkeypatch):
    """GLM-5-FP8: its FP8 fused_qkv_a_proj keeps wk / weights_proj inside the
    indexer (``projected_k_weights=None``), so the routed op could not unpack
    them: refused before any layer is hooked."""
    monkeypatch.setattr(bind, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(bind, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(bind, "get_tensor_model_parallel_rank", lambda: 0)
    why = bind.routing_refusal(_cfg(), {3: _module(external_k_weights=False)})
    assert "inside the indexer" in why
    # the same deployment with k projected outside routes (its cache unread)
    assert bind.routing_refusal(_cfg(), {3: _module()}) == ""


def test_the_wrapper_leaves_a_refused_models_indexers_unrouted(monkeypatch):
    monkeypatch.setattr(bind, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(dispatch, "config_refusal", lambda cfg: None)
    monkeypatch.setattr(bind, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(bind, "get_tensor_model_parallel_rank", lambda: 0)
    indexer = _module(external_k_weights=False)
    layer = SimpleNamespace(self_attn=SimpleNamespace(indexer=indexer))
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    assert dispatch.install_mono_decode(model, _cfg(), None) is model
    assert "forward" not in indexer.__dict__ and bind._HOOKED not in indexer.__dict__


def test_a_layer_is_refused_on_its_own_flags_and_no_model_attribute():
    assert bind._layer_refusal(_indexer(), _SHAPE) == ""
    assert "bf16" in bind._layer_refusal(_indexer(quantized_q=True), _SHAPE)
    assert "inside the indexer" in bind._layer_refusal(
        _indexer(external_k_weights=False), _SHAPE
    )
    # a RoPE indexer: only on the fused writer, whose numerics the kernel has
    rope = {"rope_dim": 64}
    assert "fused writer" in bind._layer_refusal(_indexer(**rope), _SHAPE)
    assert (
        bind._layer_refusal(_indexer(**rope, use_qk_rope_cache_fusion=True), _SHAPE)
        == ""
    )
