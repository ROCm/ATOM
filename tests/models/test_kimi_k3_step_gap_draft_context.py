# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The draft context path's host side (``step_gap.draft_context``) without a
GPU launch: the switch, the hook in ``KimiK3DSpark``, a layer's binding and its
refusals, the steps it takes, the one-launch store's plan."""

from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip("aiter", reason="the model code imports aiter")
pytest.importorskip(
    "triton", reason="the attention the draft model imports needs Triton"
)

from aiter import QuantType, dtypes
from torch import nn

from atom.model_ops import layernorm
from atom.models.dspark_draft import DSparkDraftModel
from atom.models.kimi_k3.step_gap import draft_context as dc
from atom.models.kimi_k3_dspark import KimiK3DSpark
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils import forward_context


class _Model(KimiK3DSpark):
    def __init__(self, on):
        nn.Module.__init__(self)
        self._try_mono_context = on
        self.mono_context = None


@pytest.fixture
def base_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        DSparkDraftModel,
        "write_context_kv",
        lambda self, *args, **kw: calls.append((args, kw)),
    )
    return calls


def test_off_the_hook_is_the_base_path(monkeypatch, base_calls):
    monkeypatch.setattr(dc, "write_context_kv", lambda *a: pytest.fail("asked"))
    _Model(False).write_context_kv("taps", "positions", "slots")
    _Model(False).write_context_kv("joined", "positions")  # a base without slots
    assert base_calls == [
        (("taps", "positions", "slots"), {}),
        (("joined", "positions"), {}),
    ]


def test_on_a_step_the_kernels_take_skips_the_base_path(monkeypatch, base_calls):
    seen = []
    monkeypatch.setattr(dc, "write_context_kv", lambda *a: seen.append(a) or True)
    model = _Model(True)
    model.write_context_kv("taps", "positions", "slots")
    model.write_context_kv("joined", "positions")
    assert seen == [
        (model, "taps", "positions", "slots"),
        (model, "joined", "positions"),
    ]
    assert base_calls == []


def test_on_a_refused_step_and_a_pcp_split_take_the_base_path(monkeypatch, base_calls):
    asked = []
    monkeypatch.setattr(dc, "write_context_kv", lambda *a: asked.append(a) and False)
    model = _Model(True)
    model.write_context_kv("taps", "positions", "slots")
    model.write_context_kv("taps", "positions", "slots", pcp_num_tokens=8)
    model.write_context_kv("taps", "positions", slot_mapping="slots")
    model.write_context_kv("taps", "positions", "slots", pcp_num_tokens=None)
    assert len(asked) == 2  # the first and the last: no keyword that is set
    assert base_calls == [
        (("taps", "positions", "slots"), {}),
        (("taps", "positions", "slots"), {"pcp_num_tokens": 8}),
        (("taps", "positions"), {"slot_mapping": "slots"}),
        (("taps", "positions", "slots"), {"pcp_num_tokens": None}),
    ]


def test_the_forward_contexts_slots_are_cut_to_the_context_rows(monkeypatch):
    slots = torch.arange(12)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: NS(attn_metadata=NS(slot_mapping=slots)),
    )
    assert dc.context_slots(8).tolist() == list(range(8))


def _context(monkeypatch, **flags):
    context = NS(is_dummy_run=False, is_prefill=False)
    context.__dict__.update(flags)
    monkeypatch.setattr(
        forward_context, "get_forward_context", lambda: NS(context=context)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="asks the CUDA stream")
def test_the_steps_the_kernels_take(monkeypatch):
    positions = torch.zeros(8, dtype=torch.int64)
    _context(monkeypatch)
    assert dc.step_rows(positions) == 8
    assert dc.step_rows(torch.zeros(64, dtype=torch.int64)) == 64
    assert dc.step_rows(torch.zeros(65, dtype=torch.int64)) == 0
    assert dc.step_rows(torch.zeros(0, dtype=torch.int64)) == 0
    assert dc.step_rows(torch.zeros(2, 4, dtype=torch.int64)) == 0
    assert dc.step_rows([0] * 8) == 0
    _context(monkeypatch, is_dummy_run=True)
    assert dc.step_rows(positions) == 0
    _context(monkeypatch, is_prefill=True)
    assert dc.step_rows(positions) == 0


def _linear(**changes):
    weight = torch.zeros(2176, dc.HIDDEN, dtype=torch.uint8).view(dtypes.fp8)
    weight.is_shuffled = True
    lin = NS(
        weight=weight,
        weight_scale=torch.zeros(17, 56, dtype=torch.uint8).view(dtypes.fp8_e8m0),
        bias=None,
        native_a8_group_rows=None,
        quant_type=QuantType.per_1x128,
        blockscale_e8m0_scale=True,
        params_dtype=dtypes.fp8,
        tp_dim=None,
    )
    lin.__dict__.update(changes)
    return lin


def _attn(linear=None, **changes):
    attn = NS(
        hidden_size=dc.HIDDEN,
        q_lora_rank=dc.Q_LORA,
        kv_lora_rank=dc.KV_LORA,
        qk_rope_head_dim=dc.ROPE,
        fused_qkv_a_proj=linear or _linear(),
        mla_attn=NS(impl=NS(write_context_kv_latent=lambda *a: None)),
    )
    attn.__dict__.update(changes)
    return attn


def test_a_layer_binds_in_place():
    attn = _attn()
    layer = dc.bind_layer(attn, "layers.0.self_attn")
    assert layer.attn is attn and layer.prefix == "layers.0.self_attn"
    assert layer.w_a.data_ptr() == attn.fused_qkv_a_proj.weight.data_ptr()
    assert layer.s_a.data_ptr() == attn.fused_qkv_a_proj.weight_scale.data_ptr()


@pytest.mark.parametrize(
    ("attn", "why"),
    [
        (lambda: _attn(_linear(bias=torch.zeros(1))), "bias"),
        (lambda: _attn(_linear(quant_type=QuantType.per_Token)), "per_1x128"),
        (lambda: _attn(_linear(blockscale_e8m0_scale=False)), "per_1x128"),
        (lambda: _attn(_linear(native_a8_group_rows=32)), "per_1x128"),
        (lambda: _attn(_linear(params_dtype=torch.bfloat16)), "activations"),
        (lambda: _attn(_linear(input_scale=torch.ones(1))), "static activation"),
        (lambda: _attn(_linear(tp_dim=1)), "reduces"),
        (lambda: _attn(kv_lora_rank=256), "MLA widths"),
        (lambda: _attn(mla_attn=NS(impl=NS())), "MLAAttention"),
        (lambda: _attn(_linear(weight=torch.zeros(2176, dc.HIDDEN))), "FP8"),
    ],
)
def test_a_layer_the_kernels_cannot_read_is_refused(attn, why):
    with pytest.raises(MonoUnsupported, match=why):
        dc.bind_layer(attn(), "layers.0.self_attn")


def test_a_model_that_does_not_bind_turns_itself_off():
    model = NS(
        context_layers=[NS(self_attn=_attn(_linear(bias=torch.zeros(1))))],
        _try_mono_context=True,
        mono_context=None,
    )
    assert dc.bind(model) is None
    assert model._try_mono_context is False and model.mono_context is None


def _store_layer(
    norm_eps=1e-5, cache=None, rope=None, fusion=True, k_scale=1.0, norm=None
):
    monkey_tp = layernorm.get_tensor_model_parallel_world_size
    layernorm.get_tensor_model_parallel_world_size = lambda: 1
    try:
        if norm is None:
            norm = layernorm.RMSNorm(dc.KV_LORA, eps=norm_eps)
            norm.weight.data = torch.ones(dc.KV_LORA, dtype=torch.bfloat16)
    finally:
        layernorm.get_tensor_model_parallel_world_size = monkey_tp
    impl = NS(
        _ctx_kv_fusion_enabled=fusion,
        kv_lora_rank=dc.KV_LORA,
        qk_rope_head_dim=dc.ROPE,
        rotary_emb=rope or ROPE_CACHE,
        _k_scale_device=torch.full((1,), k_scale),
        kv_cache_dtype="fp8",
    )
    cache = torch.zeros(16, 1, dc.ENTRY, dtype=torch.uint8) if cache is None else cache
    attn = NS(mla_attn=NS(impl=impl, kv_cache=cache), kv_a_layernorm=norm)
    return NS(attn=attn)


ROPE_CACHE = NS(
    cos_cache=torch.zeros(64, 32, dtype=torch.bfloat16),
    sin_cache=torch.zeros(64, 32, dtype=torch.bfloat16),
    is_neox_style=False,
)


def test_the_store_plan_of_five_like_layers():
    layers = [_store_layer() for _ in range(5)]
    plan = dc.build_store_plan(layers)
    assert plan is not None and plan.apply_scale and not plan.is_neox
    assert plan.eps == pytest.approx(1e-5)
    expected = [
        p
        for layer in layers
        for p in (
            layer.attn.kv_a_layernorm.weight.data_ptr(),
            layer.attn.mla_attn.kv_cache.data_ptr(),
        )
    ]
    assert plan.tab.tolist() == expected and plan.tab.dtype == torch.int64
    assert plan.cache_like is layers[0].attn.mla_attn.kv_cache
    assert plan.cos is ROPE_CACHE.cos_cache and plan.sin is ROPE_CACHE.sin_cache


def _other_rope():
    return NS(
        cos_cache=torch.zeros(64, 32, dtype=torch.bfloat16),
        sin_cache=torch.zeros(64, 32, dtype=torch.bfloat16),
        is_neox_style=False,
    )


@pytest.mark.parametrize(
    "odd",
    [
        lambda: _store_layer(fusion=False),
        lambda: _store_layer(cache=torch.tensor([])),
        lambda: _store_layer(cache=torch.zeros(8, 2, dc.ENTRY, dtype=torch.uint8)),
        lambda: _store_layer(
            cache=torch.zeros(16, 1, dc.ENTRY - 64, dtype=torch.uint8)
        ),
        lambda: _store_layer(cache=torch.zeros(16, 1, dc.ENTRY, dtype=torch.int8)),
        lambda: _store_layer(rope=_other_rope()),
        lambda: _store_layer(norm_eps=1e-6),
        lambda: _store_layer(k_scale=2.0),
        lambda: _store_layer(
            norm=NS(weight=torch.ones(dc.KV_LORA, dtype=torch.bfloat16), eps=1e-5)
        ),
    ],
)
@pytest.mark.parametrize("at", (0, 3))
def test_one_unlike_layer_leaves_the_writes_to_production(odd, at):
    layers = [_store_layer() for _ in range(5)]
    layers[at] = odd()
    assert dc.build_store_plan(layers) is None


def test_the_plan_follows_what_it_depends_on():
    layers = [_store_layer() for _ in range(5)]
    runner = dc.DraftContextRunner.__new__(dc.DraftContextRunner)
    runner.layers, runner.parts = layers, frozenset(("kv", "store"))
    runner._store_sig = runner._store_plan = None
    plan = runner.store_plan()
    assert plan is not None and runner.store_plan() is plan
    layers[2].attn.mla_attn.kv_cache = torch.zeros(16, 1, dc.ENTRY, dtype=torch.uint8)
    moved = runner.store_plan()
    assert moved is not None and moved is not plan
    assert moved.tab.tolist()[5] == layers[2].attn.mla_attn.kv_cache.data_ptr()
    layers[4].attn.mla_attn.impl._ctx_kv_fusion_enabled = False
    assert runner.store_plan() is None
    runner.parts = frozenset(("kv",))
    layers[4].attn.mla_attn.impl._ctx_kv_fusion_enabled = True
    assert runner.store_plan() is None
