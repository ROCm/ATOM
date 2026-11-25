# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``dsa_indexer.bind``: the hook that routes a model's DSA indexers through
the mono launch on the steps a runner serves with it, and a layer's refusal
read from the Indexer alone (its k projected outside, its q unscaled bf16)."""

from dataclasses import replace
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
    monkeypatch.setattr(bind.torch.cuda, "current_stream", lambda: None)
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


@pytest.mark.parametrize("changed", ["rows", "qlen", "world", "debug", 0, 1, 2, 3])
def test_local_step_codegen_settings_have_distinct_launcher_caches(changed):
    from atom.mono.dsa_indexer.local_step import build

    args = {"rows": 48, "qlen": 6, "world": 8, "compact": (512, 1024, 49664, 53760)}
    baseline = build(**args)
    if isinstance(changed, int):
        compact = list(args["compact"])
        compact[changed] += 16
        args["compact"] = tuple(compact)
    else:
        args[changed] = {"rows": 54, "qlen": 8, "world": 4, "debug": True}[changed]
    other = build(**args)
    assert baseline.func.__name__ != other.func.__name__
    assert build(**args) is other


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
        "_indexer_fp4": False, "_index_cache_layout": "preshuffle",
        "rope_dim": 0, "external_k_weights": True,
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


@pytest.mark.parametrize("model_kind,q_lora", [("glm", 2048), ("k3", 1536)])
@pytest.mark.parametrize("dcp", [1, 8])
def test_install_prepares_shared_weight_before_warmup(
    monkeypatch, model_kind, q_lora, dcp
):
    from atom.models.kimi_k3.mono import dispatch as k3_dispatch

    installer = dispatch if model_kind == "glm" else k3_dispatch
    monkeypatch.setattr(installer, "config_refusal", lambda cfg: None)
    monkeypatch.setattr(bind, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(bind, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(bind, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(bind, "get_dcp_rank", lambda: 0)
    indexer = _module(q_lora_rank=q_lora)
    weight = torch.nn.Parameter(
        torch.randn(_SHAPE.qrows, q_lora, dtype=torch.bfloat16), requires_grad=False
    )
    indexer.wq_b.weight = weight
    original, address = weight.clone(), weight.data_ptr()
    inner = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(indexer=indexer))]
    )
    model = torch.nn.Module()
    if model_kind == "k3":
        model.language_model = SimpleNamespace(model=inner)
    else:
        model.model = inner

    cfg = _cfg(dcp)
    cfg.tensor_parallel_size = 8
    cfg.dcp_config = SimpleNamespace(interleave_size=1)
    wrapped = installer.install_mono_decode(model, cfg, None)
    assert weight.is_shuffled
    assert indexer.wq_b.weight is weight and weight.data_ptr() == address
    assert not hasattr(indexer, "_mono_indexer_wq_packed")
    restored = (
        weight.view(256, q_lora // 32, 4, 16, 8)
        .permute(0, 3, 1, 2, 4)
        .reshape_as(weight)
    )
    assert torch.equal(restored, original)
    assert wrapped is not model
    shuffled = weight.clone()
    bind.prepare_weights({0: indexer})
    assert indexer.wq_b.weight is weight and weight.data_ptr() == address
    assert torch.equal(weight, shuffled)


def test_weight_updates_refuse_before_changing_storage(monkeypatch):
    monkeypatch.setattr(bind, "_takes_weight_updates", lambda cfg: True)
    assert "weight updates" in bind._static_refusal(None, 8, 0, {0: object()})


@pytest.mark.parametrize("q_lora", [0, 1281, 1312])
def test_unaligned_q_projection_refuses_before_shuffle(q_lora):
    module = _module(q_lora_rank=q_lora)
    module.wq_b.weight = torch.empty(_SHAPE.qrows, q_lora)
    assert "K divisible by 128" in bind._operand_refusal(
        module, replace(_SHAPE, q_lora=q_lora)
    )


def test_binding_never_shuffles_after_model_warmup(monkeypatch):
    module = _module()
    module.wq_b.weight.zero_()
    monkeypatch.setattr(bind, "indexer_modules", lambda inner: {0: module})
    monkeypatch.setattr(bind, "get_current_atom_config", _cfg)
    monkeypatch.setattr(bind, "_refusal", lambda *args: "")
    original, address = module.wq_b.weight.clone(), module.wq_b.weight.data_ptr()
    assert bind.bind_indexers(None, 8, 0) is None
    assert module.wq_b.weight.data_ptr() == address
    assert not getattr(module.wq_b.weight, "is_shuffled", False)
    assert torch.equal(module.wq_b.weight, original)


def _runner_shape(q_lora=2048, cache="fp8"):
    return bind.IndexerShape(
        32,
        128,
        2048,
        q_lora,
        "bf16",
        cache,
        64 if cache == "fp4" else 128,
        64 if q_lora == 2048 else 0,
        cache_layout="mfma" if cache == "fp4" else "preshuffle",
    )


@pytest.mark.parametrize(
    "q_lora,cache,replicated",
    [
        (1536, "fp8", False),
        (2048, "fp8", False),
        (1536, "fp4", False),
        (1536, "fp4", True),
        (2048, "fp8", True),
    ],
)
@pytest.mark.parametrize("context", [8192, 163840])
@pytest.mark.parametrize(
    "scheduled,cap,expected", [(2, 256, 64), (256, 128, 128), (256, 256, 256)]
)
def test_runner_buffers_cover_every_reachable_layout(
    monkeypatch, q_lora, cache, replicated, context, scheduled, cap, expected
):
    from atom.mono.dsa_indexer import kernel, runner
    from atom.mono.dsa_indexer.bind import IndexerBinding

    module = SimpleNamespace(wq_b=SimpleNamespace(weight=torch.empty(1)))
    binding = IndexerBinding(
        {0: module},
        _runner_shape(q_lora, cache),
        8,
        context,
        replicated=replicated,
        dual_score=cache == "fp8" and not replicated,
        short_gemm=replicated,
    )
    monkeypatch.setattr(runner, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(runner, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(runner, "bind_indexers", lambda *args: binding)
    monkeypatch.setattr(
        runner, "logits_plane", lambda *_: SimpleNamespace(numel=lambda: 256 * context)
    )
    monkeypatch.setattr(
        runner,
        "get_current_atom_config",
        lambda: SimpleNamespace(max_num_seqs=scheduled),
    )
    instance = runner.IndexerRunner(None, None, max_rows=cap, peer_offset=4096)
    assert instance.rows == expected
    assert instance.serves(expected, 1)
    assert not instance.serves(expected + 1, 1)
    verify_rows = min(252, scheduled * 6)
    assert instance.verify_rows == verify_rows
    assert instance.serves(verify_rows, 6)
    assert not instance.serves(verify_rows + 6, 6)
    # Radix buckets can need less storage than smaller widths.
    keys = [binding.key(r) for r in range(1, expected + 1)]
    keys += [binding.key(r, 6) for r in range(6, verify_rows + 1, 6)]
    assert instance.plain_bytes == max(map(kernel.plain_bytes, keys))
    assert instance.scratch_bytes == max(map(kernel.scratch_bytes, keys))
    assert instance.peer_bytes == 4096 + max(map(kernel.peer_bytes, keys))
    if replicated:
        monkeypatch.setattr(
            runner,
            "logits_plane",
            lambda *_: SimpleNamespace(numel=lambda: 64 * context),
        )
        limited = runner.IndexerRunner(None, None)
        assert limited.rows == 64
        assert limited.verify_rows == min(60, scheduled * 6)
        assert not limited.serves(65, 1)
        assert not limited.serves(66, 6)


@pytest.mark.parametrize(
    "arch,cus,expected",
    [
        ("gfx950:sramecc+:xnack-", 256, True),
        ("gfx942", 256, False),
        ("gfx950", 255, False),
        ("gfx950", 304, False),
    ],
)
def test_dual_grid_device_guard(monkeypatch, arch, cus, expected):
    from atom.mono.dsa_indexer import kernel

    for name in ("ROC_GLOBAL_CU_MASK", "HSA_CU_MASK", "HIP_CU_MASK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(gcnArchName=arch, multi_processor_count=cus),
    )
    assert kernel.dual_score_device() == expected
    monkeypatch.setenv("ROC_GLOBAL_CU_MASK", "0:0-127")
    assert not kernel.dual_score_device()


def test_binding_rejects_wrong_scale_shape_and_type():
    shape = replace(_runner_shape(cache="fp4"), rope_dim=0)
    cache = torch.empty((2, 4, 64, 16), dtype=torch.uint8)
    scale = torch.empty((2, 2, 32, 4), dtype=torch.uint8)
    m = SimpleNamespace(k_cache=SimpleNamespace(kv_cache=[cache], kv_cache_scale=scale))
    assert not bind._cache_refusal(m, shape)
    m.k_cache.kv_cache_scale = scale.reshape(2, 4, 64)
    assert "OPUS" in bind._cache_refusal(m, shape)
    m.k_cache.kv_cache_scale = scale.float()
    assert "OPUS" in bind._cache_refusal(m, shape)
