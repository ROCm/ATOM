# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 independent shared/routed MoE stream contracts."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("aiter")

from atom.models.minimax_m3 import model as m3_model


class _Stream:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def wait_stream(self, other: "_Stream") -> None:
        self.log.append(f"{self.name} waits {other.name}")


def test_dual_stream_overlaps_shared_and_joins_before_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []
    main, alt = _Stream("main", log), _Stream("alt", log)

    @contextmanager
    def use_stream(stream: _Stream):
        log.append(f"enter {stream.name}")
        yield
        log.append(f"leave {stream.name}")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)

    hidden = MagicMock()
    hidden.shape = (2, 64)
    flat = MagicMock()
    hidden.view.return_value = flat
    shared = MagicMock()
    shared.record_stream.side_effect = lambda stream: log.append(
        f"shared records {stream.name}"
    )
    routed = MagicMock()
    summed = MagicMock()
    result = object()
    routed.__add__.return_value = summed
    summed.view.return_value = result

    moe = SimpleNamespace(alt_stream=alt)
    moe.shared_experts = MagicMock(
        side_effect=lambda _: (log.append("shared"), shared)[1]
    )
    moe.routed_expert_forward = MagicMock(
        side_effect=lambda _: (log.append("routed"), routed)[1]
    )

    actual = m3_model.MiniMaxM3MoE.dual_stream_moe_forward(moe, hidden)

    assert actual is result
    assert log == [
        "alt waits main",
        "enter alt",
        "shared",
        "leave alt",
        "routed",
        "main waits alt",
        "shared records main",
    ]
    routed.__add__.assert_called_once_with(shared)


class _Epilogue:
    """`aiter moe_output_epilogue` double: records the call, and `take`
    decides whether the routed MoE consumed it (running `before` as aiter's
    gather-reduce would)."""

    def __init__(self, log: list[str], take: bool) -> None:
        self.log, self.take, self.args = log, take, None

    @contextmanager
    def __call__(self, addend, scale, before=None):
        self.args = (addend, scale)
        handle = SimpleNamespace(consumed=False)
        self.log.append("epilogue open")
        yield handle
        if self.take:
            if before is not None:
                before()
            handle.consumed = True
        self.log.append("epilogue close")


def _glue_moe(log, epilogue, routed, scale=2.0):
    moe = SimpleNamespace(
        _glue_epilogue=epilogue,
        fuse_shared_experts=False,
        routed_scaling_factor=scale,
    )
    moe._routed_scale = lambda: m3_model.MiniMaxM3MoE._routed_scale(moe)
    moe._scale_routed = lambda r: m3_model.MiniMaxM3MoE._scale_routed(moe, r)
    moe._routed_plus_shared = lambda h, s, before_add=None: (
        m3_model.MiniMaxM3MoE._routed_plus_shared(moe, h, s, before_add)
    )
    moe._front_fusion = False
    moe._route_and_experts = lambda h: m3_model.MiniMaxM3MoE._route_and_experts(moe, h)
    moe.gate = MagicMock(side_effect=lambda _: (log.append("gate"), "logits")[1])
    moe.experts = MagicMock(side_effect=lambda **_: (log.append("experts"), routed)[1])
    return moe


@pytest.mark.parametrize("taken", [True, False])
def test_dual_stream_glue_joins_inside_the_routed_epilogue(
    monkeypatch: pytest.MonkeyPatch, taken: bool
) -> None:
    log: list[str] = []
    main, alt = _Stream("main", log), _Stream("alt", log)

    @contextmanager
    def use_stream(stream: _Stream):
        log.append(f"enter {stream.name}")
        yield
        log.append(f"leave {stream.name}")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)

    hidden = MagicMock()
    hidden.shape = (2, 64)
    hidden.view.return_value = hidden
    shared = MagicMock()
    shared.record_stream.side_effect = lambda stream: log.append(
        f"shared records {stream.name}"
    )
    routed = MagicMock()
    routed.view.return_value = "fused"
    scaled = MagicMock()
    routed.__mul__.return_value = scaled
    scaled.__add__.return_value = MagicMock(view=lambda *_: "separate")
    epilogue = _Epilogue(log, take=taken)
    moe = _glue_moe(log, epilogue, routed)
    moe.alt_stream = alt
    moe.shared_experts = MagicMock(
        side_effect=lambda _: (log.append("shared"), shared)[1]
    )

    actual = m3_model.MiniMaxM3MoE.dual_stream_moe_forward(moe, hidden)

    assert epilogue.args == (shared, 2.0)
    head = ["alt waits main", "enter alt", "shared", "leave alt", "epilogue open"]
    if taken:
        # aiter added shared into the routed gather-reduce, after the join.
        assert actual == "fused"
        assert log == head + [
            "gate",
            "experts",
            "main waits alt",
            "epilogue close",
            "shared records main",
        ]
    else:
        assert actual == "separate"
        routed.__mul__.assert_called_once_with(2.0)
        scaled.__add__.assert_called_once_with(shared)
        assert log == head + [
            "gate",
            "experts",
            "epilogue close",
            "main waits alt",
            "shared records main",
        ]


def test_serial_glue_runs_shared_first_into_the_epilogue() -> None:
    log: list[str] = []
    hidden = MagicMock()
    hidden.shape = (2, 64)
    hidden.view.return_value = hidden
    routed, shared = MagicMock(), MagicMock()
    routed.view.return_value = "fused"
    epilogue = _Epilogue(log, take=True)
    moe = _glue_moe(log, epilogue, routed, scale=1.0)
    moe._use_dual_stream = True
    moe.shared_input_from_moe_quant = False
    moe.shared_experts = MagicMock(
        side_effect=lambda _: (log.append("shared"), shared)[1]
    )

    actual = m3_model.MiniMaxM3MoE.single_stream_moe_forward(moe, hidden)

    assert actual == "fused"
    assert epilogue.args == (shared, 1.0)
    assert log == ["shared", "epilogue open", "gate", "experts", "epilogue close"]


def test_serial_forward_preserves_routed_then_shared_order() -> None:
    log: list[str] = []
    hidden = MagicMock()
    hidden.shape = (2, 64)
    flat = MagicMock()
    hidden.view.return_value = flat
    routed, shared, summed = MagicMock(), MagicMock(), MagicMock()
    result = object()
    routed.__add__.return_value = summed
    summed.view.return_value = result

    # The stack's serial path can also take the shared input from the routed
    # quant (ATOM_MINIMAX_M3_MOE_QUANT_SHARED); off here.
    moe = SimpleNamespace(shared_input_from_moe_quant=False)
    moe.routed_expert_forward = MagicMock(
        side_effect=lambda _: (log.append("routed"), routed)[1]
    )
    moe.shared_experts = MagicMock(
        side_effect=lambda _: (log.append("shared"), shared)[1]
    )

    actual = m3_model.MiniMaxM3MoE.single_stream_moe_forward(moe, hidden)

    assert actual is result
    assert log == ["routed", "shared"]


def test_moe_registers_dual_stream_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context: dict[str, object] = {}
    atom_config = SimpleNamespace(
        enable_tbo=False,
        compilation_config=SimpleNamespace(static_forward_context=context),
    )
    monkeypatch.setattr(m3_model, "get_current_atom_config", lambda: atom_config)
    monkeypatch.setattr(m3_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(m3_model, "ReplicatedLinear", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        m3_model,
        "FusedMoE",
        lambda **kwargs: SimpleNamespace(
            quant_method=None,
            num_fused_shared_experts=0,
        ),
    )
    monkeypatch.setattr(m3_model, "MiniMaxM3MLP", lambda **kwargs: object())
    monkeypatch.setattr(
        m3_model.envs, "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", 1024, raising=False
    )
    config = SimpleNamespace(
        hidden_size=64,
        num_local_experts=8,
        num_experts_per_tok=2,
        intermediate_size=32,
        n_shared_experts=1,
    )
    stream = object()
    prefix = "model.layers.3.block_sparse_moe"

    moe = m3_model.MiniMaxM3MoE(
        config,
        layer_id=3,
        prefix=prefix,
        alt_stream=stream,
    )

    assert moe._use_dual_stream
    assert moe.alt_stream is stream
    assert context[prefix] is moe


def test_decoder_threads_model_stream_into_moe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m3_model, "_sparse_attention_layer_ids", lambda _: set())
    monkeypatch.setattr(m3_model, "MiniMaxM3Attention", lambda **kwargs: object())
    monkeypatch.setattr(m3_model, "GemmaRMSNorm", lambda *args, **kwargs: object())
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        m3_model,
        "MiniMaxM3MoE",
        lambda **kwargs: (calls.append(kwargs), object())[1],
    )
    config = SimpleNamespace(
        hidden_size=64,
        moe_layer_freq=[1],
        rms_norm_eps=1e-6,
    )
    stream = object()

    m3_model.MiniMaxM3DecoderLayer(
        config,
        prefix="model.layers.0",
        layer_num=0,
        alt_stream=stream,
    )

    assert calls[0]["alt_stream"] is stream


def test_model_creates_one_shared_stream_for_all_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = object()
    layer_calls: list[dict[str, object]] = []

    def layer_type(*args, **kwargs):
        layer_calls.append(kwargs)
        return object()

    def make_layers(_count, factory, **_kwargs):
        return 0, 1, [factory("model.layers.0", layer_num=0)]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(m3_model, "_validate_sparse_flydsl_config", lambda *_: None)
    # main-only helper; this stack reads get_pp_group() directly.
    monkeypatch.setattr(m3_model, "pp_holds_embedding", lambda _: True, raising=False)
    monkeypatch.setattr(
        m3_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        m3_model, "VocabParallelEmbedding", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(m3_model, "GemmaRMSNorm", lambda *args, **kwargs: object())
    monkeypatch.setattr(m3_model, "make_layers", make_layers)
    monkeypatch.setattr(
        m3_model,
        "make_empty_intermediate_tensors_factory",
        lambda *args, **kwargs: object(),
    )
    config = SimpleNamespace(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=1,
        n_shared_experts=1,
        rms_norm_eps=1e-6,
    )
    atom_config = SimpleNamespace(
        hf_config=config,
        kv_cache_dtype="fp8",
        index_cache_dtype="fp8",
        quant_config=None,
        torch_dtype=torch.bfloat16,
        compilation_config=SimpleNamespace(level=0),
    )

    model = m3_model.MiniMaxM3Model(atom_config, layer_type=layer_type)

    assert model.alt_stream is stream
    assert layer_calls[0]["alt_stream"] is stream


class _Fold:
    """`aiter moe_shared_expert_fold` double: `take` marks it consumed."""

    def __init__(self, log: list[str], take: bool) -> None:
        self.log, self.take, self.calls = log, take, []

    @contextmanager
    def __call__(self, weights, scale):
        self.calls.append((weights, scale))
        fold = SimpleNamespace(consumed=False)
        self.log.append("fold enter")
        yield fold
        fold.consumed = self.take
        self.log.append("fold exit")


def _fold_moe(log: list[str], fold: _Fold, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(m3_model, "_moe_shared_expert_fold", fold)
    routed, shared = MagicMock(name="routed"), MagicMock(name="shared")
    moe = SimpleNamespace(
        alt_stream=None,
        _shared_fold_weights={"w13": "W"},
        _shared_fold_declined=set(),
        gate=lambda h: (log.append("gate"), "logits")[1],
        experts=MagicMock(side_effect=lambda **kw: (log.append("routed"), routed)[1]),
        shared_experts=MagicMock(
            side_effect=lambda h: (log.append("shared"), shared)[1]
        ),
        _routed_scale=lambda: 2.0,
        _scale_routed=lambda out: out,
        single_stream_moe_forward=MagicMock(name="single"),
    )
    moe._shared_fold_forward = lambda h: m3_model.MiniMaxM3MoE._shared_fold_forward(
        moe, h
    )
    moe._front_fusion = False
    moe._route_and_experts = lambda h: m3_model.MiniMaxM3MoE._route_and_experts(moe, h)
    hidden = MagicMock()
    hidden.shape = (96, 64)  # past the T<=64 floor that keeps the separate MLP
    hidden.view.return_value = hidden
    return moe, hidden, routed, shared


def test_shared_fold_runs_the_shared_expert_inside_the_routed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []
    fold = _Fold(log, take=True)
    moe, hidden, _, _ = _fold_moe(log, fold, monkeypatch)

    m3_model.MiniMaxM3MoE.dual_stream_moe_forward(moe, hidden)

    assert log == ["fold enter", "gate", "routed", "fold exit"]
    assert fold.calls == [({"w13": "W"}, 2.0)]
    moe.shared_experts.assert_not_called()
    assert moe._shared_fold_declined == set()


def test_declined_shared_fold_adds_the_shared_mlp_then_skips_the_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []
    fold = _Fold(log, take=False)
    moe, hidden, routed, shared = _fold_moe(log, fold, monkeypatch)

    m3_model.MiniMaxM3MoE.dual_stream_moe_forward(moe, hidden)

    assert log == ["fold enter", "gate", "routed", "fold exit", "shared"]
    routed.__add__.assert_called_once_with(shared)
    assert moe._shared_fold_declined == {96}
    # The same token count later goes to the stream path (here: no alt stream).
    m3_model.MiniMaxM3MoE.dual_stream_moe_forward(moe, hidden)
    assert len(fold.calls) == 1
    moe.single_stream_moe_forward.assert_called_once_with(hidden)


@pytest.mark.parametrize("rows", [1, 33, 64])
def test_shared_fold_keeps_the_separate_mlp_at_small_batches(
    monkeypatch: pytest.MonkeyPatch, rows: int
) -> None:
    """Batches of <= 64 tokens never enter the fold (its T<=64 launch is
    nondeterministic); they take the stream path as a declined count would."""
    log: list[str] = []
    fold = _Fold(log, take=True)
    moe, hidden, _, _ = _fold_moe(log, fold, monkeypatch)
    hidden.shape = (rows, 64)

    m3_model.MiniMaxM3MoE.dual_stream_moe_forward(moe, hidden)

    assert fold.calls == []
    moe.single_stream_moe_forward.assert_called_once_with(hidden)
    assert moe._shared_fold_declined == set()


@pytest.mark.parametrize("fused", [False, True])
def test_front_fusion_routes_through_the_fused_router(fused: bool) -> None:
    """With the fused router the routed branch hands the gate's weight to the
    experts instead of running the gate GEMM."""
    log: list[str] = []
    routed = MagicMock()
    moe = SimpleNamespace(
        _front_fusion=fused,
        fuse_shared_experts=False,
        routed_scaling_factor=1.0,
    )
    moe._routed_scale = lambda: m3_model.MiniMaxM3MoE._routed_scale(moe)
    moe._scale_routed = lambda r: m3_model.MiniMaxM3MoE._scale_routed(moe, r)
    moe._route_and_experts = lambda h: m3_model.MiniMaxM3MoE._route_and_experts(moe, h)
    moe.gate = MagicMock(side_effect=lambda _: (log.append("gate"), "logits")[1])
    moe.gate.weight = "gate_weight"
    moe.experts = MagicMock(side_effect=lambda **_: (log.append("experts"), routed)[1])
    moe.experts.forward_fused_router = MagicMock(
        side_effect=lambda h, w: (log.append(f"fused {w}"), routed)[1]
    )

    actual = m3_model.MiniMaxM3MoE.routed_expert_forward(moe, "hidden")

    assert actual is routed
    assert log == (["fused gate_weight"] if fused else ["gate", "experts"])


class _LateEpilogue(_Epilogue):
    """`_Epilogue` whose handle carries the addend (set late by the router hook)
    and is taken only when it has one."""

    @contextmanager
    def __call__(self, addend, scale, before=None):
        self.args = (addend, scale)
        handle = SimpleNamespace(consumed=False, addend=addend)
        self.log.append("epilogue open")
        yield handle
        if self.take and handle.addend is not None:
            if before is not None:
                before()
            handle.consumed = True
        self.args = (handle.addend, scale)
        self.log.append("epilogue close")


class _Event:
    def __init__(self, log):
        self.log = log

    def record(self, stream):
        self.log.append(f"fork recorded on {stream.name}")


def _routed_first_moe(monkeypatch, log, epilogue, routed, shared, fused=True):
    from atom.model_ops import moe as moe_ops

    main, alt = _Stream("main", log), _Stream("alt", log)
    alt.wait_event = lambda ev: log.append("alt waits fork")

    @contextmanager
    def use_stream(stream: _Stream):
        log.append(f"enter {stream.name}")
        yield
        log.append(f"leave {stream.name}")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: _Event(log))
    moe = _glue_moe(log, epilogue, routed)
    moe._routed_plus_shared = lambda h, s, before_add=None, launch_shared=None: (
        m3_model.MiniMaxM3MoE._routed_plus_shared(moe, h, s, before_add, launch_shared)
    )
    moe._routed_first = True
    moe.alt_stream = alt
    moe.shared_experts = MagicMock(
        side_effect=lambda _: (log.append("shared"), shared)[1]
    )
    if fused:
        moe._front_fusion = True
        moe.gate = SimpleNamespace(weight="w")

        def fused_router(h, w):
            log.append("router")
            moe_ops._run_after_router_hook()
            log.append("experts")
            return routed

        moe.experts = SimpleNamespace(forward_fused_router=fused_router)
    return moe


@pytest.mark.parametrize("fused", [True, False])
def test_routed_first_launches_shared_after_the_router(
    monkeypatch: pytest.MonkeyPatch, fused: bool
) -> None:
    log: list[str] = []
    hidden = MagicMock()
    hidden.shape = (2, 64)
    hidden.view.return_value = hidden
    shared = MagicMock()
    shared.record_stream.side_effect = lambda stream: log.append(
        f"shared records {stream.name}"
    )
    routed = MagicMock()
    routed.view.return_value = "fused"
    epilogue = _LateEpilogue(log, take=True)
    moe = _routed_first_moe(monkeypatch, log, epilogue, routed, shared, fused)

    actual = m3_model.MiniMaxM3MoE.dual_stream_moe_forward(moe, hidden)

    assert actual == "fused"
    assert epilogue.args == (shared, 2.0)
    launch = ["alt waits fork", "enter alt", "shared", "leave alt"]
    if fused:
        # The router is the fork's first dependent launch; the shared MLP
        # still waits only for the fork (the MoE input), not for the router.
        assert log == ["fork recorded on main", "epilogue open", "router"] + launch + [
            "experts",
            "main waits alt",
            "epilogue close",
            "shared records main",
        ]
    else:
        # No router hook ran: the shared MLP after the routed one, its add
        # separate (the epilogue had no addend when the routed MoE ran).
        assert log[:4] == ["fork recorded on main", "epilogue open", "gate", "experts"]
        assert log[4:8] == launch


def test_tail_in_ar_defers_only_without_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[str] = []

    @contextmanager
    def defer():
        log.append("defer open")
        yield SimpleNamespace(deferred=None)
        log.append("defer close")

    monkeypatch.setattr(m3_model, "_moe_defer_tail", defer)
    hidden = MagicMock()
    hidden.shape = (80, 64)
    routed, shared = MagicMock(), MagicMock()
    epilogue = _Epilogue(log, take=True)
    moe = _glue_moe(log, epilogue, routed)
    moe._tail_in_ar = True
    owner = SimpleNamespace(_forward_hooks={}, _forward_pre_hooks={})
    moe._owner_layer = lambda: owner
    moe._defer_tail_now = lambda h: m3_model.MiniMaxM3MoE._defer_tail_now(moe, h)

    assert m3_model.MiniMaxM3MoE._routed_plus_shared(moe, hidden, shared) is routed
    assert log == [
        "epilogue open",
        "defer open",
        "gate",
        "experts",
        "defer close",
        "epilogue close",
    ]

    log.clear()
    owner._forward_hooks = {0: "dspark tap"}
    m3_model.MiniMaxM3MoE._routed_plus_shared(moe, hidden, shared)
    assert "defer open" not in log

    log.clear()
    owner._forward_hooks = {}
    nxt = SimpleNamespace(_forward_pre_hooks={0: "input tap"})
    owner._next_layer = lambda: nxt
    m3_model.MiniMaxM3MoE._routed_plus_shared(moe, hidden, shared)
    assert "defer open" not in log

    log.clear()
    nxt._forward_pre_hooks = {}
    hidden.shape = (m3_model._TAIL_IN_AR_MAX_ROWS + 1, 64)
    m3_model.MiniMaxM3MoE._routed_plus_shared(moe, hidden, shared)
    assert "defer open" not in log


def test_tail_in_ar_wraps_the_folded_call(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[str] = []

    @contextmanager
    def defer():
        log.append("defer open")
        yield SimpleNamespace(deferred=None)
        log.append("defer close")

    monkeypatch.setattr(m3_model, "_moe_defer_tail", defer)
    fold = _Fold(log, take=True)
    moe, hidden, routed, _ = _fold_moe(log, fold, monkeypatch)
    moe._tail_in_ar = True
    owner = SimpleNamespace(_forward_hooks={}, _forward_pre_hooks={})
    moe._owner_layer = lambda: owner
    moe._defer_tail_now = lambda h: m3_model.MiniMaxM3MoE._defer_tail_now(moe, h)

    assert m3_model.MiniMaxM3MoE._shared_fold_forward(moe, hidden) is routed
    assert log == [
        "fold enter",
        "defer open",
        "gate",
        "routed",
        "defer close",
        "fold exit",
    ]
