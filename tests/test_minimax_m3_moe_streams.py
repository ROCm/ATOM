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
