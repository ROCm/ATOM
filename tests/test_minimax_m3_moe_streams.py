# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.abc
import importlib.util
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import _ExternalModuleLoader, atom_under_stubs


class _FlyDSLStub(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.partition(".")[0] == "flydsl":
            return importlib.util.spec_from_loader(
                name, _ExternalModuleLoader(), is_package=True
            )
        return None


@pytest.fixture(scope="module")
def m3():
    # The model's mono imports reach FlyDSL, which the CPU job does not have.
    saved = {n: m for n, m in sys.modules.items() if n.partition(".")[0] == "flydsl"}
    for name in saved:
        del sys.modules[name]
    finder = _FlyDSLStub()
    sys.meta_path.insert(0, finder)
    try:
        with atom_under_stubs() as load:
            yield load("atom.models.minimax_m3.model"), load(
                "atom.model_ops.module_dispatch_ops"
            )
    finally:
        sys.meta_path.remove(finder)
        for name in [n for n in sys.modules if n.partition(".")[0] == "flydsl"]:
            del sys.modules[name]
        sys.modules.update(saved)


def _fake_streams(monkeypatch, log):
    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            log.append(f"{self.name} waits {other.name}")

    @contextmanager
    def on(stream):
        log.append(f"enter {stream.name}")
        yield
        log.append(f"leave {stream.name}")

    main = Stream("main")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "stream", on)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda t, s: log.append(f"record {s.name}")
    )
    return Stream("alt")


def test_dual_stream_matches_single_stream(m3, monkeypatch):
    model, _ = m3
    log = []
    moe = SimpleNamespace(
        alt_stream=_fake_streams(monkeypatch, log),
        shared_experts=lambda h: (log.append("shared"), h * 2)[1],
        routed_expert_forward=lambda h: (log.append("routed"), h + 1)[1],
    )
    x = torch.randn(2, 3, 8)

    dual = model.MiniMaxM3MoE.dual_stream_moe_forward(moe, x)

    assert log == ["alt waits main", "enter alt", "shared", "leave alt"] + [
        "routed",
        "main waits alt",
        "record main",
    ]
    assert torch.equal(dual, model.MiniMaxM3MoE.single_stream_moe_forward(moe, x))


@pytest.mark.parametrize("tbo", [False, True])
def test_dual_stream_dispatch(m3, monkeypatch, tbo):
    model, ops = m3
    cfg = SimpleNamespace(
        enable_tbo=tbo, compilation_config=SimpleNamespace(static_forward_context={})
    )
    decode = SimpleNamespace(context=SimpleNamespace(is_prefill=False))
    monkeypatch.setattr(model, "get_current_atom_config", lambda: cfg)
    monkeypatch.setattr(ops, "get_current_atom_config", lambda: cfg)
    monkeypatch.setattr(ops, "get_forward_context", lambda: decode)
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(model, "ReplicatedLinear", lambda *a, **k: None)
    monkeypatch.setattr(model, "MiniMaxM3MLP", lambda **k: object())
    monkeypatch.setattr(
        model,
        "FusedMoE",
        lambda **k: SimpleNamespace(quant_method=None, num_fused_shared_experts=0),
    )
    monkeypatch.setattr(
        model.envs, "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", 4, raising=False
    )
    hf = SimpleNamespace(
        hidden_size=8,
        num_local_experts=8,
        num_experts_per_tok=2,
        intermediate_size=8,
        n_shared_experts=1,
    )
    moe = model.MiniMaxM3MoE(hf, layer_id=0, prefix="moe", alt_stream=object())
    moe.dual_stream_moe_forward = lambda h: "dual"
    moe.single_stream_moe_forward = lambda h: "single"

    assert moe._use_dual_stream is not tbo
    if tbo:
        assert moe.forward(torch.empty(4, 8)) == "single"
    else:
        rows = (4, 5)  # at and past the threshold
        got = [ops.maybe_dual_stream_forward(torch.empty(n, 8), "moe") for n in rows]
        assert got == ["dual", "single"]
