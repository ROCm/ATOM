# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""ATOM_M3_QKV_CHAIN: ``qkv_chain.take`` (which decode steps take the chain), ``layer_ok`` (static eligibility) and the
park / pop hand-off between the qkv linear and the attention op."""

from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

try:
    from atom.model_ops.minimax_m3 import qkv_chain
except ImportError as exc:  # no aiter on a CPU-only runner
    skip_if_dependency_missing(exc, "atom.model_ops.minimax_m3.qkv_chain")


def _ctx(monkeypatch, **kw):
    import atom.utils.forward_context as fc

    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=False, is_dummy_run=False, **kw)
    )
    monkeypatch.setattr(fc, "get_forward_context", lambda: ctx)


@pytest.mark.parametrize(
    "rows, want",
    [
        (8, False),
        (16, False),
        (17, True),
        (40, True),
        (64, True),
        (65, False),
        (128, False),
    ],
)
def test_take_row_window(monkeypatch, rows, want):
    monkeypatch.setattr(qkv_chain, "split_available", lambda: False)
    _ctx(monkeypatch)
    assert qkv_chain.take(rows) is want


@pytest.mark.parametrize(
    "rows, want",
    [
        (16, False),
        (17, True),
        (64, True),
        (65, True),
        (96, True),
        (128, True),
        (129, False),
    ],
)
def test_take_row_window_with_split_chain(monkeypatch, rows, want):
    monkeypatch.setattr(qkv_chain, "split_available", lambda: True)
    _ctx(monkeypatch)
    assert qkv_chain.take(rows) is want


def test_code_rows_without_split_chain(monkeypatch):
    monkeypatch.setattr(qkv_chain, "split_available", lambda: False)
    assert [qkv_chain.code_rows(m) for m in (17, 32, 33, 64)] == [32, 32, 48, 64]


@pytest.mark.parametrize("flags", [{"is_prefill": True}, {"is_dummy_run": True}])
def test_take_refuses_prefill_and_dummy(monkeypatch, flags):
    import atom.utils.forward_context as fc

    base = {"is_prefill": False, "is_dummy_run": False, **flags}
    ctx = SimpleNamespace(context=SimpleNamespace(**base))
    monkeypatch.setattr(fc, "get_forward_context", lambda: ctx)
    assert qkv_chain.take(40) is False


def test_layer_ok(monkeypatch):
    monkeypatch.setattr(qkv_chain, "enabled", lambda: True)
    ok = (16, 1, 1, 64, "nvfp4", "nvfp4", False, False)
    assert qkv_chain.layer_ok(*ok)
    for i, bad in enumerate([8, 2, 2, 32, "fp8", "fp8", True, True]):
        args = list(ok)
        args[i] = bad
        assert not qkv_chain.layer_ok(*args), i
    monkeypatch.setattr(qkv_chain, "enabled", lambda: False)
    assert not qkv_chain.layer_ok(*ok)


def test_pop_hands_off_once_and_checks_rows():
    qkv_chain._set((40, "codes", "scales", "w", "s"))
    assert qkv_chain.pop(40)[0] == 40
    assert qkv_chain.pop(40) is None  # consumed
    qkv_chain._set((40, "codes", "scales", "w", "s"))
    with pytest.raises(AssertionError):
        qkv_chain.pop(48)
    qkv_chain._set(None)


def test_front_fake_shape():
    x = torch.zeros((3, 6144), dtype=torch.bfloat16)
    w = torch.zeros((2560, 6144), dtype=torch.uint8)
    y = qkv_chain._front_fake(x, w, None, None, False)
    assert y.shape == (3, 2560) and y.dtype == torch.bfloat16


def test_take_refuses_token_graph(monkeypatch):
    import atom.utils.forward_context as fc

    for token_graph, want in ((True, False), (False, True)):
        ctx = SimpleNamespace(
            context=SimpleNamespace(
                is_prefill=False,
                is_dummy_run=False,
                forward_mode=SimpleNamespace(token_graph=token_graph),
            )
        )
        monkeypatch.setattr(fc, "get_forward_context", lambda ctx=ctx: ctx)
        assert qkv_chain.take(40) is want


@pytest.mark.parametrize("tr_graph", ["record", "replay"])
def test_take_refuses_tr_hybrid_record_and_replay(monkeypatch, tr_graph):
    # the TR_HYBRID record pass has forward_mode None; only ForwardContext.tr_graph marks it
    import atom.utils.forward_context as fc

    ctx = SimpleNamespace(
        tr_graph=tr_graph,
        context=SimpleNamespace(
            is_prefill=False, is_dummy_run=False, forward_mode=None
        ),
    )
    monkeypatch.setattr(fc, "get_forward_context", lambda: ctx)
    assert qkv_chain.take(40) is False
    ctx.tr_graph = None
    assert qkv_chain.take(40) is True


def test_enabled_needs_gfx950(monkeypatch):
    monkeypatch.setattr(qkv_chain.envs, "ATOM_M3_QKV_CHAIN", True, raising=False)
    monkeypatch.setattr(qkv_chain, "available", lambda: True)
    monkeypatch.setattr(qkv_chain, "_is_gfx950", lambda: False)
    assert not qkv_chain.enabled()
    monkeypatch.setattr(qkv_chain, "_is_gfx950", lambda: True)
    assert qkv_chain.enabled()


def test_linear_ok_requires_native_1x32_mxfp8():
    ok = SimpleNamespace(
        native_a8_group_rows=1,
        params_dtype=torch.float8_e4m3fn,
        runs_bf16=False,
        bias=None,
    )
    assert qkv_chain.linear_ok(ok)
    for field, bad in (
        ("native_a8_group_rows", None),  # online requant / generic path
        ("native_a8_group_rows", 32),  # 32x32 source
        ("params_dtype", torch.bfloat16),
        ("params_dtype", torch.float4_e2m1fn_x2),  # MXFP4 weights
        ("runs_bf16", True),
        ("bias", object()),
    ):
        assert not qkv_chain.linear_ok(
            SimpleNamespace(**{**ok.__dict__, field: bad})
        ), field
    assert not qkv_chain.linear_ok(SimpleNamespace())


def test_handoff_is_per_thread():
    import threading

    qkv_chain._set((40, "codes", "scales", "w", "s"))
    seen = []
    t = threading.Thread(target=lambda: seen.append(qkv_chain.pop(40)))
    t.start()
    t.join()
    assert seen == [None]  # another thread (TBO micro-batch) never sees it
    assert qkv_chain.pop(40)[0] == 40


def test_weight_cache_key_tracks_the_chain_flag():
    from atom.model_loader.weight_cache import key

    assert "ATOM_M3_QKV_CHAIN" in key._ENV_VARS


def test_one_shot_activity_log(monkeypatch, caplog):
    import logging

    import atom.utils.forward_context as fc

    st = {r: 0 for r in qkv_chain._REASONS}
    st.update(calls=0, layers_on=3, layers_not_preshuffled=1, done=False)
    monkeypatch.setattr(qkv_chain, "_stats", st)
    monkeypatch.setattr(qkv_chain, "_LOG_AFTER", 5)
    monkeypatch.setattr(qkv_chain, "split_available", lambda: False)  # the 17-64 window
    monkeypatch.setattr(
        qkv_chain, "aiter_features", lambda: "qkv_chain_cooperative_insert"
    )

    def ctx(**kw):
        base = {"is_prefill": False, "is_dummy_run": False, **kw}
        return SimpleNamespace(context=SimpleNamespace(**base))

    seq = [
        (ctx(), 40, True),
        (ctx(is_prefill=True), 40, False),
        (ctx(is_dummy_run=True), 40, False),
        (ctx(), 128, False),
        (ctx(forward_mode=SimpleNamespace(token_graph=True)), 40, False),
        (ctx(), 40, True),  # after the tally: counting stopped
    ]
    with caplog.at_level(logging.INFO, logger="atom"):
        for c, rows, want in seq:
            monkeypatch.setattr(fc, "get_forward_context", lambda c=c: c)
            assert qkv_chain.take(rows) is want
    msgs = [
        r.getMessage() for r in caplog.records if "ATOM_M3_QKV_CHAIN" in r.getMessage()
    ]
    assert len(msgs) == 2
    assert "3 layer(s) with the chain" in msgs[0] and "max_rows=64" in msgs[0]
    assert "aiter features: qkv_chain_cooperative_insert" in msgs[0]
    assert "taken=1, prefill=1, dummy=1, token_graph=1, out_of_window=1" in msgs[1]
    assert st["calls"] == 5 and st["done"]


def test_linear_imports_qkv_chain_before_the_preshuffle_branch():
    """The not-preshuffled fallback calls ``qkv_chain.note_layer``: the import must run on both branches
    (it used to sit inside the is_shuffled branch -> UnboundLocalError at load with preshuffle off).
    """
    import ast
    import importlib.util

    # read the source without importing linear (that needs a visible GPU for the arch probe)
    with open(importlib.util.find_spec("atom.model_ops.linear").origin) as f:
        src = f.read()
    hits = []
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and ast.unparse(node.test) == "getattr(self, '_qkv_chain', False)"
        ):
            hits.append(node)
    assert len(hits) == 1
    first = hits[0].body[0]
    assert isinstance(first, ast.ImportFrom) and first.names[0].name == "qkv_chain"
