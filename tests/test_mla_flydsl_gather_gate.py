# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Which backend `_kv_b_proj_gather` dispatches to, and what happens on refusal.

`ATOM_USE_FLYDSL_GATHER_KV_B_PROJ` swaps the Triton gather for the flydsl
gather-GEMM. The flydsl backend covers exactly one shape family and validates
that itself, raising ValueError before it launches anything -- ATOM does not
restate those preconditions, it just calls and catches. So what needs pinning
here is the dispatch: off means Triton, on means flydsl, and a refusal must fall
back to Triton (with the outputs still produced) rather than take the server
down, and must not re-attempt on every later chunk.
"""

from types import SimpleNamespace

import pytest
import torch

try:
    from atom.model_ops import attention_mla
    from atom.model_ops.attention_mla import MLAAttention

    _IMPORT_ERR = None
except ImportError as _e:  # aiter/triton absent on a CPU-only runner
    _IMPORT_ERR = str(_e)

pytestmark = pytest.mark.skipif(
    _IMPORT_ERR is not None, reason=f"requires full atom import env: {_IMPORT_ERR}"
)


@pytest.fixture(autouse=True)
def _reset_once_log():
    """The one-time log latch is module state; keep tests independent."""
    attention_mla._flydsl_gather_logged.clear()
    yield
    attention_mla._flydsl_gather_logged.clear()


def _fake_self(use_flydsl):
    weight = torch.empty(3072, 512, dtype=torch.float8_e4m3fn, device="meta")
    return SimpleNamespace(
        kv_b_proj=SimpleNamespace(
            weight=weight,
            weight_scale=torch.empty(3072, 1, dtype=torch.float32, device="meta"),
        ),
        _k_scale=torch.ones(1, device="meta"),
        use_flydsl_gather_kv_b_proj=use_flydsl,
    )


def _call(obj):
    """Invoke the unbound method -- building a real MLAAttention needs a GPU."""
    args = [torch.empty(0, device="meta")] * 5 + [torch.empty(0, device="meta")]
    MLAAttention._kv_b_proj_gather(obj, *args)


@pytest.fixture
def _spies(monkeypatch):
    calls = []
    monkeypatch.setattr(
        attention_mla,
        "gather_kv_b_proj",
        lambda *a, **k: calls.append("triton"),
    )
    return calls


def test_flag_off_uses_triton(_spies, monkeypatch):
    monkeypatch.setattr(
        attention_mla,
        "_load_flydsl_gather_kv_b_proj",
        lambda: pytest.fail("flydsl must not even be imported while the flag is off"),
    )
    _call(_fake_self(use_flydsl=False))
    assert _spies == ["triton"]


def test_flag_on_uses_flydsl(_spies, monkeypatch):
    monkeypatch.setattr(
        attention_mla,
        "_load_flydsl_gather_kv_b_proj",
        lambda: (lambda *a, **k: _spies.append("flydsl")),
    )
    _call(_fake_self(use_flydsl=True))
    assert _spies == ["flydsl"]


def test_unavailable_flydsl_falls_back_to_triton(_spies, monkeypatch):
    monkeypatch.setattr(attention_mla, "_load_flydsl_gather_kv_b_proj", lambda: None)
    obj = _fake_self(use_flydsl=True)
    _call(obj)
    assert _spies == ["triton"]
    assert obj.use_flydsl_gather_kv_b_proj is False


def test_rejected_shape_falls_back_and_stops_retrying(_spies, monkeypatch, caplog):
    def _reject(*a, **k):
        _spies.append("flydsl")
        raise ValueError("[FlyDSL gather_kv_b_proj] page_size 1 only")

    monkeypatch.setattr(attention_mla, "_load_flydsl_gather_kv_b_proj", lambda: _reject)
    obj = _fake_self(use_flydsl=True)
    with caplog.at_level("WARNING"):
        _call(obj)
    assert _spies == ["flydsl", "triton"]
    assert obj.use_flydsl_gather_kv_b_proj is False
    assert "page_size 1 only" in caplog.text

    # A second chunk must go straight to Triton, not re-raise through flydsl.
    _spies.clear()
    _call(obj)
    assert _spies == ["triton"]


def test_non_value_error_is_not_swallowed(_spies, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("HIP error: illegal memory access")

    monkeypatch.setattr(attention_mla, "_load_flydsl_gather_kv_b_proj", lambda: _boom)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        _call(_fake_self(use_flydsl=True))
    assert _spies == []
