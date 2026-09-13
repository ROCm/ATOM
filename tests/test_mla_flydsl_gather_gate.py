# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch and fallback contracts for FlyDSL ``gather_kv_b_proj``."""

import logging
from types import SimpleNamespace

import pytest
import torch

try:
    from atom.model_ops import attention_mla
    from atom.model_ops.attention_mla import MLAAttention

    _IMPORT_ERROR = None
except ImportError as error:  # aiter/triton are absent on CPU-only runners
    _IMPORT_ERROR = str(error)

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"requires full atom import environment: {_IMPORT_ERROR}",
)


@pytest.fixture(autouse=True)
def _reset_once_log():
    attention_mla._flydsl_gather_logged.clear()
    yield
    attention_mla._flydsl_gather_logged.clear()


def _fake_attention(use_flydsl: bool):
    weight = torch.empty(3072, 512, dtype=torch.float8_e4m3fn, device="meta")
    return SimpleNamespace(
        kv_b_proj=SimpleNamespace(
            weight=weight,
            weight_scale=torch.empty(3072, 1, dtype=torch.float32, device="meta"),
        ),
        _k_scale=torch.ones(1, device="meta"),
        use_flydsl_gather_kv_b_proj=use_flydsl,
    )


def _call(attention) -> None:
    tensors = [torch.empty(0, device="meta") for _ in range(6)]
    MLAAttention._kv_b_proj_gather(attention, *tensors)


@pytest.fixture
def _backend_calls(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        attention_mla,
        "gather_kv_b_proj",
        lambda *args, **kwargs: calls.append("triton"),
    )
    monkeypatch.setattr(attention_mla, "_FLYDSL_GATHER_AVAILABLE", True)
    monkeypatch.setattr(
        attention_mla,
        "gather_kv_b_proj_flydsl",
        lambda *args, **kwargs: calls.append("flydsl"),
        raising=False,
    )
    return calls


def test_disabled_flydsl_uses_triton(_backend_calls):
    _call(_fake_attention(use_flydsl=False))

    assert _backend_calls == ["triton"]


def test_supported_call_uses_flydsl(_backend_calls):
    _call(_fake_attention(use_flydsl=True))

    assert _backend_calls == ["flydsl"]


def test_unavailable_flydsl_uses_triton(_backend_calls, monkeypatch):
    monkeypatch.setattr(attention_mla, "_FLYDSL_GATHER_AVAILABLE", False)
    attention = _fake_attention(use_flydsl=True)

    _call(attention)

    assert _backend_calls == ["triton"]
    assert attention.use_flydsl_gather_kv_b_proj is False


def test_rejected_call_falls_back_without_retrying(
    _backend_calls,
    monkeypatch,
    caplog,
):
    def reject(*args, **kwargs):
        _backend_calls.append("flydsl")
        raise ValueError("unsupported dtype")

    monkeypatch.setattr(attention_mla, "gather_kv_b_proj_flydsl", reject)
    attention = _fake_attention(use_flydsl=True)

    with caplog.at_level(logging.WARNING, logger="atom"):
        _call(attention)

    assert _backend_calls == ["flydsl", "triton"]
    assert attention.use_flydsl_gather_kv_b_proj is False
    assert "unsupported dtype" in caplog.text

    _backend_calls.clear()
    _call(attention)
    assert _backend_calls == ["triton"]


def test_runtime_error_is_not_swallowed(_backend_calls, monkeypatch):
    def fail_after_launch(*args, **kwargs):
        raise RuntimeError("HIP error: illegal memory access")

    monkeypatch.setattr(attention_mla, "gather_kv_b_proj_flydsl", fail_after_launch)

    with pytest.raises(RuntimeError, match="illegal memory access"):
        _call(_fake_attention(use_flydsl=True))

    assert _backend_calls == []
