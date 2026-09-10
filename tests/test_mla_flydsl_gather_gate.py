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
def _reset_once_log(monkeypatch):
    """The one-time log latch is module state; keep tests independent."""
    attention_mla._flydsl_gather_logged.clear()
    monkeypatch.setattr(attention_mla, "_FLYDSL_GATHER_AVAILABLE", True)
    monkeypatch.setattr(attention_mla, "_FLYDSL_GATHER_FP8_AVAILABLE", True)
    monkeypatch.setattr(
        attention_mla,
        "gather_kv_b_proj_flydsl",
        lambda *a, **k: pytest.fail("unexpected FlyDSL gather call"),
        raising=False,
    )
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
        use_flydsl_gather_kv_b_proj=bool(use_flydsl),
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
        "gather_kv_b_proj_flydsl",
        lambda *a, **k: pytest.fail("flydsl must not be called while the flag is off"),
    )
    _call(_fake_self(use_flydsl=False))
    assert _spies == ["triton"]


def test_flag_on_uses_flydsl(_spies, monkeypatch):
    monkeypatch.setattr(
        attention_mla,
        "gather_kv_b_proj_flydsl",
        lambda *a, **k: _spies.append("flydsl"),
    )
    _call(_fake_self(use_flydsl=True))
    assert _spies == ["flydsl"]


@pytest.mark.parametrize("symbol_present", [False, True])
def test_unavailable_flydsl_falls_back_to_triton(_spies, monkeypatch, symbol_present):
    monkeypatch.setattr(attention_mla, "_FLYDSL_GATHER_AVAILABLE", False)
    monkeypatch.setattr(attention_mla, "_FLYDSL_GATHER_FP8_AVAILABLE", False)
    if not symbol_present:
        monkeypatch.delattr(attention_mla, "gather_kv_b_proj_flydsl")
    obj = _fake_self(use_flydsl=True)
    _call(obj)
    assert _spies == ["triton"]
    assert obj.use_flydsl_gather_kv_b_proj is True  # Env preference is unchanged.


def test_rejected_shape_falls_back_and_stops_retrying(_spies, monkeypatch, caplog):
    def _reject(*a, **k):
        _spies.append("flydsl")
        raise ValueError("[FlyDSL gather_kv_b_proj] page_size 1 only")

    monkeypatch.setattr(attention_mla, "gather_kv_b_proj_flydsl", _reject)
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

    monkeypatch.setattr(attention_mla, "gather_kv_b_proj_flydsl", _boom)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        _call(_fake_self(use_flydsl=True))
    assert _spies == []


@pytest.mark.parametrize("reject", [False, True])
def test_fp8_output_alias_and_bf16_fallback(monkeypatch, reject):
    obj = _fake_self(True)
    k = torch.full((17, 12, 192), 9, dtype=torch.bfloat16)
    v = torch.full((17, 12, 128), 9, dtype=torch.bfloat16)
    ks, vs = torch.tensor([0.25]), torch.tensor([0.5])
    seen = []

    def fly(*args, **kwargs):
        ko, vo = args[7:9]
        assert ko.dtype == vo.dtype == torch.float8_e4m3fn
        assert ko.data_ptr() == k.data_ptr()
        assert vo.data_ptr() == v.data_ptr()
        assert ko.is_contiguous() and vo.is_contiguous()
        assert kwargs["k_out_scale"] is ks
        assert kwargs["v_out_scale"] is vs
        if reject:
            raise ValueError("unsupported shape")
        ko.fill_(2)
        vo.fill_(3)
        seen.extend([ko, vo])

    def triton(*args, **kwargs):
        assert args[7] is k and args[8] is v
        k.fill_(4)
        v.fill_(5)

    monkeypatch.setattr(attention_mla, "gather_kv_b_proj_flydsl", fly)
    monkeypatch.setattr(attention_mla, "gather_kv_b_proj", triton)
    unused = torch.empty(0, device="meta")
    result = MLAAttention._kv_b_proj_gather(
        obj, unused, unused, unused, unused, k, v, kv_out_scales=(ks, vs)
    )
    if reject:
        assert result is None
        assert (k == 4).all() and (v == 5).all()
        assert not obj.use_flydsl_gather_kv_b_proj
    else:
        assert result[0] is seen[0] and result[1] is seen[1]
        assert result[2] is ks and result[3] is vs
        assert (result[0].float() == 2).all() and (result[1].float() == 3).all()


def test_bf16_only_gather_retains_dynamic_quantization(monkeypatch):
    monkeypatch.setattr(attention_mla, "_FLYDSL_GATHER_FP8_AVAILABLE", False)
    obj = _fake_self(True)
    k = torch.empty((17, 12, 192), dtype=torch.bfloat16)
    v = torch.empty((17, 12, 128), dtype=torch.bfloat16)

    def bf16_gather(*args, **kwargs):
        assert args[7] is k and args[8] is v
        assert "k_out_scale" not in kwargs and "v_out_scale" not in kwargs
        k.fill_(4)
        v.fill_(5)

    monkeypatch.setattr(attention_mla, "gather_kv_b_proj_flydsl", bf16_gather)
    unused = torch.empty(0, device="meta")
    result = MLAAttention._kv_b_proj_gather(
        obj,
        unused,
        unused,
        unused,
        unused,
        k,
        v,
        kv_out_scales=(torch.ones(1), torch.ones(1)),
    )
    assert result is None  # Caller must quantize the BF16 outputs.
    assert (k == 4).all() and (v == 5).all()
