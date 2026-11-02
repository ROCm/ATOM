# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_PREFILL_AR_QR_MIN_TOKENS: when the fused all-reduce + Gemma RMSNorm
splits into quick-reduce + a separate norm.

The first part needs no AITER. The second part checks the GemmaRMSNorm wiring
and the op bodies with every collective and kernel replaced; it imports
`atom.model_ops.layernorm`, which needs a ROCm device to import, but launches
nothing on it.
"""

import pytest
import torch

from atom.model_ops.prefill_ar_route import prefill_ar_takes_split, takes_gemma_norm
from atom.utils import envs


def _never():
    raise AssertionError("the communicator must not be consulted")


@pytest.mark.parametrize("num_tokens", [1, 4, 96, 2047, 100_000])
def test_off_never_asks_the_communicator(num_tokens):
    assert prefill_ar_takes_split(num_tokens, 0, _never) is False
    assert prefill_ar_takes_split(num_tokens, -1, _never) is False


@pytest.mark.parametrize("num_tokens", [1, 4, 96, 2047])
def test_below_the_threshold_never_asks_the_communicator(num_tokens):
    assert prefill_ar_takes_split(num_tokens, 2048, _never) is False


@pytest.mark.parametrize("num_tokens", [2048, 3106, 6386, 8191, 8192, 16384])
def test_from_the_threshold_quick_reduce_decides(num_tokens):
    assert prefill_ar_takes_split(num_tokens, 2048, lambda: True) is True
    # quick-reduce off, or the tensor outside its size range: fused path
    assert prefill_ar_takes_split(num_tokens, 2048, lambda: False) is False


def test_env_default_is_off(monkeypatch):
    monkeypatch.delenv("ATOM_PREFILL_AR_QR_MIN_TOKENS", raising=False)
    assert envs.ATOM_PREFILL_AR_QR_MIN_TOKENS == 0
    monkeypatch.setenv("ATOM_PREFILL_AR_QR_MIN_TOKENS", "2048")
    assert envs.ATOM_PREFILL_AR_QR_MIN_TOKENS == 2048


# aiter's rmsnorm2d_fwd_with_add before and after ROCm/aiter#3872: the seventh
# parameter changed from use_model_sensitive_rmsnorm to gemma_norm.


def _old_aiter_add_rmsnorm(
    out,
    input,
    residual_in,
    residual_out,
    weight,
    epsilon,
    use_model_sensitive_rmsnorm=0,
):
    pass


def _new_aiter_add_rmsnorm(
    out,
    input,
    residual_in,
    residual_out,
    weight,
    epsilon,
    gemma_norm=False,
    use_model_sensitive_rmsnorm=0,
):
    pass


def _registered(schema):
    from types import SimpleNamespace

    return SimpleNamespace(default=SimpleNamespace(_schema=schema))


_OLD_SCHEMA = (
    "aiter::rmsnorm2d_fwd_with_add(Tensor(a0!) out, Tensor input, "
    "Tensor residual_in, Tensor(a3!) residual_out, Tensor weight, "
    "float epsilon, SymInt use_model_sensitive_rmsnorm=0) -> ()"
)
_NEW_SCHEMA = (
    "aiter::rmsnorm2d_fwd_with_add(Tensor(a0!) out, Tensor input, "
    "Tensor residual_in, Tensor(a3!) residual_out, Tensor weight, "
    "float epsilon, bool gemma_norm=False, "
    "SymInt use_model_sensitive_rmsnorm=0) -> ()"
)


def _wrapper(*args, **kwargs):  # what aiter's torch_compile_guard returns
    pass


@pytest.mark.parametrize(
    "op,fn,expect",
    [
        (None, _old_aiter_add_rmsnorm, False),
        (None, _new_aiter_add_rmsnorm, True),
        # registered op: the schema decides, the (*args, **kwargs) wrapper does not
        (_registered(_OLD_SCHEMA), _wrapper, False),
        (_registered(_NEW_SCHEMA), _wrapper, True),
    ],
)
def test_gemma_norm_support_is_read_off_aiter(op, fn, expect):
    assert takes_gemma_norm(op, fn) is expect


def test_old_aiter_rejects_gemma_norm_by_keyword():
    # Passed by keyword, the flag cannot land on use_model_sensitive_rmsnorm:
    # an old aiter fails loudly instead of normalizing with x * w.
    args = (None,) * 6
    _new_aiter_add_rmsnorm(*args, gemma_norm=True)
    with pytest.raises(TypeError):
        _old_aiter_add_rmsnorm(*args, gemma_norm=True)


# ---------------------------------------------------------------------------
# GemmaRMSNorm wiring and op bodies (ROCm device needed to import aiter)
# ---------------------------------------------------------------------------

HIDDEN = 256


@pytest.fixture
def ln(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("atom.model_ops.layernorm imports aiter, which needs a ROCm device")
    from atom.model_ops import layernorm

    monkeypatch.setattr(layernorm, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(layernorm, "fused_ar_gemma_per_group_supported", lambda: True)
    monkeypatch.setattr(layernorm, "fused_ar_gemma_per_token_supported", lambda: True)
    monkeypatch.setattr(layernorm, "add_rmsnorm_gemma_supported", lambda: True)
    return layernorm


@pytest.fixture
def seen(ln, monkeypatch):
    """Record which collective ran; every one returns correctly shaped CPU tensors."""
    calls = []

    def fused(x, residual, weight, eps, **kw):
        calls.append(("fused", x.shape[0], kw))
        return torch.empty_like(x), torch.empty_like(residual)

    def fused_quant(x, residual, weight, eps, **kw):
        calls.append(("fused_quant", x.shape[0], kw))
        m, n = x.shape
        outs = (
            torch.empty(m, n, dtype=torch.uint8),
            torch.empty_like(residual),
            torch.empty(m, 1),
        )
        return outs + (torch.empty_like(x),) if kw.get("emit_bf16") else outs

    class _QR:
        disabled = False

        def should_quick_allreduce(self, x):
            return True

        def quick_all_reduce(self, x):
            calls.append(("qr", x.shape[0], {}))
            return x.clone()

    # keyword-only: a positional seventh argument is use_model_sensitive_rmsnorm
    # on older aiter
    def add_rmsnorm(out, inp, residual, residual_out, weight, eps, *, gemma_norm):
        calls.append(("norm", inp.shape[0], {"gemma_norm": gemma_norm}))

    def hip_quant(quant_type):
        def run(x, quant_dtype=None):
            calls.append(("quant", x.shape[0], {"quant_type": quant_type}))
            return torch.empty(x.shape, dtype=torch.uint8), torch.empty(x.shape[0], 1)

        return run

    import aiter

    monkeypatch.setattr(ln, "tensor_model_parallel_fused_allreduce_rmsnorm", fused)
    monkeypatch.setattr(
        ln, "tensor_model_parallel_fused_allreduce_rmsnorm_quant", fused_quant
    )
    monkeypatch.setattr(ln, "_tp_quick_reduce", lambda: _QR())
    monkeypatch.setattr(ln, "rmsnorm2d_fwd_with_add", add_rmsnorm)
    monkeypatch.setattr(aiter, "get_hip_quant", hip_quant)
    return calls


def _norm(ln, monkeypatch, min_tokens, quant=None, write_bf16=False):
    from types import SimpleNamespace

    from aiter import QuantType

    if min_tokens is None:
        monkeypatch.delenv("ATOM_PREFILL_AR_QR_MIN_TOKENS", raising=False)
    else:
        monkeypatch.setenv("ATOM_PREFILL_AR_QR_MIN_TOKENS", str(min_tokens))
    monkeypatch.setenv("ATOM_FUSED_NORM_PER_TOKEN_QUANT", "1")
    qc = {
        None: None,
        "per_token": SimpleNamespace(quant_type=QuantType.per_Token),
        "per_group": SimpleNamespace(quant_type=QuantType.per_1x128),
    }[quant]
    return ln.GemmaRMSNorm(
        HIDDEN, quant_config=qc, write_bf16=write_bf16, fused_allreduce=True
    )


def _run(norm, m):
    return norm(torch.randn(m, HIDDEN), torch.randn(m, HIDDEN))


def test_unset_env_keeps_the_fused_call(ln, seen, monkeypatch):
    norm = _norm(ln, monkeypatch, None)
    assert norm.ar_split_min_tokens == 0
    _run(norm, 4096)
    assert [c[0] for c in seen] == ["fused"]
    assert seen[0][2]["gemma_norm"] is True


@pytest.mark.parametrize("m,expect", [(16, ["fused"]), (4096, ["qr", "norm"])])
def test_bf16_norm_routes_by_token_count(ln, seen, monkeypatch, m, expect):
    norm = _norm(ln, monkeypatch, 2048)
    assert norm.ar_split_min_tokens == 2048
    out, res = _run(norm, m)
    assert [c[0] for c in seen] == expect
    assert out.shape == (m, HIDDEN) and res.shape == (m, HIDDEN)
    if "norm" in expect:
        assert seen[-1][2]["gemma_norm"] is True  # (1 + w), as the fused kernel


@pytest.mark.parametrize("write_bf16", [False, True])
@pytest.mark.parametrize(
    "m,expect", [(16, ["fused_quant"]), (4096, ["qr", "norm", "quant"])]
)
def test_per_token_norm_routes_and_keeps_its_contract(
    ln, seen, monkeypatch, m, expect, write_bf16
):
    from aiter import QuantType

    norm = _norm(ln, monkeypatch, 2048, quant="per_token", write_bf16=write_bf16)
    out_fp8, scale, out_bf16, res = _run(norm, m)
    assert [c[0] for c in seen] == expect
    assert out_fp8.shape == (m, HIDDEN) and scale.shape == (m, 1)
    assert res.shape == (m, HIDDEN)
    if write_bf16:
        assert out_bf16 is not None and out_bf16.shape == (m, HIDDEN)
    else:
        assert out_bf16 is None
    if expect[0] == "fused_quant":
        kw = seen[0][2]
        assert kw["quant_type"] == "per_token" and kw["gemma_norm"] is True
        assert kw["emit_bf16"] is write_bf16
    else:
        assert seen[-1][2]["quant_type"] == QuantType.per_Token


def test_per_group_norm_is_never_routed(ln, seen, monkeypatch):
    norm = _norm(ln, monkeypatch, 2048, quant="per_group", write_bf16=True)
    assert norm.ar_split_min_tokens == 0
    _run(norm, 4096)
    assert [c[0] for c in seen] == ["fused_quant"]
    assert seen[0][2]["quant_type"] == "per_group"


def test_quick_reduce_off_falls_back_to_fused(ln, seen, monkeypatch):
    monkeypatch.setattr(ln, "_tp_quick_reduce", lambda: None)
    norm = _norm(ln, monkeypatch, 2048)
    _run(norm, 4096)
    assert [c[0] for c in seen] == ["fused"]


def test_aiter_without_gemma_norm_keeps_the_fused_call(ln, seen, monkeypatch):
    monkeypatch.setattr(ln, "add_rmsnorm_gemma_supported", lambda: False)
    norm = _norm(ln, monkeypatch, 2048)
    assert norm.ar_split_min_tokens == 0
    _run(norm, 4096)
    assert [c[0] for c in seen] == ["fused"]
