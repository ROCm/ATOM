# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The TP-fused GemmaRMSNorm hands the kernel its raw weight.

The fused all-reduce + RMSNorm (+ per-group FP8 quant) kernel applies the Gemma
`(1 + w)` itself (`gemma_norm=True`), so the norm keeps no derived copy of its
weight -- a rollout weight sync rewrites `weight` in place and the next forward
uses it. An aiter whose per-group path cannot do the Gemma norm drops the flag
silently, so that combination must be refused, not run.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    pytest.skip(
        "atom.model_ops.layernorm imports aiter, which needs a ROCm device",
        allow_module_level=True,
    )

from aiter import QuantType

from atom.model_ops import layernorm as layernorm_module
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.rollout.weight_updater import WeightUpdaterMixin

HIDDEN = 256
PER_GROUP = SimpleNamespace(quant_type=QuantType.per_1x128)
PER_TOKEN = SimpleNamespace(quant_type=QuantType.per_Token)


@pytest.fixture
def calls(monkeypatch):
    """TP=4, a gemma-capable aiter, and the fused collectives recorded."""
    seen = []

    def fused_quant(x, residual, weight, eps, **kw):
        seen.append(("quant", weight, weight.detach().clone(), kw))
        m, n = x.shape
        return (
            torch.empty_like(x),
            torch.empty_like(residual),
            torch.empty(m, n // 128),
            torch.empty_like(x),
        )

    def fused(x, residual, weight, eps, **kw):
        seen.append(("bf16", weight, weight.detach().clone(), kw))
        return torch.empty_like(x), torch.empty_like(residual)

    monkeypatch.setattr(
        layernorm_module, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(
        layernorm_module, "fused_ar_gemma_per_group_supported", lambda: True
    )
    monkeypatch.setattr(
        layernorm_module, "fused_ar_gemma_per_token_supported", lambda: True
    )
    monkeypatch.setattr(
        layernorm_module,
        "tensor_model_parallel_fused_allreduce_rmsnorm_quant",
        fused_quant,
    )
    monkeypatch.setattr(
        layernorm_module, "tensor_model_parallel_fused_allreduce_rmsnorm", fused
    )
    return seen


def _norm(quant):
    norm = GemmaRMSNorm(
        HIDDEN,
        quant_config=PER_GROUP if quant else None,
        write_bf16=quant,
        fused_allreduce=True,
    )
    norm.weight.data.fill_(0.5)
    return norm


def _run(norm):
    x = torch.randn(3, HIDDEN)
    return norm(x, torch.randn(3, HIDDEN))


@pytest.mark.parametrize("quant", [True, False])
def test_the_kernel_gets_the_raw_weight_and_gemma_norm(calls, quant):
    norm = _norm(quant)
    _run(norm)
    kind, weight, _, kw = calls[-1]
    assert kind == ("quant" if quant else "bf16")
    assert weight is norm.weight
    assert kw["gemma_norm"] is True
    if quant:
        assert kw["quant_type"] == "per_group"
        assert kw["emit_bf16"] is True


def test_no_derived_copy_of_the_weight_is_kept(calls):
    norm = _norm(quant=True)
    assert list(norm.buffers()) == []
    assert not hasattr(norm, "process_weights_after_loading")


@pytest.mark.parametrize("quant", [True, False])
def test_a_rollout_sync_reaches_the_next_forward(calls, quant):
    norm = _norm(quant)
    model = nn.Module()
    model.norm = norm

    class _Runner(WeightUpdaterMixin):
        device = torch.device("cpu")
        label = "test"
        rank = 0
        world_size = 1

        def clear_kv_cache(self):
            pass

    runner = _Runner()
    runner.model = model
    new = torch.randn(HIDDEN)
    assert runner.update_weights([("norm.weight", new)]) == 1

    _run(norm)
    _, _, weight_seen, _ = calls[-1]
    assert torch.equal(weight_seen, new)


def test_an_aiter_without_per_group_gemma_is_refused(calls, monkeypatch):
    monkeypatch.setattr(
        layernorm_module, "fused_ar_gemma_per_group_supported", lambda: False
    )
    with pytest.raises(ValueError, match="gemma_norm"):
        _norm(quant=True)
    # The bf16 fused AR + Gemma RMSNorm has taken gemma_norm all along.
    _run(_norm(quant=False))


def test_support_is_read_off_the_group_method(monkeypatch):
    from aiter.dist.parallel_state import GroupCoordinator

    probe = layernorm_module.fused_ar_gemma_per_group_supported
    for has_flag in (True, False):
        if has_flag:

            def method(self, input_, residual_inp_, weight_, eps, gemma_norm=False):
                pass

        else:

            def method(self, input_, residual_inp_, weight_, eps):
                pass

        monkeypatch.setattr(
            GroupCoordinator, "fused_allreduce_rmsnorm_quant_per_group", method
        )
        probe.cache_clear()
        assert probe() is has_flag
    probe.cache_clear()


# ---------------------------------------------------------------------------
# Per-token FP8 quant folded into the norm (ATOM_FUSED_NORM_PER_TOKEN_QUANT)
# ---------------------------------------------------------------------------


def _per_token_norm(monkeypatch, on):
    monkeypatch.setenv("ATOM_FUSED_NORM_PER_TOKEN_QUANT", "1" if on else "0")
    norm = GemmaRMSNorm(
        HIDDEN,
        quant_config=PER_TOKEN,
        write_bf16=False,
        fused_allreduce=True,
    )
    norm.weight.data.fill_(0.5)
    return norm


def test_per_token_quant_is_off_by_default(calls, monkeypatch):
    # The flag changes the norm's return arity for every per-token model that
    # reads `use_fused_quant`, so it stays opt-in until measured.
    norm = _per_token_norm(monkeypatch, on=False)
    assert norm.use_fused_quant is False
    assert norm.per_token_quant is False
    _run(norm)
    assert calls[-1][0] == "bf16"


def test_per_token_quant_asks_the_kernel_for_per_token(calls, monkeypatch):
    norm = _per_token_norm(monkeypatch, on=True)
    assert norm.use_fused_quant is True
    assert norm.per_token_quant is True
    _run(norm)
    kind, weight, _, kw = calls[-1]
    assert kind == "quant"
    assert weight is norm.weight
    assert kw["gemma_norm"] is True
    assert kw["quant_type"] == "per_token"
    # per-group-only arguments must not be sent: aiter's one-kernel per-token
    # branch is keyed on the default group_size, and a per-token scale is a
    # single column so there is nothing to transpose.
    assert "group_size" not in kw
    assert "transpose_scale" not in kw


def test_per_group_arguments_are_unchanged(calls, monkeypatch):
    monkeypatch.setenv("ATOM_FUSED_NORM_PER_TOKEN_QUANT", "1")
    norm = _norm(quant=True)
    _run(norm)
    _, _, _, kw = calls[-1]
    assert kw["quant_type"] == "per_group"
    assert kw["group_size"] == 128
    assert "transpose_scale" in kw


def test_an_aiter_without_gemma_per_token_is_refused(calls, monkeypatch):
    monkeypatch.setattr(
        layernorm_module, "fused_ar_gemma_per_token_supported", lambda: False
    )
    with pytest.raises(ValueError, match="per-token"):
        _per_token_norm(monkeypatch, on=True)


def test_per_token_without_allreduce_norms_then_quantizes(monkeypatch):
    """Layer 0 and any all-reduce-free configuration take the two-kernel path.

    aiter's `rmsnorm_quant` has no `gemma_norm`, so this path cannot fuse
    without silently normalizing with `x * w`. It must still return the same
    (fp8, scale, bf16[, residual]) contract the all-reduce path returns, or the
    decoder layer unpacks a different number of values depending on the layer
    index.
    """
    monkeypatch.setenv("ATOM_FUSED_NORM_PER_TOKEN_QUANT", "1")
    monkeypatch.setattr(
        layernorm_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    norm = GemmaRMSNorm(HIDDEN, quant_config=PER_TOKEN, write_bf16=True)
    assert norm.per_token_quant is True
    assert norm.fused_allreduce is False

    normed = torch.randn(3, HIDDEN)
    res_out = torch.randn(3, HIDDEN)
    quantized = torch.empty(3, HIDDEN, dtype=torch.uint8)
    scale = torch.empty(3, 1)
    seen = {}

    monkeypatch.setattr(
        GemmaRMSNorm, "forward_cuda", lambda self, x, r=None: (normed, res_out)
    )

    def hip_quant(quant_type):
        seen["quant_type"] = quant_type

        def run(x, quant_dtype=None):
            seen["x"] = x
            return quantized, scale

        return run

    import aiter

    monkeypatch.setattr(aiter, "get_hip_quant", hip_quant)

    out_fp8, out_scale, out_bf16, residual_out = norm(
        torch.randn(3, HIDDEN), torch.randn(3, HIDDEN)
    )
    assert out_fp8 is quantized
    assert out_scale is scale
    assert out_bf16 is normed  # write_bf16: the pre-quant normed output
    assert residual_out is res_out
    assert seen["x"] is normed
    assert seen["quant_type"] == QuantType.per_Token
