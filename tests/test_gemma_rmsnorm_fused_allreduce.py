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
