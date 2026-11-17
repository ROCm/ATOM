# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which kernel runs the KDA gating update of Kimi-K3 decode and spec verify.

`_use_aiter_kda_gating` decides once: aiter's gfx1250 kernel for the
lower-bounded gate over an fp32 state pool on gfx1250, ATOM's Triton kernel
otherwise. A routed call that aiter declines raises instead of running ATOM's
kernel. The kernels are recorders here, and AITER, Triton and einops are
stood in for where they are not installed, so these tests run on the CPU job.
"""

import importlib.util
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, Mock

import pytest
import torch
from aiter_stub import atom_without_aiter


@contextmanager
def _einops_where_missing():
    """model.py imports einops at the top, and the CPU job does not install it."""
    if "einops" in sys.modules or importlib.util.find_spec("einops") is not None:
        yield
        return
    stand_in = types.ModuleType("einops")
    stand_in.rearrange = MagicMock(name="einops.rearrange")
    sys.modules["einops"] = stand_in
    try:
        yield
    finally:
        sys.modules.pop("einops", None)


@pytest.fixture(scope="module")
def model():
    with atom_without_aiter() as load, _einops_where_missing():
        yield load("atom.models.kimi_k3.model")


@pytest.fixture
def decide(model, monkeypatch):
    """`_use_aiter_kda_gating` for an arch and state pool dtype, decided afresh."""
    monkeypatch.setattr(model, "logger", Mock())

    def decide(lower_bound=-5.0, gfx="gfx1250", ssm_dtype="fp32"):
        monkeypatch.setattr(model, "get_gfx", lambda: gfx)
        monkeypatch.setenv("ATOM_GDN_SSM_DTYPE", ssm_dtype)
        model._use_aiter_kda_gating.cache_clear()
        return model._use_aiter_kda_gating(lower_bound)

    yield decide
    model._use_aiter_kda_gating.cache_clear()


def test_gfx1250_lower_bound_fp32_state_routes_to_aiter(decide):
    assert decide() is True


@pytest.mark.parametrize(
    "case",
    [{"gfx": "gfx950"}, {"lower_bound": None}, {"ssm_dtype": "fp16"}],
    ids=["gfx950", "no-lower-bound", "fp16-state"],
)
def test_other_deployments_stay_on_atom_kernel(decide, case):
    assert decide(**case) is False


def test_route_is_decided_and_logged_once(model, decide, monkeypatch):
    assert decide() is True
    monkeypatch.setattr(model, "get_gfx", lambda: "gfx950")
    assert all(model._use_aiter_kda_gating(-5.0) for _ in range(3))
    model.logger.info.assert_called_once()


def _spec_verify_args():
    def act(*shape):
        return torch.empty(*shape, dtype=torch.bfloat16)

    return {
        "A_log": torch.empty(12),
        "a": act(1, 6, 12, 128),
        "b": act(1, 6, 12),
        "dt_bias": torch.empty(12 * 128),
        "q": act(1, 6, 12, 128),
        "k": act(1, 6, 12, 128),
        "v": act(1, 6, 12, 128),
        "o": act(6, 12, 128),
        "initial_state": torch.empty(6, 12, 128, 128),
        "cu_seqlens": torch.tensor([0, 3, 6], dtype=torch.int32),
        "ssm_state_indices": torch.arange(6, dtype=torch.int32).view(2, 3),
        "ssm_state_indices_in": None,
        "num_accepted_tokens": torch.tensor([1, 2], dtype=torch.int32),
        "lower_bound": -5.0,
    }


@pytest.fixture
def kernels(model, monkeypatch):
    """Both kernels replaced by recorders of the keywords they receive."""
    calls = {}

    def recorder(name):
        def kernel(**kwargs):
            calls[name] = kwargs
            return kwargs["o"], kwargs["initial_state"]

        return kernel

    monkeypatch.setattr(model, "aiter_kda_gating_update", recorder("aiter"))
    monkeypatch.setattr(
        model, "fused_sigmoid_gating_delta_rule_update", recorder("atom")
    )
    return calls


@pytest.mark.parametrize("on_aiter, chosen", [(True, "aiter"), (False, "atom")])
def test_call_reaches_only_the_chosen_kernel(model, kernels, on_aiter, chosen):
    args = _spec_verify_args()
    model._kda_gating_update(on_aiter=on_aiter, **args)
    assert list(kernels) == [chosen]
    sent = kernels[chosen]
    assert all(sent[name] is value for name, value in args.items())
    assert sent["is_kda"] and sent["inplace_final_state"]
    assert sent["use_qk_l2norm_in_kernel"]


def test_routed_call_that_aiter_declines_raises(model, kernels, monkeypatch):
    monkeypatch.setattr(model, "aiter_kda_gating_update", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="declined a call routed to it"):
        model._kda_gating_update(on_aiter=True, **_spec_verify_args())
    assert kernels == {}
