# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which kernel runs the KDA gating update of Kimi-K3 decode and spec verify.

aiter's gfx1250 kernel when aiter's `supports` takes the model setup, ATOM's
kernel otherwise. AITER, Triton and einops are stood in for where they are not
installed, so these tests run on the CPU job.
"""

import inspect
from unittest.mock import Mock

import pytest
import torch
from aiter_stub import atom_without_aiter, einops_where_missing


@pytest.fixture(scope="module")
def model():
    with atom_without_aiter() as load, einops_where_missing():
        yield load("atom.models.kimi_k3.model")


@pytest.fixture
def decide(model, monkeypatch):
    """`_use_aiter_kda_gating` decided afresh, with aiter's `supports` mocked."""
    monkeypatch.setattr(model, "logger", Mock())

    def decide(supported=True, ssm_dtype="fp32"):
        supports = Mock(return_value=supported)
        monkeypatch.setattr(model, "aiter_kda_gating_supports", supports)
        monkeypatch.setenv("ATOM_GDN_SSM_DTYPE", ssm_dtype)
        model._use_aiter_kda_gating.cache_clear()
        return model._use_aiter_kda_gating(-5.0, 128, torch.bfloat16), supports

    yield decide
    model._use_aiter_kda_gating.cache_clear()


@pytest.mark.parametrize("supported", [True, False], ids=["supported", "unsupported"])
@pytest.mark.parametrize(
    "ssm_dtype, state_dtype",
    [("fp32", torch.float32), ("bf16", torch.bfloat16)],
    ids=["fp32-pool", "bf16-pool"],
)
def test_route_is_what_aiter_supports(decide, supported, ssm_dtype, state_dtype):
    on_aiter, supports = decide(supported, ssm_dtype)
    assert on_aiter is supported
    supports.assert_called_once_with(
        lower_bound=-5.0,
        state_dtype=state_dtype,
        head_k_dim=128,
        head_v_dim=128,
        act_dtype=torch.bfloat16,
    )


def test_route_is_decided_and_logged_once(model, decide, monkeypatch):
    assert decide()[0] is True
    monkeypatch.setattr(model, "aiter_kda_gating_supports", Mock(return_value=False))
    assert model._use_aiter_kda_gating(-5.0, 128, torch.bfloat16) is True
    model.logger.info.assert_called_once()


def test_routed_call_that_aiter_declines_raises(model, monkeypatch):
    monkeypatch.setattr(model, "aiter_kda_gating_decode", lambda **kwargs: None)
    args = dict.fromkeys(inspect.signature(model._kda_gating_update).parameters)
    args["on_aiter"] = True
    with pytest.raises(RuntimeError, match="declined a call routed to it"):
        model._kda_gating_update(**args)
