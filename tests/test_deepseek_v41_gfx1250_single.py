# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which deployments and which steps the gfx1250 DeepSeek-V4.1 single-launch decode step takes (no GPU)."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")

from atom.models.deepseek_v41.mono import dispatch as mono_dispatch
from atom.models.deepseek_v41.mono import gfx1250_single
from atom.models.deepseek_v41.mono.gfx1250_single import config, dispatch
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.mono.runtime import deployment


def atom_config(**overrides):
    fields = {
        "tensor_parallel_size": 4,
        "enable_expert_parallel": False,
        "enable_dp_attention": False,
        "parallel_config": NS(data_parallel_size=1),
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_tbo": False,
        "enable_tbo_decode": False,
        "compilation_config": NS(level=0),
        "kv_cache_dtype": "bf16",
        "index_cache_dtype": "fp8",
        "speculative_config": NS(method="dspark", num_speculative_tokens=5),
    }
    fields.update(overrides)
    return NS(**fields)


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(deployment, "_gfx", lambda: "gfx1250")
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)
    monkeypatch.setattr(deployment, "_takes_weight_updates", lambda c: False)
    monkeypatch.setattr(deployment, "_uses_tr_hybrid", lambda c: False)
    monkeypatch.setattr(deployment, "pcp_group_size_of", lambda c: 1)
    monkeypatch.setenv("ATOM_MONO_SINGLE_SHADOW", "0")


def test_the_supported_deployment_is_not_refused():
    assert config.config_refusal(atom_config(), Dims) is None


@pytest.mark.parametrize(
    "change, why",
    [
        ({"tensor_parallel_size": 8}, "TP 8"),
        ({"kv_cache_dtype": "fp8"}, "kv cache fp8"),
        ({"index_cache_dtype": "fp4"}, "index cache fp4"),
        (
            {"speculative_config": NS(method="mtp", num_speculative_tokens=5)},
            "speculative method mtp",
        ),
        (
            {"speculative_config": NS(method="dspark", num_speculative_tokens=3)},
            "DSpark block size",
        ),
        ({"speculative_config": None}, "speculative method None"),
        ({"enable_expert_parallel": True}, "expert parallel"),
        ({"pipeline_parallel_size": 2}, "PP > 1"),
    ],
)
def test_other_deployments_are_refused(change, why):
    assert why in config.config_refusal(atom_config(**change), Dims)


def test_the_arch_refusal_of_the_gfx950_kernels_does_not_apply():
    # the shared refusals reject every arch but gfx950; the single launch is gfx1250 code
    assert config.config_refusal(atom_config(), Dims) is None


class _Lazy:
    """LazyRunner stand-in: records the widths it was asked to make ready."""

    def __init__(self, create, name):
        self.asked = []
        self.runner = object()

    def ready(self, rows):
        self.asked.append(rows)
        return True


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(dispatch, "LazyRunner", _Lazy)

    class Inner(nn.Module):
        def forward(self, input_ids, positions, inputs_embeds=None):
            return ("inner", input_ids, inputs_embeds)

        def compute_logits(self, hidden_states):
            return hidden_states

    return dispatch.SingleStepModel(Inner(), drafter=None, atom_config=atom_config())


@pytest.mark.parametrize(
    "rows", range(config.ROWS_PER_REQ, config.MAX_ROWS + 1, config.ROWS_PER_REQ)
)
def test_every_whole_request_width_up_to_eight_requests_is_served(model, rows):
    assert model.step_ready(rows)
    assert model._single.asked == [rows]


@pytest.mark.parametrize("rows", [0, 1, 5, 7, 13, 47, 54, 96])
def test_other_widths_take_the_production_path(model, rows):
    assert not model.step_ready(rows)
    assert model._single.asked == []


def test_the_forward_is_the_production_one(model):
    ids = torch.arange(6)
    assert model(ids, ids)[0] == "inner"
    assert model.compute_logits(3) == 3


def test_the_wrapper_forwards_the_model_attributes(model):
    model.model.extra = 7
    assert model.extra == 7


def test_off_unless_switched_on(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_SINGLE", "0")
    assert not gfx1250_single.enabled()


def test_a_refused_deployment_keeps_the_model(monkeypatch):
    inner = nn.Linear(1, 1)
    out = dispatch.install_single_step(
        inner, atom_config(kv_cache_dtype="fp8"), drafter=None
    )
    assert out is inner


def test_mono_dispatch_routes_to_the_single_launch_when_enabled(monkeypatch):
    monkeypatch.setattr(gfx1250_single, "enabled", lambda: True)
    seen = {}

    def install(model, cfg, drafter):
        seen["args"] = (model, cfg, drafter)
        return "single"

    monkeypatch.setattr(dispatch, "install_single_step", install)
    cfg = atom_config()
    assert mono_dispatch.install_mono_decode("m", cfg, "d") == "single"
    assert seen["args"] == ("m", cfg, "d")
