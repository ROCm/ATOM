# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K3 mono refuses a checkpoint whose MoE router is not the one it bakes.

`kernels/stages/moe.py`'s route stage does not read `moe_router_activation_func`
or `moe_renormalize`: it scores with sigmoid and renormalizes the pre-bias
sigmoids by an IEEE `1 / sum`, full stop. The production path reads both off
the config and hands them to `FusedMoE` (`kimi_k3/model.py`). Today's shipped
`moonshotai/Kimi-K3` is sigmoid + renormalize, so these refusals are latent --
they fix no live miscomputation. They exist because the failure they prevent is
silent: a softmax checkpoint scored as sigmoid, or unnormalized routing weights
normalized anyway, is a different model with no error raised anywhere.
"""

from types import SimpleNamespace

import pytest

from atom.models.kimi_k3.mono import config as k3_config
from atom.models.kimi_k3.mono.config import (
    ATTN_RES_BLOCK,
    BLOCKS,
    CONV_WIDTH,
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    KDA_HEADS,
    LATENT,
    MOE_INTER,
    SHARED_EXPERTS,
    SITU_BETA,
    SITU_LINEAR_BETA,
    TOPK,
    config_refusal,
)
from atom.mono.runtime import deployment


def _hf_config(**overrides):
    fields = {
        "hidden_size": HIDDEN,
        "linear_num_key_heads": KDA_HEADS,
        "linear_key_head_dim": HEAD_DIM,
        "linear_attn_config": {
            "short_conv_kernel_size": CONV_WIDTH,
            "use_full_rank_gate": True,
            "gate_lower_bound": 0.0,
        },
        "attn_res_block_size": ATTN_RES_BLOCK,
        "num_experts": EXPERTS,
        "num_experts_per_token": TOPK,
        "moe_intermediate_size": MOE_INTER,
        "num_shared_experts": SHARED_EXPERTS,
        "routed_expert_hidden_size": LATENT,
        "latent_moe_use_norm": True,
        "routed_scaling_factor": 1.0,
        "num_expert_group": 1,
        # what moonshotai/Kimi-K3 actually ships
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "activation_situ_beta": SITU_BETA,
        "activation_situ_linear_beta": SITU_LINEAR_BETA,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _atom_config(**hf_overrides):
    return SimpleNamespace(
        hf_config=_hf_config(**hf_overrides),
        tensor_parallel_size=8,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_tbo=False,
        enable_tbo_decode=False,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(
            level=3,
            cudagraph_mode=SimpleNamespace(
                requires_piecewise_compilation=lambda: False
            ),
        ),
        speculative_config=SimpleNamespace(method="dspark", num_speculative_tokens=3),
    )


@pytest.fixture
def servable(monkeypatch):
    """Everything outside the model config already satisfied, so
    `config_refusal` is reporting on the checkpoint and nothing else."""
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)
    monkeypatch.setattr(k3_config, "_cu_count", lambda: BLOCKS)
    monkeypatch.setattr(k3_config.envs, "ATOM_MONO_ENABLE", True)
    monkeypatch.setattr(k3_config.envs, "ATOM_GDN_SSM_DTYPE", "fp32")
    monkeypatch.setattr(k3_config.envs, "ATOM_ENABLE_REPLAYSSM", False)


def test_the_shipped_k3_router_is_served(servable):
    # sigmoid + renormalize -- so the two refusals below are latent today, and
    # this is the assertion that keeps them latent.
    assert config_refusal(_atom_config()) is None


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"moe_router_activation_func": "softmax"}, "router activation"),
        ({"moe_renormalize": False}, "does not renormalize"),
    ],
)
def test_a_router_the_route_stage_does_not_implement_is_refused(
    servable, override, reason
):
    assert reason in config_refusal(_atom_config(**override))
