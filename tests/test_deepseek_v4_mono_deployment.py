# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which deployments the DeepSeek-V4 Mono decode admits."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

# Loaded from its file: the module needs only dataclasses, but importing it by
# name runs atom.models.deepseek_v4's __init__, which needs aiter.
_spec = importlib.util.spec_from_file_location(
    "_deepseek_v4_mono_profile",
    Path(__file__).parents[1] / "atom/models/deepseek_v4/mono/profile.py",
)
profile = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = profile  # @dataclass looks its module up by name
_spec.loader.exec_module(profile)
ORDINARY_PROFILE = profile.ORDINARY_PROFILE
deployment_refusal = profile.deployment_refusal


class _Mode:
    name = "FULL"

    def uses_tr_hybrid(self):
        return False

    def decode_mode(self):
        return self


_MODEL_ARGS = SimpleNamespace(
    use_index_cache=False, index_topk_freq=1, index_topk_pattern=None
)


def _atom_config(**overrides):
    fields = {
        "tensor_parallel_size": 8,
        "enable_expert_parallel": False,
        "enable_dp_attention": False,
        "enable_tbo": False,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "is_multinode_tp": False,
        "parallel_config": SimpleNamespace(data_parallel_size=1),
        "kv_cache_dtype": "fp8",
        "index_cache_dtype": "fp4",
        "compilation_config": SimpleNamespace(level=0, cudagraph_mode=_Mode()),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_pure_tp8_on_one_node_is_admitted():
    config = _atom_config()
    assert deployment_refusal(config, ORDINARY_PROFILE, model_args=_MODEL_ARGS) is None


def test_tp_group_across_nodes_is_refused():
    config = _atom_config(is_multinode_tp=True)
    why = deployment_refusal(config, ORDINARY_PROFILE, model_args=_MODEL_ARGS)
    assert why == "TP group spans nodes (HIP IPC peer memory)"
