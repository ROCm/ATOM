# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest
from vllm.config import KVTransferConfig

from atom.plugin.vllm.model_wrapper import _enable_lmcache_v3_for_dsa


def _config(connector: str | None):
    return (
        SimpleNamespace(kv_transfer_config=None)
        if connector is None
        else SimpleNamespace(
            kv_transfer_config=KVTransferConfig(
                kv_connector=connector,
                kv_role="kv_both",
                kv_connector_extra_config={},
            )
        )
    )


def test_glm_dsa_enables_lmcache_v3() -> None:
    config = _config("LMCacheConnectorV1")

    _enable_lmcache_v3_for_dsa(config, "GlmMoeDsaForCausalLM")

    assert config.kv_transfer_config.kv_connector_extra_config == {
        "lmcache.use_gpu_connector_v3": True
    }


def test_deepseek_v32_enables_lmcache_v3() -> None:
    config = _config("LMCacheConnectorV1")

    _enable_lmcache_v3_for_dsa(config, "DeepseekV32ForCausalLM")

    assert config.kv_transfer_config.kv_connector_extra_config == {
        "lmcache.use_gpu_connector_v3": True
    }


def test_lmcache_v3_explicit_true_is_preserved() -> None:
    config = _config("LMCacheConnectorV1")
    config.kv_transfer_config.kv_connector_extra_config[
        "lmcache.use_gpu_connector_v3"
    ] = True

    _enable_lmcache_v3_for_dsa(config, "GlmMoeDsaForCausalLM")

    assert (
        config.kv_transfer_config.kv_connector_extra_config[
            "lmcache.use_gpu_connector_v3"
        ]
        is True
    )


@pytest.mark.parametrize(
    "extra_config",
    [
        {"lmcache.use_gpu_connector_v3": False},
        {"use_native": True},
    ],
)
def test_incompatible_lmcache_modes_fail_early(extra_config) -> None:
    config = _config("LMCacheConnectorV1")
    config.kv_transfer_config.kv_connector_extra_config.update(extra_config)

    with pytest.raises(ValueError, match="require"):
        _enable_lmcache_v3_for_dsa(config, "GlmMoeDsaForCausalLM")


def test_missing_non_dsa_or_non_lmcache_config_is_unchanged() -> None:
    missing = _config(None)
    non_dsa = _config("LMCacheConnectorV1")
    non_lmcache = _config("MooncakeConnector")

    _enable_lmcache_v3_for_dsa(missing, "GlmMoeDsaForCausalLM")
    _enable_lmcache_v3_for_dsa(non_dsa, "Qwen3ForCausalLM")
    _enable_lmcache_v3_for_dsa(non_lmcache, "GlmMoeDsaForCausalLM")

    assert missing.kv_transfer_config is None
    assert non_dsa.kv_transfer_config.kv_connector_extra_config == {}
    assert non_lmcache.kv_transfer_config.kv_connector_extra_config == {}
