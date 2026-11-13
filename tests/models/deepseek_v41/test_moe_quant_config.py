# SPDX-License-Identifier: MIT
"""Model-owned MoE quantization, including engine online overrides."""

from unittest.mock import Mock

import pytest

pytest.importorskip("aiter", reason="the quant types are AITER enums")

from aiter import QuantType

from atom.models.deepseek_v41 import dspark, model
from atom.models.deepseek_v41.layers import native_quant_config


@pytest.mark.parametrize(
    "online",
    [
        None,
        {"global_quant_config": "ptpc_fp8"},
        {"layer_quant_config": {"*.ffn.experts*": "ptpc_fp8"}},
        # Names the shared experts too, which stay pinned.
        {"layer_quant_config": {"*experts*": "ptpc_fp8"}},
    ],
)
@pytest.mark.parametrize("entrypoint", ["runtime", "offline", "draft", "draft_offline"])
def test_model_owns_and_forwards_expert_quantization(
    monkeypatch, single_rank, unallocated_moe, build_v41, entrypoint, online
):
    hf = unallocated_moe
    owner = dspark if entrypoint.startswith("draft") else model
    build = Mock(wraps=owner.make_v4_quant_config)
    monkeypatch.setattr(owner, "make_v4_quant_config", build)
    instance = build_v41(entrypoint, online)

    expected_online = None if entrypoint == "draft_offline" else online
    build.assert_called_once()
    assert build.call_args.kwargs == {
        "online_quant_config": expected_online,
        "shared_expert_spec": native_quant_config().global_spec,
        "online_routed_experts": True,
    }
    config = instance.moe_quant_config
    assert config.online_quant_config_raw is expected_online
    assert config.online_quant == bool(expected_online)
    draft = entrypoint.startswith("draft")
    blocks = instance.mtp if draft else instance.layers
    assert len(blocks) == (3 if draft else 40)
    assert hf.n_routed_experts == 384  # Draft must not mutate the target config.
    for index, block in enumerate(blocks):
        moe = block.ffn
        assert moe.quant_config is config
        assert moe.prefix == f"{'mtp' if draft else 'layers'}.{index}.ffn"
        assert moe.n_routed_experts == (128 if draft else 384)
        assert moe.n_activated_experts == (3 if draft else 6)
        routed = f"{moe.prefix}.experts"
        # A global online spec keeps the checkpoint FP4 experts; a layer
        # pattern that names them converts them.
        source = config.get_layer_quant_config(routed)
        assert source.quant_type == QuantType.per_1x32
        online_routed = config.get_layer_quant_config(routed, use_online_quant=True)
        if expected_online and "layer_quant_config" in expected_online:
            assert online_routed.quant_type == QuantType.per_Token
        else:
            assert online_routed == source
        # Shared experts keep the native spec on both lookups.
        shared = f"{moe.prefix}.shared_experts.gate_up_proj"
        assert config.get_layer_quant_config(shared).quant_type == QuantType.per_1x32
        online_shared = config.get_layer_quant_config(shared, use_online_quant=True)
        assert online_shared.quant_type == QuantType.per_1x32


def test_shared_loading_follows_any_fused_layer():
    """Online quant can leave a layer unfused; the loader then decides per layer."""
    from torch import nn

    from atom.models.deepseek_v4 import DeepseekV4ForCausalLM

    moe = type("MoE", (nn.Module,), {})
    layers = nn.ModuleList([moe(), moe()])
    layers[0]._fuse_shared_into_routed = False
    layers[1]._fuse_shared_into_routed = True
    probe = DeepseekV4ForCausalLM.disable_fused_shared_loading.fget
    assert probe(Mock(model=layers)) is False
    layers[1]._fuse_shared_into_routed = False
    assert probe(Mock(model=layers)) is True
