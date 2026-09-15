# SPDX-License-Identifier: MIT
"""Model-owned MoE quantization, including engine online overrides."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from aiter import QuantType
from torch import nn

from atom.config import get_hf_config
from atom.models.deepseek_v4 import MoE as V4MoE
from atom.models.deepseek_v4 import make_v4_quant_config
from atom.models.deepseek_v41 import dspark, model, multimodal, runtime

from .reference import FIXTURES


class UnallocatedModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


@pytest.mark.parametrize("online", [None, {"global_quant_config": "ptpc_fp8"}])
@pytest.mark.parametrize("entrypoint", ["runtime", "offline", "draft", "draft_offline"])
def test_model_owns_and_forwards_expert_quantization(
    monkeypatch, single_rank, entrypoint, online
):
    # Exercise real model, Block, DraftBlock and V4.1 MoE constructors.
    # Only weight allocation / the V4 constructor seam is replaced.
    hf = get_hf_config(str(FIXTURES))
    hf.engram_layer_ids = ()
    engine = SimpleNamespace(
        hf_config=hf,
        max_model_len=32,
        enforce_eager=True,
        online_quant_config=online,
    )
    for module, names in (
        (model, ("VocabParallelEmbedding", "LogitsHead", "FusedRMSNorm")),
        (dspark, ("ReplicatedLinear", "MarkovHead", "ConfidenceHead")),
        (multimodal, ("ViT", "Aligner")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, UnallocatedModule)
    monkeypatch.setattr(model.Block, "attention_cls", UnallocatedModule)
    monkeypatch.setattr(dspark.DraftBlock, "attention_cls", UnallocatedModule)

    def capture_v4(self, layer_id, args, prefix=""):
        nn.Module.__init__(self)
        self.gate = nn.Module()
        self.quant_config = args.quant_config
        self.prefix = prefix
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts

    monkeypatch.setattr(V4MoE, "__init__", capture_v4)
    owner = dspark if entrypoint.startswith("draft") else model
    build = Mock(wraps=make_v4_quant_config)
    monkeypatch.setattr(owner, "make_v4_quant_config", build)
    with torch.device("meta"):
        if entrypoint == "runtime":
            instance = runtime.DeepseekV41RuntimeModel(engine)
        elif entrypoint == "offline":
            instance = model.DeepseekV41ForCausalLM(
                hf, max_length=32, online_quant_config=online
            )
        elif entrypoint == "draft":
            instance = dspark.DeepseekV41DSpark(engine)
        else:
            instance = dspark.DeepseekV41DSpark(hf, max_length=32)

    expected_online = None if entrypoint == "draft_offline" else online
    build.assert_called_once()
    assert build.call_args.kwargs == {"online_quant_config": expected_online}
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
        # Online overrides must retain V4's protected checkpoint FP4 experts.
        source = config.get_layer_quant_config(routed)
        assert source.quant_type == QuantType.per_1x32
        assert config.get_layer_quant_config(routed, use_online_quant=True) == source
        shared = f"{moe.prefix}.shared_experts.gate_up_proj"
        assert config.get_layer_quant_config(shared).quant_type == QuantType.per_1x32
        if expected_online:
            assert (
                config.get_layer_quant_config(shared, use_online_quant=True).quant_type
                == QuantType.per_Token
            )
