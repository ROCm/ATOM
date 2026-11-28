# SPDX-License-Identifier: MIT
"""Shared-slot policy and early mono refusal, without loading a checkpoint."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter, stubbed_aiter
from torch import nn


@pytest.fixture
def modules():
    # No router arithmetic is run here; constructing its FlyDSL kernel would
    # require the GPU stack before the host policy can be exercised.
    with (
        stubbed_aiter("atom.model_ops.deepseek_v41.router"),
        atom_without_aiter() as load,
    ):
        yield SimpleNamespace(
            dspark=load("atom.models.deepseek_v41.dspark"),
            hi=load("atom.models.deepseek_v41.mono.hi_weights"),
            tables=load("atom.models.deepseek_v41.mono.gfx1250.tables"),
            consensus=load("atom.mono.runtime.consensus"),
        )


def test_dspark_disables_shared_slot_in_its_own_quant_config(modules, monkeypatch):
    mod = modules.dspark
    qc = SimpleNamespace(fuse_shared_experts=True)
    seen = []
    monkeypatch.setattr(mod, "make_v4_quant_config", lambda *a, **kw: qc)
    monkeypatch.setattr(mod, "converted_linear_spec", lambda *a: None)
    monkeypatch.setattr(mod, "build_attention_topology", lambda *a: [None])
    monkeypatch.setattr(mod, "DraftSrcBuffers", lambda *a: None)
    monkeypatch.setattr(mod, "RotaryEmbedding", lambda *a, **kw: nn.Identity())

    def block(*args, moe_quant_config, **kwargs):
        seen.append(moe_quant_config.fuse_shared_experts)
        return nn.Identity()

    monkeypatch.setattr(mod, "DraftBlock", block)
    hf = SimpleNamespace(
        num_nextn_predict_layers=1,
        dspark_block_size=2,
        sliding_window=16,
        vocab_size=32,
        dspark_n_routed_experts=2,
        dspark_num_experts_per_tok=1,
        num_hidden_layers=0,
        fp4ptpc=False,
        qk_rope_head_dim=8,
        rope_theta=10000,
        max_position_embeddings=64,
    )
    config = SimpleNamespace(
        hf_config=hf,
        dspark=SimpleNamespace(draft_kv_dtype=torch.bfloat16),
        max_num_seqs=2,
    )
    draft = mod.DeepseekV41DSpark(config)
    assert seen == [False]
    assert draft.quant_config is qc and not qc.fuse_shared_experts
    mapping = draft.get_expert_mapping()
    assert {entry[2] for entry in mapping} == {0, 1}


@pytest.mark.parametrize("entry", ["check_block", "block_tensors"])
def test_hi_refuses_fused_block_before_reading_weights(modules, entry):
    block = SimpleNamespace(attn=None, ffn=SimpleNamespace(shared_experts=None))
    args = (block, 2304, 4) if entry == "check_block" else (block, "layers.0.")
    with pytest.raises(modules.consensus.MonoUnsupported, match="separate FP8 shared"):
        getattr(modules.hi, entry)(*args)


def test_gfx1250_tables_refuse_before_allocating_or_packing(modules, monkeypatch):
    block = SimpleNamespace(ffn=SimpleNamespace(shared_experts=None))
    mod = modules.tables
    monkeypatch.setattr(mod, "_find", lambda *a: block)
    monkeypatch.setattr(mod, "_base", lambda *a: None)
    kernels = {name: None for name in ("step", "tail", "draft_ctx")}
    drafter = SimpleNamespace(model=None)
    scratch = {"fake": torch.empty(0)}
    with pytest.raises(modules.consensus.MonoUnsupported, match="separate FP8 shared"):
        mod.StepTables(kernels, None, drafter, scratch, None, 0, None)
