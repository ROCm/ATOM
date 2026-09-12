# SPDX-License-Identifier: MIT
"""Pinned model math, isolated per differential test."""

import os
from types import SimpleNamespace

import pytest

from .reference import load_reference


@pytest.fixture
def reference():
    directory = os.environ.get("ATOM_DSV41_REFERENCE")
    if not directory:
        pytest.skip("Set ATOM_DSV41_REFERENCE for pinned model methods")
    with load_reference(directory) as module:
        yield module


@pytest.fixture
def single_rank(monkeypatch):
    from atom.model_ops import embed_head, layernorm, linear
    from atom.models.deepseek_v41 import attention, layers, model, moe

    group = SimpleNamespace(rank_in_group=0, world_size=1)
    for module in (linear, attention, layers, model, moe, embed_head):
        monkeypatch.setattr(module, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        layernorm, "get_tensor_model_parallel_world_size", lambda: group.world_size
    )
    return group


@pytest.fixture
def small_config():
    from atom.models.deepseek_v41.config import DeepseekV41TextConfig

    return DeepseekV41TextConfig(
        hidden_size=64,
        head_dim=64,
        num_attention_heads=4,
        q_lora_rank=32,
        o_groups=4,
        o_lora_rank=32,
        rms_norm_eps=1e-20,
        sliding_window=4,
        index_n_heads=32,
        index_head_dim=32,
        index_topk=4,
        candidate_block_size=2,
        candidate_topk_blocks=4,
        max_position_embeddings=32,
        num_hidden_layers=5,
        num_nextn_predict_layers=0,
        compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layer_ids=(1, 3),
        index_source_layer_ids=(1, 3, 4),
        candidate_source_layer_id=3,
    )
