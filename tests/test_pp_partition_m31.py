# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_PP_LAYER_PARTITION=kv_balanced (`atom.models.pp_partition`): the split
that evens out MiniMax-M3.1's per-stage KV bytes per token under PP4 x TP1 with
the DSpark drafter's KV on the last stage, and its install for every process's
`get_pp_indices`. Pure CPU."""

import types

import pytest

# MiniMax-M3.1-Flash (as in test_moriio_fabric.py): 60 sparse layers, 4 KV heads
# of 128, a 128-wide FP4 index key; DSpark has 5 layers of 4 fp8 KV heads.
M31_LAYERS, M31_KV_HEADS, M31_HEAD_DIM, M31_INDEX_DIM, M31_DRAFT_LAYERS = (
    60,
    4,
    128,
    128,
    5,
)
PARTITION = (17, 17, 17, 9)


@pytest.fixture
def pp_partition():
    from atom.models import pp_partition

    return pp_partition


M31_LAYER = M31_KV_HEADS * 2 * (64 + 8) + (64 + 8)  # NVFP4 K/V + FP4 index
M31_DRAFT = M31_DRAFT_LAYERS * M31_KV_HEADS * 2 * (M31_HEAD_DIM + 4)


class TestKvBalanced:
    def test_m31_even_split_and_the_balanced_ones(self, pp_partition):
        layers = [M31_LAYER] * M31_LAYERS
        assert M31_LAYER == 648 and M31_DRAFT == 5280
        loose = pp_partition.kv_balanced_partition(layers, 4, M31_DRAFT, 2)
        assert loose.even == (15, 15, 15, 15)
        assert loose.even_stage_bytes == (9720, 9720, 9720, 15000)
        assert loose.partition == PARTITION
        assert loose.stage_bytes == (11016, 11016, 11016, 11112)
        # Within a few percent of the per-GPU average, 44,160 / 4.
        assert max(loose.stage_bytes) / (sum(loose.stage_bytes) / 4) < 1.01
        assert round(loose.pool_gain, 3) == 1.350
        tight = pp_partition.kv_balanced_partition(layers, 4, M31_DRAFT, 1)
        assert tight.partition == (16, 16, 16, 12)
        assert tight.stage_bytes == (10368, 10368, 10368, 13056)
        free = pp_partition.kv_balanced_partition(layers, 4, M31_DRAFT, None)
        assert free.partition == PARTITION

    def test_no_imbalance_keeps_the_even_split(self, pp_partition):
        plan = pp_partition.kv_balanced_partition([100] * 60, 4, 0, 3)
        assert plan.partition == (15, 15, 15, 15)
        # Within the tolerance of the best still counts as even.
        plan = pp_partition.kv_balanced_partition([100] * 60, 4, 20, 3)
        assert plan.partition == (15, 15, 15, 15)

    def test_heavier_layers_move_the_split(self, pp_partition):
        # Three dense layers at 1,056 B (fp8 K/V of four heads) in front.
        layers = [1056] * 3 + [M31_LAYER] * 57
        plan = pp_partition.kv_balanced_partition(layers, 4, M31_DRAFT, 2)
        assert plan.partition == (16, 17, 17, 10)
        assert max(plan.stage_bytes) == 11760
        brute = min(
            max(pp_partition.stage_bytes(layers, (a, b, c, 60 - a - b - c), M31_DRAFT))
            for a in range(1, 18)
            for b in range(1, 18)
            for c in range(1, 18)
            if 1 <= 60 - a - b - c <= 17
        )
        assert max(plan.stage_bytes) <= brute * 1.02

    def test_refusals(self, pp_partition):
        with pytest.raises(ValueError, match="non-empty"):
            pp_partition.kv_balanced_partition([1, 1], 4)
        with pytest.raises(ValueError, match="negative"):
            pp_partition.kv_balanced_partition([1] * 8, 4, -1)

    def test_the_m3_estimate_is_the_pools(self, pp_partition, monkeypatch):
        from atom import config as config_mod

        monkeypatch.setenv("ATOM_M3_SPARSE_KV_NVFP4", "1")
        monkeypatch.setattr(config_mod, "_is_minimax_m3_config", lambda hf: True)
        text = types.SimpleNamespace(
            num_hidden_layers=M31_LAYERS,
            num_key_value_heads=M31_KV_HEADS,
            num_attention_heads=64,
            hidden_size=6144,
            head_dim=M31_HEAD_DIM,
            sparse_attention_config={
                "sparse_attention_freq": [1] * M31_LAYERS,
                "sparse_index_dim": M31_INDEX_DIM,
            },
        )
        draft = types.SimpleNamespace(
            num_hidden_layers=M31_DRAFT_LAYERS,
            num_key_value_heads=M31_KV_HEADS,
            head_dim=M31_HEAD_DIM,
        )
        config = types.SimpleNamespace(
            hf_config=types.SimpleNamespace(text_config=text),
            tensor_parallel_size=1,
            pipeline_parallel_size=4,
            kv_cache_dtype="fp8",
            index_cache_dtype="fp4",
            speculative_config=types.SimpleNamespace(draft_model_hf_config=draft),
        )
        layers, extra = pp_partition.minimax_m3_stage_kv_costs(config)
        assert layers == [M31_LAYER] * M31_LAYERS and extra == M31_DRAFT
        # At TP4 a rank holds one head: the decode's 14,280 B/token.
        config.tensor_parallel_size = 4
        layers, extra = pp_partition.minimax_m3_stage_kv_costs(config)
        assert sum(layers) + extra == 14280

        monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
        config.tensor_parallel_size = 1
        monkeypatch.setenv("ATOM_PP_LAYER_PARTITION", "kv_balanced")
        monkeypatch.setenv("ATOM_PP_PARTITION_MAX_EXTRA_LAYERS", "2")
        assert pp_partition.resolve_pp_layer_partition(config) == PARTITION
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "15,15,15,15")
        assert pp_partition.resolve_pp_layer_partition(config) is None
        monkeypatch.delenv("VLLM_PP_LAYER_PARTITION")
        monkeypatch.setenv("ATOM_PP_LAYER_PARTITION", "even")
        assert pp_partition.resolve_pp_layer_partition(config) is None
        monkeypatch.setenv("ATOM_PP_LAYER_PARTITION", "balanced")
        with pytest.raises(ValueError, match="ATOM_PP_LAYER_PARTITION"):
            pp_partition.resolve_pp_layer_partition(config)

    def test_an_installed_split_reaches_every_get_pp_indices(self, monkeypatch):
        from atom.config import Config
        from atom.models import utils

        monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
        monkeypatch.setattr(utils, "_installed_pp_partitions", {})
        state = {
            "pp_layer_partition": PARTITION,
            "hf_config": types.SimpleNamespace(
                text_config=types.SimpleNamespace(num_hidden_layers=M31_LAYERS)
            ),
        }
        # What unpickling a resolved config does in a stage's process.
        Config.__setstate__(object.__new__(Config), state)
        assert [utils.get_pp_indices(60, s, 4) for s in range(4)] == [
            (0, 17),
            (17, 34),
            (34, 51),
            (51, 60),
        ]
        assert utils.get_pp_indices(60, 0, 2) == (0, 30)  # another PP size
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "15,15,15,15")
        assert utils.get_pp_indices(60, 3, 4) == (45, 60)
        with pytest.raises(ValueError, match="non-empty"):
            utils.install_pp_layer_partition(60, (60, 0))
