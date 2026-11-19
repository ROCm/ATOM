# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A PP stage's compile cache key must follow the layers the stage serves."""

from types import SimpleNamespace

import pytest
from transformers import PretrainedConfig

from atom.config import ParallelConfig
from tests.test_config_compile_hash import _hash as _config_hash

PARTITION = "VLLM_PP_LAYER_PARTITION"


def _flat(num_hidden_layers=24):
    return PretrainedConfig(num_hidden_layers=num_hidden_layers)


def _nested(num_hidden_layers=24):
    # A separate text_config carries the layer count the model splits.
    return SimpleNamespace(
        architectures=["SomeVLM"],
        text_config=PretrainedConfig(num_hidden_layers=num_hidden_layers),
    )


def _keys(pp_size=2, make_hf=_flat):
    """Every stage's compile cache key, in stage order."""
    return [
        _config_hash(
            pipeline_parallel_size=pp_size,
            parallel_config=ParallelConfig(pipeline_parallel_rank=rank),
            hf_config=make_hf(),
        )
        for rank in range(pp_size)
    ]


@pytest.mark.parametrize("make_hf", [_flat, _nested])
def test_a_new_partition_moves_the_stage_keys(monkeypatch, make_hf):
    # The reported case: stage 0 went from 11 to 13 layers and loaded the
    # artifact compiled for its 11.
    monkeypatch.setenv(PARTITION, "11,13")
    before = _keys(make_hf=make_hf)
    monkeypatch.setenv(PARTITION, "13,11")
    after = _keys(make_hf=make_hf)
    assert before[0] != after[0]
    assert before[1] != after[1]


@pytest.mark.parametrize("make_hf", [_flat, _nested])
def test_the_same_partition_keeps_the_stage_keys(monkeypatch, make_hf):
    monkeypatch.setenv(PARTITION, "11,13")
    assert _keys(make_hf=make_hf) == _keys(make_hf=make_hf)


def test_the_key_follows_the_stage_layers_not_the_spelling(monkeypatch):
    # 24 layers over 2 stages split 12,12 by default.
    monkeypatch.delenv(PARTITION, raising=False)
    default = _keys()
    monkeypatch.setenv(PARTITION, "12,12")
    assert _keys() == default
    # Stage 0 serves layers 0..7 under both splits, so only its key stays.
    monkeypatch.setenv(PARTITION, "8,8,8")
    before = _keys(pp_size=3)
    monkeypatch.setenv(PARTITION, "8,7,9")
    after = _keys(pp_size=3)
    assert after[0] == before[0]
    assert after[1] != before[1]
    assert after[2] != before[2]


def test_stage_rank_and_pp_size_key_the_stage(monkeypatch):
    monkeypatch.setenv(PARTITION, "12,12")
    pp2 = _keys()
    assert pp2[0] != pp2[1]
    # Stage 0 serves layers 0..11 under PP2 and under this PP3.
    monkeypatch.setenv(PARTITION, "12,6,6")
    assert _keys(pp_size=3)[0] != pp2[0]


def test_runs_without_pp_keep_their_key(monkeypatch):
    # pp_size 1 adds no factor, and the partition is read only under PP.
    base = _config_hash(hf_config=_flat())
    monkeypatch.setenv(PARTITION, "11,13")
    assert _config_hash(hf_config=_flat()) == base
    assert _config_hash(pipeline_parallel_size=1, hf_config=_flat()) == base
    assert _config_hash(
        pipeline_parallel_size=1,
        parallel_config=ParallelConfig(),
        hf_config=_flat(),
    ) == _config_hash(parallel_config=ParallelConfig(), hf_config=_flat())
