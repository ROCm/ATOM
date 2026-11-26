# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from atom.model_ops.indexer.schedule import (
    any_layer_skips_index_topk,
    skips_index_topk,
)


def _config(**fields):
    return SimpleNamespace(num_hidden_layers=6, **fields)


def test_a_model_without_index_sharing_skips_no_layer():
    config = _config()
    assert not any(skips_index_topk(config, layer) for layer in range(6))
    assert not any_layer_skips_index_topk(config)


def test_indexer_types_name_the_shared_layers():
    types = ["full", "shared", "shared", "full", "shared", "full"]
    config = _config(use_index_cache=True, indexer_types=types)
    assert [skips_index_topk(config, layer) for layer in range(6)] == [
        kind == "shared" for kind in types
    ]
    assert any_layer_skips_index_topk(config)


def test_indexer_types_without_the_index_cache_share_nothing():
    config = _config(indexer_types=["full", "shared"] * 3)
    assert not any_layer_skips_index_topk(config)


def test_pattern_marks_skipping_layers_with_s():
    config = _config(use_index_cache=True, index_topk_pattern="FSSFSF")
    assert [skips_index_topk(config, layer) for layer in range(6)] == [
        c == "S" for c in "FSSFSF"
    ]


def test_freq_turns_the_index_cache_on_and_skips_between_selecting_layers():
    config = _config(index_topk_freq=2)
    assert [skips_index_topk(config, layer) for layer in range(6)] == [
        False,
        False,
        True,
        False,
        True,
        False,
    ]
    assert config.use_index_cache


@pytest.mark.parametrize("layer", [6, 7])
def test_mtp_layers_past_the_stack_select_their_own(layer):
    config = _config(use_index_cache=True, indexer_types=["shared"] * 6)
    assert not skips_index_topk(config, layer)
