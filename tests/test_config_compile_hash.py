# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The compile cache key must change with every value baked into the graph."""

from types import SimpleNamespace

from atom.config import Config


def _hash(**overrides):
    fields = dict(
        quant_config=None,
        compilation_config=None,
        parallel_config=None,
        tensor_parallel_size=4,
        prefill_context_parallel_size=1,
        dcp_config=None,
        enable_dp_attention=False,
        index_cache_dtype=None,
        hf_config=SimpleNamespace(),
        max_model_len=4096,
        max_num_seqs=256,
    )
    fields.update(overrides)
    return Config.compute_hash(SimpleNamespace(**fields))


def test_hash_is_stable():
    assert _hash() == _hash()


def test_hash_changes_with_max_model_len():
    # The sparse-attention indexer bakes max_model_len into the graph.
    assert _hash(max_model_len=12288) != _hash()


def test_hash_changes_with_max_num_seqs():
    assert _hash(max_num_seqs=512) != _hash()
