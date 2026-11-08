# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The compile cache key must change with every value baked into the graph."""

from types import SimpleNamespace

from atom.config import Config


def _hash(**overrides):
    fields = {
        "quant_config": None,
        "compilation_config": None,
        "parallel_config": None,
        "tensor_parallel_size": 4,
        "prefill_context_parallel_size": 1,
        "sequence_parallel_size": 1,
        "dcp_config": None,
        "enable_dp_attention": False,
        "index_cache_dtype": None,
        "index_cache_layout": None,
        "hf_config": SimpleNamespace(),
        "max_model_len": 4096,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 16384,
        "attn_prefill_chunk_size": 16384,
        "kv_cache_block_size": 16,
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.9,
        "speculative_config": None,
    }
    fields.update(overrides)
    return Config.compute_hash(SimpleNamespace(**fields))


def test_hash_is_stable():
    assert _hash() == _hash()


def test_hash_changes_with_max_model_len():
    # The sparse-attention indexer bakes max_model_len into the graph.
    assert _hash(max_model_len=12288) != _hash()


def test_hash_changes_with_max_num_seqs():
    assert _hash(max_num_seqs=512) != _hash()


def test_hash_changes_with_max_num_batched_tokens():
    # Buffers sized by it reach the graph as fixed-shape inputs; reusing a
    # 16384 artifact at 8192 fails assert_size_stride on the first forward.
    assert _hash(max_num_batched_tokens=8192) != _hash()
    assert _hash(max_num_batched_tokens=4096) != _hash(max_num_batched_tokens=8192)


def test_hash_changes_with_kv_cache_dtype():
    assert _hash(kv_cache_dtype="bf16") != _hash()


def test_hash_ignores_settings_that_stay_out_of_the_graph():
    # These size attention-metadata workspaces, KV pages or the KV pool, all
    # outside the traced graph; changing them must keep the cached artifact.
    base = _hash()
    assert _hash(attn_prefill_chunk_size=8192) == base
    assert _hash(kv_cache_block_size=256) == base
    assert _hash(gpu_memory_utilization=0.8) == base
