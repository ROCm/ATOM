# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The MLA metadata builder under ATOM_MLA_NVFP4: what it refuses at startup,
the tail and CSR it allocates before KV sizing, and that releasing the KV pools
drops every reference to the FP4 planes."""

import gc
import weakref
from types import SimpleNamespace

import pytest

# `aiter_mla` imports AITER at module scope, so this module cannot even be
# collected on the CPU-only Pre Checkin runner without the guard.
pytest.importorskip("aiter")

from atom.model_ops.attentions.aiter_mla import MLA_ROWS, AiterMLAMetadataBuilder
from atom.model_ops.attentions.nvfp4_mla import Nvfp4MlaCache


class _Layer:
    """An attention wrapper as the bind walk sees it: hashable, with an impl."""

    def __init__(self, cache=None):
        self.impl = SimpleNamespace(nvfp4=cache)


def _builder(**config):
    settings = {
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "speculative_config": None,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_tbo": False,
        "enable_tbo_decode": False,
        "pipeline_parallel_size": 1,
        "enable_rapidserve": False,
        "kv_transfer_config": {},
        "hf_config": SimpleNamespace(
            kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
        ),
    }
    settings.update(config)
    builder = AiterMLAMetadataBuilder.__new__(AiterMLAMetadataBuilder)
    builder.model_runner = SimpleNamespace(
        block_size=64,
        has_mla_indexer=False,
        device="cpu",
        config=SimpleNamespace(**settings),
    )
    builder.num_attention_heads = 128
    builder.max_bs = 3
    builder.block_table_cols = 5
    builder.kv_pool = None
    builder.nvfp4_cache = None
    return builder


@pytest.mark.parametrize(
    "config,refused",
    [
        ({"pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"enable_rapidserve": True}, "RapidServe"),
        ({"kv_transfer_config": {"kv_connector": "mooncake"}}, "KV transfer"),
        ({"enable_chunked_prefill": True}, "chunked prefill"),
    ],
)
def test_startup_refuses(config, refused):
    _builder()._check_nvfp4_supported()
    with pytest.raises(RuntimeError, match=refused):
        _builder(**config)._check_nvfp4_supported()


def test_tail_and_csr_are_sized_by_requests_before_the_pool():
    builder = _builder()
    builder._pool_rows = {MLA_ROWS: {_Layer(): 0, _Layer(): 1}}
    builder._allocate_nvfp4_buffers()
    ring = builder._nvfp4_ring
    # One row per request plus the padding row, every layer, the 576 latent.
    assert (ring.num_reqs, ring.layers, ring.head_dim) == (4, 2, 576)
    assert [t.numel() for t in builder._nvfp4_csr] == [4, 3 * 5, 3]
    cache = Nvfp4MlaCache(2, 8, 3, 576, device="cpu", ring=ring)
    assert cache.ring is ring


def test_release_drops_every_reference_to_the_fp4_planes():
    builder = _builder()
    cache = Nvfp4MlaCache(2, 4, 3, 576, device="cpu")
    layers = [_Layer(cache), _Layer(cache)]
    builder._pool_rows = {MLA_ROWS: {layer: row for row, layer in enumerate(layers)}}
    builder.nvfp4_cache = cache
    alive = weakref.ref(cache)
    del cache
    builder.release_kv_pools()
    gc.collect()
    assert builder.nvfp4_cache is None
    assert all(layer.impl.nvfp4 is None for layer in layers)
    assert alive() is None
