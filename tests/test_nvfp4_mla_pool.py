# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Under NVFP4 the paged MLA pool holds the FP4 planes and no FP8 rows."""

import pytest
import torch

from atom.model_ops.attentions.mla_kv_pool import (
    NVFP4_AUX_BYTES,
    NVFP4_NOPE_BYTES,
    MlaKvPool,
)
from atom.model_ops.attentions.nvfp4_hp_ring import Nvfp4HpRing
from atom.model_ops.attentions.nvfp4_mla import Nvfp4MlaCache

LAYERS = 3
DIM = 576


def _pool(nvfp4, block_size=64):
    return MlaKvPool(
        layers=LAYERS,
        block_size=block_size,
        entry_dim=DIM,
        kv_dtype=torch.float8_e4m3fn,
        nvfp4=nvfp4,
    )


def test_nvfp4_block_costs_fp4_planes_only():
    fp8, fp4 = _pool(False), _pool(True)
    assert fp8.entry_bytes == LAYERS * 64 * DIM
    assert fp4.entry_bytes == LAYERS * (NVFP4_NOPE_BYTES + NVFP4_AUX_BYTES)
    # 384 B per token per layer against FP8's 576.
    assert fp4.entry_bytes * 3 == fp8.entry_bytes * 2


def test_nvfp4_planes_bind_as_contiguous_aligned_layers():
    pool = _pool(True)
    blocks = 5
    pool.allocate(blocks, "cpu")
    nope, aux = pool.field("kv_nope"), pool.field("kv_aux")
    assert tuple(nope.shape) == (LAYERS, blocks, NVFP4_NOPE_BYTES)
    assert tuple(aux.shape) == (LAYERS, blocks, NVFP4_AUX_BYTES)
    base = nope.untyped_storage().data_ptr()
    for layer in range(LAYERS):
        for plane in (nope[layer], aux[layer]):
            assert plane.is_contiguous()
            assert (plane.data_ptr() - base) % 256 == 0
    # The two planes and every layer are disjoint.
    nope[1].fill_(7)
    assert nope[0].sum() == 0 and nope[2].sum() == 0 and aux.sum() == 0
    assert pool.layer("kv_nope", 1).data_ptr() == nope[1].data_ptr()
    with pytest.raises(KeyError):
        pool.layer("kv", 0)

    cache = Nvfp4MlaCache(
        LAYERS, blocks, 2, DIM, device="cpu", kv_nope=nope, kv_aux=aux
    )
    assert cache.kv_nope[1].data_ptr() == nope[1].data_ptr()


def test_nvfp4_pool_needs_page64_blocks():
    with pytest.raises(ValueError, match="64 tokens"):
        _pool(True, block_size=16)


def test_cache_rejects_a_tail_of_the_wrong_size():
    ring = Nvfp4HpRing(3, LAYERS, DIM)
    assert Nvfp4MlaCache(LAYERS, 4, 2, DIM, device="cpu", ring=ring).ring is ring
    with pytest.raises(ValueError, match="tail"):
        Nvfp4MlaCache(LAYERS, 4, 3, DIM, device="cpu", ring=ring)


def test_cache_rejects_wrong_plane_shape():
    with pytest.raises(ValueError, match="kv_nope"):
        Nvfp4MlaCache(
            LAYERS,
            4,
            2,
            DIM,
            device="cpu",
            kv_nope=torch.empty((LAYERS, 4, 100), dtype=torch.uint8),
        )
