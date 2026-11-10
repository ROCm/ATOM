# SPDX-License-Identifier: MIT
"""OPUS gfx950 cache ABI: full permutations, not just equal tensor sizes."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from atom.mono.dsa_indexer.plan import (
    IndexerShape,
    fp4_cache_byte,
    fp4_scale_byte,
    unsupported,
)


def shape(page=64):
    return IndexerShape(
        32, 128, 2048, 2048, "bf16", "fp4", page, 0, cache_layout="mfma"
    )


@pytest.mark.parametrize("page", [64, 128, 256])
def test_opus_bytes_equal_independent_tensor_permutations(page):
    values = torch.arange(page * 64).reshape(page // 64, 64, 4, 16)
    data = values.permute(0, 2, 1, 3).contiguous().flatten()
    natural = values.reshape(page, 64)
    offsets = [fp4_cache_byte(r, b) for r in range(page) for b in range(64)]
    assert sorted(offsets) == list(range(page * 64))
    assert torch.equal(data[offsets].reshape(page, 64), natural)
    scales = torch.arange(page * 4).reshape(page // 64, 2, 32, 2, 2)
    permuted = scales.permute(0, 4, 2, 3, 1).contiguous().flatten()
    offsets = [fp4_scale_byte(r, g) for r in range(page) for g in range(4)]
    assert sorted(offsets) == list(range(page * 4))
    assert torch.equal(permuted[offsets].reshape(page, 4), scales.reshape(page, 4))
    assert shape(page).block_bytes == page * 64
    assert shape(page).block_scale_bytes == page * 4


def test_no_silent_reinterpretation_of_legacy_fp4():
    assert not unsupported(shape())
    assert "OPUS" in unsupported(replace(shape(), cache_layout="preshuffle"))
    assert "64-row" in unsupported(replace(shape(), block_rows=16))
    assert "64-row" in unsupported(replace(shape(), block_rows=96))
    assert shape().tag != replace(shape(), cache_layout="preshuffle").tag


def test_page_bases_do_not_wrap_at_four_gib():
    sh = shape()
    for page in ((1 << 20) - 1, 1 << 20, (1 << 20) + 17):
        for row in (0, 31, 32, 63):
            assert (
                page * sh.block_bytes + fp4_cache_byte(row, 63)
                == page * 4096 + 3072 + row * 16 + 15
            )


def test_fp8_contract_unchanged():
    sh = replace(shape(), kv_dtype="fp8", cache_layout="preshuffle")
    assert sh.row_bytes == 144 and sh.block_bytes == 9216
    assert sh.block_scale_bytes == 0 and not unsupported(sh)


def test_pool_abi_is_explicitly_opt_in():
    from atom.model_ops.sparse_indexer_fp4 import (
        fp4_index_block_shapes,
        fp4_q_scale_shape,
    )

    assert fp4_index_block_shapes(64, 128) == ((1, 4, 64, 16), (1, 4, 64))
    assert fp4_index_block_shapes(64, 128, layout="mfma") == ((4, 64, 16), (2, 32, 4))
    assert fp4_q_scale_shape(8, 32, 128, layout="mfma") == (8, 2, 32, 2)
    assert fp4_q_scale_shape(8, 64, 128, layout="mfma") == (8, 2, 32, 4)
    with pytest.raises(ValueError, match="Unsupported"):
        fp4_index_block_shapes(64, 128, layout="natural")
    with pytest.raises(ValueError, match="H=32/64"):
        fp4_q_scale_shape(8, 16, 128, layout="mfma")


def test_binding_rejects_wrong_scale_shape_and_type():
    pytest.importorskip("flydsl")
    from atom.mono.dsa_indexer.bind import _cache_refusal

    cache = torch.empty((2, 4, 64, 16), dtype=torch.uint8)
    scale = torch.empty((2, 2, 32, 4), dtype=torch.uint8)
    m = SimpleNamespace(k_cache=SimpleNamespace(kv_cache=[cache], kv_cache_scale=scale))
    assert not _cache_refusal(m, shape())
    m.k_cache.kv_cache_scale = scale.reshape(2, 4, 64)
    assert "OPUS" in _cache_refusal(m, shape())
    m.k_cache.kv_cache_scale = scale.float()
    assert "OPUS" in _cache_refusal(m, shape())


def test_dcp_prefill_preserves_opus_rows_and_all_four_exponents(monkeypatch):
    pytest.importorskip("aiter")
    from atom.model_ops.indexer import indexer_forward as mod

    torch.manual_seed(17)
    data = torch.randint(0, 256, (4, 4, 64, 16), dtype=torch.uint8)
    natural_scale = torch.randint(0, 256, (4, 64, 4), dtype=torch.uint8)
    scale = (
        natural_scale.reshape(4, 2, 32, 2, 2)
        .permute(0, 4, 2, 3, 1)
        .contiguous()
        .reshape(4, 2, 32, 4)
    )
    src = torch.randperm(256)[:64]
    gather = torch.arange(128).reshape(2, 64).T.flatten()
    dst = torch.arange(128)
    monkeypatch.setattr(
        mod,
        "get_dcp_group",
        lambda: SimpleNamespace(all_gather=lambda t, dim: torch.cat((t, t), dim=dim)),
    )
    staged, ss = mod._dcp_stage_indexer_fp4_prefill(
        data,
        scale,
        SimpleNamespace(
            dcp_indexer_fp4_read_page=src // 64,
            dcp_indexer_fp4_read_row=src % 64,
            dcp_indexer_gather_index=gather,
            dcp_indexer_fp4_stage_page=dst // 64,
            dcp_indexer_fp4_stage_row=dst % 64,
            dcp_indexer_fp4_stage_pages=2,
        ),
        "mfma",
    )
    source = src[gather % 64]
    assert torch.equal(
        staged[dst // 64, :, dst % 64, :], data[source // 64, :, source % 64, :]
    )
    unpacked = (
        ss.reshape(2, 2, 32, 2, 2).permute(0, 4, 2, 3, 1).contiguous().reshape(128, 4)
    )
    assert torch.equal(unpacked, natural_scale.reshape(-1, 4)[source])


@pytest.mark.parametrize("rope,dcp", [(0, 1), (64, 1), (0, 8), (64, 8)])
def test_native_forward_reaches_aiter_opus_writer(monkeypatch, rope, dcp):
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires gfx950")
    from atom.model_ops import attention_mla
    from atom.model_ops.indexer import indexer_forward as mod
    from atom.model_ops.sparse_indexer_fp4 import require_opus_api

    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")
    require_opus_api()
    rows, heads = 6, 32
    q = torch.ones((rows, heads, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((rows, heads), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((rows, 128), device="cuda", dtype=torch.bfloat16)
    cache = torch.full((1, 4, 64, 16), 0xA5, device="cuda", dtype=torch.uint8)
    scales = torch.full((1, 2, 32, 4), 0xA5, device="cuda", dtype=torch.uint8)
    slots = torch.tensor([-1, 1, -1, 3, -1, 5], device="cuda", dtype=torch.int64)
    norm = torch.ones(128, device="cuda", dtype=torch.float32)
    pos = torch.arange(rows, device="cuda", dtype=torch.int64)
    cos = torch.ones((rows, 32), device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    empty = torch.empty(0, device="cuda", dtype=torch.int32)
    module = SimpleNamespace(
        _indexer_fp4=True, _index_cache_layout="mfma", rope_dim=rope
    )
    config = SimpleNamespace(
        kv_cache_block_size=64,
        dcp_config=SimpleNamespace(interleave_size=1),
        compilation_config=SimpleNamespace(static_forward_context={"indexer": module}),
    )
    context = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=False, is_prefill=True, scheduled_bs=1),
        attn_metadata=SimpleNamespace(slot_mapping=slots, run_sparse=False),
    )
    monkeypatch.setattr(mod, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(mod, "get_forward_context", lambda: context)
    monkeypatch.setattr(mod, "get_dcp_world_size", lambda: dcp)
    monkeypatch.setattr(mod, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(attention_mla, "get_dcp_world_size", lambda: dcp)
    writer = attention_mla._indexer_qk_rope_quant_and_cache
    emitted = []

    def observe(*args, **kwargs):
        writer(*args, **kwargs)
        emitted.append((args[1], args[3], kwargs["q_scale_out"]))

    monkeypatch.setattr(attention_mla, "_indexer_qk_rope_quant_and_cache", observe)
    mod.sparse_attn_indexer(
        k,
        "indexer.k_cache",
        cache,
        q,
        k,
        weights,
        32,
        "ue8m0",
        2048,
        128,
        8192,
        rows,
        empty,
        empty,
        empty,
        norm,
        norm,
        1e-6,
        pos if rope else None,
        cos if rope else None,
        sin if rope else None,
        0.125,
        True,
        False,
        False,
        kv_cache_scale=scales,
    )
    assert len(emitted) == 1
    q_out, weights_out, q_scales = emitted[0]
    assert (q_out == 0x66).all() and (q_scales == 125).all()
    torch.testing.assert_close(weights_out, weights, rtol=0, atol=0)
    assert (cache[0, :, [0, 2, 4], :] == 0xA5).all()
    assert (cache[0, :, [1, 3, 5], :] == 0x66).all()
