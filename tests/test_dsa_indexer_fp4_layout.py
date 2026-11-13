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
            dcp_indexer_fp4_read_scale_at=fp4_scale_byte(src[:, None], torch.arange(4)),
            dcp_indexer_gather_index=gather,
            dcp_indexer_fp4_stage_page=dst // 64,
            dcp_indexer_fp4_stage_row=dst % 64,
            dcp_indexer_fp4_stage_scale_at=fp4_scale_byte(
                dst[:, None], torch.arange(4)
            ),
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


@pytest.mark.parametrize("layout,rope", [("flydsl", 64), ("opus", 0), ("opus", 64)])
@pytest.mark.parametrize("dcp", [1, 8])
def test_hip_writer_adapter_preserves_outputs_and_dcp_semantics(
    monkeypatch, layout, rope, dcp
):
    try:
        from atom.model_ops import attention_mla
    except ImportError as exc:
        from import_guard import skip_if_dependency_missing

        skip_if_dependency_missing(exc, "attention_mla imports Triton and AITER")

    captured = []
    monkeypatch.setattr(attention_mla, "get_dcp_world_size", lambda: dcp)
    monkeypatch.setattr(
        attention_mla,
        "_indexer_qk_rope_quant_and_cache",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )
    tensors = tuple(object() for _ in range(12))
    q_scale, k_scale = object(), object()
    attention_mla.indexer_qk_rope_quant_and_cache(
        *tensors,
        1e-6,
        32,
        "ue8m0",
        0.125,
        q_scale_out=q_scale,
        kv_cache_scale=k_scale,
        scale_layout=layout,
        rope_dim=rope,
    )
    assert len(captured) == 1
    args, kwargs = captured[0]
    assert all(actual is original for actual, original in zip(args[:12], tensors))
    assert kwargs["q_scale_out"] is q_scale
    assert kwargs["kv_cache_scale"] is k_scale
    assert kwargs["compute_all_q_rope"] == (layout == "opus" or dcp > 1)
    if layout == "opus":
        assert kwargs["scale_layout"] == "opus" and kwargs["rope_dim"] == rope
    else:
        assert "scale_layout" not in kwargs and "rope_dim" not in kwargs
