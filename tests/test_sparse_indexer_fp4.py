"""The FP4 sparse-indexer predicate, the shapes it derives, and the KV-pool
width it picks -- plus that the FP8 default is untouched by any of them."""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops import sparse_indexer_fp4
from atom.model_ops.attentions.mla_kv_pool import MlaKvPool
from atom.model_ops.sparse_indexer_fp4 import (
    assert_fp4_indexer_supported,
    fp4_decode_parallel_units,
    fp4_q_scale_shape,
    sparse_indexer_fp4_enabled,
)

# The DSA indexer geometry GLM-5.2 and DeepSeek-V3.2 share.
DSA = SimpleNamespace(index_topk=2048, index_n_heads=32, index_head_dim=128)


@pytest.fixture
def gfx950(monkeypatch):
    monkeypatch.setattr(sparse_indexer_fp4, "get_gfx", lambda: "gfx950")


def _pool(**overrides):
    args = {
        "layers": 2,
        "block_size": 64,
        "entry_dim": 576,
        "kv_dtype": torch.bfloat16,
        "index_layers": 3,
        "index_rows_per_block": 64,
        "index_dim": 144,
        "index_dtype": torch.uint8,
        "index_head_dim": 128,
    }
    args.update(overrides)
    return MlaKvPool(**args)


def test_predicate_reads_geometry_not_model_type(gfx950):
    assert sparse_indexer_fp4_enabled("fp4", DSA)
    # An MTP draft: `_MTP_TYPE_MAP` rewrote its model_type but not its indexer,
    # and it shares the target's cache, so it must reach the target's verdict.
    assert sparse_indexer_fp4_enabled(
        "fp4", SimpleNamespace(model_type="deepseek_mtp", **vars(DSA))
    )
    assert not sparse_indexer_fp4_enabled("fp8", DSA)
    assert not sparse_indexer_fp4_enabled(None, DSA)


@pytest.mark.parametrize(
    ("override", "why"),
    [
        ({"index_topk": 0}, "no sparse indexer"),
        ({"index_head_dim": 64}, "index_head_dim is 64"),
        ({"index_head_dim": 256}, "index_head_dim is 256"),
        ({"index_n_heads": 24}, "index_n_heads is 24"),
    ],
)
def test_predicate_falls_back_on_geometry_the_kernels_cannot_tile(
    gfx950, caplog, override, why
):
    config = SimpleNamespace(**{**vars(DSA), **override})
    assert not sparse_indexer_fp4_enabled("fp4", config)
    with caplog.at_level("WARNING", logger="atom"):
        assert not sparse_indexer_fp4_enabled("fp4", config, warn=True)
    assert why in caplog.text


def test_predicate_falls_back_off_gfx950(monkeypatch, caplog):
    monkeypatch.setattr(sparse_indexer_fp4, "get_gfx", lambda: "gfx942")
    assert not sparse_indexer_fp4_enabled("fp4", DSA)
    with caplog.at_level("WARNING", logger="atom"):
        assert not sparse_indexer_fp4_enabled("fp4", DSA, warn=True)
    assert "gfx942" in caplog.text


def test_predicate_does_not_probe_the_chip_for_the_fp8_default(monkeypatch):
    def boom():
        raise AssertionError("get_gfx() must not be reached on the FP8 default")

    monkeypatch.setattr(sparse_indexer_fp4, "get_gfx", boom)
    assert not sparse_indexer_fp4_enabled("fp8", DSA)
    assert not sparse_indexer_fp4_enabled(None, DSA)


def test_unsupported_fp4_requests_name_the_knob_that_blocked_them():
    assert_fp4_indexer_supported(fused_writer=True, context_parallel=False)
    with pytest.raises(ValueError, match="fused QK/RoPE/cache"):
        assert_fp4_indexer_supported(fused_writer=False, context_parallel=False)
    with pytest.raises(ValueError, match="DCP or PCP"):
        assert_fp4_indexer_supported(fused_writer=True, context_parallel=True)


def test_decode_parallel_units_are_a_multiple_of_next_n_covering_the_batch():
    for next_n in (1, 2, 3, 4, 8):
        for max_bs in (1, 16, 512, 8192):
            units = fp4_decode_parallel_units(max_bs, next_n)
            assert units % next_n == 0
            assert units // next_n >= max_bs
            assert units >= sparse_indexer_fp4.FP4_MQA_PARALLEL_UNIT_NUM


def test_q_scale_shape_pads_the_m_tile_axis_to_one_dword():
    # H=32 is two M-tiles, still loaded as one dword of four scale bytes.
    assert fp4_q_scale_shape(7, 32, 128) == (7, 1, 4, 16, 4)
    assert fp4_q_scale_shape(7, 64, 128) == (7, 1, 4, 16, 4)
    assert fp4_q_scale_shape(7, 128, 128) == (7, 1, 4, 16, 8)


def test_fp8_pool_is_one_index_field_and_two_arenas():
    pool = _pool()
    assert len(pool.field_groups) == 2
    assert [f.name for f in pool.index_fields] == ["index"]
    assert pool.entry_bytes == 2 * 64 * 576 * 2 + 3 * 64 * 144

    pool.allocate(3, "cpu")
    assert set(pool._views) == {"kv", "index"}
    assert pool.layer("index", 0).shape == (3, 64, 144)
    assert pool.layer("index", 0).dtype is torch.uint8


def test_fp4_pool_switches_the_index_field_without_adding_an_arena():
    pool = _pool(index_fp4=True)
    # Still two groups: the FP4 cache is the same region at another width.
    assert len(pool.field_groups) == 2
    assert [f.name for f in pool.index_fields] == ["index", "index_scale"]
    assert pool.entry_bytes == 2 * 64 * 576 * 2 + 3 * (4 * 64 * 16) + 3 * (4 * 64)
    assert pool.entry_bytes < _pool().entry_bytes

    pool.allocate(3, "cpu")
    assert set(pool._views) == {"kv", "index", "index_scale"}
    data, scale = pool.layer("index", 0), pool.layer("index_scale", 0)
    assert data.shape == (3, 1, 4, 64, 16) and data.dtype is torch.uint8
    assert scale.shape == (3, 1, 4, 64) and scale.dtype is torch.uint8

    spans = [
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
        for t in (pool.layer("kv", 0), data, scale)
    ]
    for i, lhs in enumerate(spans):
        for rhs in spans[i + 1 :]:
            assert lhs[1] <= rhs[0] or rhs[1] <= lhs[0]


def test_fp4_pool_constrains_indexer_rows_not_the_kv_block_size():
    _pool(index_fp4=True, block_size=32, index_rows_per_block=64)
    with pytest.raises(ValueError, match="--block-size 64"):
        _pool(index_fp4=True, index_rows_per_block=32)
