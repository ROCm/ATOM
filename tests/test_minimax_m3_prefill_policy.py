# SPDX-License-Identifier: MIT

import pytest

from atom.model_ops.minimax_m3.prefill_policy import (
    DEFAULT_PREFILL_SCORE_CONFIG,
    PrefillScoreConfig,
    parallel_prefill_selector_wins,
    prefill_score_config,
)

TOPK = 16

MULTI_PLANE = PrefillScoreConfig(
    block_size_q=128,
    chunk_blocks=28,
    num_warps=2,
    num_stages=1,
    store_group=1,
    prefetch_pages=False,
    post_reduce_scale=True,
)


def q64(chunk_blocks: int) -> PrefillScoreConfig:
    return PrefillScoreConfig(
        block_size_q=64,
        chunk_blocks=chunk_blocks,
        num_warps=2,
        num_stages=3,
        store_group=4,
        prefetch_pages=True,
        post_reduce_scale=True,
    )


def q256(chunk_blocks: int) -> PrefillScoreConfig:
    return PrefillScoreConfig(
        block_size_q=256,
        chunk_blocks=chunk_blocks,
        num_warps=4,
        num_stages=1,
        store_group=1,
        prefetch_pages=False,
        post_reduce_scale=True,
    )


@pytest.mark.parametrize(
    "arch,width,rows,emit,heads,decode_max_q,want",
    [
        ("gfx950", 1280, 512, True, 1, 0, False),
        ("gfx1250", 1279, 512, True, 1, 0, False),
        ("gfx1250", 1280, 511, True, 1, 0, False),
        ("gfx1250", 1280, 512, True, 1, 0, True),
        ("gfx1250", 4096, 170, True, 1, 0, False),
        ("gfx1250", 4096, 171, True, 1, 0, True),
        ("gfx1250", 4097, 512, True, 1, 0, False),
        ("gfx1250", 1280, 512, False, 1, 0, False),
        ("gfx1250", 1280, 512, True, 4, 0, False),
        ("gfx1250", 1280, 512, True, 1, 1, False),
    ],
)
def test_parallel_prefill_selector_resource_bounds(
    arch, width, rows, emit, heads, decode_max_q, want
):
    assert (
        parallel_prefill_selector_wins(
            arch=arch,
            width=width,
            rows=rows,
            compute_units=256,
            emit=emit,
            topk=TOPK,
            num_idx_heads=heads,
            decode_max_q=decode_max_q,
        )
        == want
    )


@pytest.mark.parametrize(
    "arch,is_fp8,q_len,blocks,batch,heads,want",
    [
        ("gfx950", True, 2517, 823, 1, 1, DEFAULT_PREFILL_SCORE_CONFIG),
        (
            "gfx1250",
            False,
            2517,
            823,
            1,
            1,
            DEFAULT_PREFILL_SCORE_CONFIG,
        ),
        ("gfx1250", True, 96, 640, 1, 4, DEFAULT_PREFILL_SCORE_CONFIG),
        ("gfx1250", True, 512, 640, 1, 4, q64(14)),
        (
            "gfx1250",
            True,
            2048,
            256,
            1,
            1,
            DEFAULT_PREFILL_SCORE_CONFIG,
        ),
        ("gfx1250", True, 2048, 640, 1, 1, q64(14)),
        ("gfx1250", True, 2517, 823, 1, 1, q64(22)),
        ("gfx1250", True, 2655, 796, 1, 1, q64(22)),
        ("gfx1250", True, 2159, 974, 1, 1, q64(16)),
        ("gfx1250", True, 3072, 640, 1, 1, q64(20)),
        ("gfx1250", True, 2858, 3942, 1, 1, q64(58)),
        ("gfx1250", True, 6259, 663, 1, 1, q64(42)),
        ("gfx1250", True, 4104, 1895, 1, 1, q64(40)),
        ("gfx1250", True, 4096, 320, 1, 1, q64(20)),
        ("gfx1250", True, 6144, 320, 1, 1, q64(30)),
        ("gfx1250", True, 5459, 1682, 1, 1, q256(74)),
        ("gfx1250", True, 2331, 2706, 1, 1, q64(33)),
        ("gfx1250", True, 3072, 4096, 1, 1, q64(64)),
        ("gfx1250", True, 640, 257, 1, 4, q64(10)),
        ("gfx1250", True, 640, 257, 4, 1, q64(10)),
        ("gfx1250", True, 4096, 640, 4, 1, q64(107)),
        ("gfx1250", True, 2048, 640, 1, 4, q64(54)),
        ("gfx1250", True, 2517, 640, 1, 4, q64(64)),
        ("gfx1250", True, 4096, 320, 1, 4, q64(80)),
        ("gfx1250", True, 8192, 320, 1, 4, q64(160)),
        ("gfx1250", True, 4096, 640, 1, 4, q64(107)),
        ("gfx1250", True, 1024, 640, 1, 1, DEFAULT_PREFILL_SCORE_CONFIG),
        ("gfx1250", True, 1025, 640, 1, 1, q64(8)),
        ("gfx1250", True, 8192, 640, 1, 1, q256(40)),
        ("gfx1250", True, 8193, 640, 1, 1, q256(43)),
        ("gfx1250", True, 8192, 640, 1, 4, q64(214)),
        ("gfx1250", True, 8193, 640, 1, 4, MULTI_PLANE),
        ("gfx1250", True, 32768, 640, 1, 4, MULTI_PLANE),
        ("gfx1250", True, 32769, 640, 1, 4, MULTI_PLANE),
        (
            "gfx1250",
            True,
            65536,
            640,
            1,
            1,
            DEFAULT_PREFILL_SCORE_CONFIG,
        ),
        ("gfx1250", True, 65536, 640, 1, 4, MULTI_PLANE),
        ("gfx1250", True, 32768, 2048, 1, 4, MULTI_PLANE),
        (
            "gfx1250",
            True,
            65536,
            2048,
            1,
            4,
            DEFAULT_PREFILL_SCORE_CONFIG,
        ),
        ("gfx1250", True, 131072, 2048, 1, 1, q256(2048)),
    ],
)
def test_prefill_score_config(arch, is_fp8, q_len, blocks, batch, heads, want):
    assert prefill_score_config(arch, is_fp8, q_len, blocks, batch, heads, 256) == want


def test_prefill_score_config_cache_is_bounded():
    assert prefill_score_config.cache_parameters() == {
        "maxsize": 128,
        "typed": False,
    }


@pytest.mark.parametrize(
    "q_len, blocks, batch, heads",
    [
        (2517, 823, 1, 1),  # Q64 grouped
        (8193, 640, 1, 4),  # multi-plane
        (131072, 2048, 1, 1),  # Q256, grouped stores from 2048 blocks
        (8192, 1024, 1, 1),  # Q256 on a short row: one store a block
        (5979, 4015, 1, 1),  # Q256, the c4 AgentX prefill (514K context)
        (1024, 640, 1, 1),  # default
        (96, 640, 1, 4),  # default
    ],
)
def test_fp4_takes_the_fp8_buckets_at_one_stage(q_len, blocks, batch, heads):
    """The FP4 index cache: the FP8 bucket for the shape, every tuned
    specialization at FP4_PREFILL_SCORE_NUM_STAGES (its rows are dequantized in
    registers), the multi-plane tile at four warps, the wide tile's stores
    grouped on a long row, and the default left to the generic launch policy."""
    from atom.model_ops.minimax_m3.prefill_policy import (
        FP4_PREFILL_SCORE_NUM_STAGES,
        FP4_WIDE_SCORE_BLOCK_SIZE_Q,
        FP4_WIDE_STORE_GROUP,
        FP4_WIDE_STORE_GROUP_MIN_BLOCKS,
    )

    fp8 = prefill_score_config("gfx1250", True, q_len, blocks, batch, heads, 256)
    fp4 = prefill_score_config(
        "gfx1250", False, q_len, blocks, batch, heads, 256, fp4=True
    )
    if fp8 is DEFAULT_PREFILL_SCORE_CONFIG:
        assert fp4 is DEFAULT_PREFILL_SCORE_CONFIG
    else:
        want = fp8._replace(num_stages=FP4_PREFILL_SCORE_NUM_STAGES)
        if fp8 == MULTI_PLANE:
            want = want._replace(num_warps=4)  # two warps: 694 VGPRs on FP4
        if (
            fp8.block_size_q == FP4_WIDE_SCORE_BLOCK_SIZE_Q
            and blocks >= FP4_WIDE_STORE_GROUP_MIN_BLOCKS
        ):
            want = want._replace(store_group=FP4_WIDE_STORE_GROUP)
        assert fp4 == want
        # The FP8 buckets themselves do not move.
        assert fp8.store_group == (4 if fp8.block_size_q == 64 else 1)
    # Off gfx1250 the FP4 cache keeps the default, as FP8 does.
    assert (
        prefill_score_config(
            "gfx950", False, q_len, blocks, batch, heads, 256, fp4=True
        )
        is DEFAULT_PREFILL_SCORE_CONFIG
    )


@pytest.mark.parametrize(
    "q_len, blocks, want_group",
    [(5979, 4015, 4), (16384, 2344, 4), (8192, 1024, 1), (131072, 2047, 1)],
)
def test_fp4_wide_tile_groups_its_stores_on_a_long_row(q_len, blocks, want_group):
    config = prefill_score_config("gfx1250", False, q_len, blocks, 1, 1, 256, fp4=True)
    assert config.block_size_q == 256
    assert config.store_group == want_group
    # The grouped loop reads its pages one at a time, as the tile was tuned.
    assert not config.prefetch_pages


@pytest.mark.parametrize(
    "q_len, blocks, batch",
    [
        (6705, 4058, 7),  # the c64 AgentX batched prefills measured
        (13623, 4487, 3),
        (9105, 1702, 4),
        (4272, 5649, 5),
        (4471, 2911, 2),
        (3760, 2896, 4),
    ],
)
def test_fp4_batched_prefill_takes_the_wide_grouped_launch(q_len, blocks, batch):
    from atom.model_ops.minimax_m3.prefill_policy import (
        FP4_BATCHED_PREFILL_SCORE_CONFIG,
    )

    fp4 = prefill_score_config("gfx1250", False, q_len, blocks, batch, 1, 256, fp4=True)
    assert fp4 == FP4_BATCHED_PREFILL_SCORE_CONFIG
    assert (fp4.block_size_q, fp4.chunk_blocks, fp4.num_warps) == (256, 64, 8)
    assert (fp4.store_group, fp4.post_reduce_scale, fp4.num_stages) == (4, True, 1)
    # FP8 and the other archs keep their buckets.
    fp8 = prefill_score_config("gfx1250", True, q_len, blocks, batch, 1, 256)
    assert fp8 != FP4_BATCHED_PREFILL_SCORE_CONFIG
    assert (
        prefill_score_config("gfx950", False, q_len, blocks, batch, 1, 256, fp4=True)
        is DEFAULT_PREFILL_SCORE_CONFIG
    )


@pytest.mark.parametrize(
    "q_len, blocks, batch",
    [
        (5979, 4015, 1),  # one request: its own (wide) bucket
        (4272, 1023, 5),  # under the block bound
        (1024, 4058, 3),  # 12 wide query tiles over the batch: under the bound
        (256, 640, 2),
    ],
)
def test_fp4_small_or_single_prefill_keeps_its_bucket(q_len, blocks, batch):
    from atom.model_ops.minimax_m3.prefill_policy import (
        FP4_BATCHED_PREFILL_SCORE_CONFIG,
        _gfx1250_prefill_score_config,
    )

    fp4 = prefill_score_config("gfx1250", False, q_len, blocks, batch, 1, 256, fp4=True)
    bucket = _gfx1250_prefill_score_config(q_len, blocks, batch, 1, 256)
    assert fp4.block_size_q == bucket.block_size_q
    assert fp4.chunk_blocks == bucket.chunk_blocks
    assert batch == 1 or fp4 != FP4_BATCHED_PREFILL_SCORE_CONFIG
