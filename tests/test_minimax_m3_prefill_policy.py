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
