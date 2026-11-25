# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DSA ownership, admission, selection and FP4 layout oracles on pure CPU.

No Torch, AITER or FlyDSL imports: planning coverage runs without the GPU stack.
"""

import math
from dataclasses import replace
from itertools import pairwise

import pytest

from atom.model_ops.indexer.layout import (
    MFMA,
    PRESHUFFLE,
    fp4_cache_byte,
    fp4_scale_byte,
)
from atom.mono.dsa_indexer.build import (
    IxBuild,
    dual_score_resources_fit,
    local_step_symbol,
    unfit,
)
from atom.mono.dsa_indexer.fp4_plan import e2m1_magnitude
from atom.mono.dsa_indexer.plan import (
    REPLICA_INTERLEAVE,
    IndexerShape,
    key_pos,
    local_len,
    unsupported,
)
from atom.mono.dsa_indexer.selection_plan import (
    LOCAL_RADIX,
    SHORT_VERIFY_CTX,
    SPLIT_MIN,
    Radix,
    prefix_row,
    request_owner,
    short_flag_count,
    split_request_keys,
)
from atom.mono.plan.build_key import kernel_symbol, symbol_params
from atom.mono.plan.execution import BLOCKS

CTXS = [0, 1, 7, 127, 128, 129, 1023, 1024, 1025, 2049, 5000, 33333, 131073]


def _shape(q_lora=2048, cache="fp8", *, block_rows=None, rope_dim=None):
    return IndexerShape(
        32,
        128,
        2048,
        q_lora,
        "bf16",
        cache,
        (64 if cache == "fp4" else 128) if block_rows is None else block_rows,
        (64 if q_lora == 2048 else 0) if rope_dim is None else rope_dim,
        cache_layout=MFMA if cache == "fp4" else PRESHUFFLE,
    )


@pytest.mark.parametrize("interleave", [1, REPLICA_INTERLEAVE])
@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_a_ranks_keys_are_the_positions_it_owns_in_order(world, interleave):
    for ctx in CTXS:
        owned = [[] for _ in range(world)]
        for pos in range(ctx):
            owned[pos // interleave % world].append(pos)
        for rank in range(world):
            n = local_len(ctx, rank, world, interleave)
            assert n == len(owned[rank]), (ctx, rank)
            got = [key_pos(j, rank, world, interleave) for j in range(n)]
            assert got == owned[rank], (ctx, rank)


@pytest.mark.parametrize("rows", [1, 48, 64, 65, 96, 128, 252, 256])
@pytest.mark.parametrize("replicated", [False, True])
def test_single_rank_build_admission(rows, replicated):
    key = IxBuild(_shape(1536, "fp4"), rows, 1, 163840, replicated)
    assert bool(unfit(key)) == (rows > 64 and not replicated)
    assert unfit(replace(key, short_gemm=True))
    assert not key.own_rows
    assert not key.short_owned_verify
    assert key.row_keys == key.max_ctx


@pytest.mark.parametrize("world", [0, 3, 16])
def test_invalid_world_is_refused_before_dividing_cache_geometry(world):
    assert unfit(IxBuild(_shape(1536, "fp4"), 64, world, 8192, replicated=True))


def test_zero_query_length_is_refused_before_request_arithmetic():
    assert unfit(IxBuild(_shape(1536, "fp4"), 64, 4, 8192, replicated=True, qlen=0))


@pytest.mark.parametrize("rows", [65, 96, 128, 192, 252, 256])
def test_wave_prefix_plan_matches_serial_csr_offsets(rows):
    counts = [(r * 37 + 11) % 2049 for r in range(rows)]
    for stop in range(rows + 1):
        lanes = []
        for lane in range(64):
            indices = [prefix_row(lane, tile) for tile in range((rows + 63) // 64)]
            lanes.append(sum(counts[i] for i in indices if i < stop))
        assert sum(lanes) == sum(counts[:stop])


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("qlen", [1, 6, 8])
def test_request_ownership_keeps_verify_rows_together(world, qlen):
    for request in range(43):
        owners = {
            request_owner(request * qlen + row, qlen, world) for row in range(qlen)
        }
        assert owners == {request % world}


@pytest.mark.parametrize("world", [2, 4, 8])
def test_short_verify_and_large_request_split_boundaries(world):
    assert not split_request_keys(SHORT_VERIFY_CTX, 0, world, low_requests=True)
    assert split_request_keys(SHORT_VERIFY_CTX + 1, 0, world, low_requests=True)
    assert not split_request_keys(SPLIT_MIN, 0, world)
    context = SPLIT_MIN + 1
    assert not split_request_keys(context, 2 * world * context, world)
    assert split_request_keys(context, 2 * world * context - 1, world)
    key = IxBuild(_shape(1536, "fp4"), 6, world, SPLIT_MIN, replicated=True, qlen=6)
    assert key.short_owned_verify and key.own_rows
    assert key.row_keys >= SHORT_VERIFY_CTX
    assert not replace(key, max_ctx=SPLIT_MIN + 1).short_owned_verify


@pytest.mark.parametrize("flag_stride", [0, 8, 24, 512, 1024, 8192])
def test_short_reset_stays_inside_each_rows_flag_slab(flag_stride):
    count = short_flag_count(flag_stride)
    assert count * 8 <= flag_stride
    assert count * 128 <= SHORT_VERIFY_CTX
    for row in range(252):
        for chunk in range(count):
            address = row * flag_stride + chunk * 8
            assert row * flag_stride <= address < (row + 1) * flag_stride


@pytest.mark.parametrize("block_rows", [64, 128, 256])
def test_opus_key_and_scale_planes_match_physical_tensor_coordinates(block_rows):
    data, scales = set(), set()
    for row in range(block_rows):
        group, within = divmod(row, 64)
        for col in range(64):
            chunk, lane = divmod(col, 16)
            expected = ((group * 4 + chunk) * 64 + within) * 16 + lane
            address = fp4_cache_byte(row, col)
            assert address == expected
            data.add(address)
        for chunk in range(4):
            expected = (
                ((group * 2 + chunk % 2) * 32 + within % 32) * 2 + chunk // 2
            ) * 2 + within // 32
            address = fp4_scale_byte(row, chunk)
            assert address == expected
            scales.add(address)
    assert data == set(range(block_rows * 64))
    assert scales == set(range(block_rows * 4))

    sh = _shape(cache="fp4", block_rows=block_rows, rope_dim=0)
    assert sh.block_bytes == block_rows * 64
    assert sh.scale_block_bytes == block_rows * 4


@pytest.mark.parametrize("block_rows", [64, 128])
def test_opus_page_addressing_does_not_wrap_at_four_gib(block_rows):
    sh = replace(_shape(1536, "fp4"), block_rows=block_rows)
    page = (1 << 32) // sh.block_bytes + 1
    for row in (0, 31, 32, block_rows - 1):
        for col in (0, 15, 16, 63):
            address = page * sh.block_bytes + fp4_cache_byte(row, col)
            assert address > 1 << 32
            assert address == fp4_cache_byte(page * block_rows + row, col)
        for group in range(4):
            address = page * sh.scale_block_bytes + fp4_scale_byte(row, group)
            assert address == fp4_scale_byte(page * block_rows + row, group)


def test_local_radix_digits_reconstruct_every_bit_of_the_order_key():
    radix = LOCAL_RADIX
    for key in [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x12345678]:
        digits = [
            (key >> shift) & ((1 << bits) - 1)
            for shift, bits in zip(radix.shifts, radix.bits)
        ]
        assert sum(d << shift for d, shift in zip(digits, radix.shifts)) == key
    for bits in [(8, 8, 8), (12, 10, 10), (0, 16, 16)]:
        with pytest.raises(ValueError):
            Radix(bits, local=True)


def test_e2m1_magnitude_is_nearest_even_at_thresholds_and_saturates():
    values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    samples = values + [float("inf"), 100.0]
    for a, b in pairwise(values):
        midpoint = (a + b) / 2
        samples.extend(
            [
                math.nextafter(midpoint, -math.inf),
                midpoint,
                math.nextafter(midpoint, math.inf),
            ]
        )
    for x in samples:
        expected = (
            7
            if math.isinf(x)
            else min(range(8), key=lambda i: (abs(values[i] - x), i % 2))
        )
        assert e2m1_magnitude(x) == expected


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("cache", ["fp8", "fp4"])
def test_supported_widths_and_query_lengths(replicated, cache):
    key = IxBuild(_shape(cache=cache), 1, 8, 163840, replicated)
    for rows in range(1, 257):
        assert not unfit(replace(key, tokens=rows))
    for rows in range(6, 253, 6):
        assert not unfit(replace(key, tokens=rows, qlen=6))
    assert not unfit(replace(key, tokens=64, qlen=8))
    for rows, qlen in ((0, 1), (257, 1), (64, 6), (256, 6), (258, 6), (72, 8)):
        assert unfit(replace(key, tokens=rows, qlen=qlen))
    factor = 1 if replicated else key.world
    assert not unfit(replace(key, tokens=256, max_ctx=factor * ((1 << 22) - 1)))
    assert unfit(replace(key, tokens=256, max_ctx=factor * (1 << 22)))


def test_short_projection_requires_supported_shapes_and_replicated_cache():
    key = IxBuild(_shape(), 96, 8, 163840, replicated=True, qlen=6, short_gemm=True)
    for q_lora in (1024, 1536, 2048):
        assert not unfit(replace(key, shape=_shape(q_lora)))
    for changes in (
        {"world": 1},
        {"replicated": False},
        {"max_ctx": 262145},
        {"shape": _shape(1280)},
    ):
        assert unfit(replace(key, **changes))


@pytest.mark.parametrize("q_lora", [1536, 2048])
def test_dual_grid_rejects_unsupported_launches(q_lora):
    key = IxBuild(_shape(q_lora), 256, 8, 163840, score_blocks=512)
    assert not unfit(key)
    for changes in (
        {"tokens": 64},
        {"tokens": 129},
        {"world": 4},
        {"qlen": 2},
        {"replicated": True},
        {"score_blocks": 384},
        {"shape": _shape(q_lora, "fp4")},
    ):
        assert unfit(replace(key, **changes))


@pytest.mark.parametrize("changed", ["rows", "qlen", "world", "debug", 0, 1, 2, 3])
def test_local_step_codegen_settings_have_distinct_kernel_symbols(changed):
    args = {"rows": 48, "qlen": 6, "world": 8, "compact": (512, 1024, 49664, 53760)}
    baseline = local_step_symbol(**args)
    stem = "mono_local_step_s48_q6_w8_debug0_prefix512_flag_stride1024_suffix49664_total53760"
    assert baseline == f"_ZN4atom{len(stem)}{stem}E"
    if isinstance(changed, int):
        compact = list(args["compact"])
        compact[changed] += 16
        args["compact"] = tuple(compact)
    else:
        args[changed] = {"rows": 54, "qlen": 8, "world": 4, "debug": True}[changed]
    assert baseline != local_step_symbol(**args)


def test_kernel_symbols_keep_boolean_and_negative_parameter_rendering():
    stem = "index_s1_debug1_rank_1"
    assert (
        kernel_symbol("index", s=1, debug=True, rank=-1)
        == f"_ZN4atom{len(stem)}{stem}E"
    )


def test_score_grid_builds_have_distinct_symbol_parameters():
    single = IxBuild(_shape(1536), 256, 8, 163840)
    dual = replace(single, score_blocks=2 * BLOCKS)
    assert not unfit(single) and not unfit(dual)
    assert symbol_params(single) != symbol_params(dual)


@pytest.mark.parametrize(
    "vgpr,lds,scratch,expected",
    [
        (128, 81920, 0, True),
        (129, 63840, 0, False),
        (119, 81921, 0, False),
        (119, 63840, 4, False),
    ],
)
def test_dual_cta_resource_limits(vgpr, lds, scratch, expected):
    assert (
        dual_score_resources_fit([{"vgpr": vgpr, "lds": lds, "scratch": scratch}])
        == expected
    )
    assert not dual_score_resources_fit([])


def test_no_silent_reinterpretation_of_legacy_fp4():
    sh = _shape(cache="fp4", rope_dim=0)
    assert not unsupported(sh)
    assert "OPUS" in unsupported(replace(sh, cache_layout=PRESHUFFLE))
    assert "64-row" in unsupported(replace(sh, block_rows=16))
    assert "64-row" in unsupported(replace(sh, block_rows=96))
    assert sh.tag != replace(sh, cache_layout=PRESHUFFLE).tag


def test_fp8_contract_unchanged():
    sh = replace(
        _shape(cache="fp4", rope_dim=0), kv_dtype="fp8", cache_layout=PRESHUFFLE
    )
    assert sh.row_bytes == 144 and sh.block_bytes == 9216
    assert sh.scale_block_bytes == 0 and not unsupported(sh)
