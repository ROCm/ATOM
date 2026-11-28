# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A KDA state shard packs to the bytes the direct path would write."""

import numpy as np

from atom.kv_transfer.disaggregation import state_gather as gs
from atom.kv_transfer.disaggregation.state_gather import (
    copy_state_pieces,
    plan_state_shard,
)

# One conv layer: 2 rows of [q | k | v], each run 4 bytes, then a 16-byte SSM
# unit and a 4-byte replicated cursor. The consumer is TP2, rank 1.
_ROWS = 2
_RUN = 4
_CONV = _ROWS * 3 * _RUN
_SSM = 16
_CURSOR = 4


def _shard():
    return plan_state_shard(
        src_units=[_CONV, _SSM, _CURSOR],
        dst_bases=[1000, 2000, 3000],
        dst_units=[_CONV // 2, _SSM // 2, _CURSOR],
        tp_segments=[(_RUN,) * (3 * _ROWS), (_SSM,), None],
        tp_replicated=[False, False, True],
        slot_map=[0, 1, 2],
        tp_ratio=2,
        tp_part=1,
        dst_slot=3,
    )


def test_conv_pieces_pack_into_one_segment_per_region():
    shard = _shard()
    # 2 rows * 3 runs * (4/2) bytes, plus half the SSM, plus the whole cursor.
    assert shard.total == _ROWS * 3 * (_RUN // 2) + _SSM // 2 + _CURSOR
    assert [seg[0] for seg in shard.segments] == [0, 1, 2]
    assert [seg[3] for seg in shard.segments] == [
        _ROWS * 3 * (_RUN // 2),
        _SSM // 2,
        _CURSOR,
    ]
    # Rank 1's conv pieces are the second half of every run, in run order.
    conv = [piece for piece in shard.pieces if piece[0] == 0]
    assert len(conv) == 3 * _ROWS
    assert [piece[1] % _RUN for piece in conv] == [_RUN // 2] * len(conv)
    assert shard.segments[0][2] == 1000 + 3 * (_CONV // 2)
    assert shard.segments[2][2] == 3000 + 3 * _CURSOR


def test_packed_bytes_match_the_direct_writes():
    rng = np.random.default_rng(0)
    conv = rng.integers(0, 256, _CONV, dtype=np.uint8)
    ssm = rng.integers(0, 256, _SSM, dtype=np.uint8)
    cursor = rng.integers(0, 256, _CURSOR, dtype=np.uint8)
    views = [conv.reshape(1, -1), ssm.reshape(1, -1), cursor.reshape(1, -1)]
    shard = _shard()
    staging = np.zeros(shard.total, dtype=np.uint8)
    copy_state_pieces(staging, views, 0, shard.pieces)

    direct = np.concatenate(
        [
            conv.reshape(_ROWS, 3, _RUN)[:, :, _RUN // 2 :].ravel(),
            ssm[_SSM // 2 :],
            cursor,
        ]
    )
    assert staging.tolist() == direct.tolist()


def test_a_torch_gather_copies_strided_runs_in_one_copy_and_matches_numpy():
    import torch

    rng = np.random.default_rng(1)
    host = [
        rng.integers(0, 256, (2, n), dtype=np.uint8) for n in (_CONV, _SSM, _CURSOR)
    ]
    shard = _shard()
    runs = list(gs._runs(shard.pieces))
    # 6 conv pieces at a pitch of one run collapse to one copy.
    assert runs[0][4] == 3 * _ROWS and runs[0][5] == _RUN
    want = np.zeros(shard.total, dtype=np.uint8)
    copy_state_pieces(want, [h for h in host], 1, shard.pieces)
    got = torch.zeros(shard.total, dtype=torch.uint8)
    copy_state_pieces(got, [torch.from_numpy(h) for h in host], 1, shard.pieces)
    assert got.numpy().tolist() == want.tolist()


def test_an_unsplit_producer_copies_every_unit_whole():
    shard = plan_state_shard(
        src_units=[_CONV, _SSM],
        dst_bases=[1000, 2000],
        dst_units=[_CONV, _SSM],
        tp_segments=[None, None],
        tp_replicated=None,
        slot_map=[0, 1],
        tp_ratio=1,
        tp_part=0,
        dst_slot=0,
    )
    assert shard.total == _CONV + _SSM
    assert [piece[2] for piece in shard.pieces] == [_CONV, _SSM]


def test_gather_segments_split_long_pieces_and_copy_the_same_bytes():
    rng = np.random.default_rng(2)
    host = [
        rng.integers(0, 256, (2, n), dtype=np.uint8) for n in (_CONV, _SSM, _CURSOR)
    ]
    shard = _shard()
    want = np.zeros(shard.total, dtype=np.uint8)
    copy_state_pieces(want, host, 1, shard.pieces)

    # One flat "address space" holding the three regions back to back.
    memory = np.concatenate([h.ravel() for h in host])
    bases = np.cumsum([0] + [h.size for h in host[:-1]])
    addrs = [int(b) + 1 * h.shape[1] for b, h in zip(bases, host)]
    region, src, dst, nbytes = gs.gather_segments(addrs, shard.pieces, max_bytes=4)
    assert sorted(set(region.tolist())) == [0, 1, 2]
    assert nbytes.max() == 4 and nbytes.sum() == shard.total
    got = np.zeros(shard.total, dtype=np.uint8)
    for s, d, n in zip(src, dst, nbytes):
        got[d : d + n] = memory[s : s + n]
    assert got.tolist() == want.tolist()


def test_a_cuda_gather_is_one_launch_with_the_per_copy_bytes():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    pytest.importorskip("triton")
    rng = np.random.default_rng(3)
    # 8-byte runs, so rank 1's 4-byte half is int32-aligned (2-byte halves
    # are not, and keep the per-copy path).
    run = 8
    conv = _ROWS * 3 * run
    # Separate allocations, as the KDA caches are; units of 2 slots each.
    host = [
        rng.integers(0, 256, (2, n), dtype=np.uint8)
        for n in (conv, gs.GATHER_SEGMENT_BYTES * 3, _CURSOR)
    ]
    views = [torch.from_numpy(h).cuda() for h in host]
    shard = plan_state_shard(
        src_units=[conv, gs.GATHER_SEGMENT_BYTES * 3, _CURSOR],
        dst_bases=[0, 0, 0],
        dst_units=[conv // 2, gs.GATHER_SEGMENT_BYTES * 3 // 2, _CURSOR],
        tp_segments=[(run,) * (3 * _ROWS), (gs.GATHER_SEGMENT_BYTES * 3,), None],
        tp_replicated=[False, False, True],
        slot_map=[0, 1, 2],
        tp_ratio=2,
        tp_part=1,
        dst_slot=0,
    )
    want = np.zeros(shard.total, dtype=np.uint8)
    copy_state_pieces(want, host, 1, shard.pieces)
    got = torch.zeros(shard.total, dtype=torch.uint8, device="cuda")
    assert gs._gather_on_gpu(got, views, 1, shard.pieces)
    torch.cuda.synchronize()
    assert got.cpu().numpy().tolist() == want.tolist()
    # Slot 0 reuses the cached table with the slot as a kernel argument.
    tables = len(gs._GATHER_TABLES)
    copy_state_pieces(want, host, 0, shard.pieces)
    assert gs._gather_on_gpu(got, views, 0, shard.pieces)
    assert len(gs._GATHER_TABLES) == tables
    # A 2-byte piece cannot be addressed in int32 words.
    assert not gs._gather_on_gpu(got, views, 1, [(0, 2, 2, 0)])
    torch.cuda.synchronize()
    assert got.cpu().numpy().tolist() == want.tolist()
