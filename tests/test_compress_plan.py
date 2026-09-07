#!/usr/bin/env python3
"""CPU unit tests for `make_compress_plans` write/compress slice capacities.

Covers the three slicing modes (eager / decode-CUDAGraph / extend-shaped verify)
and the CUDAGraph invariance that lets the decode write grid shrink to
`running_bs * min(qlen, K_pool)` while keeping `[bs, running_bs)` padding rows
sentinel-filled. No GPU required — a fake CpuGpuBuffer backs the plan tensors.
"""

import importlib.util
import pathlib

import numpy as np
import pytest
import torch

# Load compress_plan.py directly by path: it has no atom imports (numpy+torch
# only), so this avoids triggering `atom.model_ops.__init__`, whose heavy
# import chain needs a full atom.config that the CPU test env stubs out.
_CP_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "atom/model_ops/v4_kernels/compress_plan.py"
)
_spec = importlib.util.spec_from_file_location("_compress_plan_under_test", _CP_PATH)
_cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cp)
make_compress_plans = _cp.make_compress_plans

# V4-Pro layer geometry: (ratio, is_overlap). CSA r=4 overlap (K_pool=8),
# HCA r=128 non-overlap (K_pool=128).
RATIOS_OVERLAP = [(4, True), (128, False)]
POSITION_COL = 2  # write/compress plan row = [ragged_id, batch_id, position, wlen]


def _k_pool(ratio, is_overlap):
    return (2 if is_overlap else 1) * ratio


class _FakeBuf:
    """Minimal CpuGpuBuffer stand-in: numpy-backed, `copy_to_gpu(n)` returns a
    from-base torch view (shares memory, so host writes are visible)."""

    def __init__(self, rows):
        # Pre-fill with a non-sentinel sentinel so we can prove the -1 fill ran.
        self.np = np.full((rows, 4), 7, dtype=np.int32)
        self._t = torch.from_numpy(self.np)

    def copy_to_gpu(self, n=None):
        return self._t if n is None else self._t[:n]


def _buffers(compress_rows=256, write_rows=256):
    return {
        ratio: {"compress": _FakeBuf(compress_rows), "write": _FakeBuf(write_rows)}
        for ratio, _ in RATIOS_OVERLAP
    }


def _uniform_decode(bs, qlen, ctx=100):
    extend = np.full(bs, qlen, dtype=np.int32)
    context = np.full(bs, ctx, dtype=np.int32)
    return extend, context


# ── eager path (running_bs=None) ────────────────────────────────────────────


def test_eager_compress_tight_write_full_buffer():
    extend, context = _uniform_decode(bs=5, qlen=4)
    bufs = _buffers(write_rows=200)
    plans = make_compress_plans(extend, context, RATIOS_OVERLAP, plan_buffers=bufs)
    for ratio, _ in RATIOS_OVERLAP:
        p = plans[ratio]
        # compress slice is tight; write slice is the full buffer (legacy).
        assert p.compress_plan_gpu.shape[0] == p.num_compress
        assert p.write_plan_gpu.shape[0] == 200


# ── decode CUDAGraph path (running_bs set) ──────────────────────────────────


def test_decode_cg_slice_equals_graph_bs_times_bound():
    running_bs, qlen = 8, 4
    extend, context = _uniform_decode(bs=5, qlen=qlen)
    bufs = _buffers()
    plans = make_compress_plans(
        extend,
        context,
        RATIOS_OVERLAP,
        plan_buffers=bufs,
        running_bs=running_bs,
        max_q_len=qlen,
    )
    for ratio, is_overlap in RATIOS_OVERLAP:
        p = plans[ratio]
        exp_ccap = running_bs * ((qlen + ratio - 1) // ratio)
        exp_wcap = running_bs * min(qlen, _k_pool(ratio, is_overlap))
        assert p.compress_plan_gpu.shape[0] == exp_ccap
        assert p.write_plan_gpu.shape[0] == exp_wcap


def test_decode_write_count_is_bs_times_bound():
    # For decode (qlen <= K_pool) every query token lands in the ring window,
    # so num_write is EXACTLY bs * min(qlen, K_pool) — content-independent.
    running_bs, qlen, bs = 8, 4, 5
    extend, context = _uniform_decode(bs=bs, qlen=qlen)
    plans = make_compress_plans(
        extend,
        context,
        RATIOS_OVERLAP,
        plan_buffers=_buffers(),
        running_bs=running_bs,
        max_q_len=qlen,
    )
    for ratio, is_overlap in RATIOS_OVERLAP:
        assert plans[ratio].num_write == bs * min(qlen, _k_pool(ratio, is_overlap))


def test_decode_padding_region_is_sentinel():
    # The [n_write, cap) tail == the [bs, running_bs) padding seqs → all sentinel.
    running_bs, qlen, bs = 8, 4, 5
    extend, context = _uniform_decode(bs=bs, qlen=qlen)
    plans = make_compress_plans(
        extend,
        context,
        RATIOS_OVERLAP,
        plan_buffers=_buffers(),
        running_bs=running_bs,
        max_q_len=qlen,
    )
    for ratio, _ in RATIOS_OVERLAP:
        p = plans[ratio]
        active = p.write_plan_gpu[: p.num_write, POSITION_COL]
        padding = p.write_plan_gpu[p.num_write :, POSITION_COL]
        assert (active >= 0).all(), "active write rows must carry real positions"
        assert (padding == -1).all(), "padding rows must be sentinel (position=-1)"


def test_decode_cg_invariant_across_real_bs():
    # Same (running_bs, qlen), different real bs → identical slice shapes. This is
    # the CUDAGraph capture/replay invariant: capture runs at bs==running_bs,
    # replay at bs<running_bs, both must dispatch the same-shaped kernel.
    running_bs, qlen = 16, 2
    shapes = {}
    for bs in (running_bs, 1, 7):
        extend, context = _uniform_decode(bs=bs, qlen=qlen)
        plans = make_compress_plans(
            extend,
            context,
            RATIOS_OVERLAP,
            plan_buffers=_buffers(),
            running_bs=running_bs,
            max_q_len=qlen,
        )
        shapes[bs] = {
            r: (plans[r].compress_plan_gpu.shape[0], plans[r].write_plan_gpu.shape[0])
            for r, _ in RATIOS_OVERLAP
        }
    assert shapes[running_bs] == shapes[1] == shapes[7]


# ── empty fwd ─────────────────────────────────────────────────────────────


def test_empty_fwd_decode_cg_matches_caps_all_sentinel():
    running_bs, qlen = 8, 4
    extend = np.zeros(running_bs, dtype=np.int32)
    context = np.zeros(running_bs, dtype=np.int32)
    plans = make_compress_plans(
        extend,
        context,
        RATIOS_OVERLAP,
        plan_buffers=_buffers(),
        running_bs=running_bs,
        max_q_len=qlen,
    )
    for ratio, is_overlap in RATIOS_OVERLAP:
        p = plans[ratio]
        assert p.num_write == 0 and p.num_compress == 0
        assert p.write_plan_gpu.shape[0] == running_bs * min(
            qlen, _k_pool(ratio, is_overlap)
        )
        assert (p.write_plan_gpu[:, POSITION_COL] == -1).all()


def test_empty_fwd_eager_compress_zero_write_full():
    extend = np.zeros(4, dtype=np.int32)
    context = np.zeros(4, dtype=np.int32)
    bufs = _buffers(write_rows=50)
    plans = make_compress_plans(extend, context, RATIOS_OVERLAP, plan_buffers=bufs)
    for ratio, _ in RATIOS_OVERLAP:
        assert plans[ratio].compress_plan_gpu.shape[0] == 0
        assert plans[ratio].write_plan_gpu.shape[0] == 50


# ── extend-shaped verify path (explicit compress cap) ─────────────────────


def test_verify_explicit_compress_cap_write_full_buffer():
    extend, context = _uniform_decode(bs=3, qlen=2)
    bufs = _buffers(compress_rows=64, write_rows=64)
    cap = {4: 40, 128: 40}
    plans = make_compress_plans(
        extend,
        context,
        RATIOS_OVERLAP,
        plan_buffers=bufs,
        decode_capacity_per_ratio=cap,
    )
    for ratio, _ in RATIOS_OVERLAP:
        p = plans[ratio]
        assert p.compress_plan_gpu.shape[0] == cap[ratio]  # explicit fixed cap
        assert p.write_plan_gpu.shape[0] == 64  # full buffer (legacy)


# ── per-ratio row content ─────────────────────────────────────────────────
#
# Every test above checks a COUNT. These check what is in the rows, which is
# what a shared row block would get wrong: the four columns are built once per
# call and only `window_len` is rewritten per ratio, so a plan published for one
# ratio must still carry that ratio's K after a later ratio has been built.


def _window_len(j_in_seq, k_pool):
    return max(0, k_pool - min(j_in_seq + 1, k_pool))


def _rows_carry_own_k(rows, extend, ratio, is_overlap):
    """Every row's `window_len` recomputed from its own (ragged_id, batch_id)."""
    cu = np.concatenate([[0], np.cumsum(extend)]).astype(np.int64)
    k_pool = _k_pool(ratio, is_overlap)
    for ragged_id, batch_id, _pos, wlen in rows:
        if ragged_id < 0:  # sentinel padding row
            continue
        j = int(ragged_id) - int(cu[int(batch_id)])
        assert wlen == _window_len(j, k_pool), (
            f"ratio={ratio}: row(ragged={ragged_id}, batch={batch_id}, j={j}) "
            f"has window_len={wlen}, expected {_window_len(j, k_pool)} for "
            f"K_pool={k_pool}"
        )


def test_each_ratio_publishes_its_own_window_len():
    # Ragged extend lens and a context long enough that both ratios have live
    # write rows; qlen > 1 so `j_in_seq` actually varies within a sequence.
    extend = np.array([6, 3, 5, 1], dtype=np.int32)
    context = np.array([600, 40, 271, 9], dtype=np.int32)
    bufs = _buffers(compress_rows=512, write_rows=512)
    plans = make_compress_plans(extend, context, RATIOS_OVERLAP, plan_buffers=bufs)
    checked = 0
    for ratio, is_overlap in RATIOS_OVERLAP:
        p = plans[ratio]
        if p.compress_plan_cpu is not None:
            _rows_carry_own_k(p.compress_plan_cpu, extend, ratio, is_overlap)
            checked += len(p.compress_plan_cpu)
        write_rows = bufs[ratio]["write"].np[: p.num_write]
        _rows_carry_own_k(write_rows, extend, ratio, is_overlap)
        checked += p.num_write
    # The two ratios must disagree somewhere, or the assertions above are vacuous.
    assert checked > 0
    assert _window_len(0, _k_pool(4, True)) != _window_len(0, _k_pool(128, False))


def test_graph_bs_and_decode_cap_mutually_exclusive():
    extend, context = _uniform_decode(bs=2, qlen=2)
    with pytest.raises(AssertionError):
        make_compress_plans(
            extend,
            context,
            RATIOS_OVERLAP,
            plan_buffers=_buffers(),
            running_bs=4,
            max_q_len=2,
            decode_capacity_per_ratio={4: 8, 128: 8},
        )
