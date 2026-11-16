# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contract of the DeepSeek-V4.1 radix (prefix) cache on the SGLang plugin.

Replay first: a prefix hit at ``P`` recomputes ``[P - R, P)`` into its freshly
reset STATE slot (``deepseek_v41_radix``), which the bridge translates into
ATOM's ``ScheduledBatch.replaying`` -- fused into the step's own forward (the
hit's span starts at ``P - R``), or in one side forward before it (A/B).
Covered here:

1. the bridge's replay branch and the main forward's stamp guard;
2. ATOM's ``_replay_layout`` over a plugin-shaped replay batch;
3. compress plans over ``[start, P)`` with ``write_from = P``;
4. ``PagedAttentionCache``: replay then main, no replay, coincidental cursor;
5. the seam: which rows replay, the one side batch, capacity splits;
6. the admission charge;
7. the refusal matrix, and that nothing is installed with radix off;
8. the fused replay: the seam's record, the widened inputs and the output
   rows, the bridge's widened spans, the runtime hook, ATOM's cache over a
   fused span, and the host-row invalidation that lets overlap run.
"""

import sys
from array import array
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.plugin.sglang import deepseek_v41_bridge as bridge
from atom.plugin.sglang import deepseek_v41_radix as radix
from atom.plugin.sglang.deepseek_v41_bridge import (
    ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE,
    build_v41_scheduled_batch,
    refuse_unsupported_sglang_features,
    sglang_radix_cache_enabled,
)
from atom.plugin.sglang.deepseek_v41_radix import (
    FUSED_ATTR,
    FusedReplay,
    ReplayHit,
    carry_fused_replay,
    check_state_stamps,
    expand_v41_forward_inputs,
    fused_layout,
    plan_prefix_replay,
    replay_admission_rows,
    resolve_replay_mode,
    resolve_replay_rows,
    run_v41_prefix_replay,
    run_with_fused_replay,
    split_replay_groups,
)

PAGE = ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE
R = 128
HISTORY = 3


class _Mode:
    """The ForwardMode surface the seam and the bridge read."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def is_decode(self):
        return self.name == "DECODE"

    def is_idle(self):
        return self.name == "IDLE"

    def is_context_parallel_extend(self, include_draft_extend_v2=False):
        return self.name in ("EXTEND", "MIXED")

    def is_extend_without_speculative(self):
        return self.name in ("EXTEND", "MIXED")


EXTEND, DECODE, MIXED = _Mode("EXTEND"), _Mode("DECODE"), _Mode("MIXED")


@pytest.fixture(autouse=True)
def _isolated_process_state(monkeypatch):
    # Radix is opt-in; these tests exercise the opted-in launch.
    monkeypatch.setenv("ATOM_V41_RADIX", "1")
    # One builder and one replay length per process; no test may leak either.
    monkeypatch.setattr(bridge, "_V41_BUILDER", None)
    monkeypatch.setattr(radix, "_REPLAY_ROWS", None)
    monkeypatch.setattr(radix, "_REPLAY_MODE", None)
    monkeypatch.setattr(radix, "_TEST_SIDE_STEPS", 0)
    monkeypatch.setattr(radix, "_STATS", dict.fromkeys(radix._STATS, 0))
    monkeypatch.setattr(bridge, "_HOST_ROWS", bridge.HostPageRows())
    monkeypatch.setattr(bridge, "_HOST_ROWS_SUPPORTED", None)
    monkeypatch.setattr(bridge, "_HOST_ROWS_RADIX_SAFE", False)


@pytest.fixture
def radix_on(monkeypatch):
    """The side-forward replay (the A/B arm)."""
    monkeypatch.setattr(radix, "_REPLAY_ROWS", R)
    monkeypatch.setattr(radix, "_REPLAY_MODE", "side")
    return R


@pytest.fixture
def radix_fused(monkeypatch):
    """The fused replay (the default)."""
    monkeypatch.setattr(radix, "_REPLAY_ROWS", R)
    monkeypatch.setattr(radix, "_REPLAY_MODE", "fused")
    return R


def _req_to_token(rows, context_len, pages_by_row):
    table = torch.zeros(rows, context_len, dtype=torch.int32)
    for row, pages in pages_by_row.items():
        for i, page in enumerate(pages):
            start = i * PAGE
            width = min(PAGE, context_len - start)
            table[row, start : start + width] = page * PAGE + torch.arange(width)
    return table


@pytest.fixture
def req_to_token(monkeypatch):
    table = _req_to_token(
        rows=6,
        context_len=4 * PAGE,
        pages_by_row={1: [7, 3, 9], 2: [5, 6], 3: [2, 11, 4, 8], 4: [10, 12]},
    )
    monkeypatch.setattr(bridge, "_req_to_token", lambda: table)
    return table


def _replay_fb(starts, ends, slots, history, mode=EXTEND, spec_info=None):
    bs = len(slots)
    return SimpleNamespace(
        forward_mode=mode,
        rids=[f"r{i}" for i in range(bs)],
        batch_size=bs,
        seq_lens=torch.tensor(ends, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(ends, dtype=torch.int64),
        req_pool_indices=torch.tensor(slots, dtype=torch.int64),
        req_pool_indices_cpu=None,
        extend_seq_lens_cpu=[e - s for s, e in zip(starts, ends)],
        extend_prefix_lens_cpu=list(starts),
        encoder_swa_replay=True,
        engram_history=history,
        spec_info=spec_info,
    )


# ---------------------------------------------------------------------------
# 1. Bridge: the replay branch and the stamp guard
# ---------------------------------------------------------------------------


class TestReplayBranch:
    def test_side_forward_maps_onto_replay_records(self, req_to_token):
        from atom.model_engine.sequence import Replay

        history = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.int32)
        batch = build_v41_scheduled_batch(
            _replay_fb([PAGE - R, 2 * PAGE - R], [PAGE, 2 * PAGE], [1, 2], history),
            PAGE,
            history=HISTORY,
        )
        assert batch.replaying == {
            0: Replay(until=PAGE, rebuilds=True, history_ids=(5, 6, 7), start=128),
            1: Replay(until=2 * PAGE, rebuilds=True, history_ids=(8, 9, 10), start=384),
        }
        assert batch.num_scheduled_tokens.tolist() == [R, R]
        assert batch.context_lens.tolist() == [PAGE, 2 * PAGE]
        assert batch.state_slots_committed == [1, 2]
        assert [row.tolist() for row in batch.block_tables] == [[7], [5, 6]]

    @pytest.mark.parametrize(("start", "kept"), [(0, ()), (2, (6, 7)), (4, (5, 6, 7))])
    def test_history_keeps_only_the_tokens_before_the_start(
        self, req_to_token, start, kept
    ):
        # SGLang left-pads a short history; native ATOM carries what exists.
        history = torch.tensor([[5, 6, 7]], dtype=torch.int32)
        batch = build_v41_scheduled_batch(
            _replay_fb([start], [PAGE], [1], history), PAGE, history=HISTORY
        )
        assert batch.replaying[0].history_ids == kept
        assert batch.replaying[0].start == start

    @pytest.mark.parametrize(
        ("starts", "ends", "match"),
        [
            ([129], [PAGE], "start even"),
            ([128], [PAGE + 2], "PAGE boundary"),
        ],
    )
    def test_misaligned_spans_are_refused(self, req_to_token, starts, ends, match):
        history = torch.zeros(1, HISTORY, dtype=torch.int32)
        with pytest.raises(ValueError, match=match):
            build_v41_scheduled_batch(
                _replay_fb(starts, ends, [1], history), PAGE, history=HISTORY
            )

    @pytest.mark.parametrize(
        "history",
        [
            None,
            torch.zeros(1, 2, dtype=torch.int32),
            torch.zeros(2, HISTORY, dtype=torch.int32),
            torch.zeros(HISTORY, dtype=torch.int32),
        ],
    )
    def test_missing_or_misshaped_history_is_refused(self, req_to_token, history):
        with pytest.raises(ValueError, match="engram_history"):
            build_v41_scheduled_batch(
                _replay_fb([128], [PAGE], [1], history), PAGE, history=HISTORY
            )

    def test_spec_info_is_refused(self, req_to_token):
        fb = _replay_fb(
            [128],
            [PAGE],
            [1],
            torch.zeros(1, HISTORY, dtype=torch.int32),
            spec_info=object(),
        )
        with pytest.raises(NotImplementedError, match="spec_info"):
            build_v41_scheduled_batch(fb, PAGE, history=HISTORY)

    def test_non_extend_modes_are_refused(self, req_to_token):
        fb = _replay_fb(
            [128], [PAGE], [1], torch.zeros(1, HISTORY, dtype=torch.int32), MIXED
        )
        with pytest.raises(NotImplementedError, match="EXTEND"):
            build_v41_scheduled_batch(fb, PAGE, history=HISTORY)
        fb.forward_mode = DECODE
        fb.extend_prefix_lens_cpu = [PAGE - 1]
        fb.extend_seq_lens_cpu = [1]
        with pytest.raises(NotImplementedError, match="EXTEND"):
            build_v41_scheduled_batch(fb, PAGE, history=HISTORY)


def _extend_fb(prefix, lengths, slots):
    fb = _replay_fb(prefix, [p + n for p, n in zip(prefix, lengths)], slots, None)
    fb.encoder_swa_replay = False
    return fb


class TestStampGuard:
    def test_radix_off_reads_no_stamps(self, monkeypatch, req_to_token):
        def boom():
            raise AssertionError("stamps read with radix off")

        monkeypatch.setattr(bridge, "_radix_stamps", boom)
        batch = build_v41_scheduled_batch(_extend_fb([PAGE], [3], [1]), PAGE)
        assert batch.replaying == {}

    def test_resuming_row_needs_its_tenant_stamp(
        self, monkeypatch, req_to_token, radix_on
    ):
        state_gen = np.array([0, 4, 7, 0, 0, 0], dtype=np.int64)
        generation = torch.tensor([0, 4, 8, 0, 0, 0], dtype=torch.int64)
        monkeypatch.setattr(bridge, "_radix_stamps", lambda: (state_gen, generation))
        # Slot 1 is stamped; slot 2 at P = 0 needs no stamp.
        build_v41_scheduled_batch(_extend_fb([PAGE, 0], [3, 5], [1, 2]), PAGE)
        # Slot 2 resuming above 0 on a stale stamp: the seam was skipped.
        with pytest.raises(RuntimeError, match=r"slots \[2\]"):
            build_v41_scheduled_batch(_extend_fb([PAGE, PAGE], [3, 5], [1, 2]), PAGE)

    def test_decode_is_not_checked(self, monkeypatch, req_to_token, radix_on):
        monkeypatch.setattr(
            bridge, "_radix_stamps", lambda: pytest.fail("decode read stamps")
        )
        fb = SimpleNamespace(
            forward_mode=DECODE,
            rids=["a"],
            batch_size=1,
            seq_lens=torch.tensor([PAGE + 1]),
            seq_lens_cpu=torch.tensor([PAGE + 1]),
            req_pool_indices=torch.tensor([1]),
            req_pool_indices_cpu=None,
        )
        build_v41_scheduled_batch(fb, PAGE)

    def test_check_state_stamps(self):
        state_gen = np.array([0, 3, 5])
        generation = torch.tensor([0, 3, 6])
        check_state_stamps(state_gen, generation, [1, 2], np.array([PAGE, 0]))
        with pytest.raises(RuntimeError, match="not rebuilt"):
            check_state_stamps(state_gen, generation, [1, 2], np.array([PAGE, PAGE]))


# ---------------------------------------------------------------------------
# 2. ATOM's replay layout over a plugin-shaped batch
# ---------------------------------------------------------------------------


def _spans(batch):
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan

    spans, offset = [], 0
    for rid, length, end, slot in zip(
        batch.req_ids,
        batch.num_scheduled_tokens,
        batch.context_lens,
        batch.state_slots_committed,
    ):
        spans.append(RequestSpan(rid, int(end - length), offset, int(length), slot))
        offset += int(length)
    return spans


def test_replay_layout_over_a_plugin_replay_batch(req_to_token):
    pytest.importorskip("aiter")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("aiter's arch probe needs a visible GPU")
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from tests.model_ops.engram.test_hash_bounds import build, tiny_config

    mapping = build(tiny_config())
    width = mapping.config.max_ngram_size - 1
    history = torch.tensor([[11, 12], [13, 14]], dtype=torch.int32)
    batch = build_v41_scheduled_batch(
        _replay_fb([PAGE - R, 2 * PAGE - R], [PAGE, 2 * PAGE], [1, 2], history),
        PAGE,
        history=width,
    )
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.engram = SimpleNamespace(mapping=mapping)
    builder.geometry = SimpleNamespace(history_size=width)
    builder.config = SimpleNamespace(image_token_id=None)
    resumes, floors, cached, histories = builder._replay_layout(
        batch.replaying, _spans(batch)
    )
    # Nothing global written below P; the window floored at the start; the
    # slot rebuilt (no state came along); Engram as a prefill leaves it.
    assert resumes.tolist() == [PAGE, 2 * PAGE]
    assert floors.tolist() == [PAGE - R, 2 * PAGE - R]
    assert cached.tolist() == [0, 0]
    for row, ids in zip(histories, ([11, 12], [13, 14])):
        expected = mapping.advance_history(
            None, mapping.compress_tokens(np.asarray(ids, np.int64))
        )[0]
        assert row.tolist() == np.asarray(expected).tolist()


# ---------------------------------------------------------------------------
# 3. Compress plans of a replay forward write no group below P
# ---------------------------------------------------------------------------


def _small_geometry():
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )

    return V41PoolGeometry(
        4, ((0, 2), (2, 1)), PAGE, 128, 512, 32, layer_ratios=(0, 1, 2), index_topk=4
    )


@pytest.mark.parametrize("prefix", [PAGE, 3 * PAGE])
def test_replay_compress_plans_write_no_group_below_the_hit(prefix):
    from atom.model_ops.v4_kernels import make_compress_plans
    from atom.utils import CpuGpuBuffer

    geo = _small_geometry()
    lengths = np.asarray([R], dtype=np.int32)
    ends = np.asarray([prefix], dtype=np.int32)

    def plans(write_from):
        buffers = {
            ratio: {
                name: CpuGpuBuffer(
                    R + 1, 4, dtype=torch.int32, device="cpu", pin_memory=False
                )
                for name in ("compress", "write")
            }
            for ratio, _ in geo.compress_ratios
        }
        return make_compress_plans(
            lengths,
            ends,
            geo.compress_ratios,
            plan_buffers=buffers,
            extra_write=0,
            write_from=write_from,
        )

    assert {ratio for ratio, _ in geo.compress_ratios} == {1, 2}
    replayed = plans(np.asarray([prefix], dtype=np.int32))
    cold = plans(None)
    for ratio, plan in replayed.items():
        assert plan.num_compress == 0, ratio
        # The compressor ring is still refilled for the main extend at P.
        assert plan.num_write > 0, ratio
        assert cold[ratio].num_compress > 0, ratio


# ---------------------------------------------------------------------------
# 4. PagedAttentionCache: replay then main; no replay; coincidental cursor
# ---------------------------------------------------------------------------


def _tiny_cache(slots=3, kv_dtype="bf16"):
    pytest.importorskip("aiter")
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )

    # A window of 8 rows, PAGEs of 32 (the CPU floor of an FP8 index plane).
    geo = V41PoolGeometry(
        2,
        ((0, 1),),
        32,
        8,
        512,
        32,
        kv_dtype=kv_dtype,
        layer_ratios=(0, 1),
        index_topk=4,
    )
    return PagedAttentionCache(geo, 16, slots, "cpu")


def _span(rid, position, length, slot):
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan

    return RequestSpan(rid, position, 0, length, slot)


class TestCacheContract:
    P, START = 64, 32

    def _dirty(self, cache, slot, cursor):
        # The last tenant's leftovers.
        cache.state_bytes[slot].fill_(77)
        cache.cursor[slot] = torch.tensor(cursor, dtype=torch.int64)

    def _main(self, cache, slot):
        step = cache.begin_step(
            [_span(9, self.P, 5, slot)], block_tables=[np.arange(1, 4, dtype=np.int32)]
        )
        return cache.prepare_state(step)

    def test_replay_then_main(self):
        cache = _tiny_cache()
        self._dirty(cache, 1, [1000, 1, 2, 3])
        step = cache.begin_step(
            [_span(9, self.START, self.P - self.START, 1)],
            block_tables=[np.arange(1, 3, dtype=np.int32)],
            floors=np.asarray([self.START], np.int32),
            cached_until=np.zeros(1, np.int32),
            start_histories=np.asarray([[4, 5, 6]], np.int64),
        )
        histories = cache.prepare_state(step)
        # A fresh slot: zeroed, its history seeded from before the start.
        assert histories.tolist() == [[4, 5, 6]]
        assert cache.state.view("window")[:, 1].count_nonzero() == 0
        cache.advance_cursor(step, histories)
        assert cache.cursor[1].tolist() == [self.P, 4, 5, 6]
        assert self._main(cache, 1).tolist() == [[4, 5, 6]]

    def test_no_replay_raises(self):
        cache = _tiny_cache()
        self._dirty(cache, 1, [1000, 1, 2, 3])
        with pytest.raises(ValueError, match=f"needs state at {self.P}"):
            self._main(cache, 1)

    def test_coincidental_cursor_is_caught_by_the_stamp_guard(self):
        # The last tenant stopped exactly at P: ATOM's cursor check passes on
        # its state. Only the per-slot stamp tells the tenants apart.
        cache = _tiny_cache()
        self._dirty(cache, 1, [self.P, 1, 2, 3])
        self._main(cache, 1)
        state_gen = np.array([0, 3, 0])
        generation = torch.tensor([0, 4, 0])
        with pytest.raises(RuntimeError, match="not rebuilt"):
            check_state_stamps(state_gen, generation, [1], np.array([self.P]))


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_page_digests_see_main_and_index_writes(kv_dtype):
    from atom.plugin.sglang.deepseek_v41_radix import page_digests

    cache = _tiny_cache(kv_dtype=kv_dtype)
    pages = torch.tensor([1, 2, 3])
    before = page_digests(cache, pages)
    assert torch.equal(page_digests(cache, pages), before)
    # page 2's main rows: the flat PAGE (BF16) or the main source's data
    # plane (FP8, its own buffer)
    main = next(
        view for name, view in cache.page_regions if name in ("page", "main_0.data")
    )
    main.view(int(cache.num_pages), -1)[2, 5] = 1
    plane = next(iter(cache.index_planes.values()))[0]
    plane[3, 0, 1] = 9  # page 3's index rows
    plane[5, 0, 0] = 9  # a page outside the set
    changed = (page_digests(cache, pages) != before).any(dim=1)
    assert changed.tolist() == [False, True, True]


# ---------------------------------------------------------------------------
# 5. The seam
# ---------------------------------------------------------------------------


def _req(tokens, return_logprob=False, logprob_start_len=-1, rid=None):
    return SimpleNamespace(
        rid=rid,
        full_untruncated_fill_ids=array("q", tokens),
        return_logprob=return_logprob,
        logprob_start_len=logprob_start_len,
    )


class _World:
    """A worker, its runner and pools, a builder, and a recorder of side
    forwards in place of ``ForwardBatch.init_new`` + ``runner.forward``."""

    def __init__(self, monkeypatch, *, rows=8, max_bs=8, max_tokens=4096):
        self.table = torch.arange(rows * 4 * PAGE, dtype=torch.int32).view(
            rows, 4 * PAGE
        )
        self.generation = torch.ones(rows, dtype=torch.int64)
        self.pool = SimpleNamespace(
            is_atom_v41_proxy_pool=True, state_gen=np.zeros(rows, dtype=np.int64)
        )
        self.runner = SimpleNamespace(
            device="cpu",
            req_to_token_pool=SimpleNamespace(
                req_to_token=self.table, req_generation=self.generation
            ),
            token_to_kv_pool=self.pool,
        )
        self.worker = SimpleNamespace(model_runner=self.runner)
        self.sides = []
        monkeypatch.setattr(
            bridge,
            "_V41_BUILDER",
            SimpleNamespace(
                block_size=PAGE,
                max_bs=max_bs,
                max_num_batched_tokens=max_tokens,
                geometry=SimpleNamespace(history_size=HISTORY),
                cache=None,
            ),
        )
        monkeypatch.setattr(
            radix, "_run_side_forward", lambda runner, side: self.sides.append(side)
        )

    def batch(self, prefix, slots, reqs=None, mode=EXTEND, lengths=None):
        reqs = reqs or [
            _req(range(1000, 1000 + p + 300), rid=f"q{i}") for i, p in enumerate(prefix)
        ]
        lengths = list(lengths or [7] * len(prefix))
        return SimpleNamespace(
            forward_mode=mode,
            reqs=reqs,
            prefix_lens=list(prefix),
            extend_lens=lengths,
            extend_num_tokens=sum(lengths),
            req_pool_indices=torch.tensor(slots),
            req_pool_indices_cpu=torch.tensor(slots),
            sampling_info="main",
            return_logprob=True,
            spec_info=None,
            engram_history=None,
            is_prefill_only=False,
        )

    def stamp(self, slots):
        self.pool.state_gen[slots] = self.generation[slots].numpy()


class TestSeam:
    def test_new_tenant_with_a_hit_replays(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        req = _req(range(1000, 1000 + 2 * PAGE + 7))
        fused = run_v41_prefix_replay(world.worker, world.batch([2 * PAGE], [3], [req]))
        assert fused is None and radix._STATS["rows"] == R
        (side,) = world.sides
        start, end = 2 * PAGE - R, 2 * PAGE
        assert side.input_ids.tolist() == list(range(1000 + start, 1000 + end))
        assert side.reqs == [req]
        assert side.prefix_lens == [start] and side.extend_lens == [R]
        assert side.extend_num_tokens == R
        assert side.seq_lens.tolist() == [end] and side.seq_lens_cpu.tolist() == [end]
        assert side.seq_lens_sum == end
        assert side.req_pool_indices.tolist() == [3]
        assert side.req_pool_indices_cpu.tolist() == [3]
        assert side.out_cache_loc.tolist() == world.table[3, start:end].tolist()
        assert side.out_cache_loc.dtype == torch.int64
        assert side.engram_history.tolist() == [
            [1000 + start - 3, 1000 + start - 2, 1000 + start - 1]
        ]
        assert side.engram_history.dtype == torch.int32
        assert side.sampling_info is None and side.spec_info is None
        assert side.return_logprob is False and side.is_prefill_only is True
        assert side.multimodal_inputs == [None]
        # The main batch is untouched and the slot stamped for its tenant.
        assert world.pool.state_gen[3] == world.generation[3]

    def test_chunk_continuation_does_not_replay(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        world.stamp([3])
        assert run_v41_prefix_replay(world.worker, world.batch([2 * PAGE], [3])) is None
        assert world.sides == []

    def test_retracted_readmission_replays(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        world.stamp([3])
        world.generation[3] += 1  # alloc_rows: a new tenant of row 3
        run_v41_prefix_replay(world.worker, world.batch([PAGE], [3]))
        assert len(world.sides) == 1
        assert world.pool.state_gen[3] == world.generation[3]

    def test_cold_row_is_stamped_without_a_replay(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        assert run_v41_prefix_replay(world.worker, world.batch([0], [2])) is None
        assert world.sides == []
        assert world.pool.state_gen[2] == world.generation[2]

    def test_every_hit_of_a_step_shares_one_side_forward(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        world.stamp([4])
        reqs = [_req(range(k * 10000, k * 10000 + 3 * PAGE)) for k in range(4)]
        batch = world.batch([PAGE, 0, 2 * PAGE, PAGE], [1, 2, 3, 4], reqs)
        assert run_v41_prefix_replay(world.worker, batch) is None
        assert radix._STATS["rows"] == 2 * R
        (side,) = world.sides
        # Rows 0 and 2 hit; row 1 is cold, row 3 continues its chunk.
        assert side.reqs == [reqs[0], reqs[2]]
        assert side.prefix_lens == [PAGE - R, 2 * PAGE - R]
        assert side.seq_lens.tolist() == [PAGE, 2 * PAGE]
        assert side.input_ids.tolist() == (
            list(range(PAGE - R, PAGE))
            + list(range(20000 + 2 * PAGE - R, 20000 + 2 * PAGE))
        )
        assert side.out_cache_loc.tolist() == (
            world.table[1, PAGE - R : PAGE].tolist()
            + world.table[3, 2 * PAGE - R : 2 * PAGE].tolist()
        )
        assert side.engram_history.tolist() == [
            [PAGE - R - 3, PAGE - R - 2, PAGE - R - 1],
            [20000 + 2 * PAGE - R - 3 + i for i in range(3)],
        ]
        assert world.pool.state_gen[[1, 2, 3, 4]].tolist() == [1, 1, 1, 1]

    def test_hits_past_the_builder_capacity_split(self, monkeypatch, radix_on):
        world = _World(monkeypatch, max_bs=2, max_tokens=4096)
        run_v41_prefix_replay(world.worker, world.batch([PAGE] * 3, [1, 2, 3]))
        assert [side.req_pool_indices.tolist() for side in world.sides] == [
            [1, 2],
            [3],
        ]
        world = _World(monkeypatch, max_bs=8, max_tokens=R)
        run_v41_prefix_replay(world.worker, world.batch([PAGE] * 2, [1, 2]))
        assert len(world.sides) == 2

    def test_short_hit_replays_from_zero(self, monkeypatch):
        monkeypatch.setattr(radix, "_REPLAY_ROWS", 512)
        world = _World(monkeypatch)
        req = _req(range(50, 50 + PAGE + 3))
        run_v41_prefix_replay(world.worker, world.batch([PAGE], [1], [req]))
        (side,) = world.sides
        assert side.prefix_lens == [0] and side.extend_lens == [PAGE]
        assert side.engram_history.tolist() == [[0, 0, 0]]

    def test_unaligned_hit_is_refused(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        with pytest.raises(ValueError, match="page boundary"):
            run_v41_prefix_replay(world.worker, world.batch([PAGE + 2], [1]))

    def test_cached_prompt_logprobs_are_refused(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        req = _req(range(2 * PAGE), return_logprob=True, logprob_start_len=10)
        with pytest.raises(ValueError, match="prompt logprobs"):
            run_v41_prefix_replay(world.worker, world.batch([PAGE], [1], [req]))

    def test_mixed_batch_is_refused(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        with pytest.raises(NotImplementedError, match="mixed-chunk"):
            run_v41_prefix_replay(world.worker, world.batch([PAGE], [1], mode=MIXED))

    @pytest.mark.parametrize("case", ["decode", "radix off", "other pool", "none"])
    def test_inert_cases_touch_nothing(self, monkeypatch, case):
        world = _World(monkeypatch)
        batch = world.batch([PAGE], [1])
        if case == "decode":
            monkeypatch.setattr(radix, "_REPLAY_ROWS", R)
            batch.forward_mode = DECODE
        elif case == "other pool":
            monkeypatch.setattr(radix, "_REPLAY_ROWS", R)
            world.pool.is_atom_v41_proxy_pool = False
        elif case == "none":
            monkeypatch.setattr(radix, "_REPLAY_ROWS", R)
            batch = None
        assert run_v41_prefix_replay(world.worker, batch) is None
        assert world.sides == []
        assert not world.pool.state_gen.any()

    def test_plan_and_split_are_pure(self):
        hits = plan_prefix_replay(
            [PAGE, 0, 2 * PAGE], [1, 2, 3], [5, 5, 5], np.array([0, 0, 0, 5]), R, PAGE
        )
        assert hits == [ReplayHit(0, 1, PAGE - R, PAGE)]
        many = [ReplayHit(i, i + 1, 0, R) for i in range(5)]
        assert [len(g) for g in split_replay_groups(many, 2, 10 * R)] == [2, 2, 1]
        assert [len(g) for g in split_replay_groups(many, 8, 3 * R)] == [3, 2]
        with pytest.raises(ValueError, match="exceeds"):
            split_replay_groups(many, 8, R - 2)


def _sglang_module(name):
    """SGLang's ``name``, or a skip where it cannot import (some SGLang
    modules probe the GPU at import)."""
    pytest.importorskip("sglang")
    import importlib

    try:
        return importlib.import_module(name)
    except RuntimeError as exc:
        pytest.skip(f"{name} does not import here: {exc}")


def test_worker_seam_runs_before_the_forward(monkeypatch, radix_on):
    TpModelWorker = _sglang_module("sglang.srt.managers.tp_worker").TpModelWorker

    calls = []

    def original(self, batch, forward_batch=None, **kwargs):
        calls.append(("forward", batch, forward_batch))
        return "result"

    monkeypatch.setattr(TpModelWorker, "forward_batch_generation", original)
    monkeypatch.setattr(
        radix,
        "run_v41_prefix_replay",
        lambda worker, batch, checks=None: calls.append("replay"),
    )
    radix._install_worker_seam()
    radix._install_worker_seam()  # idempotent
    patched = TpModelWorker.forward_batch_generation
    assert patched._atom_v41_radix_patched
    assert patched(object(), "batch") == "result"
    assert calls == ["replay", ("forward", "batch", None)]
    calls.clear()
    patched(object(), batch=None, forward_batch="fb")
    assert calls == [("forward", None, "fb")]


# ---------------------------------------------------------------------------
# 6. Admission accounting
# ---------------------------------------------------------------------------


def _admitted(pool_idx=None, retracted=False):
    return SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=pool_idx), is_retracted=retracted
    )


@pytest.mark.parametrize(
    ("req", "prefix", "charge"),
    [
        (_admitted(), 2 * PAGE, R),
        (_admitted(), PAGE, R),
        (_admitted(), 0, 0),
        (_admitted(pool_idx=3), 2 * PAGE, 0),  # a chunk continuation's row
        (_admitted(pool_idx=3, retracted=True), 2 * PAGE, R),
    ],
)
def test_replay_admission_rows(req, prefix, charge):
    assert replay_admission_rows(req, prefix, R) == charge
    assert replay_admission_rows(req, prefix, 512) == (
        min(prefix, 512) if charge else 0
    )


def test_admission_charge_wraps_sglang(monkeypatch):
    PrefillAdder = _sglang_module("sglang.srt.managers.schedule_policy").PrefillAdder

    seen = []
    monkeypatch.setattr(
        PrefillAdder,
        "_account_prefill_cache_admission",
        lambda self, req, prefix_len: seen.append(prefix_len),
    )
    radix._install_admission_accounting()
    radix._install_admission_accounting()  # idempotent
    account = PrefillAdder._account_prefill_cache_admission
    adder = SimpleNamespace(
        log_replay_tokens=0, rem_input_tokens=1000, rem_chunk_tokens=500
    )
    account(adder, _admitted(), 2 * PAGE)  # radix off: no charge
    assert (adder.log_replay_tokens, adder.rem_input_tokens) == (0, 1000)
    monkeypatch.setattr(radix, "_REPLAY_ROWS", R)
    account(adder, _admitted(), 2 * PAGE)
    account(adder, _admitted(pool_idx=2), 2 * PAGE)
    assert adder.log_replay_tokens == R
    assert (adder.rem_input_tokens, adder.rem_chunk_tokens) == (1000 - R, 500 - R)
    adder.rem_chunk_tokens = None
    account(adder, _admitted(retracted=True), PAGE)
    assert adder.rem_input_tokens == 1000 - 2 * R
    assert seen == [2 * PAGE] * 3 + [PAGE]


# ---------------------------------------------------------------------------
# 7. Refusal matrix, and nothing installed with radix off
# ---------------------------------------------------------------------------


def _radix_bags(**overrides):
    bags = {
        "memory": SimpleNamespace(
            disable_radix_cache=False,
            enable_hierarchical_cache=False,
            enable_hisparse=False,
            enable_session_radix_cache=False,
            enable_unified_memory=False,
            radix_cache_backend=None,
            enable_lmcache=False,
            enable_flexkv=False,
            enable_unified_cache_external_linker=False,
        ),
        "exec": SimpleNamespace(
            features=SimpleNamespace(enable_encoder_swa_bounded_replay=False),
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(backend="disabled")
                )
            ),
        ),
        "spec": SimpleNamespace(speculative_algorithm=None),
        "disagg": SimpleNamespace(disaggregation_mode="null", language_model_only=True),
        "schedule": SimpleNamespace(
            chunked_prefill_size=8192,
            disable_overlap_schedule=True,
            enable_mixed_chunk=False,
        ),
    }
    bags.update(overrides)
    return bags


@pytest.fixture
def launch(monkeypatch):
    import atom.plugin.config as plugin_config

    state = SimpleNamespace(
        bags=_radix_bags(),
        args=SimpleNamespace(
            enable_dp_attention=False,
            pp_size=1,
            enable_two_batch_overlap=False,
            page_size=PAGE,
        ),
    )
    monkeypatch.setattr(bridge, "_sglang_bag", lambda name: state.bags.get(name))
    monkeypatch.setattr(plugin_config, "get_sglang_server_args", lambda: state.args)
    monkeypatch.delenv(radix.REPLAY_ROWS_ENV, raising=False)
    monkeypatch.delenv(radix.REPLAY_MODE_ENV, raising=False)
    return state


class TestRefusals:
    def test_supported_radix_launch_passes(self, launch):
        assert sglang_radix_cache_enabled()
        refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("bag", "path", "value", "match"),
        [
            ("schedule", "enable_mixed_chunk", True, "mixed chunk"),
            ("memory", "enable_session_radix_cache", True, "session radix cache"),
            ("memory", "radix_cache_backend", "custom", "custom radix cache"),
            ("memory", "enable_unified_memory", True, "unified memory"),
            ("memory", "enable_lmcache", True, "KV storage backend"),
            ("memory", "enable_flexkv", True, "KV storage backend"),
            (
                "memory",
                "enable_unified_cache_external_linker",
                True,
                "KV storage backend",
            ),
            ("schedule", "chunked_prefill_size", 64, "--chunked-prefill-size 64"),
            ("spec", "speculative_algorithm", "DSPARK", "speculative decoding with"),
        ],
    )
    def test_radix_refusals(self, launch, bag, path, value, match):
        setattr(launch.bags[bag], path, value)
        with pytest.raises(ValueError, match=match):
            refuse_unsupported_sglang_features()

    @pytest.mark.parametrize("backend", ["full", "breakable"])
    def test_prefill_graphs_are_refused(self, launch, backend):
        launch.bags["exec"].graph.cuda_graph_config.prefill.backend = backend
        with pytest.raises(ValueError, match="prefill CUDA graphs"):
            refuse_unsupported_sglang_features()

    def test_overlap_scheduling_runs_with_radix(self, launch):
        # The seam runs inside forward_batch_generation under overlap too, and
        # a radix checkpoint drops the slot's host rows (section 8).
        launch.bags["schedule"].disable_overlap_schedule = False
        refuse_unsupported_sglang_features()

    def test_raw_args_are_the_fallback(self, launch):
        launch.bags = {"disagg": launch.bags["disagg"]}
        launch.args.disable_radix_cache = False
        launch.args.cuda_graph_backend_prefill = "full"
        launch.args.enable_lmcache = True
        with pytest.raises(ValueError, match="storage backend.*prefill CUDA graphs"):
            refuse_unsupported_sglang_features()
        launch.args.enable_lmcache = False
        launch.args.cuda_graph_backend_prefill = "disabled"
        refuse_unsupported_sglang_features()

    @pytest.mark.parametrize("value", ["", "both", "SIDEWAYS"])
    def test_invalid_replay_mode_is_refused(self, launch, monkeypatch, value):
        monkeypatch.setenv(radix.REPLAY_MODE_ENV, value)
        with pytest.raises(ValueError, match=radix.REPLAY_MODE_ENV):
            refuse_unsupported_sglang_features()

    def test_replay_mode(self, monkeypatch):
        monkeypatch.delenv(radix.REPLAY_MODE_ENV, raising=False)
        assert resolve_replay_mode() == "fused"
        monkeypatch.setenv(radix.REPLAY_MODE_ENV, " Side ")
        assert resolve_replay_mode() == "side"

    def test_disabled_chunking_is_not_below_r(self, launch):
        launch.bags["schedule"].chunked_prefill_size = -1
        refuse_unsupported_sglang_features()

    @pytest.mark.parametrize("value", ["0", "127", "-128"])
    def test_invalid_replay_rows_are_refused(self, launch, monkeypatch, value):
        monkeypatch.setenv(radix.REPLAY_ROWS_ENV, value)
        with pytest.raises(ValueError, match=radix.REPLAY_ROWS_ENV):
            refuse_unsupported_sglang_features()

    def test_replay_rows_must_cover_the_window(self, monkeypatch):
        monkeypatch.setenv(radix.REPLAY_ROWS_ENV, "64")
        assert resolve_replay_rows() == 64
        with pytest.raises(ValueError, match="sliding window"):
            resolve_replay_rows(window=128)
        monkeypatch.setenv(radix.REPLAY_ROWS_ENV, "256")
        assert resolve_replay_rows(window=128) == 256
        monkeypatch.delenv(radix.REPLAY_ROWS_ENV)
        assert resolve_replay_rows(window=128) == radix.DEFAULT_REPLAY_ROWS

    def test_radix_off_ignores_radix_only_refusals(self, launch):
        launch.bags["memory"].disable_radix_cache = True
        launch.bags["schedule"].enable_mixed_chunk = True
        launch.bags["schedule"].disable_overlap_schedule = False
        launch.bags["schedule"].chunked_prefill_size = 64
        assert not sglang_radix_cache_enabled()
        refuse_unsupported_sglang_features()

    def test_encoder_replay_flag_stays_refused(self, launch):
        launch.bags["exec"].features.enable_encoder_swa_bounded_replay = True
        with pytest.raises(ValueError, match="encoder SWA bounded replay"):
            refuse_unsupported_sglang_features()


def _v41_atom_config():
    return SimpleNamespace(
        max_num_batched_tokens=16384,
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
        kv_cache_block_size=128,
        enable_decoder_swa_bounded_replay=True,
        enable_encoder_swa_bounded_replay=True,
        online_quant_config={"layer_quant_config": {}},
        quant_config=None,
        hf_config=SimpleNamespace(sliding_window=128),
        kv_cache_dtype="bf16",
        index_cache_dtype="fp8",
    )


@pytest.mark.parametrize("mode", ["fused", "side"])
@pytest.mark.parametrize("decoder_replay", ["1", "0"])
@pytest.mark.parametrize("radix_cache", [False, True])
def test_prepare_installs_the_replay_only_with_radix(
    launch, monkeypatch, radix_cache, decoder_replay, mode
):
    from atom.plugin.sglang.runtime import model_arch

    # Decoder bounded replay is independent of radix cache: on unless
    # ATOM_SGLANG_V41_DECODER_REPLAY=0, with radix on or off.
    monkeypatch.setenv("ATOM_SGLANG_V41_DECODER_REPLAY", decoder_replay)
    installed = []
    monkeypatch.setattr(
        radix,
        "install_deepseek_v41_radix",
        lambda window=None: installed.append(window) or (R, mode),
    )
    launch.bags["memory"].disable_radix_cache = not radix_cache
    launch.bags["schedule"] = None  # leave the builder sizing alone
    launch.args.disable_overlap_schedule = True
    config = _v41_atom_config()
    model_arch._prepare_deepseek_v41_config(config, "DeepseekV41ForCausalLM")
    assert config.enable_encoder_swa_bounded_replay is radix_cache
    assert config.enable_decoder_swa_bounded_replay is (decoder_replay == "1")
    assert installed == ([128] if radix_cache else [])
    assert config.kv_cache_block_size == PAGE
    # A fused step runs its hits' replay rows on top of its chunk.
    fused = radix_cache and mode == "fused"
    assert config.max_num_batched_tokens == 16384 + (R if fused else 0)


def test_side_forward_runs_every_decoder_row(monkeypatch, radix_on):
    """The side forward is prefill-only, so under decoder bounded replay it
    computes the decoder over all of ``[P - R, P)``: the ring rows below the
    hit that a main extend shorter than the window then reads."""
    world = _World(monkeypatch)
    run_v41_prefix_replay(world.worker, world.batch([2 * PAGE], [3]))
    (side,) = world.sides
    fb = SimpleNamespace(is_prefill_only=side.is_prefill_only)
    lengths = np.asarray(side.extend_lens, np.int32)
    scheduled = SimpleNamespace(
        num_scheduled_tokens=lengths,
        context_lens=np.asarray(side.prefix_lens, np.int64) + lengths,
    )
    assert not bridge.v41_extend_replays(fb, scheduled, 128)
    # The same rows as an ordinary extend would replay its decoder tail.
    fb.is_prefill_only = False
    assert bridge.v41_extend_replays(fb, scheduled, 128)


def test_proxy_pool_owns_a_stamp_per_request_slot(monkeypatch):
    pytest.importorskip("sglang")
    from tests.plugin.test_sglang_deepseek_v41_bridge import (
        _SGLANG_MODULES,
        _install_builder,
        _proxy,
    )
    from tests.plugin.test_sglang_deepseek_v41_bridge import (
        _small_geometry as bridge_geometry,
    )

    if not _SGLANG_MODULES:
        pytest.skip("SGLang is not importable")
    for name, module in _SGLANG_MODULES.items():
        monkeypatch.setitem(sys.modules, name, module)
    _install_builder(monkeypatch, bridge_geometry())
    pool = _proxy()
    assert pool.state_gen.dtype == np.int64
    assert pool.state_gen.tolist() == [0] * pool.num_req_slots
    with pytest.raises(NotImplementedError, match="relocation"):
        pool.move_kv_cache(torch.zeros(1), torch.zeros(1))


# ---------------------------------------------------------------------------
# 8. The fused replay: the rows below each hit ride in the step's own forward
# ---------------------------------------------------------------------------


class TestFusedSeam:
    def test_hit_rides_in_the_step(self, monkeypatch, radix_fused):
        world = _World(monkeypatch)
        req = _req(range(1000, 1000 + 2 * PAGE + 7), rid="a")
        fused = run_v41_prefix_replay(world.worker, world.batch([2 * PAGE], [3], [req]))
        # No side forward: the rows ride in the step's forward.
        assert world.sides == []
        start, end = 2 * PAGE - R, 2 * PAGE
        assert fused.rows.tolist() == [0] and fused.slots.tolist() == [3]
        assert fused.rids == ["a"]
        assert fused.starts.tolist() == [start] and fused.ends.tolist() == [end]
        assert fused.ids.tolist() == list(range(1000 + start, 1000 + end))
        assert fused.positions().tolist() == list(range(start, end))
        assert fused.histories.tolist() == [
            [1000 + start - 3, 1000 + start - 2, 1000 + start - 1]
        ]
        assert fused.expanded is None
        # Not stamped by the seam: the bridge stamps it when the forward
        # consumes the record, so a lost record fails the guard first.
        assert world.pool.state_gen[3] != world.generation[3]
        assert radix._STATS["fused_steps"] == 1 and radix._STATS["rows"] == R

    def test_only_new_tenants_above_zero_ride(self, monkeypatch, radix_fused):
        world = _World(monkeypatch)
        world.stamp([4])
        reqs = [
            _req(range(k * 10000, k * 10000 + 3 * PAGE), rid=f"r{k}") for k in range(4)
        ]
        batch = world.batch([PAGE, 0, 2 * PAGE, PAGE], [1, 2, 3, 4], reqs)
        fused = run_v41_prefix_replay(world.worker, batch)
        # Row 1 is cold, row 3 continues its chunk.
        assert fused.rows.tolist() == [0, 2]
        assert fused.rids == ["r0", "r2"]
        assert fused.replay_rows(4).tolist() == [R, 0, R, 0]
        assert fused.ids.tolist() == (
            list(range(PAGE - R, PAGE))
            + list(range(20000 + 2 * PAGE - R, 20000 + 2 * PAGE))
        )
        assert world.sides == []
        # The cold row and the continuation are stamped; the hits wait for
        # their forward to consume the record.
        assert world.pool.state_gen[[1, 2, 3, 4]].tolist() == [0, 1, 0, 1]

    @pytest.mark.parametrize("prefix,stamped", [([0], False), ([PAGE], True)])
    def test_no_hit_carries_nothing(self, monkeypatch, radix_fused, prefix, stamped):
        world = _World(monkeypatch)
        if stamped:
            world.stamp([1])
        assert run_v41_prefix_replay(world.worker, world.batch(prefix, [1])) is None
        assert world.sides == [] and radix._STATS["fused_steps"] == 0

    def test_a_step_past_the_builder_falls_back_to_a_side_forward(
        self, monkeypatch, radix_fused
    ):
        world = _World(monkeypatch, max_tokens=R + 6)
        batch = world.batch([PAGE], [1], lengths=[7])
        assert run_v41_prefix_replay(world.worker, batch) is None
        (side,) = world.sides
        assert side.prefix_lens == [PAGE - R]
        assert radix._STATS["fallbacks"] == 1 and radix._STATS["fused_steps"] == 0
        assert world.pool.state_gen[1] == world.generation[1]

    def test_short_hit_replays_from_zero(self, monkeypatch):
        monkeypatch.setattr(radix, "_REPLAY_ROWS", 512)
        monkeypatch.setattr(radix, "_REPLAY_MODE", "fused")
        world = _World(monkeypatch)
        req = _req(range(50, 50 + PAGE + 3), rid="z")
        fused = run_v41_prefix_replay(world.worker, world.batch([PAGE], [1], [req]))
        assert fused.starts.tolist() == [0] and fused.ends.tolist() == [PAGE]
        assert fused.histories.tolist() == [[0, 0, 0]]

    def test_cached_prompt_logprobs_are_refused(self, monkeypatch, radix_fused):
        world = _World(monkeypatch)
        req = _req(range(2 * PAGE), True, 10, rid="x")
        with pytest.raises(ValueError, match="prompt logprobs"):
            run_v41_prefix_replay(world.worker, world.batch([PAGE], [1], [req]))


def _fused(rows, slots, rids, starts, ends, history=((1, 2, 3),), ids=None):
    if ids is None:
        ids = [
            1000 * (k + 1) + t
            for k, (s, e) in enumerate(zip(starts, ends))
            for t in range(e - s)
        ]
    histories = np.asarray(history, np.int64)
    if histories.shape[0] != len(rows):
        histories = np.repeat(histories[:1], len(rows), axis=0)
    return FusedReplay(rows, slots, rids, starts, ends, ids, histories)


class TestFusedRecord:
    def test_record_shape_is_checked(self):
        with pytest.raises(ValueError, match="ascend"):
            _fused([2, 1], [1, 2], ["a", "b"], [0, 0], [4, 4])
        with pytest.raises(ValueError, match="non-empty"):
            _fused([0], [1], ["a"], [4], [4])
        with pytest.raises(ValueError, match="ids for"):
            _fused([0], [1], ["a"], [0], [4], ids=[1, 2, 3])
        with pytest.raises(ValueError, match="disagree"):
            FusedReplay([0], [1, 2], ["a"], [0], [4], [1, 2, 3, 4], [[1, 2, 3]])
        with pytest.raises(ValueError, match="outside a batch"):
            _fused([3], [1], ["a"], [0], [4]).replay_rows(2)

    def test_layout_puts_each_replay_before_its_request(self):
        keep, replay_at = fused_layout([3, 2, 4], [0, 5, 2])
        # req0: 0..2; req1: replay 3..7, own 8..9; req2: replay 10..11, own 12..15
        assert keep.tolist() == [0, 1, 2, 8, 9, 12, 13, 14, 15]
        assert replay_at.tolist() == [3, 4, 5, 6, 7, 10, 11]
        keep, replay_at = fused_layout([4], [0])
        assert keep.tolist() == [0, 1, 2, 3] and replay_at.tolist() == []


def _fused_fb(lengths, fused, mode=EXTEND):
    fb = SimpleNamespace(
        forward_mode=mode,
        batch_size=len(lengths),
        extend_seq_lens_cpu=list(lengths),
    )
    setattr(fb, FUSED_ATTR, fused)
    return fb


class TestFusedInputs:
    def _case(self):
        fused = _fused([1], [4], ["b"], [6], [10], ids=[60, 70, 80, 90])
        fb = _fused_fb([3, 2], fused)
        ids = torch.tensor([11, 12, 13, 21, 22])
        positions = torch.tensor([0, 1, 2, 10, 11])
        return fused, fb, ids, positions

    def test_rows_below_the_hit_are_prepended(self):
        fused, fb, ids, positions = self._case()
        wide_ids, wide_pos, rows = expand_v41_forward_inputs(fb, ids, positions)
        assert wide_ids.tolist() == [11, 12, 13, 60, 70, 80, 90, 21, 22]
        assert wide_pos.tolist() == [0, 1, 2, 6, 7, 8, 9, 10, 11]
        assert wide_ids.dtype == ids.dtype and wide_pos.dtype == positions.dtype
        # SGLang's rows of the widened output, in order.
        assert rows.tolist() == [0, 1, 2, 7, 8]
        hidden = torch.arange(9)[:, None].expand(9, 4)
        assert hidden.index_select(0, rows)[:, 0].tolist() == [0, 1, 2, 7, 8]
        # Kept on the record for the bridge; one expansion per forward.
        assert fused.expanded.input_ids is wide_ids
        assert expand_v41_forward_inputs(fb, ids, positions) is fused.expanded

    def test_other_forwards_are_not_widened(self):
        fb = _fused_fb([3], None)
        assert expand_v41_forward_inputs(fb, torch.zeros(3), torch.zeros(3)) is None

    def test_mismatched_rows_are_refused(self):
        _, fb, ids, positions = self._case()
        with pytest.raises(ValueError, match="extend rows"):
            expand_v41_forward_inputs(fb, ids[:4], positions[:4])

    def test_embeddings_and_mrope_are_refused(self):
        _, fb, ids, positions = self._case()
        with pytest.raises(NotImplementedError, match="input_ids"):
            expand_v41_forward_inputs(fb, ids, positions, torch.zeros(5, 2))
        with pytest.raises(NotImplementedError, match="1-D positions"):
            expand_v41_forward_inputs(fb, ids, positions.expand(3, 5))

    def test_the_v41_adapter_hook(self):
        from atom.plugin.sglang.runtime import model_arch

        spec = model_arch.MODEL_ADAPTER_SPECS[bridge.DEEPSEEK_V41_ARCH]
        _, fb, ids, positions = self._case()
        assert spec.expand_forward_inputs(fb, ids, positions, None)[0].numel() == 9
        plain = _fused_fb([5], None)
        assert spec.expand_forward_inputs(plain, ids, positions, None) is None


class TestFusedBridge:
    def _fb(self):
        fb = _extend_fb([2 * PAGE, 0, PAGE], [5, 9, 300], [1, 2, 3])
        fused = _fused(
            [0, 2],
            [1, 3],
            ["r0", "r2"],
            [2 * PAGE - R, PAGE - R],
            [2 * PAGE, PAGE],
            history=[[1, 2, 3], [4, 5, 6]],
        )
        setattr(fb, FUSED_ATTR, fused)
        return fb, fused

    def test_hits_are_scheduled_from_below_the_hit(self, req_to_token):
        from atom.model_engine.sequence import Replay

        fb, _ = self._fb()
        batch = build_v41_scheduled_batch(fb, PAGE, history=HISTORY)
        assert batch.num_scheduled_tokens.tolist() == [5 + R, 9, 300 + R]
        assert batch.context_lens.tolist() == [2 * PAGE + 5, 9, PAGE + 300]
        assert batch.replay_rows.tolist() == [R, 0, R]
        assert batch.total_tokens_num == 5 + 9 + 300 + 2 * R
        assert [span.position for span in _spans(batch)] == [2 * PAGE - R, 0, PAGE - R]
        assert batch.replaying == {
            0: Replay(2 * PAGE, True, (1, 2, 3), 2 * PAGE - R),
            2: Replay(PAGE, True, (4, 5, 6), PAGE - R),
        }
        # The block tables cover the request, prefix pages included.
        assert [row.tolist() for row in batch.block_tables] == [
            [7, 3, 9],
            [5],
            [2, 11, 4],
        ]

    def test_history_keeps_only_real_tokens(self, req_to_token):
        fb = _extend_fb([PAGE], [3], [1])
        setattr(fb, FUSED_ATTR, _fused([0], [1], ["r0"], [2], [PAGE], [[0, 7, 8]]))
        batch = build_v41_scheduled_batch(fb, PAGE, history=HISTORY)
        assert batch.replaying[0].history_ids == (7, 8)

    @pytest.mark.parametrize(
        "field,value",
        [("slots", [2, 3]), ("ends", [PAGE, PAGE]), ("rids", ["r0", "rX"])],
    )
    def test_a_record_of_another_forward_is_refused(self, req_to_token, field, value):
        fb, fused = self._fb()
        setattr(fused, field, value if field == "rids" else np.asarray(value, np.int64))
        with pytest.raises(ValueError, match="does not describe this forward"):
            build_v41_scheduled_batch(fb, PAGE, history=HISTORY)

    def test_spans_must_be_page_aligned_and_history_sized(self, req_to_token):
        fb = _extend_fb([PAGE + 2], [3], [1])
        setattr(fb, FUSED_ATTR, _fused([0], [1], ["r0"], [2], [PAGE + 2]))
        with pytest.raises(ValueError, match="PAGE boundary"):
            build_v41_scheduled_batch(fb, PAGE, history=HISTORY)
        fb, _ = self._fb()
        with pytest.raises(ValueError, match="history width"):
            build_v41_scheduled_batch(fb, PAGE, history=HISTORY + 1)

    def test_side_forwards_and_decode_carry_none(self, req_to_token):
        fb, _ = self._fb()
        fb.encoder_swa_replay = True
        with pytest.raises(ValueError, match="side forward carries a fused"):
            build_v41_scheduled_batch(fb, PAGE, history=HISTORY)
        decode = SimpleNamespace(
            forward_mode=DECODE,
            rids=["a"],
            batch_size=1,
            seq_lens=torch.tensor([PAGE + 1]),
            seq_lens_cpu=torch.tensor([PAGE + 1]),
            req_pool_indices=torch.tensor([1]),
            req_pool_indices_cpu=None,
        )
        setattr(decode, FUSED_ATTR, _fused([0], [1], ["a"], [PAGE - R], [PAGE]))
        with pytest.raises(NotImplementedError, match="rides on an EXTEND"):
            build_v41_scheduled_batch(decode, PAGE)

    def test_the_stamp_guard_still_reads_sglang_prefix(
        self, monkeypatch, req_to_token, radix_fused
    ):
        seen, stamped = [], []
        monkeypatch.setattr(
            radix,
            "check_state_stamps",
            lambda state_gen, generation, slots, prefix, carried: seen.append(
                (prefix.tolist(), carried.tolist())
            ),
        )
        monkeypatch.setattr(
            radix,
            "stamp_fused_hits",
            lambda state_gen, generation, fused: stamped.append(fused.slots.tolist()),
        )
        monkeypatch.setattr(bridge, "_radix_stamps", lambda: (None, None))
        fb, _ = self._fb()
        build_v41_scheduled_batch(fb, PAGE, history=HISTORY)
        # SGLang's prefix, and the slots the carried record rebuilds.
        assert seen == [([2 * PAGE, 0, PAGE], [1, 3])]
        assert stamped == [[1, 3]]

    def test_a_short_suffix_still_replays_its_decoder(self, req_to_token):
        """The fused span is at least R >= window rows, so even a hit whose
        own suffix is shorter than the window runs the decoder on its tail,
        as native ATOM's hit forward from ``P - 128`` does."""
        fb, _ = self._fb()
        fb.is_prefill_only = False
        fb.return_logprob = False
        batch = build_v41_scheduled_batch(fb, PAGE, history=HISTORY)
        assert bridge.v41_extend_replays(fb, batch, 128)
        # Without the fused rows, the 5-row hit could not.
        setattr(fb, FUSED_ATTR, None)
        plain = build_v41_scheduled_batch(fb, PAGE, history=HISTORY)
        assert not bridge.v41_extend_replays(fb, plain, 128)

    @pytest.mark.parametrize(
        "starts,replays", [([5, 9, 300], True), ([4, 9, 300], False)]
    )
    def test_input_logprobs_count_sglang_rows_only(self, req_to_token, starts, replays):
        fb, _ = self._fb()
        fb.return_logprob = True
        fb.extend_logprob_start_lens_cpu = starts
        batch = build_v41_scheduled_batch(fb, PAGE, history=HISTORY)
        assert bridge.v41_extend_replays(fb, batch, 128) is replays


class TestFusedMetadata:
    def _builder(self, monkeypatch, captured):
        def prepare_prefill(batch, running_bs, running_tokens):
            captured.update(batch=batch, bs=running_bs, tokens=running_tokens)
            return SimpleNamespace(engram_embeddings=None), None

        builder = SimpleNamespace(
            cache=object(),
            block_size=PAGE,
            geometry=SimpleNamespace(history_size=HISTORY, window_size=128),
            max_bs=8,
            max_num_batched_tokens=4096,
            _decoder_replays=True,
            prepare_prefill=prepare_prefill,
            prepare_model_inputs=lambda ids, md: captured.update(ids=ids),
        )
        monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
        return builder

    def _fb(self):
        fb = _extend_fb([PAGE, 0], [3, 2], [1, 2])
        fb.input_ids = torch.tensor([11, 12, 13, 21, 22])
        fb.is_prefill_only = False
        fb.return_logprob = False
        fused = _fused([0], [1], ["r0"], [PAGE - R], [PAGE])
        setattr(fb, FUSED_ATTR, fused)
        return fb, fused

    def test_atom_stages_the_widened_inputs(self, monkeypatch, req_to_token):
        captured = {}
        self._builder(monkeypatch, captured)
        fb, _ = self._fb()
        positions = torch.tensor([PAGE, PAGE + 1, PAGE + 2, 0, 1])
        wide_ids, wide_pos, _ = expand_v41_forward_inputs(fb, fb.input_ids, positions)
        metadata = bridge.build_deepseek_v41_forward_metadata(fb, wide_pos)
        assert metadata.staged_outside_forward
        assert captured["ids"] is wide_ids
        assert captured["tokens"] == 5 + R and captured["bs"] == 2
        assert captured["batch"].num_scheduled_tokens.tolist() == [3 + R, 2]
        assert captured["batch"].decoder_replays

    def test_unexpanded_inputs_are_refused(self, monkeypatch, req_to_token):
        self._builder(monkeypatch, {})
        fb, _ = self._fb()
        with pytest.raises(RuntimeError, match="widened inputs"):
            bridge.build_deepseek_v41_forward_metadata(fb, torch.zeros(5 + R))


class TestRuntimeExpansion:
    def _runtime(self, monkeypatch, expand, mode=EXTEND):
        from atom.plugin.sglang.runtime import model_arch
        from atom.plugin.sglang.runtime.forward_context import SGLangPluginRuntime

        spec = model_arch.SGLangModelAdapterSpec(expand_forward_inputs=expand)
        monkeypatch.setattr(
            model_arch, "resolve_model_arch_spec", lambda config: ("X", spec)
        )
        return SGLangPluginRuntime(
            atom_config=SimpleNamespace(hf_config=None),
            forward_batch=SimpleNamespace(forward_mode=mode),
            positions=torch.tensor([0, 1, 2]),
            input_ids=torch.tensor([5, 6, 7]),
            set_forward_context=False,
        )

    def test_widened_inputs_and_sglang_rows_out(self, monkeypatch):
        seen = []

        def expand(fb, ids, positions, embeds):
            seen.append((ids.tolist(), positions.tolist(), embeds))
            return (
                torch.tensor([9, 9, 5, 6, 7]),
                torch.tensor([8, 9, 10, 11, 12]),
                torch.tensor([2, 3, 4]),
            )

        with self._runtime(monkeypatch, expand) as runtime:
            assert runtime.input_ids.tolist() == [9, 9, 5, 6, 7]
            assert runtime.positions.tolist() == [8, 9, 10, 11, 12]
            hidden = torch.arange(10).view(5, 2)
            assert runtime.trim_output(hidden).tolist() == [[4, 5], [6, 7], [8, 9]]
            pair = runtime.trim_output((hidden, None))
            assert pair[0].shape == (3, 2) and pair[1] is None
            with pytest.raises(TypeError, match="widened forward"):
                runtime.trim_output([hidden])
        assert seen == [([5, 6, 7], [0, 1, 2], None)]

    def test_forwards_the_hook_declines_are_untouched(self, monkeypatch):
        with self._runtime(monkeypatch, lambda *args: None) as runtime:
            assert runtime.input_ids.tolist() == [5, 6, 7]
            hidden = torch.ones(3, 2)
            assert runtime.trim_output(hidden) is hidden

    def test_dummy_forwards_are_not_expanded(self, monkeypatch):
        idle = _Mode("IDLE")

        def expand(*args):
            raise AssertionError("an IDLE forward was expanded")

        with self._runtime(monkeypatch, expand, mode=idle) as runtime:
            assert runtime.trim_output(torch.ones(1, 2)).shape == (0, 2)


class TestFusedCarrier:
    def test_init_new_carries_the_record_onto_an_extend(self):
        fused = _fused([0], [1], ["a"], [0], [4])
        batch = SimpleNamespace()
        setattr(batch, FUSED_ATTR, fused)
        fb = SimpleNamespace(forward_mode=EXTEND)
        carry_fused_replay(batch, fb)
        assert getattr(fb, FUSED_ATTR) is fused
        # A batch with none (the side batch copy, any other step) adds none.
        for none in (SimpleNamespace(), SimpleNamespace(**{FUSED_ATTR: None})):
            fb = SimpleNamespace(forward_mode=EXTEND)
            carry_fused_replay(none, fb)
            assert getattr(fb, FUSED_ATTR, None) is None
        with pytest.raises(NotImplementedError, match="rides on an EXTEND"):
            carry_fused_replay(batch, SimpleNamespace(forward_mode=DECODE))

    def test_the_record_lives_for_one_call(self):
        fused = _fused([0], [1], ["a"], [0], [4])
        batch = SimpleNamespace()
        seen = []

        def original(worker, batch, *args, **kwargs):
            seen.append((getattr(batch, FUSED_ATTR), args, kwargs))
            getattr(batch, FUSED_ATTR).expanded = "widened"  # the model ran it
            return "result"

        out = run_with_fused_replay(original, "w", batch, fused, (1,), {"k": 2})
        assert out == "result" and seen == [(fused, (1,), {"k": 2})]
        assert getattr(batch, FUSED_ATTR) is None

        def failing(worker, batch, *args, **kwargs):
            raise RuntimeError("forward failed")

        with pytest.raises(RuntimeError, match="forward failed"):
            run_with_fused_replay(failing, "w", batch, fused, (), {})
        assert getattr(batch, FUSED_ATTR) is None

    def test_a_forward_that_never_ran_the_replay_raises(self):
        # The second line behind the bridge's guard: a call that returned
        # without the model consuming the record.
        fused = _fused([0], [7], ["a"], [0], [4])
        batch = SimpleNamespace()
        with pytest.raises(RuntimeError, match=r"slots \[7\] never reached"):
            run_with_fused_replay(lambda *a, **k: "result", "w", batch, fused, (), {})
        assert getattr(batch, FUSED_ATTR) is None

    def test_the_eager_runners_view_carries_the_record(self, monkeypatch):
        import dataclasses
        import types

        @dataclasses.dataclass
        class ForwardBatch:
            forward_mode: object
            input_ids: object = None

        class EagerRunner:
            def load_batch(self, forward_batch, pp_proxy_tensors=None, **kwargs):
                # SGLang's view over its static buffers: fields only.
                return dataclasses.replace(forward_batch, input_ids="static")

        module = types.ModuleType("sglang.srt.model_executor.runner.eager_runner")
        module.EagerRunner = EagerRunner
        monkeypatch.setitem(
            sys.modules, "sglang.srt.model_executor.runner.eager_runner", module
        )
        radix._install_eager_view_carrier()
        radix._install_eager_view_carrier()  # idempotent
        fused = _fused([0], [1], ["a"], [0], [4])
        fb = ForwardBatch(EXTEND)
        setattr(fb, FUSED_ATTR, fused)
        view = EagerRunner().load_batch(fb)
        assert view is not fb and view.input_ids == "static"
        assert getattr(view, FUSED_ATTR) is fused
        # Every other forward's view stays without one.
        other = EagerRunner().load_batch(ForwardBatch(EXTEND))
        assert getattr(other, FUSED_ATTR, None) is None
        same = ForwardBatch(EXTEND)
        radix.carry_fused_view(same, same)
        assert getattr(same, FUSED_ATTR, None) is None

    def test_the_side_batch_carries_none(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        batch = world.batch([PAGE], [1])
        setattr(batch, FUSED_ATTR, _fused([0], [1], ["q0"], [0], [4]))
        run_v41_prefix_replay(world.worker, batch)
        (side,) = world.sides
        assert getattr(side, FUSED_ATTR) is None

    def test_worker_seam_attaches_the_record_around_the_forward(
        self, monkeypatch, radix_fused
    ):
        import types

        calls, reports = [], []

        class TpModelWorker:
            def forward_batch_generation(self, batch=None, *args, **kwargs):
                fused = getattr(batch, FUSED_ATTR, None)
                calls.append(fused)
                if fused is not None:
                    fused.expanded = "widened"  # the model ran it
                return "result"

        module = types.ModuleType("sglang.srt.managers.tp_worker")
        module.TpModelWorker = TpModelWorker
        monkeypatch.setitem(sys.modules, "sglang.srt.managers.tp_worker", module)
        fused = _fused([0], [1], ["a"], [0], [4])

        def plan(worker, batch, checks=None):
            checks.append(SimpleNamespace(report=lambda: reports.append(calls[:])))
            return fused

        monkeypatch.setattr(radix, "run_v41_prefix_replay", plan)
        radix._install_worker_seam()
        radix._install_worker_seam()  # idempotent
        batch = SimpleNamespace()
        assert TpModelWorker().forward_batch_generation(batch) == "result"
        # Set for the call, gone after it; debug checks report after it.
        assert calls == [fused] and getattr(batch, FUSED_ATTR) is None
        assert reports == [[fused]]
        monkeypatch.setattr(radix, "run_v41_prefix_replay", lambda *a: None)
        TpModelWorker().forward_batch_generation(batch)
        assert calls == [fused, None]


class TestFusedCache:
    """ATOM's cache over a step in which one request's span starts below its
    hit (the fused replay) beside a continuing request."""

    P, START = 64, 32

    def test_the_fused_span_resets_its_slot_and_resumes_past_the_hit(self):
        from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan

        cache = _tiny_cache()
        cache.state_bytes[1].fill_(77)
        cache.cursor[1] = torch.tensor([1000, 1, 2, 3])
        cache.state_bytes[2].fill_(55)
        cache.cursor[2] = torch.tensor([40, 7, 8, 9])
        length = self.P + 5 - self.START
        step = cache.begin_step(
            [
                RequestSpan(9, self.START, 0, length, 1),
                RequestSpan(8, 40, length, 6, 2),
            ],
            block_tables=[
                np.arange(1, 4, dtype=np.int32),
                np.arange(4, 6, dtype=np.int32),
            ],
            floors=np.asarray([self.START, 0], np.int32),
            cached_until=np.zeros(2, np.int32),
            start_histories=np.asarray([[4, 5, 6], [-1, -1, -1]], np.int64),
        )
        histories = cache.prepare_state(step)
        # The hit's slot starts over at its floor with its seeded history; the
        # continuation keeps its state and history.
        assert histories.tolist() == [[4, 5, 6], [7, 8, 9]]
        window = cache.state.view("window")
        assert window[:, 1].count_nonzero() == 0
        assert bool((window[:, 2] != 0).all())
        cache.advance_cursor(step, histories)
        assert cache.cursor[1, 0] == self.P + 5 and cache.cursor[2, 0] == 46


@pytest.mark.parametrize("prefix,suffix", [(PAGE, 6), (3 * PAGE, 200)])
def test_fused_compress_plans_write_only_from_the_hit(prefix, suffix):
    """One span over ``[P - R, P + suffix)`` with ``write_from = P``: every
    compressed group from the hit on is written, none below it."""
    from atom.model_ops.v4_kernels import make_compress_plans
    from atom.utils import CpuGpuBuffer

    geo = _small_geometry()
    lengths = np.asarray([R + suffix], dtype=np.int32)
    ends = np.asarray([prefix + suffix], dtype=np.int32)

    def plans(lengths, ends, write_from):
        width = int(lengths.max()) + 1
        buffers = {
            ratio: {
                name: CpuGpuBuffer(
                    width, 4, dtype=torch.int32, device="cpu", pin_memory=False
                )
                for name in ("compress", "write")
            }
            for ratio, _ in geo.compress_ratios
        }
        return make_compress_plans(
            lengths,
            ends,
            geo.compress_ratios,
            plan_buffers=buffers,
            extra_write=0,
            write_from=write_from,
        )

    fused = plans(lengths, ends, np.asarray([prefix], dtype=np.int32))
    # What an extend resuming at P (the side-forward path's main forward)
    # compresses.
    resumed = plans(
        np.asarray([suffix], dtype=np.int32), ends, np.zeros(1, dtype=np.int32)
    )
    for ratio, plan in fused.items():
        assert plan.num_compress == resumed[ratio].num_compress == suffix // ratio


class TestHostRowsUnderRadix:
    def _fake_tree(self, monkeypatch):
        import types

        class UnifiedRadixCache:
            def checkpoint_into_tree(self, req, *, up_to, **kwargs):
                if up_to < 0:
                    raise RuntimeError("insert failed")
                return up_to

        module = types.ModuleType("sglang.srt.mem_cache.unified_radix_cache")
        module.UnifiedRadixCache = UnifiedRadixCache
        monkeypatch.setitem(
            sys.modules, "sglang.srt.mem_cache.unified_radix_cache", module
        )
        return UnifiedRadixCache

    def test_a_checkpoint_drops_its_slots_host_rows(self, monkeypatch, launch):
        tree_cls = self._fake_tree(monkeypatch)
        rows = [np.asarray([7, 3], np.int32), np.asarray([5], np.int32)]
        bridge._HOST_ROWS.record([3, 4], ["a", "b"], rows)
        # Radix on: host rows are off until every rewrite drops its copy.
        assert not bridge.host_page_rows_supported()
        radix._install_checkpoint_invalidation()
        radix._install_checkpoint_invalidation()  # idempotent
        assert bridge.host_page_rows_supported()
        req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=3))
        assert tree_cls().checkpoint_into_tree(req, up_to=512) == 512
        assert bridge._HOST_ROWS.lookup([3], ["a"], [1]) is None
        assert bridge._HOST_ROWS.lookup([4], ["b"], [1])[0].tolist() == [5]
        # A failed insert may have rewritten part of the row: dropped too.
        req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=4))
        with pytest.raises(RuntimeError, match="insert failed"):
            tree_cls().checkpoint_into_tree(req, up_to=-1)
        assert bridge._HOST_ROWS.lookup([4], ["b"], [1]) is None

    def test_radix_off_keeps_host_rows(self, launch):
        launch.bags["memory"].disable_radix_cache = True
        assert bridge.host_page_rows_supported()


def test_replay_layout_over_a_fused_step(req_to_token):
    pytest.importorskip("aiter")
    if not torch.cuda.is_available():
        pytest.skip("aiter's arch probe needs a visible GPU")
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from tests.model_ops.engram.test_hash_bounds import build, tiny_config

    mapping = build(tiny_config())
    width = mapping.config.max_ngram_size - 1
    fb = _extend_fb([2 * PAGE, 0], [5, 9], [1, 2])
    setattr(
        fb,
        FUSED_ATTR,
        _fused([0], [1], ["r0"], [2 * PAGE - R], [2 * PAGE], [[11, 12, 13][-width:]]),
    )
    batch = build_v41_scheduled_batch(fb, PAGE, history=width)
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.engram = SimpleNamespace(mapping=mapping)
    builder.geometry = SimpleNamespace(history_size=width)
    builder.config = SimpleNamespace(image_token_id=None)
    resumes, floors, cached, histories = builder._replay_layout(
        batch.replaying, _spans(batch)
    )
    # The hit: nothing global below P, its window floored at P - R, its slot
    # rebuilt; the cold row beside it untouched by the replay.
    assert resumes.tolist() == [2 * PAGE, 0]
    assert floors.tolist() == [2 * PAGE - R, 0]
    assert cached.tolist() == [0, 0]
    expected = mapping.advance_history(
        None, mapping.compress_tokens(np.asarray([11, 12, 13][-width:], np.int64))
    )[0]
    assert histories[0].tolist() == np.asarray(expected).tolist()


# ---------------------------------------------------------------------------
# 9. Risk-audit fixes: token-only admission, the pinned H2D, the stamp guard
#    before compute, the forced fused -> side fallback
# ---------------------------------------------------------------------------


def _text_req(**fields):
    req = SimpleNamespace(
        rid="t",
        input_embeds=None,
        positional_embed_overrides=None,
        multimodal_inputs=None,
        is_retracted=False,
        return_logprob=False,
    )
    for name, value in fields.items():
        setattr(req, name, value)
    return req


class TestTokenOnlyAdmission:
    @pytest.mark.parametrize(
        "field,name",
        [
            ("input_embeds", "input_embeds"),
            ("positional_embed_overrides", "positional_embed_overrides"),
            ("multimodal_inputs", "multimodal inputs"),
        ],
    )
    def test_non_token_requests_are_refused_with_radix_on(
        self, radix_fused, field, name
    ):
        error = radix.radix_request_refusal(_text_req(**{field: [[0.0]]}))
        assert name in error and "--disable-radix-cache" in error
        assert radix.radix_request_refusal(_text_req()) is None

    def test_radix_off_refuses_nothing(self):
        req = _text_req(input_embeds=[[0.0]], positional_embed_overrides="x")
        assert radix.radix_request_refusal(req) is None

    def test_admission_answers_new_requests_only(self, monkeypatch, radix_fused):
        rejected = []
        monkeypatch.setattr(
            radix,
            "reject_request",
            lambda scheduler, req, error: rejected.append((scheduler, req.rid)),
        )
        embeds = _text_req(rid="e", input_embeds=[[0.0]])
        assert not radix.admit_or_reject("sched", embeds, False)
        assert rejected == [("sched", "e")]
        # Text, and anything re-entering after a retraction, is SGLang's.
        assert radix.admit_or_reject("sched", _text_req(), False)
        assert radix.admit_or_reject("sched", embeds, True)
        embeds.is_retracted = True
        assert radix.admit_or_reject("sched", embeds, False)
        assert len(rejected) == 1

    def test_reject_answers_400_without_queueing(self, monkeypatch, radix_fused):
        _sglang_module("sglang.srt.disaggregation.utils")
        streamed, released, aborted = [], [], []
        scheduler = SimpleNamespace(
            output_streamer=SimpleNamespace(
                stream_output=lambda reqs, return_logprob: streamed.append(
                    ([r.rid for r in reqs], return_logprob)
                )
            )
        )
        req = _text_req(
            rid="e",
            multimodal_inputs="mm",
            release_mm_inputs_on_abort=lambda: released.append(True),
            time_stats=SimpleNamespace(
                trace_ctx=SimpleNamespace(abort=lambda abort_info: aborted.append(1))
            ),
        )
        error = radix.radix_request_refusal(req)
        radix.reject_request(scheduler, req, error)
        reason = req.finished_reason
        assert type(reason).__name__ == "FINISH_ABORT"
        assert reason.message == error and int(reason.status_code) == 400
        assert streamed == [(["e"], False)]
        assert released == [True] and aborted == [1]
        assert radix._STATS["refused_requests"] == 1

    def test_the_scheduler_queue_door_is_wrapped(self, monkeypatch, radix_fused):
        Scheduler = _sglang_module("sglang.srt.managers.scheduler").Scheduler

        queued, rejected = [], []

        def original(self, req, is_retracted=False):
            queued.append((req.rid, is_retracted))

        monkeypatch.setattr(Scheduler, "_add_request_to_queue", original)
        monkeypatch.setattr(
            radix,
            "reject_request",
            lambda scheduler, req, error: rejected.append(req.rid),
        )
        radix._install_request_admission()
        radix._install_request_admission()  # idempotent
        door = Scheduler._add_request_to_queue
        assert door._atom_v41_radix_patched
        door("s", _text_req(rid="text"))
        door("s", _text_req(rid="emb", input_embeds=[[0.0]]))
        door("s", _text_req(rid="score", positional_embed_overrides="pe"))
        door("s", _text_req(rid="back", input_embeds=[[0.0]]), is_retracted=True)
        assert queued == [("text", False), ("back", True)]
        assert rejected == ["emb", "score"]
        # Radix off (not installed in this process): every request queues.
        monkeypatch.setattr(radix, "_REPLAY_ROWS", None)
        door("s", _text_req(rid="emb2", input_embeds=[[0.0]]))
        assert queued[-1] == ("emb2", False) and rejected == ["emb", "score"]

    @pytest.mark.parametrize("mode", ["fused", "side"])
    def test_the_replay_never_recomputes_placeholder_ids(self, monkeypatch, mode):
        monkeypatch.setattr(radix, "_REPLAY_ROWS", R)
        monkeypatch.setattr(radix, "_REPLAY_MODE", mode)
        world = _World(monkeypatch)
        req = _req([1] * (PAGE + 44), rid="e")
        req.input_embeds = [[0.0]] * (PAGE + 44)
        with pytest.raises(ValueError, match="admission should have refused"):
            run_v41_prefix_replay(world.worker, world.batch([PAGE], [1], [req]))
        assert world.sides == []

    def test_prepare_installs_admission_only_with_radix(self, launch, monkeypatch):
        """The admission door is part of ``install_deepseek_v41_radix``, which
        ``_prepare_deepseek_v41_config`` calls with radix cache on only
        (``test_prepare_installs_the_replay_only_with_radix``)."""
        installed = []
        for name in (
            "_install_worker_seam",
            "_install_forward_batch_carrier",
            "_install_eager_view_carrier",
            "_install_checkpoint_invalidation",
            "_install_admission_accounting",
            "_install_request_admission",
        ):
            monkeypatch.setattr(radix, name, lambda name=name: installed.append(name))
        monkeypatch.setenv("ATOM_V41_RADIX_TEST_SIDE_STEPS", "2")
        assert radix.install_deepseek_v41_radix(window=128) == (R, "fused")
        assert "_install_request_admission" in installed
        assert radix._TEST_SIDE_STEPS == 2


class TestPinnedStaging:
    def test_the_host_layout_rides_on_the_record(self):
        fused = _fused([1], [4], ["b"], [6], [10], ids=[60, 70, 80, 90])
        fb = _fused_fb([3, 2], fused)
        ids = torch.tensor([11, 12, 13, 21, 22])
        positions = torch.tensor([0, 1, 2, 10, 11])
        expand_v41_forward_inputs(fb, ids, positions)
        # keep (5), replay_at (4), ids (4), positions (4), in that order.
        assert fused.host.tolist() == (
            [0, 1, 2, 7, 8] + [3, 4, 5, 6] + [60, 70, 80, 90] + [6, 7, 8, 9]
        )
        assert not fused.host.is_pinned()  # CPU forward: nothing to pin

    def test_a_device_forward_copies_from_pinned_memory(self, monkeypatch):
        if not torch.cuda.is_available():
            pytest.skip("needs a visible GPU")
        copies = []
        original_to = torch.Tensor.to

        def to(self, *args, **kwargs):
            if self.device.type == "cpu":
                copies.append((self.is_pinned(), kwargs.get("non_blocking")))
            return original_to(self, *args, **kwargs)

        fused = _fused([1], [4], ["b"], [6], [10], ids=[60, 70, 80, 90])
        fb = _fused_fb([3, 2], fused)
        ids = torch.tensor([11, 12, 13, 21, 22], device="cuda")
        positions = torch.tensor([0, 1, 2, 10, 11], device="cuda")
        monkeypatch.setattr(torch.Tensor, "to", to)
        wide_ids, wide_pos, rows = expand_v41_forward_inputs(fb, ids, positions)
        monkeypatch.undo()
        assert copies == [(True, True)]
        assert fused.host.is_pinned()
        assert wide_ids.tolist() == [11, 12, 13, 60, 70, 80, 90, 21, 22]
        assert wide_pos.tolist() == [0, 1, 2, 6, 7, 8, 9, 10, 11]
        assert rows.tolist() == [0, 1, 2, 7, 8]


class TestStampOnConsume:
    """The seam leaves a fused hit unstamped; the bridge's guard accepts it
    only on a forward carrying its record, and stamps it there."""

    def _seam(self, monkeypatch):
        world = _World(monkeypatch)
        req = _req(range(1000, 1000 + 2 * PAGE + 7), rid="a")
        batch = world.batch([2 * PAGE, 0], [3, 2], [req, _req(range(9), rid="c")])
        fused = run_v41_prefix_replay(world.worker, batch)
        monkeypatch.setattr(
            bridge,
            "_radix_stamps",
            lambda: (world.pool.state_gen, world.generation),
        )
        table = _req_to_token(
            rows=8, context_len=4 * PAGE, pages_by_row={3: [7, 3, 9], 2: [5]}
        )
        monkeypatch.setattr(bridge, "_req_to_token", lambda: table)
        fb = _extend_fb([2 * PAGE, 0], [7, 9], [3, 2])
        fb.rids = ["a", "c"]
        return world, fused, fb

    def test_a_forward_that_lost_the_record_fails_before_compute(
        self, monkeypatch, radix_fused
    ):
        world, fused, fb = self._seam(monkeypatch)
        assert fused is not None and world.pool.state_gen[3] == 0
        # The ForwardBatch reached the bridge without the record.
        with pytest.raises(RuntimeError, match=r"slots \[3\].*not rebuilt"):
            build_v41_scheduled_batch(fb, PAGE, history=HISTORY)
        assert world.pool.state_gen[3] == 0

    def test_the_forward_carrying_the_record_stamps_its_hits(
        self, monkeypatch, radix_fused
    ):
        world, fused, fb = self._seam(monkeypatch)
        setattr(fb, FUSED_ATTR, fused)
        ids = torch.arange(16)
        expand_v41_forward_inputs(fb, ids, torch.arange(16))
        batch = build_v41_scheduled_batch(fb, PAGE, history=HISTORY)
        assert batch.replay_rows.tolist() == [R, 0]
        assert world.pool.state_gen[3] == world.generation[3]
        # The next chunk of the same tenant is a plain continuation.
        setattr(fb, FUSED_ATTR, None)
        build_v41_scheduled_batch(fb, PAGE, history=HISTORY)

    def test_an_unexpanded_record_stamps_nothing(self, monkeypatch, radix_fused):
        world, fused, fb = self._seam(monkeypatch)
        setattr(fb, FUSED_ATTR, fused)
        with pytest.raises(RuntimeError, match="widened inputs"):
            build_v41_scheduled_batch(fb, PAGE, history=HISTORY)
        assert world.pool.state_gen[3] == 0

    def test_carried_slots_pass_the_guard(self):
        state_gen = np.array([0, 3, 5])
        generation = torch.tensor([0, 3, 6])
        prefix = np.array([PAGE, PAGE])
        with pytest.raises(RuntimeError, match=r"slots \[2\]"):
            check_state_stamps(state_gen, generation, [1, 2], prefix)
        check_state_stamps(state_gen, generation, [1, 2], prefix, np.array([2]))
        with pytest.raises(RuntimeError, match=r"slots \[2\]"):
            check_state_stamps(state_gen, generation, [1, 2], prefix, np.array([1]))


class TestForcedSideFallback:
    def test_the_first_fused_steps_take_the_side_forward(
        self, monkeypatch, radix_fused
    ):
        monkeypatch.setattr(radix, "_TEST_SIDE_STEPS", 1)
        world = _World(monkeypatch)
        assert run_v41_prefix_replay(world.worker, world.batch([PAGE], [1])) is None
        (side,) = world.sides
        assert side.prefix_lens == [PAGE - R]
        # A side replay's slot is stamped once its side forward has run.
        assert world.pool.state_gen[1] == world.generation[1]
        assert radix._STATS["forced_fallbacks"] == 1
        assert radix._STATS["fallbacks"] == 1
        # The hook is spent: the next hit fuses.
        fused = run_v41_prefix_replay(world.worker, world.batch([PAGE], [2]))
        assert fused is not None and fused.slots.tolist() == [2]
        assert len(world.sides) == 1 and radix._STATS["fused_steps"] == 1

    def test_off_by_default(self, monkeypatch, radix_fused):
        world = _World(monkeypatch)
        assert run_v41_prefix_replay(world.worker, world.batch([PAGE], [1])) is not None
        assert world.sides == [] and radix._STATS["forced_fallbacks"] == 0
