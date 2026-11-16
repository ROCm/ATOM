# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contract of the DeepSeek-V4.1 radix (prefix) cache on the SGLang plugin.

Replay first: a prefix hit at ``P`` recomputes ``[P - R, P)`` into its freshly
reset STATE slot in one side forward per step (``deepseek_v41_radix``), which
the bridge translates into ATOM's ``ScheduledBatch.replaying``. Covered here:

1. the bridge's replay branch and the main forward's stamp guard;
2. ATOM's ``_replay_layout`` over a plugin-shaped replay batch;
3. compress plans over ``[start, P)`` with ``write_from = P``;
4. ``PagedAttentionCache``: replay then main, no replay, coincidental cursor;
5. the seam: which rows replay, the one side batch, capacity splits;
6. the admission charge;
7. the refusal matrix, and that nothing is installed with radix off.
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
    ReplayHit,
    check_state_stamps,
    plan_prefix_replay,
    replay_admission_rows,
    resolve_replay_rows,
    run_v41_prefix_replay,
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
    monkeypatch.setattr(
        radix, "_STATS", {"side_forwards": 0, "hits": 0, "rows": 0, "steps": 0}
    )


@pytest.fixture
def radix_on(monkeypatch):
    monkeypatch.setattr(radix, "_REPLAY_ROWS", R)
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


def _req(tokens, return_logprob=False, logprob_start_len=-1):
    return SimpleNamespace(
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

    def batch(self, prefix, slots, reqs=None, mode=EXTEND):
        reqs = reqs or [_req(range(1000, 1000 + p + 300)) for p in prefix]
        return SimpleNamespace(
            forward_mode=mode,
            reqs=reqs,
            prefix_lens=list(prefix),
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
        replayed = run_v41_prefix_replay(
            world.worker, world.batch([2 * PAGE], [3], [req])
        )
        assert replayed == R
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
        assert run_v41_prefix_replay(world.worker, world.batch([2 * PAGE], [3])) == 0
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
        assert run_v41_prefix_replay(world.worker, world.batch([0], [2])) == 0
        assert world.sides == []
        assert world.pool.state_gen[2] == world.generation[2]

    def test_every_hit_of_a_step_shares_one_side_forward(self, monkeypatch, radix_on):
        world = _World(monkeypatch)
        world.stamp([4])
        reqs = [_req(range(k * 10000, k * 10000 + 3 * PAGE)) for k in range(4)]
        batch = world.batch([PAGE, 0, 2 * PAGE, PAGE], [1, 2, 3, 4], reqs)
        assert run_v41_prefix_replay(world.worker, batch) == 2 * R
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
        assert run_v41_prefix_replay(world.worker, batch) == 0
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


def test_worker_seam_runs_before_the_forward(monkeypatch, radix_on):
    pytest.importorskip("sglang")
    from sglang.srt.managers.tp_worker import TpModelWorker

    calls = []

    def original(self, batch, forward_batch=None, **kwargs):
        calls.append(("forward", batch, forward_batch))
        return "result"

    monkeypatch.setattr(TpModelWorker, "forward_batch_generation", original)
    monkeypatch.setattr(
        radix, "run_v41_prefix_replay", lambda worker, batch: calls.append("replay")
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
    pytest.importorskip("sglang")
    from sglang.srt.managers.schedule_policy import PrefillAdder

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
    return state


class TestRefusals:
    def test_supported_radix_launch_passes(self, launch):
        assert sglang_radix_cache_enabled()
        refuse_unsupported_sglang_features()

    @pytest.mark.parametrize(
        ("bag", "path", "value", "match"),
        [
            ("schedule", "enable_mixed_chunk", True, "mixed chunk"),
            ("schedule", "disable_overlap_schedule", False, "overlap scheduling"),
            ("memory", "enable_session_radix_cache", True, "session radix cache"),
            ("memory", "radix_cache_backend", "custom", "custom radix cache"),
            ("memory", "enable_unified_memory", True, "unified memory"),
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

    def test_raw_args_are_the_fallback(self, launch):
        launch.bags = {"disagg": launch.bags["disagg"]}
        launch.args.disable_radix_cache = False
        launch.args.cuda_graph_backend_prefill = "full"
        with pytest.raises(ValueError, match="overlap.*prefill CUDA graphs"):
            refuse_unsupported_sglang_features()
        launch.args.disable_overlap_schedule = True
        launch.args.cuda_graph_backend_prefill = "disabled"
        refuse_unsupported_sglang_features()

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


@pytest.mark.parametrize("decoder_replay", ["1", "0"])
@pytest.mark.parametrize("radix_cache", [False, True])
def test_prepare_installs_the_replay_only_with_radix(
    launch, monkeypatch, radix_cache, decoder_replay
):
    from atom.plugin.sglang.runtime import model_arch

    # Decoder bounded replay is independent of radix cache: on unless
    # ATOM_SGLANG_V41_DECODER_REPLAY=0, with radix on or off.
    monkeypatch.setenv("ATOM_SGLANG_V41_DECODER_REPLAY", decoder_replay)
    installed = []
    monkeypatch.setattr(
        radix,
        "install_deepseek_v41_radix",
        lambda window=None: installed.append(window) or R,
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
