# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for intra-GPU disagg constrained vs unconstrained modes.

Only the scheduler-level shm gating is exercised here; the IPC handshake
and CUDA stream pool are out of scope for the no-GPU test environment.
"""

import pickle
from threading import Lock

import pytest
import torch
from conftest import MockConfig


@pytest.fixture
def prefill_executor_unconstrained():
    from atom.model_engine.scheduler import PrefillExecutor

    return PrefillExecutor(MockConfig())


@pytest.fixture
def decode_scheduler_unconstrained():
    from atom.model_engine.scheduler import DecodeScheduler

    return DecodeScheduler(MockConfig())


def _batch_msg(chunks, cu_stream_fraction=None):
    from atom.model_engine.disagg_types import PrefillBatchMsg

    return PrefillBatchMsg(
        batch_id=0,
        target_rank=0,
        cu_stream_fraction=cu_stream_fraction,
        chunks=chunks,
    )


def _chunk(seq, start, num_tokens, is_final, block_table=None):
    from atom.model_engine.disagg_types import PrefillChunk

    return PrefillChunk(
        seq_id=seq.id,
        start=start,
        num_tokens=num_tokens,
        is_final_chunk=is_final,
        block_table=block_table,
    )


@pytest.fixture
def seq_factory():
    from atom.model_engine.sequence import Sequence
    from atom.sampling_params import SamplingParams

    def make(token_ids, block_size=4):
        return Sequence(token_ids, block_size, sampling_params=SamplingParams())

    return make


# ── Unconstrained: no shm handle attached ────────────────────────────────


def test_no_scheduler_opens_a_shared_memory_channel(
    prefill_executor_unconstrained, decode_scheduler_unconstrained
):
    """The CU shm is gone, on both sides.

    It existed to carry decode's in-flight token count across to prefill,
    because prefill sized its own batches and had to guess how much GPU decode
    wanted. Decode plans both batches now, so it holds both operands and the
    fraction travels inside the batch message.
    """
    for sched in (prefill_executor_unconstrained, decode_scheduler_unconstrained):
        assert not hasattr(sched, "_cu_shm")
        assert not hasattr(sched, "_cu_shm_offset")


def test_decode_handoff_builds_full_mtp_window(seq_factory):
    class SpecConfig:
        num_speculative_tokens = 3

        @staticmethod
        def use_dspark():
            return True

    from atom.model_engine.disagg_types import PrefillBatchDone, PrefillChunkResult
    from atom.model_engine.scheduler import DecodeScheduler

    scheduler = DecodeScheduler(MockConfig(speculative_config=SpecConfig()))
    prompt = [11, 22, 33, 44, 55]
    t0 = 14
    seq = seq_factory(prompt)
    scheduler.block_manager.allocate(seq)
    scheduler.prefill_inflight[seq.id] = seq
    scheduler._inflight_batch_id = 0
    scheduler._inflight_spans[seq.id] = (0, len(prompt), 0)

    scheduler.on_prefill_batch_done(
        PrefillBatchDone(
            batch_id=0,
            results=[
                PrefillChunkResult(
                    seq_id=seq.id, prefill_pos=len(prompt), sampled_token_id=t0
                )
            ],
        )
    )
    scheduler.drain_prefill_completions()

    assert seq.is_first_decode is True
    assert list(seq.token_ids[-4:]) == [t0, 2, 2, 2]
    assert not set(seq.token_ids[-4:]) & set(prompt)

    batch, _ = scheduler.schedule()
    assert batch.is_first_decode_without_local_prefill == [True]
    assert seq.is_first_decode is False


# ── Unconstrained: batches carry cu_stream_fraction=None ─────────────────


def test_unconstrained_prefill_batch_has_none_cu_fraction(
    prefill_executor_unconstrained, seq_factory
):
    """The executor forwards the fraction decode chose, unchanged.

    Unconstrained, decode sends None and the batch must be keyed by the plain
    stream — a fractional mask here would index a stream pool that was never
    built (KeyError on the first forward).
    """
    seq = seq_factory([10, 20, 30, 40])
    prefill_executor_unconstrained.add(seq)

    batch, seqs = prefill_executor_unconstrained.apply_batch(
        _batch_msg([_chunk(seq, 0, 4, True, block_table=[0, 1])])
    )
    assert batch is not None
    assert batch.cu_stream_fraction is None
    assert list(seqs) == [seq.id]


# ── Asymmetric rapidserve: reserved KV dump index ────────────────────────
#
# Under asymmetric rapidserve (prefill TP=N, decode TP=1) MLA's TP-replicated
# KV latent leaves every prefill rank holding an identical copy, but only the
# rank co-located with the target GPU may commit it. The others redirect their
# writes to block_manager.DUMP_INDEX, which must therefore never be handed out
# by any of the three pools.


# ── Asymmetric rapidserve: TP=1 weight twins ─────────────────────────────
#
# Prefill runs TP=N and decode TP=1, so a few weights (attention, shared
# experts, embed/head) have no counterpart among prefill's shards. Rather than
# let decode load them — which meant 8 concurrent checkpoint reads and a
# partial-load path that kept writing through IPC aliases into prefill's memory
# — prefill builds a TP=1 twin of those modules and exports it. Selecting the
# wrong module set is the failure that matters: twinning a FusedMoE would
# duplicate ~90GB that already aliases correctly, and MISSING a module leaves
# decode with an un-materialized meta parameter.


# ── Asymmetric rapidserve: decode-rank selection ─────────────────────────
#
# DisaggCoreManager's engine list is [prefill, decode_0 .. decode_N-1], so a
# request's target must be chosen from the decode slice only. Picking index 0
# would route a sequence to the prefill process, which owns no KV pool.


class TestBoundedRankSelection:
    class _Mgr:
        """Just the selector and the counters it reads."""

        def __init__(self, n_engines, strategy="least_requests"):
            from atom.model_engine.engine_core_mgr import CoreManager

            self.local_engine_count = n_engines
            # main's #2238 made the selector read `_routable_engine_count`
            # (a CoreManager property that falls back to local_engine_count).
            # This stub borrows one method, not the class, so it has to supply
            # the attribute the method actually reads.
            self._routable_engine_count = n_engines
            self._dp_lb_strategy = strategy
            self._dp_lb_req_equiv = 0
            self._rank_reqs = [0] * n_engines
            self._rank_tokens = [0] * n_engines
            self._rank_rotation_cursor = 0
            self._select = CoreManager._select_dp_rank_locked.__get__(self)

    def test_never_selects_the_prefill_index(self):
        m = self._Mgr(9)  # 1 prefill + 8 decode
        picks = {m._select(lo=1, hi=9) for _ in range(50)}
        assert 0 not in picks
        assert picks <= set(range(1, 9))

    def test_round_robin_stays_in_range(self):
        m = self._Mgr(9, strategy="round_robin")
        picks = [m._select(lo=1, hi=9) for _ in range(24)]
        assert set(picks) == set(range(1, 9))
        assert picks[0] == 1 and picks[8] == 1  # wraps within the slice

    def test_prefers_the_least_loaded_decode_rank(self):
        m = self._Mgr(5)  # 1 prefill + 4 decode
        m._rank_reqs = [99, 3, 1, 7, 5]  # index 0 is prefill and must be ignored
        assert m._select(lo=1, hi=5) == 2

    def test_default_bounds_cover_every_engine(self):
        """Non-disagg callers pass no bounds and must be unaffected."""
        m = self._Mgr(4, strategy="round_robin")
        assert {m._select() for _ in range(12)} == {0, 1, 2, 3}


# ── Asymmetric rapidserve: per-row KV write ownership ────────────────────


# ── Asymmetric rapidserve: GPU pinning of decode ranks ───────────────────
#
# ModelRunner derives its GPU from
#   (data_parallel_rank_local * pp + pp_rank) * (tp * pcp) + rank
# (model_runner.py:970-983). Decode runs at tp=1, so data_parallel_rank_local is
# the ONLY term that varies between its ranks — omit it and all N processes
# compute device 0, pile onto GPU 0, and then try to open IPC handles belonging
# to prefill ranks on other GPUs. That surfaces far away as "storage on
# different device", so the arithmetic is pinned down here.


def _local_device_rank(dp_rank_local, rank, tp, pp=1, pp_rank=0, pcp=1):
    engine_index = dp_rank_local * pp + pp_rank
    return engine_index * (tp * pcp) + rank


class TestDecodeGpuPinning:
    def test_prefill_tp_ranks_span_all_gpus(self):
        got = [_local_device_rank(0, r, tp=8) for r in range(8)]
        assert got == list(range(8))

    def test_decode_dp_ranks_span_all_gpus(self):
        """tp=1, so the DP-local rank alone must carry the device index."""
        got = [_local_device_rank(k, 0, tp=1) for k in range(8)]
        assert got == list(range(8))

    def test_unset_dp_rank_local_collapses_onto_gpu0(self):
        """The exact bug: without dp_rank_local every decode rank lands on 0."""
        got = [_local_device_rank(0, 0, tp=1) for _ in range(8)]
        assert got == [0] * 8

    def test_decode_rank_k_pairs_with_prefill_rank_k(self):
        """Both sides must resolve to the same physical GPU for IPC to work."""
        for k in range(8):
            assert _local_device_rank(k, 0, tp=1) == _local_device_rank(0, k, tp=8)


# ── rapidserve: where decode's device cost comes from ────────────────────
#
# prefill's `non_torch` = (total - free) - its own reserved, i.e. every byte the
# DEVICE says is used that prefill does not own. On 8xMI355X / V4-Pro that came
# to 150.10 GB while decode's allocator held ~5 GB at import time, and the terms
# reconcile with device-used (111.64 + 150.10 = 261.7 vs 260.2 actually used) —
# so the accounting is self-consistent, NOT double counting.
#
# An earlier attempt netted the IPC-mapped weights out of `non_torch` on the
# theory that they were prefill's own pages counted twice. That was wrong: it
# handed out a KV budget that did not exist and turned a clean startup error
# into an OOM inside allocate_kv_cache. `non_torch` must stay as written.
#
# Which of map_cost vs load_cost dominates the 150 GB is still open; the
# instrumentation in import_model_weight_ipc_handles measures both.


GB = 1 << 30


def _non_torch(total, free, reserved):
    """Mirrors the expression in ModelRunner.get_num_blocks."""
    return max((total - free) - reserved, 0)


class TestNonTorchAccounting:
    def test_reports_memory_held_by_other_processes(self):
        assert _non_torch(288 * GB, 100 * GB, 180 * GB) == 8 * GB

    def test_peer_memory_is_not_netted_out(self):
        """Regression guard for the reverted 'double count' fix."""
        # Device used 260, prefill allocator 110 -> 150 unavailable to prefill.
        # Subtracting the ~105 GB decode has mapped would report 45 and OOM later.
        assert _non_torch(288 * GB, 28 * GB, 110 * GB) == 150 * GB

    def test_never_goes_negative(self):
        assert _non_torch(288 * GB, 290 * GB, 80 * GB) == 0

    def test_terms_reconcile_with_device_used(self):
        """peak_torch + non_torch should sum to device-used, not exceed it.

        111.64 + 150.10 = 261.74 against 260.16 GB used: agreement to ~1.6GB is
        what shows the two terms are not counting the same bytes twice.
        """
        total, free = 287.98, 27.82
        peak_torch, non_torch = 111.64, 150.10
        assert abs((peak_torch + non_torch) - (total - free)) < 2.0


# ── rapidserve: exported tensors must outlive the export call ────────────
#
# decode's parameters are CUDA IPC views into prefill's allocations, so anything
# prefill materialises during export has to stay referenced for the life of the
# process. `_export_tensor` calls `.contiguous()`, which returns a NEW tensor for
# a non-contiguous input — the handle is taken from that copy's storage, and if
# it is left to go out of scope the consumer reads freed memory.


class TestExportRetention:
    def test_contiguous_input_is_not_retained(self):
        """Nothing was materialised, so there is nothing extra to hold."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        retain = []
        t = torch.zeros(4, 4)
        try:
            ipc._export_tensor(t, retain)
        except RuntimeError:
            pass  # CPU tensor: _share_cuda_ raises, but only AFTER .contiguous()
        assert retain == []

    def test_materialised_copy_is_retained(self):
        """A non-contiguous input produces a copy that MUST be kept alive."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        retain = []
        base = torch.zeros(4, 8)
        view = base[:, ::2]  # strided
        assert not view.is_contiguous()
        try:
            ipc._export_tensor(view, retain)
        except RuntimeError:
            pass
        assert len(retain) == 1
        assert retain[0].is_contiguous()
        assert retain[0].shape == view.shape

    def test_retain_is_optional(self):
        """Callers that do not care must not crash."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        base = torch.zeros(4, 8)
        try:
            ipc._export_tensor(base[:, ::2])
        except RuntimeError:
            pass


# ── Twin module attributes must override, not just fill gaps ─────────────
#
# process_weights_after_loading emits more than tensors. LinearBase sets
# `is_output_padded` (linear.py:798) and `_output_size_before_padding` (:827),
# and forward slices padded columns off using both (:985). Whether padding is
# needed is decided from output_size — precisely what differs between the TP=N
# module and its TP=1 twin — so the consumer must take the TWIN's values. It
# cannot derive them: it never runs the hook, and __init__ leaves
# is_output_padded=False (linear.py:508).


# ── Ancestor hooks must run against the twins too ────────────────────────
#
# A post-load hook does not always live on the module owning the weight.
# DeepSeek-V4's attention dequantizes wo_a from FP8 to BF16 for the grouped-LoRA
# einsum (deepseek_v4.py:2453) — the hook is on the PARENT, and wo_a is a
# twinned child. Running only the twins' own hooks leaves the twin FP8, and
# decode dies in _wo_a_grouped_lora with "expected scalar type BFloat16 but
# found Float8_e4m3fn".


# ── Decode DP lockstep ───────────────────────────────────────────────────
#
# The MoE all_gather / reduce_scatter (moe.py:362) is a COLLECTIVE across the
# decode DP ranks, run once per MoE layer per step. Every rank must issue a
# forward on every step or the ones that do block forever on the ones that do
# not. That is exactly what happens while draining — ranks run out of sequences
# at different times — and it cost 7 in-flight requests in debug9.log, with the
# hang surfacing as aiter "shared memory broadcast block" timeouts.
#
# DPEngineCoreProc's loop prevents it: all-reduce `has_unfinished` each step and
# run `_execute_dummy_batch()` on any rank with nothing real to do.


class TestDecodeLockstep:
    def test_decode_engine_inherits_the_dp_loop(self):
        ec = pytest.importorskip("atom.model_engine.engine_core")
        assert issubclass(ec.DecodeEngineCore, ec.DPEngineCoreProc)

    def test_lockstep_selected_when_a_dp_group_exists(self):
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DecodeEngineCore)
        obj.dp_group = object()  # stand-in for a real group
        assert obj._dp_lockstep is True

    def test_plain_loop_when_single_decode_rank(self):
        """Symmetric rapidserve: one decode engine, no peer to stay in step with."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DecodeEngineCore)
        obj.dp_group = None
        assert obj._dp_lockstep is False

    def test_missing_attribute_degrades_to_plain_loop(self):
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DecodeEngineCore)
        assert obj._dp_lockstep is False

    def test_an_idle_decode_group_skips_the_dummy_forward(self):
        """Decode must keep ticking to drive prefill, but `busy_loop` reads
        "the step ran nothing" as "issue a dummy forward". While every request
        is mid-prefill that is a real forward per tick, on the same GPU as the
        prefill chunk it competes with, matching a collective nobody entered.

        Measured on a 670k-token agentic warmup: decode sat at `Running: 0`
        for 15 minutes while prefill ran at ~60% occupancy.
        """
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DecodeEngineCore)
        obj.dp_group = None  # dp=1: answer locally, no collective
        obj.scheduler = type(
            "S",
            (),
            {
                "prefill_batch_inflight": False,
                "cu_masking": False,
                "set_cu_fraction": lambda self, f: None,
                "decode_tokens_estimate": lambda self: 0,
                "prefill_pending_estimate": lambda self: 0,
                "has_decode_work": lambda self: False,
                "schedule": lambda self: None,
                "has_requests": lambda self: True,
                "build_prefill_batch": lambda self, **kw: None,
                "drain_prefill_completions": lambda self: None,
                "pending_aborts": [],
            },
        )()
        assert obj._sync_decode_group_state().will_forward is False
        assert obj._process_engine_step() is True, (
            "a tick that drove prefill but had nothing to decode must not be "
            "reported as 'ran nothing'"
        )

    def test_a_rank_with_decode_work_still_forwards(self):
        """The gate must not swallow real work: the proxy is `running`, and a
        non-empty queue falls through to `schedule()` exactly as before."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DecodeEngineCore)
        obj.dp_group = None
        obj.scheduler = type(
            "S",
            (),
            {
                "prefill_batch_inflight": False,
                "cu_masking": False,
                "set_cu_fraction": lambda self, f: None,
                "decode_tokens_estimate": lambda self: 0,
                "prefill_pending_estimate": lambda self: 0,
                "has_decode_work": lambda self: True,
            },
        )()
        state = obj._sync_decode_group_state()
        assert (state.busy, state.will_forward) == (False, True)

    def test_the_gate_runs_before_every_per_rank_return(self):
        """gloo matches collectives by ARRIVAL ORDER, not by tag.

        A rank that skipped this reduction while its peers entered it would
        pair its next `_sync_dp_state` (3 int64) against their 1-int64
        reduction here, and the group wedges. Placed behind `has_requests()`,
        a burst whose ADDs reached the decode ranks a tick apart hung the whole
        replica after a single dispatched batch -- 7 ranks stuck inside a dummy
        forward with `dummy_inflight=4.0s` and nothing else in any log.

        Checked on the AST, not the source text: the comments here name the
        very constructs being ordered, so a substring search matches prose and
        passes whatever the code does.
        """
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        fn = ast.parse(
            inspect.getsource(ec.DecodeEngineCore._process_engine_step).lstrip()
        ).body[0]
        gate = min(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_sync_decode_group_state"
        )
        first_return = min(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Return))
        assert gate < first_return, (
            "a per-rank early return before the cross-DP reduction lets ranks "
            "disagree about how many collectives they have entered"
        )
        per_rank = min(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("has_requests", "schedule")
        )
        assert gate < per_rank

    def test_skipping_the_forward_is_a_group_decision(self):
        """`busy_loop` hands a dummy forward to whoever returns False, so every
        rank runs exactly one forward per step -- real or dummy -- and that is
        what keeps the MoE collective inside it matched.

        So on the no-batch path the answer must come from the all-reduced value
        and nothing else. One rank dummying while the others skip is the same
        hang from the other side.
        """
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        fn = ast.parse(
            inspect.getsource(ec.DecodeEngineCore._process_engine_step).lstrip()
        ).body[0]
        guards = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Return)
            and isinstance(n.value, ast.UnaryOp)
            and isinstance(n.value.op, ast.Not)
            and isinstance(n.value.operand, ast.Name)
            and n.value.operand.id == "group_will_forward"
        ]
        assert len(guards) == 1, (
            "the no-batch path must return exactly `not group_will_forward` -- "
            "a local condition there desynchronises the dummy"
        )

    def test_schedule_is_never_skipped_by_the_gate(self):
        """`schedule()` is not just a batch builder.

        It promotes `prefill_done` into `running`, and it runs
        `complete_previous_state_batch()` -- "without which the pins taken by
        every resume accumulate forever and admission starves". The ticks this
        gate fires on are exactly the ones where decode has no batch, which
        during a long prefill is all of them, so a gate that skipped
        `schedule()` would starve the state pool for the whole prompt.
        """
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        fn = ast.parse(
            inspect.getsource(ec.DecodeEngineCore._process_engine_step).lstrip()
        ).body[0]
        sched = min(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "schedule"
        )
        first_return = min(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Return))
        assert sched < first_return, (
            "no return may precede schedule(): it must run on every path, "
            "whatever the gate decided"
        )

    def test_a_rank_with_no_requests_still_dummies_when_peers_forward(self):
        """Returning True there would leave the peers' MoE collective short."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DecodeEngineCore)
        obj.dp_group = None
        obj.scheduler = type(
            "S",
            (),
            {
                "prefill_batch_inflight": False,
                "cu_masking": False,
                "set_cu_fraction": lambda self, f: None,
                "decode_tokens_estimate": lambda self: 0,
                "prefill_pending_estimate": lambda self: 0,
                "has_decode_work": lambda self: True,  # group forwards
                "schedule": lambda self: None,
                "has_requests": lambda self: False,
                "build_prefill_batch": lambda self, **kw: None,
                "drain_prefill_completions": lambda self: None,
                "pending_aborts": [],
            },
        )()
        assert obj._process_engine_step() is False, "must earn a dummy forward"

    def test_the_gate_agrees_with_what_schedule_actually_does(self):
        """The predicate and `_schedule()` must never disagree.

        The gate decides whether `_schedule()` is called AT ALL, so a predicate
        that says "no work" when `_schedule()` would have produced a batch is a
        livelock, not a missed optimisation -- nothing else promotes
        `prefill_done` into `running`.

        Checked behaviourally against the real scheduler over every queue state
        that matters, rather than by reading either implementation: that is the
        only form of this test that would have caught `bool(running)`.
        """
        from conftest import MockConfig

        from atom.model_engine.scheduler import DecodeScheduler
        from atom.model_engine.sequence import Sequence
        from atom.sampling_params import SamplingParams

        def seq():
            return Sequence([1, 2, 3, 4], 4, sampling_params=SamplingParams())

        for name, place in (
            ("empty", lambda s: None),
            ("running", lambda s: s.running.append(seq())),
            ("prefill_done", lambda s: s.prefill_done.append(seq())),
        ):
            sched = DecodeScheduler(MockConfig())
            place(sched)
            for q in (sched.running, sched.prefill_done):
                for sq in q:
                    sched.block_manager.allocate(sq)
            claimed = sched.has_decode_work()
            produced = sched.schedule() is not None
            assert claimed == produced, (
                f"queue state {name!r}: has_decode_work()={claimed} but "
                f"schedule() produced a batch={produced}"
            )

    def test_a_finished_prefill_is_not_stranded(self):
        """The exact livelock: prefill drains, every request lands in
        `prefill_done`, and the gate -- reading `running` -- reports no work,
        so `_schedule()` is never called and nothing ever promotes them.

        Observed as 16 requests per rank reported `Running`, 0 tokens/s, zero
        decode forwards for the whole run, and the delayer firing `vacuous`
        41000 times because the prefill side was empty too.
        """
        from conftest import MockConfig

        from atom.model_engine.scheduler import DecodeScheduler
        from atom.model_engine.sequence import Sequence
        from atom.sampling_params import SamplingParams

        sched = DecodeScheduler(MockConfig())
        sq = Sequence([1, 2, 3, 4], 4, sampling_params=SamplingParams())
        sched.block_manager.allocate(sq)
        sched.prefill_done.append(sq)

        assert not sched.running, "the request has not been promoted yet"
        assert (
            sched.has_decode_work()
        ), "a request whose prefill just finished is decodable this tick"

    def test_the_delayer_skip_is_group_wide_never_local(self):
        """ "Busy" is PER-RANK -- decode ranks finish their prefills at
        different times -- and the delayer's own tick is a cross-DP all_reduce.
        A rank that skipped it on its own `prefill_batch_inflight` would
        desynchronise the group: gloo matches by arrival order, so it would
        pair its next `_sync_dp_state` against a peer's delayer reduction.

        So the engine must pass the ALL-REDUCED answer down, and the scheduler
        must take the skip only from that argument.
        """
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")
        sched_mod = pytest.importorskip("atom.model_engine.scheduler")

        step = inspect.getsource(ec.DecodeEngineCore._process_engine_step)
        assert "skip_delayer=group_busy" in step, (
            "the skip must come from the reduction, not from the scheduler's "
            "own in-flight flag"
        )

        build = ast.parse(
            inspect.getsource(sched_mod.DecodeScheduler.build_prefill_batch).lstrip()
        ).body[0]
        guards = [
            n
            for n in build.body
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name)
            and n.test.id == "skip_delayer"
        ]
        assert len(guards) == 1, "exactly one guard, reading the argument"
        # ...and it must sit above the delayer tick it is skipping.
        delayer = min(
            n.lineno
            for n in ast.walk(build)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "should_allow_prefill"
        )
        assert guards[0].lineno < delayer

    def test_one_busy_peer_holds_the_whole_group(self):
        """Dispatch is gated on NO rank being mid-flight, not on all of them
        being so.

        The pair's prefill ranks run one forward per step and its MoE
        all-to-all spans every rank, so a rank handed a batch early does not
        get a head start -- its peers simply block inside the collective in a
        dummy until it is done. Letting each rank dispatch the moment it frees
        up split the group: 4.05 of 8 ranks doing real work per step, 51%
        prefill occupancy, one ~1.8s blocked dummy per real 1.9s chunk.
        """
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        fn = ast.parse(
            inspect.getsource(ec.DecodeEngineCore._sync_decode_group_state).lstrip()
        ).body[0]
        ret = max(
            (n for n in ast.walk(fn) if isinstance(n, ast.Return)),
            key=lambda n: n.lineno,
        )
        # `DecodeGroupState(busy, ...)` -- the first construction argument is
        # the busy predicate, whatever the container around it.
        value = ret.value
        args = value.args if isinstance(value, ast.Call) else value.elts
        busy = args[0]
        # Shape, not spelling: `<reduced busy slot> > 0`. Pinning the exact
        # text breaks on a variable rename and says nothing about meaning;
        # what matters is that it is a comparison against ZERO -- "any rank" --
        # and not against a group size, which would let a rank dispatch as soon
        # as it alone was free.
        assert isinstance(busy, ast.Compare), ast.unparse(busy)
        assert len(busy.ops) == 1 and isinstance(busy.ops[0], ast.Gt), (
            f"dispatch must hold while ANY rank is busy, got {ast.unparse(busy)!r}"
        )
        bound = busy.comparators[0]
        assert isinstance(bound, ast.Constant) and bound.value == 0, (
            f"compared against {ast.unparse(bound)!r}, not 0 -- a threshold of "
            "dp_size lets ranks dispatch independently and splits the prefill "
            "group between real work and blocked dummies"
        )

    def test_a_busy_rank_alone_does_not_skip_the_delayer(self):
        """Only the group's answer may skip it. A locally busy rank whose
        peers are free still has to tick the delayer -- that is where the
        `n_prefillable < dp_size` alignment gate holds the free ranks until
        this one catches up.
        """
        from conftest import MockConfig

        from atom.model_engine.scheduler import DecodeScheduler

        sched = DecodeScheduler(MockConfig())
        sched._inflight_batch_id = 7  # locally busy
        assert sched.prefill_batch_inflight is True

        calls = []

        class _Delayer:
            is_local = True

            def should_allow_prefill(self, **kw):
                calls.append(kw)
                return True

        sched.prefill_delayer = _Delayer()
        assert sched.build_prefill_batch(skip_delayer=False) is None
        assert len(calls) == 1, "a busy rank still ticks the delayer"
        assert calls[0]["prefillable"] is False

        calls.clear()
        assert sched.build_prefill_batch(skip_delayer=True) is None
        assert calls == [], "the group said skip, so the tick is skipped"

    def test_the_gate_fails_open(self):
        """A peer dying must not wedge the step; the guarded DP-state barrier
        drives shutdown, as it does for `_sync_dp_state`."""
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        src = inspect.getsource(ec.DecodeEngineCore._sync_decode_group_state)
        assert "except RuntimeError" in src
        assert "DecodeGroupState(False, True, 0, 0)" in src, (
            "the error path must not skip the delayer, must assume decode work "
            "exists, and must mask no CUs -- i.e. leave every rank on the path "
            "it took before this method existed"
        )

    def test_step_reports_whether_it_ran(self):
        """The loop uses this to decide when a dummy batch is needed."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DecodeEngineCore)
        # All three run before the early return:
        # `drain_prefill_completions` applies batches that retired since the
        # last pass, and `build_prefill_batch` ticks the delayer's collective.
        # A rank that skips either is out of step with the others, so the stub
        # has to provide them even when there is nothing to dispatch.
        obj.dp_group = None
        obj.scheduler = type(
            "S",
            (),
            {
                # A peer has decode work, so the group forwards and this rank
                # owes it a dummy.
                "prefill_batch_inflight": False,
                "cu_masking": False,
                "set_cu_fraction": lambda self, f: None,
                "decode_tokens_estimate": lambda self: 0,
                "prefill_pending_estimate": lambda self: 0,
                "has_decode_work": lambda self: True,
                "schedule": lambda self: None,
                "has_requests": lambda self: False,
                "build_prefill_batch": lambda self, **kw: None,
                "drain_prefill_completions": lambda self: None,
                "pending_aborts": [],
            },
        )()
        assert obj._process_engine_step() is False


# ── Symmetric rapidserve init path ───────────────────────────────────────
#
# `--enable-rapidserve` without `--enable-dp-attention` runs ONE decode process
# at the same TP as prefill, so decode worker `r` sits on GPU r and pairs with
# prefill TP rank r. `disagg_decode_rank` is 0 for all of them, which is what
# made indexing the IPC handle files by decode rank alone collapse every worker
# onto prefill rank 0's file: workers 1..N-1 opened a handle exported on cuda:0
# from another GPU and died, and the engine only noticed two RPCs later, when
# the manager had already torn the broadcast shm down ("'NoneType' object is
# not subscriptable" out of aiter's shm_broadcast).
#
# Decode also has to keep taking the plain EngineCore loop here: with a single
# rank there is no collective to stay in step with, and DPEngineCoreProc's loop
# would all-reduce over a group that was never formed.


def _rapidserve_runner(*, rank, decode, dp_rank_local=0, tp=8, device=None, pcp=1):
    """A RapidServeModelRunner with only the attributes the pairing logic reads.

    `tp` is THIS process's TP size — 8 for symmetric rapidserve (one process
    per side), 1 for the paired DP topology (N processes per side).
    `dp_rank_local` is the process's DP rank, which carries the GPU when tp==1.
    `device` defaults to the correctly-paired GPU, so a test spells it out only
    when it is deliberately breaking the pairing.
    """
    import types

    mr = pytest.importorskip("atom.model_engine.model_runner")

    obj = object.__new__(mr.RapidServeModelRunner)
    obj.rank = rank
    obj.config = types.SimpleNamespace(
        disagg_is_decode=decode,
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        parallel_config=types.SimpleNamespace(data_parallel_rank_local=dp_rank_local),
    )
    if device is None:
        device = dp_rank_local * tp * pcp + rank
    obj.device = torch.device(f"cuda:{device}")
    return obj


class TestDisaggPairRank:
    """The arithmetic itself. `disagg_types` is stdlib-only on purpose, so these
    keep running under the module-shadowing that skips the ModelRunner tests
    below when the whole suite runs in one process."""

    @staticmethod
    def _pair_rank(*args):
        from atom.model_engine.disagg_types import disagg_pair_rank

        return disagg_pair_rank(*args)

    def test_symmetric_decode_worker_pairs_with_its_own_gpu(self):
        """The regression: TP=8 decode, one process, decode_rank 0 throughout."""
        got = [self._pair_rank(0, r, 8) for r in range(8)]
        assert got == list(range(8))

    def test_symmetric_decode_does_not_collapse_onto_prefill_rank_0(self):
        """Every worker reading paths[0] is what killed ranks 1..7."""
        assert len({self._pair_rank(0, r, 8) for r in range(8)}) == 8

    def test_asymmetric_decode_indexes_by_decode_rank(self):
        """TP=1 decode: the worker rank is always 0, so decode_rank is the GPU."""
        got = [self._pair_rank(k, 0, 1) for k in range(8)]
        assert got == list(range(8))

    def test_matches_the_device_arithmetic_in_both_topologies(self):
        """The pair rank IS a GPU index, so it must equal _local_device_rank."""
        for r in range(8):  # symmetric: one process at TP=8
            assert self._pair_rank(0, r, 8) == _local_device_rank(0, r, tp=8)
        for k in range(8):  # asymmetric: eight processes at TP=1
            assert self._pair_rank(k, 0, 1) == _local_device_rank(k, 0, tp=1)

    def test_context_parallel_widens_the_stage(self):
        """stage_span is tp*pcp, matching _setup_device_and_distributed."""
        assert self._pair_rank(1, 1, 2 * 2) == 5


class TestDisaggIndexSplit:
    """The GPU index and the handle-paths index are different things.

    They coincide whenever prefill is a single process, which is why they were
    one property until prefill itself became data-parallel. The GPU index says
    which device an IPC handle is valid on; the paths index says where in the
    paired prefill's per-worker file list to look.
    """

    def test_symmetric_gpu_index_is_the_worker_rank(self):
        """One process at TP=8: dp_rank_local is 0, so rank carries the GPU."""
        for r in range(8):
            mr = _rapidserve_runner(rank=r, decode=True, tp=8)
            assert mr._disagg_gpu_index == r
            assert mr._disagg_paths_index == r

    def test_paired_gpu_index_is_the_dp_rank(self):
        """N processes at TP=1: rank is always 0, dp_rank_local carries the GPU."""
        for k in range(8):
            mr = _rapidserve_runner(rank=0, decode=True, dp_rank_local=k, tp=1)
            assert mr._disagg_gpu_index == k
            # ...but the paired prefill published ONE file, so index 0.
            assert mr._disagg_paths_index == 0

    def test_paired_prefill_and_decode_land_on_the_same_gpu(self):
        for k in range(8):
            pre = _rapidserve_runner(rank=0, decode=False, dp_rank_local=k, tp=1)
            dec = _rapidserve_runner(rank=0, decode=True, dp_rank_local=k, tp=1)
            assert pre._disagg_gpu_index == dec._disagg_gpu_index == k

    def test_symmetric_prefill_and_decode_land_on_the_same_gpu(self):
        for r in range(8):
            pre = _rapidserve_runner(rank=r, decode=False, tp=8)
            dec = _rapidserve_runner(rank=r, decode=True, tp=8)
            assert pre._disagg_gpu_index == dec._disagg_gpu_index == r

    def test_paths_index_would_be_out_of_range_if_it_used_the_gpu(self):
        """The regression this split prevents: a paired prefill publishes one
        path, so indexing it by GPU would raise on every rank but 0."""
        mr = _rapidserve_runner(rank=0, decode=True, dp_rank_local=5, tp=1)
        published = ["only-one-file"]
        assert published[mr._disagg_paths_index] == "only-one-file"
        assert mr._disagg_gpu_index >= len(published)


class TestDisaggPairingGuard:
    """The producer and consumer of an IPC handle must be on the same GPU."""

    @staticmethod
    def _assert(mr):
        return mr._assert_disagg_pairing()

    def test_symmetric_correct_pairing_passes(self):
        for r in range(8):
            self._assert(_rapidserve_runner(rank=r, decode=True, tp=8))

    def test_paired_correct_pairing_passes(self):
        for k in range(8):
            self._assert(_rapidserve_runner(rank=0, decode=True, dp_rank_local=k, tp=1))

    def test_collapse_onto_gpu0_is_caught(self):
        """Forgetting data_parallel_rank_local puts every process on cuda:0."""
        mr = _rapidserve_runner(rank=0, decode=True, dp_rank_local=5, tp=1, device=0)
        with pytest.raises(RuntimeError, match="pairing broken"):
            self._assert(mr)

    def test_symmetric_mismatch_is_caught(self):
        mr = _rapidserve_runner(rank=3, decode=True, tp=8, device=0)
        with pytest.raises(RuntimeError, match="pairing broken"):
            self._assert(mr)


class TestSymmetricDecodeInit:
    def test_single_rank_forms_no_dp_group(self):
        """dp_size == 1: nothing to rendezvous with, so never build a group."""
        import types

        ec = pytest.importorskip("atom.model_engine.engine_core")

        def _boom():
            raise AssertionError("stateless_init_dp_group must not be called")

        obj = object.__new__(ec.DPEngineCoreProc)
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                data_parallel_rank=0,
                data_parallel_size=1,
                # None is what a non-DP launch leaves this at; the DP path
                # asserts on it, so the early return has to come first.
                data_parallel_rank_local=None,
                stateless_init_dp_group=_boom,
            )
        )
        obj._init_data_parallel(config)
        assert obj.dp_group is None
        assert obj.dp_rank == 0

    def test_symmetric_takes_the_plain_engine_loop(self, monkeypatch):
        ec = pytest.importorskip("atom.model_engine.engine_core")

        picked = []
        monkeypatch.setattr(
            ec.EngineCore, "busy_loop", lambda self: picked.append("plain")
        )
        monkeypatch.setattr(
            ec.DPEngineCoreProc, "busy_loop", lambda self: picked.append("lockstep")
        )
        obj = object.__new__(ec.DecodeEngineCore)
        obj.dp_group = None
        obj.busy_loop()
        assert picked == ["plain"]

    def test_asymmetric_takes_the_lockstep_loop(self, monkeypatch):
        """Load-bearing: dropping it reintroduces the drain deadlock."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        picked = []
        monkeypatch.setattr(
            ec.EngineCore, "busy_loop", lambda self: picked.append("plain")
        )
        monkeypatch.setattr(
            ec.DPEngineCoreProc, "busy_loop", lambda self: picked.append("lockstep")
        )
        obj = object.__new__(ec.DecodeEngineCore)
        obj.dp_group = object()
        obj.busy_loop()
        assert picked == ["lockstep"]

    def test_decode_never_attaches_a_prefill_delayer(self, monkeypatch):
        """Decode has no prefill to delay, and no scheduler yet at this point."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        monkeypatch.setattr(ec.envs, "ATOM_ENABLE_PREFILL_DELAYER", True)

        class _Scheduler:
            def set_prefill_delayer(self, delayer):
                raise AssertionError("decode must not get a prefill delayer")

        obj = object.__new__(ec.DecodeEngineCore)
        obj.scheduler = _Scheduler()
        obj.dp_group = None
        assert obj._maybe_attach_prefill_delayer(None) is None


# ── Asymmetric rapidserve: twin/shard cross-check geometry ───────────────
#
# `verify_against_shards` asserts that a twin holds the same bytes its real
# module's TP shard received from the one shared `loaded_weight`. That is only
# meaningful if the shard geometry is mapped correctly — a wrong map either
# compares the wrong bytes (silent) or fails a correct run (fatal). Merged
# layers are the case worth pinning down: the real tensor is
# [gate_shard, up_shard] while the twin is [gate_full, up_full], so a single
# narrow reads across the partition boundary.


# ── Asymmetric rapidserve: masking the prefill KV WRITE ──────────────────
#
# prepare_block_tables masks the paged block table, but that is the READ side
# and it is published only when the batch has prefix-cache hits. Where prefill
# WRITES its KV is decided by slot_mapping, built in prepare_prefill from the
# raw batch.block_tables lists. Unmasked, every prefill TP rank commits every
# row into its own GPU's cache at block ids owned by a different decode rank's
# sequences — silent cross-sequence corruption.


# ── Asymmetric rapidserve: post-load module attributes across the IPC ────
#
# The consumer builds on meta and imports; it NEVER runs
# process_weights_after_loading. So every attribute that hook produces must
# travel in the payload, and the consumer's own value is a construction-time
# default rather than a computed result worth preserving. A "only fill what is
# None" policy restored 0 of 1343 modules in practice, because the flags that
# matter default to False, not None.


class TestModuleMetaAttrs:
    @staticmethod
    def _mod(**attrs):
        m = torch.nn.Module()
        for k, v in attrs.items():
            object.__setattr__(m, k, v)
        return m

    def test_false_bools_are_exported(self):
        """is_output_padded defaults False; it must still travel."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        attrs = ipc._module_meta_attrs(self._mod(is_output_padded=False))
        assert attrs["is_output_padded"] is False

    def test_true_bools_are_exported(self):
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        attrs = ipc._module_meta_attrs(self._mod(is_output_padded=True))
        assert attrs["is_output_padded"] is True

    def test_layout_strings_are_exported(self):
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        attrs = ipc._module_meta_attrs(self._mod(w13_swizzle_layout="foo"))
        assert attrs["w13_swizzle_layout"] == "foo"

    def test_padding_width_travels_despite_being_int_and_underscored(self):
        """_output_size_before_padding is read by forward (linear.py:986)."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        attrs = ipc._module_meta_attrs(self._mod(_output_size_before_padding=4096))
        assert attrs["_output_size_before_padding"] == 4096

    def test_parallel_config_ints_are_not_carried(self):
        """tp_size/tp_rank differ by design between a TP=8 and a TP=1 process."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        attrs = ipc._module_meta_attrs(self._mod(tp_size=8, tp_rank=3))
        assert "tp_size" not in attrs and "tp_rank" not in attrs

    def test_training_flag_is_not_carried(self):
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        m = torch.nn.Module()
        assert "training" not in ipc._module_meta_attrs(m)


class TestKvDimShipping:
    """Dims KV binding needs that the CONSUMER cannot derive correctly.

    The runner computes num_kv_heads as num_key_value_heads // world_size.
    Under asymmetric rapidserve prefill is TP=8 and decode TP=1, so decode
    derives an 8x larger value and would bind views that size over prefill's
    allocation. MLA/V4 never reads it, but the MHA backends size their KV with
    it, so the producer's value has to travel.
    """

    def test_num_kv_heads_is_shipped(self):
        mr = pytest.importorskip("atom.model_engine.model_runner")

        assert "num_kv_heads" in mr._KV_DIM_ATTRS

    def test_swa_and_slot_dims_still_shipped(self):
        mr = pytest.importorskip("atom.model_engine.model_runner")

        assert "num_swa_blocks" in mr._KV_DIM_ATTRS
        assert "max_per_req_cache_slots" in mr._KV_DIM_ATTRS

    def test_asymmetric_world_sizes_derive_different_head_counts(self):
        """The arithmetic that makes shipping necessary."""

        def derive(num_kv_heads_cfg, world_size):
            if num_kv_heads_cfg >= world_size:
                return num_kv_heads_cfg // world_size
            return 1

        assert derive(128, 8) == 16  # prefill
        assert derive(128, 1) == 128  # decode — 8x, and wrong for binding
        assert derive(128, 8) != derive(128, 1)

    def test_symmetric_world_sizes_agree(self):
        """Why symmetric rapidserve never hit this."""

        def derive(num_kv_heads_cfg, world_size):
            if num_kv_heads_cfg >= world_size:
                return num_kv_heads_cfg // world_size
            return 1

        assert derive(128, 8) == derive(128, 8)


# ── Paired rapidserve: process layout and request routing ────────────────
#
# --enable-rapidserve --enable-dp-attention gives N prefill + N decode
# processes, all TP=1, paired one-to-one per GPU. A sequence assigned to pair k
# prefills on GPU k and decodes on GPU k, so its KV never crosses a device —
# which is what removed the weight twins, the KV write masking and the dump
# blocks that an asymmetric (prefill TP=N) topology needed.


class TestPairedTopologyLayout:
    @staticmethod
    def _layout(tp, dp, dp_attention):
        """Mirror of DisaggCoreManager.__init__'s index arithmetic."""
        n_pairs = tp * dp if dp_attention else 1
        return {
            "n_pairs": n_pairs,
            "local_engine_count": 2 * n_pairs,
            "decode_idx0": n_pairs,
        }

    def test_paired_spawns_two_processes_per_gpu(self):
        got = self._layout(tp=8, dp=1, dp_attention=True)
        assert got["n_pairs"] == 8
        assert got["local_engine_count"] == 16
        assert got["decode_idx0"] == 8

    def test_symmetric_keeps_one_process_per_side(self):
        """--enable-rapidserve alone: one prefill at TP=N, one decode at TP=N."""
        got = self._layout(tp=8, dp=1, dp_attention=False)
        assert got["n_pairs"] == 1
        assert got["local_engine_count"] == 2
        assert got["decode_idx0"] == 1

    def test_dp_multiplies_into_the_pair_count(self):
        got = self._layout(tp=4, dp=2, dp_attention=True)
        assert got["n_pairs"] == 8 and got["local_engine_count"] == 16

    def test_prefill_and_decode_indices_never_overlap(self):
        got = self._layout(tp=8, dp=1, dp_attention=True)
        prefill = set(range(got["decode_idx0"]))
        decode = set(range(got["decode_idx0"], got["local_engine_count"]))
        assert not (prefill & decode)
        assert len(prefill) == len(decode) == got["n_pairs"]


class TestPairedDataParallelBounds:
    """The paired configs must carry data_parallel_size_local, not just _size.

    ParallelConfig resolves size_local once, in __post_init__. `_pair_config`
    deepcopies an already-built config and only then raises
    data_parallel_size to n_pairs, so size_local does not follow it: a plain
    `-tp 8` launch left it at 1 while local ranks ran 0..7, and every rank but
    0 died in DPEngineCoreProc._init_data_parallel with
    `local_dp_rank=k outside [0,1)`. The same stale value reaches MoRI as
    gpu_per_node (moe.py local_ep_size), so the assert was the first symptom,
    not the only one.
    """

    @staticmethod
    def _dp_engine_core_proc():
        """Bind the class against a fake AITER — async_proc needs a real build
        at import time, which the no-GPU runner does not have."""
        from aiter_stub import stubbed_aiter

        with stubbed_aiter():
            from atom.model_engine.engine_core import DPEngineCoreProc

        return DPEngineCoreProc

    @staticmethod
    def _pair_parallel_config(k, n_pairs):
        """The parallel_config `_pair_config` hands one paired process."""
        import copy

        from atom.config import ParallelConfig

        # As built by a `-tp 8` launch, before DisaggCoreManager reshapes it:
        # dp_size 1, so __post_init__ pins size_local to 1.
        pc = copy.deepcopy(ParallelConfig())
        pc.data_parallel_size = n_pairs
        pc.data_parallel_size_local = n_pairs
        pc.data_parallel_rank = k
        pc.data_parallel_rank_local = k
        return pc

    def test_size_local_does_not_track_a_later_dp_size(self):
        """The trap itself: assigning dp_size after construction is not enough."""
        from atom.config import ParallelConfig

        pc = ParallelConfig()
        assert pc.data_parallel_size_local == 1
        pc.data_parallel_size = 8
        assert pc.data_parallel_size_local == 1

    @pytest.mark.parametrize("k", range(8))
    def test_every_pair_rank_passes_the_bound_check(self, k):
        import types

        DPEngineCoreProc = self._dp_engine_core_proc()

        pc = self._pair_parallel_config(k, n_pairs=8)
        pc.stateless_init_dp_group = lambda: "dp_group"
        obj = object.__new__(DPEngineCoreProc)

        DPEngineCoreProc._init_data_parallel(
            obj, types.SimpleNamespace(parallel_config=pc)
        )

        assert obj.dp_rank == k
        assert obj.dp_group == "dp_group"

    def test_stale_size_local_is_what_broke_rank_4(self):
        """Guard the guard: without the fix the bound check must still fire."""
        import types

        DPEngineCoreProc = self._dp_engine_core_proc()

        pc = self._pair_parallel_config(4, n_pairs=8)
        pc.data_parallel_size_local = 1  # the pre-fix value
        obj = object.__new__(DPEngineCoreProc)

        with pytest.raises(AssertionError, match=r"local_dp_rank=4 outside \[0,1\)"):
            DPEngineCoreProc._init_data_parallel(
                obj, types.SimpleNamespace(parallel_config=pc)
            )


class _RecordingSocket:
    def __init__(self):
        self.messages = []

    def send_multipart(self, parts, copy=False):
        self.messages.append(parts)


class _PairedSeq:
    """Minimal Sequence stand-in: routing reads id/num_prompt_tokens/max_tokens."""

    def __init__(self, seq_id, num_prompt_tokens=1, max_tokens=64, **attrs):
        self.id = seq_id
        self.num_prompt_tokens = num_prompt_tokens
        self.max_tokens = max_tokens
        self.stream_callback = None
        for name, value in attrs.items():
            setattr(self, name, value)


def _make_paired_mgr(n_pairs, strategy="least_tokens", session_affinity=False):
    """A DisaggCoreManager with only its routing/socket state, no processes.

    Driven through the REAL inherited `add_request`, so these tests fail if the
    paired topology ever stops sharing the base router -- which is exactly the
    regression they exist for: a private copy of the dispatch loop quietly lost
    hints, session affinity, token-aware placement and the per-add log.
    """
    from atom.model_engine.engine_core_mgr import DisaggCoreManager

    mgr = DisaggCoreManager.__new__(DisaggCoreManager)
    mgr.label = "DisaggCoreManager"
    mgr.pp_size = 1
    mgr._n_pairs = n_pairs
    mgr._decode_idx0 = n_pairs
    mgr.local_engine_count = 2 * n_pairs
    mgr.global_engine_count = 2 * n_pairs
    mgr._dp_lb_strategy = strategy
    mgr._dp_lb_req_equiv = 512
    mgr._dp_session_affinity_enabled = session_affinity
    mgr._dp_session_owners = {}
    mgr._dp_session_prompt_tokens = {}
    mgr._dp_route_counters = {
        "affinity_new_total": 0,
        "affinity_owner_hit_total": 0,
        "affinity_spill_total": 0,
        "affinity_parent_ignored_total": 0,
        "explicit_total": 0,
        "load_balanced_total": 0,
    }
    mgr._rank_routed_total = [0] * (2 * n_pairs)
    mgr._rank_rotation_cursor = 0
    mgr._rank_reqs = [0] * (2 * n_pairs)
    mgr._rank_tokens = [0] * (2 * n_pairs)
    mgr._seq_load = {}
    mgr._lb_lock = Lock()
    mgr._seq_id_to_callback = {}
    mgr.engine_core_identities = [f"e{i}".encode() for i in range(2 * n_pairs)]
    mgr.input_sockets = [_RecordingSocket() for _ in range(2 * n_pairs)]
    return mgr


def _added(mgr, idx):
    """Sequence ids delivered to engine index `idx`, in order."""
    out = []
    for _identity, payload in mgr.input_sockets[idx].messages:
        request_type, seqs = pickle.loads(payload)
        assert request_type.name == "ADD"
        out.extend(seqs)
    return out


class TestPairedRequestRouting:
    """A sequence must reach BOTH members of exactly one pair, and no other.

    Sending it to a second decode rank would have two ranks allocate blocks for
    the same sequence; sending it to a second prefill rank would have two ranks
    write its KV into different GPUs' pools.
    """

    def test_sequence_goes_to_both_members_of_one_pair(self):
        mgr = _make_paired_mgr(8)
        # Make pair 3 the only attractive destination.
        for rank in range(8, 16):
            mgr._rank_tokens[rank] = 0 if rank == 11 else 10_000
        mgr.add_request([_PairedSeq("s", num_prompt_tokens=100)])

        got = [i for i in range(16) if mgr.input_sockets[i].messages]
        assert got == [3, 11]

    def test_routing_never_selects_a_prefill_index(self):
        """Prefill ranks are the other half of a pair, not a destination.

        They are indices 0..n_pairs-1 of the same engine list, so a router that
        did not know about the window would hand them requests directly.
        """
        mgr = _make_paired_mgr(4)
        mgr.add_request([_PairedSeq(i, num_prompt_tokens=i + 1) for i in range(12)])
        # Charged load lands only on decode indices.
        assert mgr._rank_reqs[:4] == [0, 0, 0, 0]
        assert sum(mgr._rank_reqs[4:]) == 12
        # Every pair got the same count on both of its sockets.
        for pair in range(4):
            assert len(_added(mgr, pair)) == len(_added(mgr, 4 + pair))

    def test_prefill_copy_is_capped_at_one_token(self):
        """Prefill keeps only the first sampled token, shipped in PrefillDone."""
        mgr = _make_paired_mgr(2)
        seq = _PairedSeq("s", num_prompt_tokens=10, max_tokens=256)
        mgr.add_request([seq])

        pair = next(p for p in range(2) if mgr.input_sockets[p].messages)
        (prefill_seq,) = _added(mgr, pair)
        (decode_seq,) = _added(mgr, 2 + pair)
        assert prefill_seq.max_tokens == 1
        # The decode copy -- and the caller's own object -- keep the real budget.
        assert decode_seq.max_tokens == 256
        assert seq.max_tokens == 256

    def test_placement_is_token_aware_not_request_count_aware(self):
        """The regression this sharing exists for.

        Equal request counts across pairs say nothing about equal work when
        prompts differ by an order of magnitude: the pair holding the long
        prompts becomes the straggler that sets wall time.
        """
        mgr = _make_paired_mgr(4, strategy="least_tokens")
        sizes = [100_000, 1_000, 1_000, 1_000, 1_000, 1_000, 1_000, 1_000]
        mgr.add_request(
            [_PairedSeq(i, num_prompt_tokens=n) for i, n in enumerate(sizes)]
        )
        decode_tokens = mgr._rank_tokens[4:]
        # The 100k prompt's pair takes no further work; the rest share the tail.
        assert max(decode_tokens) == 100_000
        assert sorted(decode_tokens)[:3] == [2_000, 2_000, 3_000]

    def test_explicit_hint_names_a_pair_not_an_engine_index(self):
        """`data_parallel_rank` is caller-facing: 0..n_pairs-1."""
        mgr = _make_paired_mgr(8)
        mgr.add_request([_PairedSeq("s", num_prompt_tokens=10, data_parallel_rank=3)])
        assert [i for i in range(16) if mgr.input_sockets[i].messages] == [3, 11]

    def test_out_of_range_hint_is_rejected_against_the_pair_count(self):
        mgr = _make_paired_mgr(8)
        # 8 is a valid engine index (decode rank 0) but not a valid pair.
        with pytest.raises(ValueError):
            mgr.add_request(
                [_PairedSeq("s", num_prompt_tokens=10, data_parallel_rank=8)]
            )
        assert not any(sock.messages for sock in mgr.input_sockets)
        assert mgr._seq_load == {}

    def test_session_affinity_keeps_later_turns_on_their_pair(self):
        """Agentic turns must come back to the pair whose prefix cache is warm."""
        mgr = _make_paired_mgr(4, session_affinity=True)
        mgr.add_request([_PairedSeq("t1", num_prompt_tokens=1_000, dp_session_id="c")])
        owner = mgr._dp_session_owners["c"]
        assert 4 <= owner < 8
        # Pile load onto the owner: affinity must still win.
        mgr._rank_tokens[owner] += 10_000_000
        mgr.add_request([_PairedSeq("t2", num_prompt_tokens=2_000, dp_session_id="c")])
        assert _added(mgr, owner)[-1].id == "t2"
        assert _added(mgr, owner - 4)[-1].id == "t2"

    def test_symmetric_routes_everything_to_pair_zero(self):
        """One pair: nothing to balance, but both members still get the seq."""
        mgr = _make_paired_mgr(1)
        mgr.add_request([_PairedSeq("s", num_prompt_tokens=10)])
        assert [i for i in range(2) if mgr.input_sockets[i].messages] == [0, 1]

    def test_every_pair_is_reachable(self):
        mgr = _make_paired_mgr(8, strategy="round_robin")
        mgr.add_request([_PairedSeq(i, num_prompt_tokens=10) for i in range(8)])
        assert all(mgr.input_sockets[i].messages for i in range(16))


class TestPrefillLockstep:
    """N prefill processes share a DP group, so their MoE collectives must match.

    Mirrors what DecodeEngineCore already does. At dp_size == 1 (symmetric)
    there is nothing to synchronise and the plain loop must be taken instead —
    DPEngineCoreProc's loop would all-reduce against a group that is None.
    """

    @staticmethod
    def _core(dp_group):
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.PrefillEngineCore)
        obj.dp_group = dp_group
        return obj, ec

    def test_paired_prefill_takes_the_dp_loop(self):
        obj, _ = self._core(dp_group=object())
        assert obj._dp_lockstep is True

    def test_symmetric_prefill_takes_the_plain_loop(self):
        obj, _ = self._core(dp_group=None)
        assert obj._dp_lockstep is False

    def test_prefill_is_a_dp_engine(self):
        ec = pytest.importorskip("atom.model_engine.engine_core")

        assert issubclass(ec.PrefillEngineCore, ec.DPEngineCoreProc)

    def test_neither_engine_uses_the_base_delayer_hook(self):
        """Both override it: prefill declines (admission is decode's call),
        decode defers until its scheduler exists after the kvcache import."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        base = ec.DPEngineCoreProc._maybe_attach_prefill_delayer
        assert ec.PrefillEngineCore._maybe_attach_prefill_delayer is not base
        assert ec.DecodeEngineCore._maybe_attach_prefill_delayer is not base

    def test_delayer_tolerates_no_dp_group(self):
        """cpu_group=None is a documented single-rank mode (prefill_delayer.py
        :19-24), so the attach must not be gated on having a DP group."""
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        src = inspect.getsource(ec.DPEngineCoreProc._maybe_attach_prefill_delayer)
        assert 'getattr(self, "dp_group", None) is None' not in src


class TestPairedRendezvousPorts:
    """Each side forms TWO groups, and they bind DIFFERENT config fields.

    data_parallel_base_port   -> ModelRunner worker group (model_runner.py:988)
    data_parallel_master_port -> EngineCore DP group (config.py:817-841)

    Prefill and decode must differ on both. Setting only the worker port was
    enough while prefill had no DP group; once both sides formed one, the
    master port stayed at its 29500 default and the second group to start died
    with EADDRINUSE.
    """

    @staticmethod
    def _ports(is_decode, *, worker_pair=(101, 102), dp_pair=(201, 202)):
        """Mirror of _pair_config's port assignment."""
        return {
            "base": worker_pair[1] if is_decode else worker_pair[0],
            "master": dp_pair[1] if is_decode else dp_pair[0],
        }

    def test_worker_group_ports_differ_between_sides(self):
        assert self._ports(False)["base"] != self._ports(True)["base"]

    def test_dp_group_ports_differ_between_sides(self):
        assert self._ports(False)["master"] != self._ports(True)["master"]

    def test_worker_and_dp_ports_are_distinct_within_a_side(self):
        for is_decode in (False, True):
            p = self._ports(is_decode)
            assert p["base"] != p["master"]

    def test_all_four_ports_are_unique(self):
        got = [self._ports(d)[k] for d in (False, True) for k in ("base", "master")]
        assert len(set(got)) == 4

    def test_dp_group_binds_master_not_base(self):
        """Guards the field mix-up itself: stateless_init_dp_group reads
        data_parallel_master_port, so setting only base_port leaves it on the
        29500 default."""
        import inspect

        cfg = pytest.importorskip("atom.config")
        src = inspect.getsource(cfg.ParallelConfig.get_next_dp_init_port)
        assert "data_parallel_master_port" in src
        assert "data_parallel_base_port" not in src


class TestPerPairChannels:
    """Every disagg channel must be per-pair once BOTH sides are multi-process.

    decode->prefill was a single shared address, which is correct with one
    prefill process (many PUSHers to one PULLer is native ZMQ). With N prefill
    processes each binding the same path, one wins and receives assignments for
    sequences the other N-1 are holding: those never prefill (lost) and their
    ranks spin on dummy batches (slow).
    """

    @staticmethod
    def _addr_fields():
        cfg = pytest.importorskip("atom.config")

        return {f.name for f in cfg.fields(cfg.Config) if "disagg_" in f.name}

    def test_d2p_is_a_per_pair_list(self):
        cfg = pytest.importorskip("atom.config")

        names = {f.name: f for f in cfg.fields(cfg.Config)}
        assert "disagg_d2p_addrs" in names, "must be plural/per-pair"
        assert "disagg_d2p_addr" not in names, "the shared singular must be gone"

    def test_every_disagg_channel_is_per_pair(self):
        """A singular disagg *_addr is the shape that misroutes; there should
        be none left."""
        singular = {
            n
            for n in self._addr_fields()
            if n.endswith("_addr") and not n.endswith("_addrs")
        }
        assert singular == set(), f"shared channels remain: {sorted(singular)}"

    def test_both_sides_index_the_same_slot(self):
        """prefill k binds d2p_addrs[k]; decode k connects d2p_addrs[k]."""
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        pre = inspect.getsource(ec.PrefillEngineCore.__init__)
        dec = inspect.getsource(ec.DecodeEngineCore.__init__)
        assert "config.disagg_d2p_addrs[k]" in pre
        assert "config.disagg_d2p_addrs[k]" in dec


class TestCuFractionTravelsInTheBatch:
    """Decode computes the CU split and ships it; there is no side channel.

    It used to go through 4 bytes of shared memory per pair, because prefill
    sized its own batches and needed decode's in-flight token count to pick a
    fraction. Decode plans both batches now, so it holds both operands at the
    moment it needs them and the answer rides in the message.
    """

    def test_message_carries_the_fraction(self):
        from atom.model_engine.disagg_types import PrefillBatchMsg

        assert "cu_stream_fraction" in PrefillBatchMsg.__dataclass_fields__

    def test_no_scheduler_mentions_shared_memory(self):
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        for cls in (sched.PrefillExecutor, sched.DecodeScheduler):
            src = inspect.getsource(cls)
            assert "shared_memory" not in src, cls.__name__
            assert "_cu_shm" not in src, cls.__name__

    def test_manager_allocates_no_shm(self):
        import inspect

        mgr = pytest.importorskip("atom.model_engine.engine_core_mgr")

        src = inspect.getsource(mgr.DisaggCoreManager)
        assert "SharedMemory" not in src
        assert "cu_shm" not in src

    def test_config_has_no_shm_field(self):
        cfg = pytest.importorskip("atom.config")

        assert not hasattr(cfg.Config, "disagg_cu_shm_name")

    def test_masking_still_gated_on_the_constrained_flag(self):
        """Emitting a fraction when the pool holds only the full-CU stream is
        a KeyError on the first forward, so the gate has to survive."""
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        src = inspect.getsource(sched.DecodeScheduler)
        assert 'self.cu_masking = bool(getattr(config, "disagg_constrained"' in src

        ec = pytest.importorskip("atom.model_engine.engine_core")

        step = inspect.getsource(ec.DecodeEngineCore._process_engine_step)
        assert "if self.scheduler.cu_masking" in step, (
            "emitting a fraction with no masked streams built is a KeyError on "
            "the first forward"
        )


class TestPrefillDelayerObservability:
    """A quiet delayer must not be indistinguishable from a dead one.

    The stats counter is per PROCESS and the periodic trigger is an exact
    multiple (default 1000). With N prefill ranks each sees ~total_prefills/N
    decisions, so a run with a few hundred prefills over 8 ranks logs nothing
    at all — which reads as "the delayer never ran".
    """

    @staticmethod
    def _delayer():
        pd = pytest.importorskip("atom.model_engine.prefill_delayer")

        import inspect
        import re

        obj = object.__new__(pd.PrefillDelayer)
        # Read the counter names off __init__ rather than listing them here.
        # A hardcoded list silently rots: main added `_stat_hold_decode_interval`
        # and `_maybe_log` read it, so every test through this helper died on an
        # AttributeError that had nothing to do with what it was testing.
        names = re.findall(
            r"self\.(_stat_\w+)\s*=", inspect.getsource(pd.PrefillDelayer.__init__)
        )
        assert names, "no _stat_* counters found on PrefillDelayer.__init__"
        for name in names:
            setattr(obj, name, 0)
        obj._stat_log_every = 1000
        return obj, pd

    def test_first_decision_is_logged(self, caplog):
        import logging as _log

        obj, pd = self._delayer()
        obj._stat_fire_nodecode = 1
        with caplog.at_level(_log.INFO, logger=pd.__name__):
            obj._maybe_log()
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[PrefillDelayer stats] total=1" in m for m in msgs)

    def test_still_silent_before_any_decision(self, caplog):
        import logging as _log

        obj, pd = self._delayer()
        with caplog.at_level(_log.INFO, logger=pd.__name__):
            obj._maybe_log()
        msgs = [r.getMessage() for r in caplog.records]
        assert not [m for m in msgs if "PrefillDelayer stats" in m]

    def test_periodic_trigger_still_fires(self, caplog):
        import logging as _log

        obj, pd = self._delayer()
        obj._stat_hold = 1000
        with caplog.at_level(_log.INFO, logger=pd.__name__):
            obj._maybe_log()
        msgs = [r.getMessage() for r in caplog.records]
        assert any("total=1000" in m for m in msgs)

    def test_disabled_when_log_every_is_zero(self, caplog):
        import logging as _log

        obj, pd = self._delayer()
        obj._stat_log_every = 0
        obj._stat_fire_fill = 1
        with caplog.at_level(_log.INFO, logger=pd.__name__):
            obj._maybe_log()
        msgs = [r.getMessage() for r in caplog.records]
        assert not [m for m in msgs if "PrefillDelayer stats" in m]


class TestDelayerLivesOnDecode:
    """There is ONE prefill gate and it is decode's, because decode builds
    the batch.

    Three things follow from decode owning the BlockManager:
      - holding the gate holds the KV allocation, so blocks are not pinned for
        a prefill that is not worth running yet;
      - the alignment that matters is across DECODE ranks, since each one
        dispatches to its own paired prefill rank;
      - decode has the inputs prefill never had: real KV usage, and a real
        decode batch to hide the hold behind.
    """

    def test_decode_gates_the_batch_build_on_the_delayer(self):
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        src = inspect.getsource(sched.DecodeScheduler.build_prefill_batch)
        assert "should_allow_prefill" in src
        assert "return None" in src, "a HOLD must dispatch nothing"

    def test_prefill_side_has_no_scheduling_at_all(self):
        """The executor chooses nothing: no delayer, no gate, no schedule()."""
        sched = pytest.importorskip("atom.model_engine.scheduler")

        for name in (
            "schedule",
            "_schedule",
            "should_fire_this_tick",
            "set_prefill_group_delayer",
            "decode_tokens",
        ):
            assert not hasattr(sched.PrefillExecutor, name), name

    def test_the_second_delayer_is_gone(self):
        """`PrefillGroupDelayer` was a per-rank barrier on prefill's own
        forward. Decode sizes the batch now, so the floor it enforced is just
        how the batch is built."""
        pd = pytest.importorskip("atom.model_engine.prefill_delayer")

        assert not hasattr(pd, "PrefillGroupDelayer")

    def test_prefill_engine_declines_the_delayer(self):
        ec = pytest.importorskip("atom.model_engine.engine_core")

        assert (
            ec.PrefillEngineCore._maybe_attach_prefill_delayer
            is not ec.DPEngineCoreProc._maybe_attach_prefill_delayer
        )

    def test_decode_uses_real_kv_usage_and_decode_batch(self):
        """The inputs prefill could not supply: it passed kv_usage=0.0, which
        disables the KV-high/KV-low release bounds entirely."""
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        src = inspect.getsource(sched.DecodeScheduler.build_prefill_batch)
        assert "kv_usage=self._kv_usage()" in src
        assert "running_decode_batch=len(self.running)" in src

    def test_an_inflight_batch_is_reported_not_skipped(self):
        """A busy rank must still enter the reduction.

        "One batch in flight" is a PER-RANK condition and decode ranks finish
        at different times, so returning before `should_allow_prefill` would
        split the all_reduce call count and hang the group. Reporting
        `prefillable=False` instead puts the condition inside the reduction,
        where the existing alignment gate re-synchronises dispatch -- which is
        also the only thing keeping the paired PREFILL ranks in step, since the
        two sides share no collective.
        """
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        src = inspect.getsource(sched.DecodeScheduler.build_prefill_batch)
        gate = src.index("should_allow_prefill")
        bail = src.index("if busy or not allow:")
        assert gate < bail, "the delayer must be consulted before bailing out"
        assert "prefillable = (not busy)" in src

    def test_decode_reuses_the_scheduler_delayer_helpers(self):
        """Not interchangeable with the obvious one-liners.

        _waiting_new_token_count skips sequences this rank could not admit;
        counting them trips the fill target before a real batch accumulates.
        _can_admit_head_prefill is stricter than a non-empty queue, which under
        a burst is True on every rank so the delayer never engages.
        _oldest_waiting_prefill_age_ms is the TTFT SLA guard.
        """
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        src = inspect.getsource(sched.DecodeScheduler.build_prefill_batch)
        for helper in (
            "_waiting_new_token_count()",
            "_can_admit_head_prefill()",
            "_oldest_waiting_prefill_age_ms()",
        ):
            assert helper in src, helper

    def test_both_call_sites_build_the_same_inputs(self):
        """Decode's call and the base scheduler's must not drift apart.

        Compared by INPUT SOURCE, not by exact kwarg spelling: the base site
        reaches these helpers through locals because of branches rapidserve
        does not have. What this pins is that neither side stops consulting one
        of the inputs altogether.
        """
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        decode = inspect.getsource(sched.DecodeScheduler.build_prefill_batch)
        base = inspect.getsource(sched.Scheduler._schedule)
        for source in (
            "self._can_admit_head_prefill()",
            "self._kv_usage()",
            "self._oldest_waiting_prefill_age_ms()",
        ):
            assert source in decode, f"decode dropped {source}"
            assert source in base, f"base dropped {source}"

    def test_partial_terms_are_real_on_this_side_now(self):
        """Chunked prefill is decode's, so a mid-prompt request is one of ours
        and its remainder belongs in the aggregate. The old admission-time call
        passed structural zeros here."""
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        src = inspect.getsource(sched.DecodeScheduler.build_prefill_batch)
        assert "self._partial_prefill_remaining_tokens()" in src
        assert "has_partial=bool(self.prefill_partial)" in src

    def test_the_partial_term_reads_this_side_s_queue(self):
        """The base helper walks `running` for `is_partial_prefill`, and a
        rapidserve partial is never there -- `running` is the DECODE queue and
        a mid-prompt request is not decodable. Inherited unchanged it returns 0
        always, silently dropping the partial term out of the delayer's
        aggregate so a rank chewing through a long chunked prompt looks idle to
        its peers."""
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        src = inspect.getsource(sched.DecodeScheduler._partial_prefill_remaining_tokens)
        assert "for seq in self.prefill_partial:" in src

    def test_the_partial_term_counts_real_remainders(self):
        from conftest import MockConfig

        sched = pytest.importorskip("atom.model_engine.scheduler")

        obj = sched.DecodeScheduler(MockConfig(max_num_batched_tokens=4096))
        assert obj._partial_prefill_remaining_tokens() == 0

        from atom.model_engine.sequence import Sequence
        from atom.sampling_params import SamplingParams

        seq = Sequence(list(range(40)), 4, sampling_params=SamplingParams())
        seq.num_cached_tokens = 16
        obj.prefill_partial.append(seq)
        assert obj._partial_prefill_remaining_tokens() == 24

    def test_execution_feedback_fires_on_completion_not_dispatch(self):
        """`notify_prefill_executed` arms post-prefill decode protection. A
        dispatch is not an execution -- the forward happens in another
        process -- so the true analogue of the baseline's call site is the
        batch result arriving."""
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        drain = inspect.getsource(sched.DecodeScheduler.drain_prefill_completions)
        build = inspect.getsource(sched.DecodeScheduler.build_prefill_batch)
        assert "notify_prefill_executed()" in drain
        assert "notify_prefill_executed" not in build

    def test_decode_attaches_after_its_scheduler_exists(self):
        """DecodeScheduler is built only after the kvcache import, so the
        attach inside super().__init__() has nothing to attach to."""
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        src = inspect.getsource(ec.DecodeEngineCore)
        build = src.index("self.scheduler = DecodeScheduler")
        attach = src.index("DPEngineCoreProc._maybe_attach_prefill_delayer(self")
        assert attach > build


class TestNoExtraKvReserveUnderRapidServe:
    """Sharing the GPU with a peer is not a reason to hold back KV.

    RapidServe used to reserve 8% of total HBM here -- 23GB on a 288GB card,
    for a stated concern of a 512MB NCCL barrier buffer -- and it was double
    counting. `non_torch` in the budget is `(total - free) - torch_reserved`,
    derived from the device's own numbers, so it already measures the peer
    process. Across a baseline and a rapidserve run on the same 8xMI355X box
    the two terms moved together and accounted for each other exactly:

        free       134.61 -> 118.39 GB   (-16.22, decode's footprint)
        non_torch   17.72 ->  33.94 GB   (+16.22, the same bytes)

    The reserve on top of that cost 23GB of KV: available_for_kv 87.94 -> 48.05
    and the pool fell from 64,924 blocks to 34,306, nearly half.
    """

    def test_rapidserve_reserves_nothing_extra(self):
        mr = pytest.importorskip("atom.model_engine.model_runner")

        assert (
            mr.RapidServeModelRunner._kv_budget_extra_reserve
            is mr.ModelRunner._kv_budget_extra_reserve
        ), "rapidserve must not override the reserve back in"

    def test_the_reserve_is_zero_whatever_the_card_size(self):
        """It used to scale with total HBM, so the bigger the card the more KV
        it gave away."""
        mr = pytest.importorskip("atom.model_engine.model_runner")

        for gb in (80, 192, 288, 512):
            assert (
                mr.ModelRunner._kv_budget_extra_reserve(None, gb * (1 << 30)) == 0
            ), gb


class TestOneGroupWideCuSplit:
    """One CU fraction, chosen at dispatch, held until the batch retires.

    The halves take COMPLEMENTARY masks off this one number -- prefill gets
    CUs `[0, split)` and decode `[split, total)` -- so a disagreement is not a
    missed optimisation: the masks overlap, decode runs on prefill's CUs, and
    the result is the contention masking exists to remove plus the cost of a
    mask.

    It used to be computed twice from different operands, once for the
    dispatched batch and once (gated on `prefill_finished`) for decode's own,
    neither reduced across ranks.
    """

    def test_the_reduction_uses_max_and_carries_four_slots(self):
        """MAX, not SUM. The two flags are 0/1 "any" predicates, for which
        `SUM > 0` and `MAX > 0` agree, and the two counts want the largest
        rank's value -- so one op serves all four and the group still needs
        only one collective per tick."""
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        fn = ast.parse(
            inspect.getsource(ec.DecodeEngineCore._sync_decode_group_state).lstrip()
        ).body[0]
        reduce_calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "all_reduce"
        ]
        assert len(reduce_calls) == 1, "one collective per tick, not two"
        op = next(k for k in reduce_calls[0].keywords if k.arg == "op")
        assert ast.unparse(op.value).endswith("MAX"), ast.unparse(op.value)

        assert len(ec.DecodeGroupState._fields) == 4
        assert ec.DecodeGroupState._fields == (
            "busy",
            "will_forward",
            "max_decode_tokens",
            "max_prefill_pending",
        )

    def test_the_split_is_decided_only_from_reduced_values(self):
        """Every operand must come off the reduction. A local one and the
        ranks disagree, which is the overlap this exists to prevent."""
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        fn = ast.parse(
            inspect.getsource(ec.DecodeEngineCore._process_engine_step).lstrip()
        ).body[0]
        call = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_optimal_cu_fraction"
        )
        for arg in call.args:
            text = ast.unparse(arg)
            assert text.startswith("group."), (
                f"{text!r} is not a reduced value; every rank must reach the "
                "same fraction or the two halves' masks overlap"
            )

    def test_the_split_is_not_recomputed_while_a_batch_is_in_flight(self):
        """The candidate is chosen on a free tick and remembered, so the batch
        dispatched under it and every decode forward that overlaps it use one
        number. Recomputing mid-flight lets them drift apart."""
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        fn = ast.parse(
            inspect.getsource(ec.DecodeEngineCore._process_engine_step).lstrip()
        ).body[0]
        guards = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.UnaryOp)
            and isinstance(n.test.op, ast.Not)
            and isinstance(n.test.operand, ast.Name)
            and n.test.operand.id == "group_busy"
            and any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == "_optimal_cu_fraction"
                for c in ast.walk(n)
            )
        ]
        assert len(guards) == 1, (
            "the candidate must be chosen under `if not group_busy`, i.e. only "
            "on a tick with no batch already running under another fraction"
        )

    def test_decodes_mask_follows_the_group_not_this_rank(self):
        """Uniformity is a correctness requirement, not a preference.

        The decode ranks share an MoE all-to-all, so a rank that masked while
        a peer did not would wedge it -- exactly as the prefill ranks did when
        three that had drained their queues dummied on full CUs while five ran
        masked batches, and the group hung for six minutes.

        `group_busy` is reduced, so every rank gets the same answer. "Did I
        dispatch" is not.
        """
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        fn = ast.parse(
            inspect.getsource(ec.DecodeEngineCore._process_engine_step).lstrip()
        ).body[0]
        call = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "set_cu_fraction"
        )
        arg = ast.unparse(call.args[0])
        assert "group_busy" in arg, (
            f"decode's mask is keyed on {arg!r}; it must come off the reduced "
            "group state so every rank agrees"
        )
        assert "prefill_batch" not in arg, "a local dispatch outcome diverges"


    def test_there_is_exactly_one_decision_site(self):
        sched = pytest.importorskip("atom.model_engine.scheduler")
        ec = pytest.importorskip("atom.model_engine.engine_core")
        import inspect

        in_scheduler = inspect.getsource(sched.DecodeScheduler).count(
            "_optimal_cu_fraction("
        )
        assert in_scheduler == 0, (
            "the scheduler must not compute its own fraction; it reads the "
            "one the group agreed on"
        )
        in_engine = inspect.getsource(ec.DecodeEngineCore).count(
            "_optimal_cu_fraction("
        )
        assert in_engine == 1

    def test_build_only_stamps_the_message(self):
        """The scheduler stamps; the engine decides what this process runs on.
        Two owners is how the two drifted apart in the first place."""
        import ast
        import inspect

        from conftest import MockConfig

        sched = pytest.importorskip("atom.model_engine.scheduler")

        fn = ast.parse(
            inspect.getsource(sched.DecodeScheduler.build_prefill_batch).lstrip()
        ).body[0]
        assert not [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "set_cu_fraction"
        ], "build_prefill_batch must not install a fraction"

        from atom.model_engine.sequence import Sequence
        from atom.sampling_params import SamplingParams

        obj = sched.DecodeScheduler(MockConfig(disagg_constrained=True))
        obj.waiting.append(
            Sequence([1, 2, 3, 4], 4, sampling_params=SamplingParams())
        )
        msg = obj.build_prefill_batch(cu_fraction=0.5)
        assert msg.cu_stream_fraction == 0.5

    def test_nothing_in_flight_means_no_mask(self):
        """A delayer hold dispatches nothing, so there is no prefill on the
        other half for decode to make room for. Keyed on pending work instead,
        decode sat on half the device for the whole hold."""
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        fn = ast.parse(
            inspect.getsource(ec.DecodeEngineCore._process_engine_step).lstrip()
        ).body[0]
        call = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "set_cu_fraction"
        )
        expr = call.args[0]
        assert isinstance(expr, ast.IfExp), ast.unparse(expr)
        assert isinstance(expr.orelse, ast.Constant) and expr.orelse.value is None, (
            "with no batch in flight the mask must come off"
        )






    def test_decode_does_not_clear_the_fraction_on_its_own(self):
        """`_schedule`'s early returns used to null it, which would drop the
        mask while a prefill was still running on the complementary half."""
        import ast
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        fn = ast.parse(
            inspect.getsource(sched.DecodeScheduler._schedule).lstrip()
        ).body[0]
        writes = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "cu_fraction"
                for t in n.targets
            )
        ]
        assert not writes, "only set_cu_fraction may write the fraction"


class TestEveryPrefillRankMeetsOnTheSameStream:
    """The prefill ranks share one MoE all-to-all per step, so they must all
    enter it on the same stream.

    The fraction reaches this process inside a `PrefillBatchMsg`, and a rank
    with no batch has no message -- so a rank that had drained its queue
    dummied on full CUs while its peers ran masked. Observed on 8 ranks:
    batch 198 dispatched by 5 of them, ranks 4/5/7 with `running=0 waiting=0`,
    and `dummy_inflight=356.4s` on exactly those three.
    """

    def test_the_reduction_precedes_every_early_return(self):
        """Same rule as the decode group's: gloo matches collectives by
        arrival order, so a rank that skips this pairs its next
        `_sync_dp_state` against a peer's reduction here."""
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        fn = ast.parse(
            inspect.getsource(ec.PrefillEngineCore._process_engine_step).lstrip()
        ).body[0]
        sync = min(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_sync_prefill_cu_fraction"
        )
        first_return = min(
            n.lineno for n in ast.walk(fn) if isinstance(n, ast.Return)
        )
        assert sync < first_return

    def test_it_reduces_with_max(self):
        """One rank with a masked batch must pull everyone onto that mask."""
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        fn = ast.parse(
            inspect.getsource(ec.PrefillEngineCore._sync_prefill_cu_fraction).lstrip()
        ).body[0]
        call = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "all_reduce"
        )
        op = next(k for k in call.keywords if k.arg == "op")
        assert ast.unparse(op.value).endswith("MAX")

    def test_it_round_trips_a_fraction_without_a_group(self):
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.PrefillEngineCore)
        obj.dp_group = None
        assert obj._sync_prefill_cu_fraction(0.5) == 0.5
        assert obj._sync_prefill_cu_fraction(None) is None

    def test_it_fails_open_to_no_mask(self, caplog, monkeypatch):
        """An unmasked group is slower than a masked one, never wedged.

        The raise is the RuntimeError gloo actually produces -- the same one
        `_sync_dp_state` catches -- not whatever a bare stub happens to throw.
        """
        import logging

        import torch

        ec = pytest.importorskip("atom.model_engine.engine_core")

        def _boom(*a, **kw):
            raise RuntimeError("peer down")

        monkeypatch.setattr(torch.distributed, "all_reduce", _boom)

        obj = object.__new__(ec.PrefillEngineCore)
        obj.dp_group = object()
        obj.label = "PrefillEngineCore"
        with caplog.at_level(logging.WARNING, logger="atom"):
            got = obj._sync_prefill_cu_fraction(0.5)
        assert got is None
        assert "running unmasked" in caplog.text

    def test_the_dummy_runs_on_the_groups_fraction(self):
        """The dummy exists to match its peers' collective; a masked stream
        and an unmasked one are not the same place to meet."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.PrefillEngineCore)
        obj._group_cu_fraction = 0.5
        assert obj._dummy_cu_fraction() == 0.5

    def test_the_base_engine_masks_nothing(self):
        """Every engine that does not partition CUs keeps the plain stream."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DPEngineCoreProc)
        assert obj._dummy_cu_fraction() is None

    def test_the_dummy_batch_carries_it(self):
        """`dummy_execution` built the batch with no fraction at all, so even
        a uniformly masked group would have met a dummy on full CUs."""
        import ast
        import inspect

        mr = pytest.importorskip("atom.model_engine.model_runner")

        fn = ast.parse(
            inspect.getsource(mr.ModelRunner.dummy_execution).lstrip()
        ).body[0]
        assert "cu_stream_fraction" in {a.arg for a in fn.args.args}
        call = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "ScheduledBatch"
        )
        assert any(k.arg == "cu_stream_fraction" for k in call.keywords)

    def test_a_real_batch_is_overridden_with_the_group_fraction(self):
        """A rank whose own batch says 0.5 and whose peers all say None has to
        run unmasked, or it is alone on half the device inside their
        collective."""
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        src = inspect.getsource(ec.PrefillEngineCore._process_engine_step)
        assert "scheduled_batch.cu_stream_fraction = self._group_cu_fraction" in src


class TestMaskingNeedsBothSides:
    """The split is complementary, so whichever side is idle is holding CUs
    the other could use. Masking only pays when both halves have work."""

    @staticmethod
    def _f(decode, prefill):
        sched = pytest.importorskip("atom.model_engine.scheduler")

        return sched._optimal_cu_fraction(decode, prefill)

    def test_no_decode_means_no_mask(self):
        """The expensive direction, and the common one.

        An agentic warmup runs for many minutes with `Running: 0` on every
        decode rank -- every request is still mid-prefill. Masking there halves
        prefill throughput for the whole phase, and prefill is the only thing
        happening.
        """
        assert self._f(0, 8) is None
        assert self._f(0, 16384) is None

    def test_no_prefill_means_no_mask(self):
        assert self._f(7, 0) is None

    def test_neither_means_no_mask(self):
        assert self._f(0, 0) is None

    def test_both_sides_busy_masks(self):
        assert self._f(7, 8) == 0.5

    def test_whatever_it_returns_is_buildable(self):
        """A fraction with no masked stream behind it falls back to full CUs
        with an error -- correct, but it means the mask silently did nothing."""
        envs = pytest.importorskip("atom.utils.envs")

        for decode, prefill in ((7, 8), (12, 16384), (1, 1)):
            f = self._f(decode, prefill)
            assert f is None or f in envs.ATOM_CU_POOL_FRACTIONS

    def test_an_idle_decode_rank_dispatches_prefill_unmasked(self):
        """End to end, not just the table: no decode anywhere, so the batch
        goes out on full CUs."""
        from conftest import MockConfig

        from atom.model_engine.scheduler import (
            DecodeScheduler,
            _optimal_cu_fraction,
        )
        from atom.model_engine.sequence import Sequence
        from atom.sampling_params import SamplingParams

        sched = DecodeScheduler(MockConfig(disagg_constrained=True))
        sched.waiting.append(
            Sequence(list(range(8)), 4, sampling_params=SamplingParams())
        )
        assert sched.decode_tokens_estimate() == 0

        msg = sched.build_prefill_batch(
            cu_fraction=_optimal_cu_fraction(
                sched.decode_tokens_estimate(), sched.prefill_pending_estimate()
            )
        )
        assert msg is not None
        assert msg.cu_stream_fraction is None, (
            "prefill is masked to half the device with nothing to decode on "
            "the other half"
        )


class TestDecodeTokenOperand:
    """The operand the split is chosen from has to be decode TOKENS."""

    @staticmethod
    def _sched(mtp):
        from conftest import MockConfig

        from atom.model_engine.scheduler import DecodeScheduler

        class _Spec:
            num_speculative_tokens = mtp

            @staticmethod
            def use_dspark():
                return True

        return DecodeScheduler(
            MockConfig(speculative_config=_Spec() if mtp else None)
        )

    @pytest.mark.parametrize("mtp", [0, 3])
    def test_the_estimate_matches_what_schedule_produces(self, mtp):
        """It is read before the batch exists, so it is an estimate -- but it
        must be the same number `_schedule` goes on to report, or the split is
        chosen from a quantity nothing else uses."""
        from atom.model_engine.sequence import Sequence
        from atom.sampling_params import SamplingParams

        sched = self._sched(mtp)
        for _ in range(3):
            sq = Sequence([1, 2, 3, 4], 4, sampling_params=SamplingParams())
            sched.block_manager.allocate(sq)
            sched.prefill_done.append(sq)

        # Before `schedule()`, which is when the engine actually reads it --
        # measuring after lets the promotion hide a queue the estimate misses.
        estimate = sched.decode_tokens_estimate()
        batch, _ = sched.schedule()
        assert estimate == batch.total_tokens_num_decode

    def test_it_is_not_the_running_context_length(self):
        """What the old call site passed. On agentic traffic that is ~186k
        tokens per request against a true decode count of `mtp_k + 1` -- four
        orders of magnitude out, invisible only because the lookup table does
        not read this operand yet."""
        from atom.model_engine.sequence import Sequence
        from atom.sampling_params import SamplingParams

        sched = self._sched(0)
        sq = Sequence(list(range(4096)), 4, sampling_params=SamplingParams())
        sched.running.append(sq)
        assert sched.decode_tokens_estimate() == 1
        assert sched.decode_tokens_estimate() != sq.num_tokens


class TestCuPoolFractions:
    def test_default_is_the_single_split_it_replaced(self):
        envs = pytest.importorskip("atom.utils.envs")

        assert envs.ATOM_CU_POOL_FRACTIONS == [0.5]

    @pytest.mark.parametrize("bad", ["0", "1.0", "-0.5", "abc", "0.5,2"])
    def test_values_outside_the_open_unit_interval_are_refused(
        self, bad, monkeypatch
    ):
        """`_cu_mask_for_fraction` clamps the split to [1, total-1] CUs, so 0.0
        and 1.0 do not mean "no mask" -- they silently become "one CU" and
        "all but one", a pathological partition that looks like a working
        one."""
        import importlib

        envs = pytest.importorskip("atom.utils.envs")

        monkeypatch.setenv("ATOM_CU_POOL_FRACTIONS", bad)
        with pytest.raises(ValueError):
            importlib.reload(envs).ATOM_CU_POOL_FRACTIONS

    def test_values_are_sorted_and_deduplicated(self, monkeypatch):
        """The pool's key order has to be stable across ranks, and a repeated
        value must not build the same masked stream twice."""
        import importlib

        envs = pytest.importorskip("atom.utils.envs")

        monkeypatch.setenv("ATOM_CU_POOL_FRACTIONS", "0.7, 0.3,0.5,0.3")
        assert importlib.reload(envs).ATOM_CU_POOL_FRACTIONS == [0.3, 0.5, 0.7]

    def test_an_unpooled_fraction_degrades_instead_of_raising(self, caplog):
        """A bare dict index here is a KeyError on the first forward, after
        model load and graph capture. The decision path should make it
        impossible; this turns "should" into a log line."""
        import logging

        mr = pytest.importorskip("atom.model_engine.model_runner")

        sentinel = object()
        pool = {None: sentinel}
        with caplog.at_level(logging.ERROR, logger="atom"):
            got = mr.RapidServeModelRunner._cu_stream(pool, 0.25, "prefill")
        assert got is sentinel
        assert "no entry for cu_stream_fraction" in caplog.text

    def test_a_pooled_fraction_is_returned_without_complaint(self, caplog):
        import logging

        mr = pytest.importorskip("atom.model_engine.model_runner")

        masked, plain = object(), object()
        with caplog.at_level(logging.ERROR, logger="atom"):
            got = mr.RapidServeModelRunner._cu_stream(
                {None: plain, 0.5: masked}, 0.5, "decode"
            )
        assert got is masked
        assert caplog.text == ""


class TestUnconstrainedEmitsOnlyTheNoneStream:
    """The pool's sole key in unconstrained mode is None, so that is the only
    fraction either side may put on a batch.

    Emitting 0.5 against a pool built without the fractional entries is a
    KeyError on the first forward, which is how this was found.
    """

    def test_only_none_key_is_emitted_when_unconstrained(self):
        import inspect

        mr_src = inspect.getsource(
            pytest.importorskip("atom.model_engine.model_runner").RapidServeModelRunner
        )
        # the fractional entries are gated; the None entry is unconditional
        assert 'if getattr(self.config, "disagg_constrained", False):' in mr_src
        assert "self._decode_streams[None] = torch.cuda.Stream()" in mr_src

    def test_decode_leaves_the_fraction_none_when_unconstrained(self):
        from atom.model_engine.scheduler import DecodeScheduler

        sched = DecodeScheduler(MockConfig(disagg_constrained=False))
        assert sched.cu_masking is False
        assert sched.cu_fraction is None


class TestDelayerDoesNotSpinWhenIdle:
    """The delayer must not tick on an idle cluster.

    busy_loop calls pull_and_process_input_queue EVERY iteration, before the
    all-reduced work check. Ticking the delayer from there spun every decode
    rank through a cross-DP all_reduce at full speed with nothing to do —
    150k decisions in three minutes, all fire_vacuous, which starved the
    process and made the server look hung. Admission belongs in
    _process_engine_step, which busy_loop reaches only when some rank has work.
    """

    def test_admission_is_not_in_the_unconditional_input_path(self):
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        src = inspect.getsource(ec.DecodeEngineCore.pull_and_process_input_queue)
        assert "build_prefill_batch" not in src

    def test_admission_is_in_the_work_gated_step(self):
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        src = inspect.getsource(ec.DecodeEngineCore._process_engine_step)
        assert "build_prefill_batch" in src

    def test_admission_precedes_every_early_return(self):
        """Every rank must reach the delayer's collective on the same
        iterations, so nothing may return before the batch build that ticks it.

        Checked on the AST: the comments around this code name the very
        constructs being ordered, so a substring search matches prose.
        """
        import ast
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        fn = ast.parse(
            inspect.getsource(ec.DecodeEngineCore._process_engine_step).lstrip()
        ).body[0]
        build = min(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "build_prefill_batch"
        )
        first_return = min(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Return))
        assert build < first_return

    def test_busy_loop_gates_the_step_on_an_all_reduced_flag(self):
        """What makes the entry condition identical on every rank."""
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        src = inspect.getsource(ec.DPEngineCoreProc.busy_loop)
        assert "global_has_unfinished" in src
        assert "self._process_engine_step()" in src


class TestMoriAll2AllBypass:
    """EP must be runnable without MoRI's all-to-all.

    MoRI misbehaves when two expert-parallel groups share a GPU, which is what
    rapidserve creates: a prefill process and a decode process per device, each
    with its own EP group and its own symmetric heap. Measured — DPA+EP alone
    is correct, rapidserve+EP with dp_size=1 (so no all-to-all) is correct, and
    only rapidserve+DPA+EP, the sole two-groups-per-GPU case, is wrong.

    The bypass routes the MoE through the DP all_gather/reduce path instead
    (moe.py "mode 3"): every rank sees every token rather than only its own
    experts' tokens. Slower, same result.
    """

    @staticmethod
    def _cfg(dp_size=8, use_ep=True):
        moe = pytest.importorskip("atom.model_ops.moe")

        obj = object.__new__(moe.FusedMoEParallelConfig)
        object.__setattr__(obj, "dp_size", dp_size)
        object.__setattr__(obj, "use_ep", use_ep)
        return obj

    def test_all2all_on_by_default_for_ep(self, monkeypatch):
        moe = pytest.importorskip("atom.model_ops.moe")
        envs = pytest.importorskip("atom.utils.envs")

        monkeypatch.setattr(envs, "ATOM_DISABLE_MORI_ALL2ALL", False)
        if not moe._has_module("mori"):
            pytest.skip("mori not installed")
        assert self._cfg().use_all2all_kernels is True

    def test_bypass_disables_it(self, monkeypatch):
        envs = pytest.importorskip("atom.utils.envs")
        pytest.importorskip("atom.model_ops.moe")

        monkeypatch.setattr(envs, "ATOM_DISABLE_MORI_ALL2ALL", True)
        assert self._cfg().use_all2all_kernels is False

    def test_bypass_does_not_resurrect_it_for_non_ep(self, monkeypatch):
        """TP-sharded MoE never used the all-to-all; the knob must be inert."""
        envs = pytest.importorskip("atom.utils.envs")
        pytest.importorskip("atom.model_ops.moe")

        monkeypatch.setattr(envs, "ATOM_DISABLE_MORI_ALL2ALL", True)
        assert self._cfg(use_ep=False).use_all2all_kernels is False

    def test_single_rank_never_uses_all2all(self, monkeypatch):
        envs = pytest.importorskip("atom.utils.envs")
        pytest.importorskip("atom.model_ops.moe")

        monkeypatch.setattr(envs, "ATOM_DISABLE_MORI_ALL2ALL", False)
        assert self._cfg(dp_size=1).use_all2all_kernels is False

    def test_env_default_is_off(self):
        """Nothing changes for anyone who does not set it."""
        envs = pytest.importorskip("atom.utils.envs")

        assert envs.ATOM_DISABLE_MORI_ALL2ALL is False


class TestHashRoutingIdsGather:
    """V4 routes its first layers on a hash of input_ids, not on logits alone.

    When the MoE hands the gate DP-GATHERED gating_output, the ids must be
    gathered to match or `_hash_topk` trips:
        input_ids length 16384 does not match gating_output num_tokens 131072
    (131072 = 16384 * dp_size).

    The old condition was `enable_dp_attention and not enable_expert_parallel`,
    which assumed EP always means MoRI's all-to-all — and MoRI routes per-rank,
    so it never gathers. That assumption breaks two ways: the
    ATOM_DISABLE_MORI_ALL2ALL bypass, and a build with no mori module. Both put
    EP on the gather path with local ids.
    """

    @staticmethod
    def _need(dp_attn, ep, hash_layers=3, bypass=False, has_mori=True):
        """Mirror of DeepseekV4ForCausalLM._need_ids_gather."""
        moe_will_gather = not ep or (bypass or not has_mori)
        return dp_attn and moe_will_gather and hash_layers > 0

    def test_dpa_without_ep_gathers(self):
        """The original supported case — MoE mode 3, ids must follow."""
        assert self._need(dp_attn=True, ep=False) is True

    def test_dpa_with_ep_and_mori_does_not(self):
        """MoRI routes per-rank; gathering ids would be wrong, not just wasteful."""
        assert self._need(dp_attn=True, ep=True) is False

    def test_dpa_with_ep_and_bypass_gathers(self):
        """The regression: bypass puts EP on the gather path."""
        assert self._need(dp_attn=True, ep=True, bypass=True) is True

    def test_dpa_with_ep_and_no_mori_gathers(self):
        """Same shape of bug without any env var — a build lacking mori."""
        assert self._need(dp_attn=True, ep=True, has_mori=False) is True

    def test_no_dp_attention_never_gathers(self):
        """Nothing to gather across; must hold for every EP/bypass combination."""
        for ep in (False, True):
            for bypass in (False, True):
                assert self._need(False, ep, bypass=bypass) is False

    def test_model_without_hash_layers_never_gathers(self):
        assert self._need(dp_attn=True, ep=False, hash_layers=0) is False

    def test_condition_matches_use_all2all_kernels(self):
        """The two predicates must agree: ids gather exactly when the MoE does.

        use_all2all_kernels = dp>1 and use_ep and has_mori and not bypass
        gather             = dp attention and not use_all2all_kernels
        """
        for ep in (False, True):
            for bypass in (False, True):
                for has_mori in (False, True):
                    all2all = ep and has_mori and not bypass
                    assert self._need(True, ep, bypass=bypass, has_mori=has_mori) is (
                        not all2all
                    )


class TestAll2AllBackendSelection:
    """aiter hardcodes the EP all-to-all backend; ATOM has to override it.

    base_device_communicator.py:126 sets `all2all_backend = "mori"` with the
    config-driven selection commented out, and the manager is built lazily on
    first read of the property. So the override has to happen before that read,
    which is why it lives at the single read site in moe.py.

    Only "mori" and "flydsl" exist in this aiter build. flydsl is the useful
    one here: it replaces MoRI's dispatch/combine kernels but still allocates
    P2P buffers from MoRI's shmem heap, so it distinguishes "the kernels are
    at fault" from "the shared heap is".
    """

    class _Comm:
        def __init__(self, created=False):
            self.all2all_backend = "mori"
            self._all2all_manager_created = created
            self.reads = 0

        @property
        def all2all_manager(self):
            self.reads += 1
            self._all2all_manager_created = True
            return f"manager:{self.all2all_backend}"

    class _Group:
        def __init__(self, comm):
            self.device_communicator = comm

    def _resolve(self, comm, backend, monkeypatch):
        moe = pytest.importorskip("atom.model_ops.moe")
        envs = pytest.importorskip("atom.utils.envs")

        monkeypatch.setattr(envs, "ATOM_ALL2ALL_BACKEND", backend)
        return moe._resolve_all2all_manager(self._Group(comm))

    def test_empty_leaves_aiter_default(self, monkeypatch):
        comm = self._Comm()
        got = self._resolve(comm, "", monkeypatch)
        assert comm.all2all_backend == "mori"
        assert got == "manager:mori"

    def test_selects_flydsl(self, monkeypatch):
        comm = self._Comm()
        got = self._resolve(comm, "flydsl", monkeypatch)
        assert comm.all2all_backend == "flydsl"
        assert got == "manager:flydsl"

    def test_override_happens_before_the_manager_is_built(self, monkeypatch):
        """The whole point: the property caches, so a late set is a silent no-op."""
        comm = self._Comm()
        assert comm.reads == 0
        self._resolve(comm, "flydsl", monkeypatch)
        assert comm.reads == 1

    def test_warns_instead_of_lying_when_already_built(self, monkeypatch, caplog):
        import logging as _log

        comm = self._Comm(created=True)
        with caplog.at_level(_log.WARNING, logger="atom"):
            got = self._resolve(comm, "flydsl", monkeypatch)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("ignored" in m for m in msgs)
        assert got == "manager:mori", "must not claim a backend it did not get"

    def test_setting_the_current_backend_is_a_no_op(self, monkeypatch, caplog):
        import logging as _log

        comm = self._Comm(created=True)
        with caplog.at_level(_log.WARNING, logger="atom"):
            self._resolve(comm, "mori", monkeypatch)
        msgs = [r.getMessage() for r in caplog.records]
        assert not [m for m in msgs if "ignored" in m]

    def test_env_default_is_empty(self):
        envs = pytest.importorskip("atom.utils.envs")

        assert envs.ATOM_ALL2ALL_BACKEND == ""


class TestAll2AllKwargFiltering:
    """The all-to-all arg set is backend-specific.

    MoRI's _make_all2all_kwargs takes `gpu_per_node`; FlyDSL's does not, and
    passing it raises

        TypeError: FlyDSLAll2AllManager._make_all2all_kwargs() got an
        unexpected keyword argument 'gpu_per_node'

    Filtering against the signature rather than special-casing that one name
    keeps any future backend working, and can only ever drop keys the callee
    would have rejected anyway.
    """

    @staticmethod
    def _filter(manager, kwargs):
        moe = pytest.importorskip("atom.model_ops.moe")

        return moe._filter_all2all_kwargs(manager, kwargs)

    def test_drops_only_unaccepted_keys(self):
        class M:
            def _make_all2all_kwargs(self, rank, world_size):
                pass

        got = self._filter(M(), {"rank": 1, "world_size": 8, "gpu_per_node": 1})
        assert got == {"rank": 1, "world_size": 8}

    def test_keeps_everything_the_backend_accepts(self):
        class M:
            def _make_all2all_kwargs(self, rank, world_size, gpu_per_node):
                pass

        args = {"rank": 1, "world_size": 8, "gpu_per_node": 1}
        assert self._filter(M(), args) == args

    def test_var_keyword_backend_is_left_alone(self):
        """**kwargs accepts anything; filtering could only lose information."""

        class M:
            def _make_all2all_kwargs(self, rank, **kw):
                pass

        args = {"rank": 1, "anything": 2}
        assert self._filter(M(), args) == args

    def test_backend_without_the_hook_is_left_alone(self):
        class M:
            pass

        args = {"rank": 1, "gpu_per_node": 1}
        assert self._filter(M(), args) == args

    def test_missing_required_arg_is_not_papered_over(self):
        """Filtering must not invent defaults — a real mismatch still raises."""

        class M:
            def _make_all2all_kwargs(self, rank, world_size):
                pass

        got = self._filter(M(), {"rank": 1})
        assert got == {"rank": 1}
        with pytest.raises(TypeError):
            M()._make_all2all_kwargs(**got)

    def test_real_backends_differ_exactly_by_gpu_per_node(self):
        a2a = pytest.importorskip("aiter.dist.device_communicators.all2all")
        import inspect

        mori = set(
            inspect.signature(a2a.MoriAll2AllManager._make_all2all_kwargs).parameters
        )
        fly = set(
            inspect.signature(a2a.FlyDSLAll2AllManager._make_all2all_kwargs).parameters
        )
        assert mori - fly == {"gpu_per_node"}
        assert fly - mori == set()


class TestDecodeMoEAll2AllInit:
    """Decode must build its own MoE all-to-all path after the IPC import.

    `init_prepare_finalize` is reachable only from load_model's post-processing
    loop, and decode never calls load_model -- it builds on meta and imports
    prefill's weights. Without an explicit call its `fused_experts` stays None
    and the MoE silently degrades to a local-only fused_moe: under expert
    parallelism every token routed to another rank's experts contributes
    nothing, so prefill's first token is right and every decode token is wrong.
    """

    @staticmethod
    def _fake_moe_method():
        moe = pytest.importorskip("atom.model_ops.moe")

        class _FakeMoEMethod(moe.FusedMoEMethodBase):
            # The base is abstract; stub its interface so it can be built.
            # This test only exercises the init hook.
            def create_weights(self, *a, **kw):
                pass

            def apply(self, *a, **kw):
                pass

            def get_fused_moe_quant_config(self, layer):
                return None

            def __init__(self):
                self.fused_experts = None
                self.init_calls = 0

            def init_prepare_finalize(self, layer):
                self.init_calls += 1
                self.fused_experts = object()  # stands in for the modular kernel

        return _FakeMoEMethod()

    def _run(self, model, rank=0):
        from types import SimpleNamespace

        mr = pytest.importorskip("atom.model_engine.model_runner")
        runner = SimpleNamespace(model=model, rank=rank)
        mr.RapidServeModelRunner._init_moe_all2all_after_import(runner)
        return runner

    def test_builds_all2all_for_every_moe_layer(self):
        model = torch.nn.Module()
        methods = []
        for i in range(3):
            layer = torch.nn.Module()
            layer.quant_method = self._fake_moe_method()
            methods.append(layer.quant_method)
            model.add_module(f"moe{i}", layer)

        self._run(model)

        assert [m.init_calls for m in methods] == [1, 1, 1]
        assert all(m.using_modular_kernel for m in methods)

    def test_is_idempotent(self):
        """init_prepare_finalize asserts it runs once, so a second pass must skip."""
        model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.quant_method = self._fake_moe_method()
        model.add_module("moe", layer)

        self._run(model)
        self._run(model)

        assert layer.quant_method.init_calls == 1

    def test_ignores_non_moe_modules(self):
        model = torch.nn.Module()
        plain = torch.nn.Module()
        plain.quant_method = object()  # a non-MoE quant method
        model.add_module("linear", plain)
        model.add_module("bare", torch.nn.Module())

        self._run(model)  # must not raise

    def test_does_not_rerun_weight_post_processing(self):
        """Prefill exports post-processed weights; shuffling twice corrupts them."""
        model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.quant_method = self._fake_moe_method()
        called = []
        layer.process_weights_after_loading = lambda: called.append(1)
        model.add_module("moe", layer)

        self._run(model)

        assert called == []
        assert layer.quant_method.init_calls == 1


# ── engine-status line: the two P/D processes must be distinguishable ─────


def test_pd_schedulers_label_their_engine_lines(
    prefill_executor_unconstrained, decode_scheduler_unconstrained
):
    """Both P/D processes run as engine index 0 and usually log to the same
    place, so the line carries a label; the aggregated engine keeps none."""
    from atom.model_engine.scheduler import Scheduler

    assert prefill_executor_unconstrained.engine_stats.label == "Prefill "
    assert decode_scheduler_unconstrained.engine_stats.label == "Decode "
    assert Scheduler(MockConfig()).engine_stats.label == ""


def test_decode_request_counts_fold_in_the_pd_queues(
    decode_scheduler_unconstrained, seq_factory
):
    """A decode engine at full load parks its requests in `prefill_inflight`
    / `prefill_partial` / `prefill_done`. Counting only the base pair reports
    an idle engine."""
    sched = decode_scheduler_unconstrained
    assert sched.get_request_counts() == (0, 0)

    inflight = seq_factory([1, 2, 3, 4])
    partial = seq_factory([9, 10, 11, 12])
    ready = seq_factory([5, 6, 7, 8])
    sched.prefill_inflight[inflight.id] = inflight
    sched.prefill_partial.append(partial)
    sched.prefill_done.append(ready)

    running, waiting = sched.get_request_counts()
    assert waiting == 2, "prefill_inflight and prefill_partial count as waiting"
    assert running == 1, "prefill_done must count as running"


def test_decode_metrics_and_status_line_agree(
    decode_scheduler_unconstrained, seq_factory
):
    """The status line and `/metrics` must not describe the same engine
    differently. `collect_metrics` reads `get_request_counts`, so overriding
    anything else would leave the dashboard reporting an idle engine while the
    log says it is at full load."""
    sched = decode_scheduler_unconstrained
    for i in range(3):
        sched.prefill_inflight[i] = seq_factory([1, 2, 3, 4])
    sched.prefill_done.append(seq_factory([5, 6, 7, 8]))

    running, waiting = sched.get_request_counts()
    assert (running, waiting) == (1, 3)
    # Everything else that counts in-flight work derives from that one pair.
    assert sched.get_num_unfinished_requests() == 4
    assert sched.has_requests() is True
    assert sched.has_unfinished_requests() is True


def test_metrics_work_on_the_prefill_executor():
    """`PrefillEngineCore` swaps in a `PrefillExecutor` and rewires the
    utility handler to it, so `collect_metrics` has to survive a scheduler
    that is not a `Scheduler` subclass and owns no BlockManager — it used to
    raise AttributeError on `block_manager.kv` inside the busy loop."""
    import queue

    from atom.model_engine.engine_utility import EngineUtilityHandler
    from atom.model_engine.scheduler import PrefillExecutor

    sched = PrefillExecutor(MockConfig(enable_log_stats=True))
    handler = EngineUtilityHandler(
        runner_mgr=None,
        output_queue=queue.Queue(),
        label="PrefillEngineCore",
        scheduler=sched,
    )

    metrics = handler.collect_metrics()
    assert metrics["enabled"] is True
    assert metrics["requests_running"] == 0
    # No KV pool on this side (decode owns the blocks), so the snapshot omits
    # those keys rather than reporting a pool of size zero. The aggregator
    # sums with `.get(key, 0)`, so the decode rank's real figures still land.
    assert not [k for k in metrics if k.startswith("kv_blocks")]

    handler.push_metrics()  # must not raise from inside the busy loop


def test_idle_heartbeat_is_available_on_every_scheduler(caplog):
    """The busy loop calls this on every idle pass, whichever scheduler the
    engine core is driving, so all three must accept it, close the window,
    and stay silent while nothing is running."""
    import logging
    import time

    from atom.model_engine.scheduler import (
        DecodeScheduler,
        PrefillExecutor,
        Scheduler,
    )

    cfg = MockConfig(enable_log_stats=True)
    for sched in (
        PrefillExecutor(cfg),
        DecodeScheduler(cfg),
        Scheduler(cfg),
    ):
        stats = sched.engine_stats
        before = stats._throughput_last_log_time
        sched.heartbeat_throughput(time.monotonic())
        assert stats._throughput_last_log_time == before, "window was not due"

        # Force the window open on an engine with nothing running or queued:
        # it must close (fresh start) without logging.
        stats._throughput_last_log_time -= 43.0
        with caplog.at_level(logging.INFO, logger="atom"):
            sched.heartbeat_throughput(time.monotonic())
        assert time.monotonic() - stats._throughput_last_log_time < 1.0
        assert "Engine" not in caplog.text
        caplog.clear()


def test_idle_heartbeat_is_inert_when_log_stats_off():
    import time

    from atom.model_engine.scheduler import Scheduler

    sched = Scheduler(MockConfig(enable_log_stats=False))
    sched.engine_stats._throughput_last_log_time -= 43.0
    before = sched.engine_stats._throughput_last_log_time
    sched.heartbeat_throughput(time.monotonic())
    assert sched.engine_stats._throughput_last_log_time == before


def test_decode_schedule_records_throughput_when_log_stats_on(seq_factory):
    """DecodeScheduler overrides schedule() without calling super(), so its
    throughput wiring is separate code — exercise it with the production
    default (log stats ON) on both the empty and the scheduled path."""
    from atom.model_engine.scheduler import DecodeScheduler

    sched = DecodeScheduler(MockConfig(enable_log_stats=True))
    assert sched.engine_stats.throughput_enabled is True

    # Empty path: `running` is empty, so schedule() takes the early return.
    # Must still tick the cadence rather than raise.
    assert sched.schedule() is None

    # Scheduled path: a promoted seq goes through the full return.
    seq = seq_factory([1, 2, 3, 4])
    sched.prefill_done.append(seq)
    batch, seqs = sched.schedule()
    assert batch is not None
    assert seq.id in seqs


# ── /metrics does not count a P/D request twice ────────────────────────────


def _aggregate(*snapshots):
    """Run the real aggregator over hand-built rank snapshots.

    `get_metrics_statistics` only reads `core_mgr.latest_metrics`, so a stub
    with that attribute exercises the actual aggregation code without an
    engine, GPUs, or a running EngineCore.
    """
    from types import SimpleNamespace

    from atom.model_engine.llm_engine import LLMEngine

    engine = SimpleNamespace(
        core_mgr=SimpleNamespace(
            latest_metrics=dict(enumerate(snapshots)),
            get_dp_router_statistics=dict,
        )
    )
    return LLMEngine.get_metrics_statistics(engine)


def _snapshot(role, running, waiting, **extra):
    return {
        "enabled": True,
        "role": role,
        "requests_running": running,
        "requests_waiting": waiting,
        **extra,
    }


def test_snapshots_carry_their_pd_role():
    """The aggregation below keys off `role`, so the snapshot has to carry it.

    Half the fix lives in `collect_metrics`; without this the aggregator could
    be perfectly correct and still see `role` missing from every rank, quietly
    falling back to the summing path that double-counts.
    """
    import queue

    from atom.model_engine.engine_utility import EngineUtilityHandler
    from atom.model_engine.scheduler import (
        DecodeScheduler,
        PrefillExecutor,
        Scheduler,
    )

    def role_of(sched):
        handler = EngineUtilityHandler(
            runner_mgr=None,
            output_queue=queue.Queue(),
            label="test",
            scheduler=sched,
        )
        return handler.collect_metrics()["role"]

    assert role_of(PrefillExecutor(MockConfig())) == "prefill"
    assert role_of(DecodeScheduler(MockConfig())) == "decode"
    # The aggregated (non-P/D) engine is neither half of a pair.
    assert role_of(Scheduler(MockConfig())) == ""


def test_pd_queue_depths_are_not_double_counted():
    """One in-flight request is held by both P/D ranks at the same instant —
    the prefill rank's `running` and the decode rank's `prefill_inflight`. The
    decode side's queues already span the whole lifetime, so summing the pair
    reports more requests than exist."""
    metrics = _aggregate(
        _snapshot("prefill", running=6, waiting=2),
        _snapshot("decode", running=3, waiting=6),
    )
    assert metrics["requests_running"] == 3
    assert metrics["requests_waiting"] == 6


def test_cumulative_counters_still_sum_across_the_pd_pair():
    """Only queue depths are deduplicated. Lifetime counters are written on
    one side only, so dropping the prefill rank from *those* would lose real
    numbers rather than remove a duplicate."""
    metrics = _aggregate(
        _snapshot("prefill", running=6, waiting=2, requests_finished=0, preemptions=1),
        _snapshot("decode", running=3, waiting=6, requests_finished=40, preemptions=2),
    )
    assert metrics["requests_finished"] == 40
    assert metrics["preemptions"] == 3


def test_queue_depths_sum_when_no_rank_is_a_decode_rank():
    """Two DP ranks of an aggregated engine. Neither is half of a pair, so
    their queues are disjoint and must add up."""
    metrics = _aggregate(
        _snapshot("", running=4, waiting=1),
        _snapshot("", running=5, waiting=2),
    )
    assert metrics["requests_running"] == 9
    assert metrics["requests_waiting"] == 3


def test_prefill_only_window_still_reports_its_queues():
    """Between the prefill rank's first metrics push and the decode rank's,
    the prefill counts are all there is — and nothing is duplicating them yet,
    so reporting 0 here would blank the dashboard instead of deduplicating."""
    metrics = _aggregate(_snapshot("prefill", running=6, waiting=2))
    assert metrics["requests_running"] == 6
    assert metrics["requests_waiting"] == 2
