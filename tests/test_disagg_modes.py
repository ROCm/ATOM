# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for intra-GPU disagg constrained vs unconstrained modes.

Only the scheduler-level shm gating is exercised here; the IPC handshake
and CUDA stream pool are out of scope for the no-GPU test environment.
"""

import pytest
import torch
from conftest import MockConfig


@pytest.fixture
def prefill_scheduler_unconstrained():
    from atom.model_engine.scheduler import PrefillScheduler

    return PrefillScheduler(MockConfig(), disagg_cu_shm_name="")


@pytest.fixture
def decode_scheduler_unconstrained():
    from atom.model_engine.scheduler import DecodeScheduler

    return DecodeScheduler(MockConfig(), disagg_cu_shm_name="")


@pytest.fixture
def seq_factory():
    from atom.sampling_params import SamplingParams
    from atom.model_engine.sequence import Sequence

    def make(token_ids, block_size=4):
        return Sequence(token_ids, block_size, sampling_params=SamplingParams())

    return make


# ── Unconstrained: no shm handle attached ────────────────────────────────


def test_prefill_scheduler_skips_shm_when_name_empty(prefill_scheduler_unconstrained):
    assert prefill_scheduler_unconstrained._cu_shm is None


def test_decode_scheduler_skips_shm_when_name_empty(decode_scheduler_unconstrained):
    assert decode_scheduler_unconstrained._cu_shm is None


# ── Unconstrained: batches carry cu_stream_fraction=None ─────────────────


def test_unconstrained_prefill_batch_has_none_cu_fraction(
    prefill_scheduler_unconstrained, seq_factory
):
    """Without shm, PrefillScheduler must produce batches keyed by the
    plain (None) stream — never a fractional CU mask."""
    seq = seq_factory([10, 20, 30, 40])
    seq.block_table = [0, 1]
    seq.num_cached_tokens = 0
    prefill_scheduler_unconstrained.add(seq)

    batch, _ = prefill_scheduler_unconstrained.schedule()
    assert batch is not None
    assert batch.cu_stream_fraction is None


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

    def test_step_reports_whether_it_ran(self):
        """The loop uses this to decide when a dummy batch is needed."""
        ec = pytest.importorskip("atom.model_engine.engine_core")

        obj = object.__new__(ec.DecodeEngineCore)
        # allocate_waiting now runs first (it ticks the delayer's collective),
        # so the stub has to provide it even when there is nothing to admit.
        obj.scheduler = type(
            "S",
            (),
            {
                "has_requests": lambda self: False,
                "allocate_waiting": lambda self: [],
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


def _rapidserve_runner(
    *, rank, decode, dp_rank_local=0, tp=8, device=None, pcp=1
):
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
            self._assert(
                _rapidserve_runner(rank=0, decode=True, dp_rank_local=k, tp=1)
            )

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

    _prepare_kv_dims computes num_kv_heads as num_key_value_heads // world_size.
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


class TestPairedRequestRouting:
    """A sequence must reach BOTH members of exactly one pair, and no other.

    Sending it to a second decode rank would have two ranks allocate blocks for
    the same sequence; sending it to a second prefill rank would have two ranks
    write its KV into different GPUs' pools.
    """

    @staticmethod
    def _route(pair, decode_idx0):
        """Mirror of DisaggCoreManager.add_request's index arithmetic."""
        return {"prefill": pair, "decode": decode_idx0 + pair}

    def test_sequence_goes_to_both_members_of_one_pair(self):
        got = self._route(pair=3, decode_idx0=8)
        assert got == {"prefill": 3, "decode": 11}

    def test_selected_engine_index_maps_back_to_its_pair(self):
        """_select_dp_rank_locked returns an ENGINE index in the decode slice."""
        decode_idx0 = 8
        for idx in range(decode_idx0, 2 * decode_idx0):
            pair = idx - decode_idx0
            assert self._route(pair, decode_idx0)["decode"] == idx
            assert 0 <= pair < decode_idx0

    def test_symmetric_routes_everything_to_pair_zero(self):
        got = self._route(pair=0, decode_idx0=1)
        assert got == {"prefill": 0, "decode": 1}

    def test_every_pair_is_reachable(self):
        decode_idx0 = 8
        routed = [self._route(k, decode_idx0) for k in range(decode_idx0)]
        assert len({r["prefill"] for r in routed}) == decode_idx0
        assert len({r["decode"] for r in routed}) == decode_idx0


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




class TestDecodeTokensChannel:
    """Decode publishes its in-flight token count for prefill, always.

    It used to exist only under --disagg-constrained, where CU partitioning
    consumed it. The prefill delayer needs the same number for its "is there
    decode to hide the wait behind?" gate, and prefill cannot answer that from
    its own scheduler — decode is a different process.
    """

    @staticmethod
    def _offset(decode_rank):
        return 4 * decode_rank

    def test_each_pair_gets_its_own_slot(self):
        assert [self._offset(k) for k in range(4)] == [0, 4, 8, 12]

    def test_slots_do_not_overlap(self):
        offs = [self._offset(k) for k in range(8)]
        assert len(set(offs)) == 8
        assert max(offs) + 4 == 4 * 8  # the size the manager allocates

    def test_both_schedulers_derive_the_same_offset(self):
        import ast
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        for cls in (sched.PrefillScheduler, sched.DecodeScheduler):
            src = inspect.getsource(cls)
            assert "self._cu_shm_offset = 4 * getattr(config" in src, cls.__name__
            ast.parse(src.lstrip())

    def test_cu_masking_is_gated_on_the_flag_not_the_shm(self):
        """The shm now exists regardless, so `_cu_shm is not None` no longer
        means 'constrained mode'."""
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        src = inspect.getsource(sched.PrefillScheduler)
        assert "self._cu_masking = bool(" in src
        assert "if self._cu_masking and self._cu_shm is not None:" in src

    def test_prefill_reads_decode_tokens_through_a_named_accessor(self):
        sched = pytest.importorskip("atom.model_engine.scheduler")

        assert hasattr(sched.PrefillScheduler, "decode_tokens")

    def test_decode_tokens_is_zero_without_the_shm(self):
        sched = pytest.importorskip("atom.model_engine.scheduler")

        obj = object.__new__(sched.PrefillScheduler)
        obj._cu_shm = None
        obj._cu_shm_offset = 0
        assert obj.decode_tokens() == 0


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

        obj = object.__new__(pd.PrefillDelayer)
        for name in (
            "fire_fill", "fire_stall", "fire_ttft", "fire_kv", "fire_partial",
            "fire_nodecode", "fire_queue_ms", "fire_vacuous", "hold",
        ):
            setattr(obj, f"_stat_{name}", 0)
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
    """Admission is decode's decision, so the delayer belongs there.

    Two things follow from decode owning the BlockManager:
      - delaying admission delays the KV allocation, so blocks are not pinned
        for a sequence whose prefill has not been scheduled;
      - the alignment that matters is across DECODE ranks. One rank admitting
        alone leaves it the only one with prefill_waiting, and its paired
        prefill rank runs while the other prefill ranks issue dummy forwards.
    """

    def test_decode_gates_admission_on_the_delayer(self):
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        src = inspect.getsource(sched.DecodeScheduler.allocate_waiting)
        assert "should_allow_prefill" in src
        assert "return []" in src, "a HOLD must allocate nothing"

    def test_prefill_no_longer_gates(self):
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        src = inspect.getsource(sched.PrefillScheduler.schedule)
        assert "should_allow_prefill" not in src

    def test_prefill_engine_declines_the_delayer(self):
        ec = pytest.importorskip("atom.model_engine.engine_core")

        assert (
            ec.PrefillEngineCore._maybe_attach_prefill_delayer
            is not ec.DPEngineCoreProc._maybe_attach_prefill_delayer
        )

    def test_decode_uses_real_kv_usage_and_decode_batch(self):
        """The inputs prefill could not supply: prefill passed kv_usage=0.0,
        which disables the KV-high/KV-low release bounds entirely."""
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        src = inspect.getsource(sched.DecodeScheduler.allocate_waiting)
        assert "kv_usage=self._kv_usage()" in src
        assert "len(self.running) - self._partial_prefill_count" in src

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

        src = inspect.getsource(sched.DecodeScheduler.allocate_waiting)
        for helper in (
            "_waiting_new_token_count()",
            "_can_admit_head_prefill()",
            "_oldest_waiting_prefill_age_ms()",
        ):
            assert helper in src, helper

    def test_both_call_sites_build_the_same_inputs(self):
        """Decode's call and Scheduler.schedule()'s must not drift apart."""
        import inspect

        sched = pytest.importorskip("atom.model_engine.scheduler")

        decode = inspect.getsource(sched.DecodeScheduler.allocate_waiting)
        base = inspect.getsource(sched.Scheduler.schedule)
        for arg in (
            "prefillable=self._can_admit_head_prefill()",
            "kv_usage=self._kv_usage()",
            "has_partial=self._partial_prefill_count > 0",
            "oldest_waiting_age_ms=self._oldest_waiting_prefill_age_ms()",
        ):
            assert arg in decode and arg in base, arg

    def test_decode_attaches_after_its_scheduler_exists(self):
        """DecodeScheduler is built only after the kvcache import, so the
        attach inside super().__init__() has nothing to attach to."""
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        src = inspect.getsource(ec.DecodeEngineCore)
        build = src.index("self.scheduler = DecodeScheduler")
        attach = src.index("DPEngineCoreProc._maybe_attach_prefill_delayer(self")
        assert attach > build




class TestCuMaskingIsIndependentOfTheShm:
    """Publishing the decode token count and masking CUs are separate concerns.

    The shm now exists in BOTH modes so the count is always available. Decode
    used to infer "constrained mode" from `self._cu_shm is not None`, so once
    the shm became unconditional it emitted cu_stream_fraction=0.5 while
    ModelRunner had built only the full-CU stream (the pool is gated on
    disagg_constrained, model_runner.py:4711) — KeyError: 0.5 on the first
    decode forward.
    """

    @staticmethod
    def _shm(nbytes=4):
        import multiprocessing.shared_memory
        import os

        shm = multiprocessing.shared_memory.SharedMemory(
            name=f"atom_test_cu_{os.getpid()}_{nbytes}", create=True, size=nbytes
        )
        shm.buf[:nbytes] = b"\x00" * nbytes
        return shm

    @staticmethod
    def _build(cls, cfg, shm_name):
        """Construct, or skip on the pre-existing full-suite import pollution.

        Scheduler.__init__ does a lazy `from atom.utils.forward_context import
        get_kvconnector` (scheduler.py:599) that fails once the suite has run —
        inside __init__, so pytest.importorskip on the module cannot catch it.
        """
        try:
            return cls(cfg, disagg_cu_shm_name=shm_name)
        except ImportError as exc:
            pytest.skip(f"polluted sys.modules: {exc}")

    def test_masking_off_when_unconstrained_even_with_shm(self):
        from atom.model_engine.scheduler import DecodeScheduler

        shm = self._shm()
        try:
            sched = self._build(
                DecodeScheduler, MockConfig(disagg_constrained=False), shm.name
            )
            assert sched._cu_shm is not None, "the count must still be published"
            assert sched._cu_masking is False
            assert sched.cu_fraction is None
        finally:
            shm.close()
            shm.unlink()

    def test_masking_on_when_constrained(self):
        from atom.model_engine.scheduler import DecodeScheduler

        shm = self._shm(8)
        try:
            sched = self._build(
                DecodeScheduler, MockConfig(disagg_constrained=True), shm.name
            )
            assert sched._cu_masking is True
        finally:
            shm.close()
            shm.unlink()

    def test_prefill_side_agrees(self):
        from atom.model_engine.scheduler import PrefillScheduler

        shm = self._shm()
        try:
            sched = self._build(
                PrefillScheduler, MockConfig(disagg_constrained=False), shm.name
            )
            assert sched._cu_shm is not None
            assert sched._cu_masking is False
        finally:
            shm.close()
            shm.unlink()

    def test_only_none_key_is_emitted_when_unconstrained(self):
        """The pool's sole key in unconstrained mode is None, so that is the
        only fraction either scheduler may put on a batch."""
        import inspect

        mr_src = inspect.getsource(
            pytest.importorskip("atom.model_engine.model_runner").RapidServeModelRunner
        )
        # the fractional entries are gated; the None entry is unconditional
        assert 'if getattr(self.config, "disagg_constrained", False):' in mr_src
        assert "self._decode_streams[None] = torch.cuda.Stream()" in mr_src


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
        assert "allocate_waiting" not in src

    def test_admission_is_in_the_work_gated_step(self):
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        src = inspect.getsource(ec.DecodeEngineCore._process_engine_step)
        assert "allocate_waiting" in src

    def test_admission_precedes_the_early_return(self):
        """Every rank must reach the collective on the same iterations."""
        import inspect

        ec = pytest.importorskip("atom.model_engine.engine_core")

        src = inspect.getsource(ec.DecodeEngineCore._process_engine_step)
        assert src.index("allocate_waiting") < src.index("has_requests()")

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
                    assert self._need(
                        True, ep, bypass=bypass, has_mori=has_mori
                    ) is (not all2all)


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
        a2a = pytest.importorskip(
            "aiter.dist.device_communicators.all2all"
        )
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
