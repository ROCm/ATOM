# SPDX-License-Identifier: MIT
# Tests for the SparseKV cold-pool admission back-pressure in Scheduler.
#
# The decode worker keeps every resident request's full KV in a paged host pool
# sized ratio × max_num_seqs × (hot+1) tokens, and that context grows one page per
# page_size decoded tokens via an UNGATED worker path (alloc_host_pages, which
# hard-raises on exhaustion). So the scheduler reserves each request's whole
# LIFETIME footprint (prompt + max_tokens) at admission: it DEFERS (parks) a
# remote-load request that would overflow the pool, and permanently REJECTS one
# whose lifetime footprint can never fit — instead of the worker over-committing
# mid-decode and killing the runner.
#
# A GPU cold tier (ATOM_SPARSEKV_GPU_COLD_PAGES > 0) does not add a second gate:
# the gate still counts HOST pages only, because prefill RDMAs a whole request
# into the host pool before any of it can be promoted. What lifts the batch
# ceiling from host to host+GPU is the promote-done signal shrinking each
# request's host reservation once the worker moves those pages to the GPU tier.


from conftest import MockConfig

from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus
from atom.sampling_params import SamplingParams


def _make_scheduler(monkeypatch, *, ratio, hot, reserve_pages, gpu_pages=0, **cfg):
    monkeypatch.setenv("ATOM_SPARSEKV_ENABLE", "1")
    monkeypatch.setenv("ATOM_SPARSEKV_HOST_TO_DEVICE_RATIO", str(ratio))
    monkeypatch.setenv("ATOM_SPARSEKV_HOT_BUFFER_SIZE", str(hot))
    monkeypatch.setenv("ATOM_SPARSEKV_ADMIT_RESERVE_PAGES", str(reserve_pages))
    monkeypatch.setenv("ATOM_SPARSEKV_GPU_COLD_PAGES", str(gpu_pages))
    return Scheduler(MockConfig(**cfg))


class _RemoteLoadConnector:
    """Consumer-side stub: every prefill parks for a remote KV load."""

    is_producer = False
    is_offload = False

    def get_num_new_matched_tokens(self, seq):
        return 0, True

    def update_state_after_alloc(self, seq):
        pass

    def build_connector_meta(self):
        return None


def test_admit_pages_matches_coordinator_formula(monkeypatch):
    # host_tokens = ratio * max_num_seqs * (hot+1); pages = ceil/page; page=block.
    sch = _make_scheduler(
        monkeypatch,
        ratio=2,
        hot=1,
        reserve_pages=0,
        max_num_seqs=4,
        kv_cache_block_size=4,
        num_kvcache_blocks=1000,
        max_model_len=4096,
    )
    # host_tokens = 2*4*2 = 16; page=4 -> 4 pages; reserve 0 -> admit 4.
    assert sch._sparsekv_enabled is True
    assert sch._sparsekv_page == 4
    assert sch._sparsekv_admit_pages == 4


def test_reserve_pages_subtracted_per_slot(monkeypatch):
    sch = _make_scheduler(
        monkeypatch,
        ratio=8,
        hot=1,
        reserve_pages=1,
        max_num_seqs=4,
        kv_cache_block_size=4,
        num_kvcache_blocks=1000,
        max_model_len=4096,
    )
    # host_tokens = 8*4*2 = 64; page=4 -> 16 pages; reserve = 1*4 -> admit 12.
    assert sch._sparsekv_admit_pages == 12


def test_disabled_when_env_off(monkeypatch):
    monkeypatch.delenv("ATOM_SPARSEKV_ENABLE", raising=False)
    sch = Scheduler(MockConfig())
    assert sch._sparsekv_enabled is False


def test_oversized_lifetime_footprint_rejected_not_deadlocked(monkeypatch, seq_factory):
    """A request whose LIFETIME footprint (prompt + max_tokens) needs more pages
    than the whole admittable pool can never fit -> _unschedulable_reason must flag
    it (so admission drops it instead of parking it forever). The decode budget,
    not just the prompt, is what counts."""
    sch = _make_scheduler(
        monkeypatch,
        ratio=2,
        hot=1,
        reserve_pages=0,
        max_num_seqs=4,
        kv_cache_block_size=4,
        num_kvcache_blocks=1000,  # not KV-block bound
        max_model_len=4096,  # not length bound
    )
    # admit_pages = 4 -> capacity 16 tokens of lifetime footprint.
    # Oversized prompt: 40 + 1 -> 11 pages > 4.
    big = seq_factory([1] * 40, sampling_params=SamplingParams(max_tokens=1))
    reason = sch._unschedulable_reason(big)
    assert reason is not None and "SparseKV host cold-pool pages" in reason
    # Small prompt but a large decode budget also overruns: 8 + 40 -> 12 pages > 4.
    big_decode = seq_factory([1] * 8, sampling_params=SamplingParams(max_tokens=40))
    reason = sch._unschedulable_reason(big_decode)
    assert reason is not None and "SparseKV host cold-pool pages" in reason
    # A request whose whole lifetime fits is not flagged: 8 + 4 -> 3 pages <= 4.
    small = seq_factory([1] * 8, sampling_params=SamplingParams(max_tokens=4))
    assert sch._unschedulable_reason(small) is None


def test_pages_in_use_reserves_lifetime_footprint(monkeypatch, seq_factory):
    """_sparsekv_pages_in_use sums each request's worst-case lifetime footprint
    (prompt + max_tokens) over running + in-flight-recv (WAITING_FOR_REMOTE_KVS)
    requests — self-correcting, no dict. Decode growth does NOT raise the count:
    the final footprint was already reserved at admission, which is what keeps the
    ungated worker growth from ever overrunning the pool."""
    sch = _make_scheduler(
        monkeypatch,
        ratio=2,
        hot=3,
        reserve_pages=0,
        max_num_seqs=8,
        kv_cache_block_size=4,
        num_kvcache_blocks=1000,
        max_model_len=4096,
    )
    assert sch._sparsekv_pages_in_use() == 0

    # Running request: prompt 16 + max_tokens 4 -> final 20 -> 5 pages.
    r = seq_factory([1] * 16, sampling_params=SamplingParams(max_tokens=4))
    r.status = SequenceStatus.RUNNING
    sch.running.append(r)
    assert sch._sparsekv_pages_in_use() == 5

    # A waiting request counts ONLY when its recv is in flight.
    # prompt 8 + max_tokens 4 -> final 12 -> 3 pages.
    w = seq_factory([1] * 8, sampling_params=SamplingParams(max_tokens=4))
    w.status = SequenceStatus.WAITING
    sch.waiting.append(w)
    assert sch._sparsekv_pages_in_use() == 5  # plain WAITING not counted
    w.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert sch._sparsekv_pages_in_use() == 8  # now counted (5 + 3)

    # Decode growth: num_tokens rises but the reservation is fixed at admission
    # (prompt + max_tokens), so the count does NOT change — the growth the worker
    # will do was already accounted for.
    r.append_token(5)
    r.append_token(6)
    assert sch._sparsekv_pages_in_use() == 8


def test_lifetime_footprint_capped_at_max_model_len(monkeypatch, seq_factory):
    """A huge max_tokens is capped at max_model_len, the true context ceiling, so
    the reservation never exceeds what a request could physically occupy."""
    sch = _make_scheduler(
        monkeypatch,
        ratio=8,
        hot=7,
        reserve_pages=0,
        max_num_seqs=4,
        kv_cache_block_size=4,
        num_kvcache_blocks=1000,
        max_model_len=64,
    )
    # prompt 16 + max_tokens 10_000 -> capped to max_model_len 64 -> 16 pages.
    r = seq_factory([1] * 16, sampling_params=SamplingParams(max_tokens=10_000))
    assert sch._sparsekv_worst_case_pages(r) == 64 // sch._sparsekv_page


def test_worst_case_includes_spec_draft_headroom(monkeypatch, seq_factory):
    """Under speculative decode the worker writes up to mtp_k extra draft positions
    per step, so the footprint must include mtp_k or a request landing near a page
    boundary overruns the reserved pages and the worker's alloc_host_pages raises."""
    sch = _make_scheduler(
        monkeypatch,
        ratio=8,
        hot=7,
        reserve_pages=0,
        max_num_seqs=4,
        kv_cache_block_size=4,
        num_kvcache_blocks=1000,
        max_model_len=4096,
    )
    # prompt 16 + max_tokens 3 -> final 19 -> ceil(19/4)=5 pages without spec.
    r = seq_factory([1] * 16, sampling_params=SamplingParams(max_tokens=3))
    assert sch.mtp_k == 0
    assert sch._sparsekv_worst_case_pages(r) == 5
    # With mtp_k=1 the peak is 20 tokens -> still 5 pages; mtp_k=2 -> 21 -> 6 pages
    # (the extra draft position crosses into a new page and must be reserved).
    sch.mtp_k = 2
    assert sch._sparsekv_worst_case_pages(r) == 6


# ---------------------------------------------------------------------------
# GPU cold tier: two admission gates + promote-done signal
# ---------------------------------------------------------------------------


def _gpu_tier_scheduler(monkeypatch, gpu_pages):
    # host_tokens = 2*8*2 = 32; page=4 -> 8 pages; reserve 0 -> host admit 8.
    return _make_scheduler(
        monkeypatch,
        ratio=2,
        hot=1,
        reserve_pages=0,
        gpu_pages=gpu_pages,
        max_num_seqs=8,
        kv_cache_block_size=4,
        num_kvcache_blocks=1000,
        max_num_batched_tokens=256,
        max_model_len=4096,
    )


def _resident(sch, seq_factory, prompt=8, max_tokens=4, promoted=0):
    """Add a request already holding cold-pool pages (recv in flight)."""
    seq = seq_factory(
        [1] * prompt, sampling_params=SamplingParams(max_tokens=max_tokens)
    )
    seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    seq.sparsekv_promoted_pages = promoted
    sch.waiting.append(seq)
    return seq


def _incoming(sch, seq_factory, prompt=8, max_tokens=4):
    seq = seq_factory(
        [1] * prompt, sampling_params=SamplingParams(max_tokens=max_tokens)
    )
    sch.waiting.append(seq)
    return seq


def test_gpu_tier_budget_from_env(monkeypatch):
    sch = _gpu_tier_scheduler(monkeypatch, gpu_pages=6)
    assert sch._sparsekv_admit_pages == 8
    assert sch._sparsekv_gpu_pages == 6
    assert sch._sparsekv_gpu_cold_enabled is True

    off = _gpu_tier_scheduler(monkeypatch, gpu_pages=0)
    assert off._sparsekv_gpu_pages == 0
    assert off._sparsekv_gpu_cold_enabled is False


def test_promoted_pages_split_host_and_gpu_usage(monkeypatch, seq_factory):
    """A promoted page moves from the host tally to the GPU tally; the combined
    total stays the request's worst-case footprint."""
    sch = _gpu_tier_scheduler(monkeypatch, gpu_pages=6)
    seq = _resident(sch, seq_factory)  # 8 + 4 = 12 tokens -> 3 pages
    assert sch._sparsekv_pages_in_use() == 3
    assert sch._sparsekv_gpu_pages_of(seq) == 0

    seq.sparsekv_promoted_pages = 2
    assert sch._sparsekv_pages_in_use() == 1  # host tally drops by the promoted 2
    assert sch._sparsekv_gpu_pages_of(seq) == 2


def test_admission_gates_on_host_pages_only(monkeypatch, seq_factory):
    """Prefill RDMAs the whole request into the HOST pool before any of it can be
    promoted, so a free GPU tier does not make an unlandable request admissible."""
    sch = _gpu_tier_scheduler(monkeypatch, gpu_pages=6)
    sch.kv_connector = _RemoteLoadConnector()
    _resident(sch, seq_factory)
    _resident(sch, seq_factory)  # host used 6 of 8, GPU tier fully free
    seq = _incoming(sch, seq_factory)  # needs 3 host pages, only 2 free

    assert sch._sparsekv_pages_in_use() == 6
    sch.schedule()

    assert seq.status == SequenceStatus.WAITING
    assert seq in sch.waiting

    # Control: the host budget is the only thing holding it back.
    sch._sparsekv_admit_pages = 9
    sch.schedule()
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS


def test_promote_report_clamped_to_footprint(monkeypatch, seq_factory):
    """A report larger than the request's whole footprint is impossible; crediting
    it verbatim would zero out that request's host reservation."""
    sch = _gpu_tier_scheduler(monkeypatch, gpu_pages=64)
    sch.kv_connector = _RemoteLoadConnector()
    seq = _resident(sch, seq_factory)  # worst case 3 pages

    sch._update_from_kv_xfer_finished(
        KVConnectorOutput(promoted_gpu_pages={seq.id: 40})
    )
    assert seq.sparsekv_promoted_pages == 3


def test_promote_done_releases_host_budget(monkeypatch, seq_factory):
    """The promote-done signal downgrades a request's host reservation, which is
    what lets the host pool cycle instead of pinning the batch ceiling."""
    sch = _gpu_tier_scheduler(monkeypatch, gpu_pages=6)
    sch.kv_connector = _RemoteLoadConnector()
    promoted_seq = _resident(sch, seq_factory)
    _resident(sch, seq_factory)
    seq = _incoming(sch, seq_factory)

    sch._update_from_kv_xfer_finished(
        KVConnectorOutput(promoted_gpu_pages={promoted_seq.id: 3})
    )
    assert promoted_seq.sparsekv_promoted_pages == 3
    assert sch._sparsekv_pages_in_use() == 3  # was 6

    sch.schedule()

    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS


def test_gpu_tier_off_keeps_single_host_gate(monkeypatch, seq_factory):
    """Kill-switch: with no GPU tier nothing is ever promoted, so no reservation
    is ever downgraded and admission behaves as it did before the tier existed."""
    sch = _gpu_tier_scheduler(monkeypatch, gpu_pages=0)
    sch.kv_connector = _RemoteLoadConnector()
    _resident(sch, seq_factory)
    fits = _incoming(sch, seq_factory)  # 3 + 3 <= 8

    sch.schedule()
    assert fits.status == SequenceStatus.WAITING_FOR_REMOTE_KVS

    _resident(sch, seq_factory)  # host used 9 > 8
    overflows = _incoming(sch, seq_factory)
    sch.schedule()
    assert overflows.status == SequenceStatus.WAITING


def test_promote_credit_dropped_on_preempt(monkeypatch, seq_factory):
    """Preempt drops the coordinator slot (both tiers freed), so the credit must
    not survive into re-admission or host usage is under-counted."""
    sch = _gpu_tier_scheduler(monkeypatch, gpu_pages=6)
    seq = seq_factory([1] * 8, sampling_params=SamplingParams(max_tokens=4))
    seq.status = SequenceStatus.RUNNING
    seq.sparsekv_promoted_pages = 3
    sch.running.append(seq)
    assert sch._sparsekv_pages_in_use() == 0

    sch.preempt(sch.running.pop())  # as the decode loop preempts
    assert seq.sparsekv_promoted_pages == 0

    # The report is fanned in across all TP ranks and can land steps later, well
    # after the preempt already freed both tiers.
    sch.kv_connector = _RemoteLoadConnector()
    sch._update_from_kv_xfer_finished(KVConnectorOutput(promoted_gpu_pages={seq.id: 3}))
    assert seq.sparsekv_promoted_pages == 0
    sch.waiting.remove(seq)  # re-admitted
    seq.status = SequenceStatus.RUNNING
    sch.running.append(seq)
    assert sch._sparsekv_pages_in_use() == 3  # full footprint, no phantom credit


def test_promote_signal_for_unknown_request_ignored(monkeypatch, seq_factory):
    """A promote report can land after the request finished; drop it rather than
    accumulate scheduler-side state keyed by a dead id."""
    sch = _gpu_tier_scheduler(monkeypatch, gpu_pages=6)
    sch.kv_connector = _RemoteLoadConnector()
    live = _resident(sch, seq_factory)
    sch._update_from_kv_xfer_finished(
        KVConnectorOutput(promoted_gpu_pages={str(live.id + 1000): 4})
    )
    assert live.sparsekv_promoted_pages == 0

    # Same request id as a str, which is how connectors report it: credited.
    sch._update_from_kv_xfer_finished(
        KVConnectorOutput(promoted_gpu_pages={str(live.id): 2})
    )
    assert live.sparsekv_promoted_pages == 2
