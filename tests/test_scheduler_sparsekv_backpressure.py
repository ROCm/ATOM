# SPDX-License-Identifier: MIT
# Tests for the SparseKV host cold-pool admission back-pressure in Scheduler.
#
# The decode worker keeps every resident request's full KV in a paged host pool
# sized ratio × max_num_seqs × (hot+1) tokens, and that context grows one page per
# page_size decoded tokens via an UNGATED worker path (alloc_host_pages, which
# hard-raises on exhaustion). So the scheduler reserves each request's whole
# LIFETIME footprint (prompt + max_tokens) at admission: it DEFERS (parks) a
# remote-load request that would overflow the pool, and permanently REJECTS one
# whose lifetime footprint can never fit — instead of the worker over-committing
# mid-decode and killing the runner.


from conftest import MockConfig

from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus
from atom.sampling_params import SamplingParams


def _make_scheduler(monkeypatch, *, ratio, hot, reserve_pages, **cfg):
    monkeypatch.setenv("ATOM_SPARSEKV_ENABLE", "1")
    monkeypatch.setenv("ATOM_SPARSEKV_HOST_TO_DEVICE_RATIO", str(ratio))
    monkeypatch.setenv("ATOM_SPARSEKV_HOT_BUFFER_SIZE", str(hot))
    monkeypatch.setenv("ATOM_SPARSEKV_ADMIT_RESERVE_PAGES", str(reserve_pages))
    return Scheduler(MockConfig(**cfg))


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
