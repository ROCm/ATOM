# SPDX-License-Identifier: MIT
# Tests for the SparseKV host cold-pool admission back-pressure in Scheduler.
#
# The decode worker keeps every resident request's full KV in a paged host pool
# sized ratio × max_num_seqs × (hot+1) tokens. The scheduler mirrors that page
# budget so admission DEFERS (parks) a remote-load request that would overflow
# the pool, and permanently REJECTS one whose full context can never fit — instead
# of the worker's alloc_host_pages raising and killing the runner.


from conftest import MockConfig

from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus


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


def test_oversized_prompt_rejected_not_deadlocked(monkeypatch, seq_factory):
    """A prompt whose full context needs more pages than the whole admittable
    pool can never fit -> _unschedulable_reason must flag it (so admission drops
    it instead of parking it forever)."""
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
    # admit_pages = 4 -> capacity 16 tokens. 40 tokens -> 10 pages > 4.
    big = seq_factory([1] * 40)
    reason = sch._unschedulable_reason(big)
    assert reason is not None and "SparseKV host cold-pool pages" in reason
    # A request that fits the pool is not flagged by the SparseKV check.
    small = seq_factory([1] * 8)  # 2 pages <= 4
    assert sch._unschedulable_reason(small) is None


def test_pages_in_use_recompute_from_state(monkeypatch, seq_factory):
    """_sparsekv_pages_in_use sums ceil(num_tokens/page) over running requests +
    in-flight-recv (WAITING_FOR_REMOTE_KVS) requests — self-correcting, no dict."""
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
    page = sch._sparsekv_page  # 4
    assert sch._sparsekv_pages_in_use() == 0

    # A running request of 16 tokens -> 4 pages.
    r = seq_factory([1] * 16)
    r.status = SequenceStatus.RUNNING
    sch.running.append(r)
    assert sch._sparsekv_pages_in_use() == 4

    # A waiting request counts ONLY when its recv is in flight.
    w = seq_factory([1] * 8)  # 2 pages
    w.status = SequenceStatus.WAITING
    sch.waiting.append(w)
    assert sch._sparsekv_pages_in_use() == 4  # plain WAITING not counted
    w.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert sch._sparsekv_pages_in_use() == 6  # now counted (4 + 2)

    # Decode growth: running req grows -> pages recomputed higher automatically.
    r.append_token(5)
    r.append_token(6)  # 18 tokens -> ceil(18/4)=5 pages
    assert sch._sparsekv_pages_in_use() == 5 + 2
