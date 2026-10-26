"""ATOM_PREFILL_WAITING_RESERVE: while new prefills wait, resumed partial
prefills leave them a share of the step's token budget (and cost target), so a
long chunked prompt cannot hold a rank's whole prefill budget until it ends."""

from conftest import MockConfig

from atom.model_engine.scheduler import Scheduler, ScheduledBatchOutput


def _sched(monkeypatch, reserve):
    monkeypatch.setenv("ATOM_PREFILL_WAITING_RESERVE", str(reserve))
    return Scheduler(
        MockConfig(
            max_num_seqs=8,
            num_kvcache_blocks=600,
            kv_cache_block_size=4,
            max_model_len=2048,
            max_num_batched_tokens=256,
            enable_chunked_prefill=True,
        )
    )


def _step(sched):
    batch, _ = sched.schedule()
    sched.postprocess(
        list(sched.running),
        ScheduledBatchOutput(
            req_ids=[],
            token_ids=[],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        batch=batch,
    )
    return dict(zip(batch.req_ids, (int(n) for n in batch.num_scheduled_tokens)))


def _start_long_prefill(sched, seq_factory):
    long_seq = seq_factory(list(range(1000, 2024)))
    sched.add(long_seq)
    assert _step(sched) == {long_seq.id: 256}
    assert long_seq.is_partial_prefill
    return long_seq


def test_without_reserve_the_partial_holds_the_whole_budget(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, 0)
    long_seq = _start_long_prefill(sched, seq_factory)
    sched.add(seq_factory(list(range(10, 18))))
    assert _step(sched) == {long_seq.id: 256}


def test_reserve_admits_a_waiting_prefill_beside_the_partial(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, 0.25)
    long_seq = _start_long_prefill(sched, seq_factory)
    short = seq_factory(list(range(10, 18)))
    sched.add(short)
    assert _step(sched) == {long_seq.id: 192, short.id: 8}


def test_reserve_does_not_shrink_a_lone_partial(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, 0.25)
    long_seq = _start_long_prefill(sched, seq_factory)
    assert _step(sched) == {long_seq.id: 256}


def test_reserve_keeps_large_uncached_prompts_waiting(monkeypatch, seq_factory):
    # A 128-token uncached prompt does not fit the 64-token reserve: it waits
    # for the partial as it would without the reserve, while a short prompt
    # queued behind it still uses the reserve, and arrival order is kept.
    sched = _sched(monkeypatch, 0.25)
    long_seq = _start_long_prefill(sched, seq_factory)
    big = seq_factory(list(range(3000, 3128)))
    short = seq_factory(list(range(10, 18)))
    sched.add(big)
    sched.add(short)
    assert _step(sched) == {long_seq.id: 192, short.id: 8}
    assert sched.waiting[0] is big
