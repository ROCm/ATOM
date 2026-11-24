"""Short requests first (ATOM_PREFILL_SHORT_FIRST_ROWS).

In arrival order a rank's prefill step goes first to the chunks of the long
prompts in progress, and a short request behind them waits a step per round.
With the switch on the order is a two-tier key: a preemption victim and a
request passed over ATOM_PREFILL_SHORT_FIRST_MAX_SKIPS times stay in arrival
order at the front; the rest goes by the prefill work left, the requests with
at most ATOM_PREFILL_SHORT_FIRST_ROWS new tokens left ahead of the longer
prompts. The long prompts in progress keep ATOM_PREFILL_SHORT_FIRST_LONG_ROWS
tokens of every step.

The schedulers below take 16 tokens a step, a long prompt 8 of them.
"""

from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.model_engine.prefill_delayer import PrefillDecision
from atom.model_engine.scheduler import Scheduler


def _sched(monkeypatch, rows=None, long_rows=8, max_skips=None, **overrides):
    for name, value in (
        ("ATOM_PREFILL_SHORT_FIRST_ROWS", rows),
        ("ATOM_PREFILL_SHORT_FIRST_LONG_ROWS", long_rows),
        ("ATOM_PREFILL_SHORT_FIRST_MAX_SKIPS", max_skips),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(value))
    config = {
        "max_num_seqs": 64,
        "num_kvcache_blocks": 400,
        "kv_cache_block_size": 4,
        "max_model_len": 256,
        "max_num_batched_tokens": 16,
        "long_prefill_token_threshold": 8,
    }
    config.update(overrides)
    return Scheduler(MockConfig(**config))


def _step(sched):
    """One prefill step: what was scheduled, as (request, tokens) in batch
    order, with the forward's bookkeeping applied as postprocess leaves it."""
    batch, seqs = sched.schedule()
    taken = []
    for seq, tokens in zip(seqs.values(), batch.num_scheduled_tokens):
        taken.append((seq, int(tokens)))
        seq.num_cached_tokens += int(tokens)
        partial = seq.num_cached_tokens < seq.num_prompt_tokens
        if not partial:
            seq.append_token(99)
        if partial != seq.is_partial_prefill:
            sched._partial_prefill_count += 1 if partial else -1
            seq.is_partial_prefill = partial
    return taken


def _prompt(seq_factory, first, length):
    return seq_factory(list(range(first, first + length)))


def _shorts(seq_factory, tick):
    first = 10_000 + 100 * tick
    return [_prompt(seq_factory, first + 10 * i, 4) for i in range(4)]


def _two_long_two_short(sched, seq_factory):
    a = _prompt(seq_factory, 1000, 40)
    b = _prompt(seq_factory, 2000, 40)
    c = _prompt(seq_factory, 3000, 4)
    d = _prompt(seq_factory, 4000, 4)
    sched.extend([a, b, c, d])
    return a, b, c, d


@pytest.mark.parametrize("rows", [None, 0])
def test_off_is_arrival_order(monkeypatch, seq_factory, rows):
    """Unset or 0: the long prompts' chunks fill every step until they are
    done, the ordering code is never reached and nothing is counted."""
    sched = _sched(monkeypatch, rows=rows, long_rows=None)
    monkeypatch.setattr(
        sched, "_short_first_order", lambda: pytest.fail("the switch is off")
    )
    a, b, c, d = _two_long_two_short(sched, seq_factory)
    for _ in range(5):
        assert _step(sched) == [(a, 8), (b, 8)]
    assert _step(sched) == [(c, 4), (d, 4)]
    assert [seq.num_skipped_steps for seq in (a, b, c, d)] == [0, 0, 0, 0]


def test_short_requests_go_ahead_of_long_prompts(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, rows=4)
    a, b, c, d = _two_long_two_short(sched, seq_factory)
    assert _step(sched) == [(c, 4), (d, 4), (a, 8)]
    for _ in range(4):
        assert _step(sched) == [(a, 8), (b, 8)]
    assert _step(sched) == [(b, 8)]


def test_long_prompts_in_progress_go_before_waiting_long_ones(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, rows=4, max_num_batched_tokens=32)
    p1 = _prompt(seq_factory, 1000, 40)
    p2 = _prompt(seq_factory, 2000, 40)
    sched.extend([p1, p2])
    assert _step(sched) == [(p1, 8), (p2, 8)]
    waiting_long = _prompt(seq_factory, 3000, 40)
    s1 = _prompt(seq_factory, 4000, 4)
    s2 = _prompt(seq_factory, 5000, 4)
    sched.extend([waiting_long, s1, s2])
    assert _step(sched) == [(s1, 4), (s2, 4), (p1, 8), (p2, 8), (waiting_long, 8)]


def test_long_prompt_in_progress_keeps_its_tokens_of_every_step(
    monkeypatch, seq_factory
):
    """Short requests arrive faster than a step can take them. They leave the
    held 8 tokens alone, so the long prompt advances as in arrival order and
    is never passed over."""
    sched = _sched(monkeypatch, rows=4)
    a = _prompt(seq_factory, 1000, 40)
    sched.add(a)
    assert _step(sched) == [(a, 8)]
    for tick in range(4):
        sched.extend(_shorts(seq_factory, tick))
        taken = _step(sched)
        assert [tokens for _, tokens in taken] == [4, 4, 8]
        assert taken[-1][0] is a
    assert a.num_skipped_steps == 0
    assert [tokens for _, tokens in _step(sched)] == [4, 4, 4, 4]


def test_request_passed_over_max_skips_times_goes_first(monkeypatch, seq_factory):
    """The bound on starving a long prompt: after ATOM_PREFILL_SHORT_FIRST_MAX_SKIPS
    steps that scheduled prefill and none of its own it is ahead of the short
    requests, in arrival order, until its prefill is done."""
    sched = _sched(monkeypatch, rows=4, max_skips=3)
    a = _prompt(seq_factory, 1000, 40)
    b = _prompt(seq_factory, 2000, 40)
    sched.extend([a, b])
    for tick in range(3):
        sched.extend(_shorts(seq_factory, tick))
        assert [tokens for _, tokens in _step(sched)] == [4, 4, 4, 4]
    assert (a.num_skipped_steps, b.num_skipped_steps) == (3, 3)
    for tick in range(3, 8):
        sched.extend(_shorts(seq_factory, tick))
        assert _step(sched) == [(a, 8), (b, 8)]
    assert [tokens for _, tokens in _step(sched)] == [4, 4, 4, 4]


class _Delayer:
    """Answers the DP cost exchange with this rank's cost followed by
    `others`, and releases prefill unless told to hold."""

    max_queue_ms = None
    is_local = False

    def __init__(self, others=()):
        self.others = list(others)
        self.rank_prefill_costs = []
        self.allow = True

    def protects_decode(self, running_decode_batch: int) -> bool:
        return False

    def should_allow_prefill(
        self, prefillable, pending_tokens, prefill_cost=0.0, **kwargs
    ):
        self.rank_prefill_costs = [prefill_cost, *self.others]
        return PrefillDecision(
            allow=self.allow, prefillable_ranks=self.allow * (1 + len(self.others))
        )


def test_skips_are_charged_only_on_steps_that_scheduled_prefill(
    monkeypatch, seq_factory
):
    sched = _sched(monkeypatch, rows=4)
    delayer = _Delayer()
    sched.set_prefill_delayer(delayer)
    long_one = _prompt(seq_factory, 1000, 40)
    sched.add(long_one)
    sched.extend(_shorts(seq_factory, 0))
    delayer.allow = False
    sched.schedule()
    assert long_one.num_skipped_steps == 0
    delayer.allow = True
    assert [tokens for _, tokens in _step(sched)] == [4, 4, 4, 4]
    assert long_one.num_skipped_steps == 1


def test_preemption_victim_stays_ahead_of_short_requests(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, rows=4)
    victim = _prompt(seq_factory, 1000, 40)
    victim.is_preempted = True
    sched.add(victim)
    sched.extend(_shorts(seq_factory, 0))
    taken = _step(sched)
    assert taken[0] == (victim, 8)
    assert [tokens for _, tokens in taken] == [8, 4, 4]
    assert not victim.is_preempted


def test_request_that_cannot_be_admitted_is_not_overtaken(monkeypatch, seq_factory):
    """Arrival order protects a request waiting for KV room: the reordering
    stops in front of it."""
    sched = _sched(monkeypatch, rows=4)
    long_one = _prompt(seq_factory, 1000, 40)
    s1 = _prompt(seq_factory, 2000, 4)
    blocked = _prompt(seq_factory, 3000, 40)
    s2 = _prompt(seq_factory, 4000, 4)
    sched.extend([long_one, s1, blocked, s2])
    probe = sched.block_manager.can_allocate_memoized
    monkeypatch.setattr(
        sched.block_manager,
        "can_allocate_memoized",
        lambda seq: -1 if seq is blocked else probe(seq),
    )
    early_partials, long_partials, late_before = sched._short_first_order()
    assert (early_partials, long_partials) == ([], [])
    assert list(sched.waiting) == [s1, long_one, blocked, s2]
    assert late_before is long_one


def test_key_is_the_work_left_after_the_prefix_hit(monkeypatch, seq_factory):
    """A long prompt whose prefix is cached has little left to prefill: it is
    a short request, whatever its length."""
    sched = _sched(monkeypatch, rows=4, enable_prefix_caching=True)
    bm = sched.block_manager
    prefix = list(range(100, 136))
    first = seq_factory(prefix)
    bm.allocate(first, 0)
    bm.hash_blocks(first, len(prefix))
    bm.deallocate(first)
    cold = _prompt(seq_factory, 1000, 40)
    hot = seq_factory(prefix + [500, 501, 502, 503])
    sched.extend([cold, hot])
    sched._short_first_order()
    assert list(sched.waiting) == [hot, cold]


@pytest.mark.parametrize("rows", [0, 8])
def test_cost_target_goes_to_short_requests_first(monkeypatch, seq_factory, rows):
    """A rank over the DP cost target: in arrival order the long prompt in
    progress takes the target and the short request waits. With the switch on
    the short request is charged first, and the long prompt still gets its
    held tokens past the target."""
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT", "0")
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_SLACK", "0.1")
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_MIN_TOKENS", "4")
    sched = _sched(
        monkeypatch,
        rows=rows,
        long_rows=16,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=16,
        enable_dp_attention=True,
        parallel_config=SimpleNamespace(data_parallel_size=8),
    )
    delayer = _Delayer(others=[1e9] * 7)
    sched.set_prefill_delayer(delayer)
    long_one = _prompt(seq_factory, 1000, 64)
    sched.add(long_one)
    assert _step(sched) == [(long_one, 16)]
    short = _prompt(seq_factory, 2000, 8)
    sched.add(short)
    # Own cost 16 * (16 + 8) + 8 * 4 = 416 against seven ranks at 8: the target
    # is 1.1 * 472 / 8 = 64.9, three tokens of the long prompt's next chunk.
    delayer.others = [8.0] * 7
    taken = _step(sched)
    if rows:
        assert taken == [(short, 8), (long_one, 16)]
    else:
        assert [seq for seq, _ in taken] == [long_one]
        assert taken[0][1] < 16


@pytest.mark.parametrize(
    ("rows", "long_rows", "max_skips"),
    [(-1, 8, 32), (4, 0, 32), (4, 16, 32), (4, 4096, 32), (4, 8, -1)],
)
def test_refuses_values_that_cannot_work(monkeypatch, rows, long_rows, max_skips):
    """The held tokens have to leave the short requests part of the 16-token
    step."""
    with pytest.raises(ValueError, match="ATOM_PREFILL_SHORT_FIRST"):
        _sched(monkeypatch, rows=rows, long_rows=long_rows, max_skips=max_skips)


def test_held_tokens_are_not_checked_while_off(monkeypatch):
    assert _sched(monkeypatch, rows=0, long_rows=4096)._short_first_rows == 0
