"""Decode balance (ATOM_PREFILL_DECODE_BALANCE_BAND).

Every DP rank decodes at the width of the rank holding the most sequences. A
rank that holds more decoding sequences than the group's mean plus the band
puts its long prompts first for that prefill step, so it releases few
sequences while it is ahead; any other step keeps the short-first order
(ATOM_PREFILL_SHORT_FIRST_ROWS). Off (unset or negative) the short-first
scheduler is untouched.

The schedulers below take 16 tokens a step, a long prompt 8 of them.
"""

import pytest
from conftest import MockConfig

from atom.model_engine.prefill_delayer import PrefillDecision
from atom.model_engine.scheduler import Scheduler, decode_balance_long_first


@pytest.mark.parametrize(
    "own, group, dp_size, band, expected",
    [
        # off: a negative band, whatever the counts
        (40, 40, 8, -1.0, False),
        # one rank has no mean to be above
        (40, 40, 1, 0.0, False),
        # mean 12: strictly more than the band above it
        (16, 96, 8, 4.0, False),
        (17, 96, 8, 4.0, True),
        (12, 96, 8, 0.0, False),
        (13, 96, 8, 0.0, True),
        (11, 96, 8, 0.0, False),
        # a fractional mean (95 / 8 = 11.875) and a fractional band
        (14, 95, 8, 2.0, True),
        (13, 95, 8, 2.0, False),
        (14, 95, 8, 2.5, False),
        # nothing decodes anywhere
        (0, 0, 8, 0.0, False),
    ],
)
def test_decision(own, group, dp_size, band, expected):
    assert decode_balance_long_first(own, group, dp_size, band) is expected


def test_decision_is_monotone_in_the_rank_s_own_count():
    for group in range(0, 200, 7):
        for band in (0.0, 1.0, 2.5, 4.0):
            verdicts = [
                decode_balance_long_first(own, group, 8, band) for own in range(64)
            ]
            assert verdicts == sorted(verdicts)
            # at most one rank short of all can be above the mean by a band
            assert not decode_balance_long_first(group // 8, group, 8, band)


class _Delayer:
    """Releases prefill; `group_running_decode` is what the test says the DP
    group decodes, this rank included."""

    max_queue_ms = None
    is_local = False
    dp_size = 8

    def __init__(self):
        self.rank_prefill_costs = []
        self.group_running_decode = 0

    def protects_decode(self, running_decode_batch: int) -> bool:
        return False

    def should_allow_prefill(
        self, prefillable, pending_tokens, prefill_cost=0.0, **kwargs
    ):
        self.rank_prefill_costs = [prefill_cost]
        return PrefillDecision(allow=True, prefillable_ranks=1)


def _sched(monkeypatch, band=None, max_skips=None, delayer=True, **overrides):
    for name, value in (
        ("ATOM_PREFILL_SHORT_FIRST_ROWS", 4),
        ("ATOM_PREFILL_SHORT_FIRST_LONG_ROWS", 8),
        ("ATOM_PREFILL_SHORT_FIRST_MAX_SKIPS", max_skips),
        ("ATOM_PREFILL_DECODE_BALANCE_BAND", band),
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
    sched = Scheduler(MockConfig(**config))
    if delayer:
        sched.set_prefill_delayer(_Delayer())
    return sched


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


def _shorts(seq_factory, tick, n=4):
    first = 10_000 + 100 * tick
    return [_prompt(seq_factory, first + 10 * i, 4) for i in range(n)]


def _run(sched, seq_factory, group_counts):
    """Two long prompts and a stream of short requests over len(group_counts)
    steps; returns each step as [(name, tokens)]."""
    names = {}
    a = _prompt(seq_factory, 1000, 40)
    b = _prompt(seq_factory, 2000, 40)
    names[a.id], names[b.id] = "a", "b"
    sched.extend([a, b])
    out = []
    for tick, group in enumerate(group_counts):
        for i, seq in enumerate(_shorts(seq_factory, tick, n=2)):
            names[seq.id] = f"s{tick}.{i}"
            sched.add(seq)
        if sched.prefill_delayer is not None:
            sched.prefill_delayer.group_running_decode = group
        out.append([(names[seq.id], tokens) for seq, tokens in _step(sched)])
    return out


@pytest.mark.parametrize("band", [None, -1, -0.5])
def test_off_is_the_short_first_scheduler(monkeypatch, seq_factory, band):
    """Unset or negative: the decision and the long-first order are never
    reached, nothing is counted, and the steps are those of a scheduler that
    has never heard of the switch -- whatever the group decodes."""
    reference = _sched(monkeypatch, band=None, delayer=False)
    expected = _run(reference, seq_factory, [0] * 8)
    sched = _sched(monkeypatch, band=band)
    monkeypatch.setattr(
        sched, "_long_first_step", lambda: pytest.fail("the switch is off")
    )
    monkeypatch.setattr(
        sched, "_long_first_order", lambda: pytest.fail("the switch is off")
    )
    assert _run(sched, seq_factory, [0, 0, 1, 2, 0, 3, 0, 0]) == expected
    assert sched._decode_balance_stats == [0, 0, 0, 0]


def test_rank_at_or_below_the_mean_keeps_short_first(monkeypatch, seq_factory):
    """On, but this rank is never above the group's mean by the band: the
    short-first steps, and every step counted."""
    reference = _sched(monkeypatch, band=None, delayer=False)
    expected = _run(reference, seq_factory, [0] * 8)
    sched = _sched(monkeypatch, band=1)
    monkeypatch.setattr(
        sched, "_long_first_order", lambda: pytest.fail("not above the mean")
    )
    # 8 ranks: a group count of 8 x (own + 1) keeps the mean above this rank
    assert _run(sched, seq_factory, [800] * 8) == expected
    assert sched._decode_balance_stats[:2] == [8, 0]


def test_rank_above_the_mean_puts_its_long_prompts_first(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, band=1)
    a = _prompt(seq_factory, 1000, 40)
    b = _prompt(seq_factory, 2000, 40)
    c = _prompt(seq_factory, 3000, 4)
    d = _prompt(seq_factory, 4000, 4)
    sched.extend([a, b, c, d])
    # nothing decodes yet: short requests first, the long prompt behind them
    assert _step(sched) == [(c, 4), (d, 4), (a, 8)]
    # c and d decode here and nowhere else: 2 against a mean of 2 / 8
    sched.prefill_delayer.group_running_decode = 2
    e, f = _shorts(seq_factory, 1, n=2)
    sched.extend([e, f])
    assert _step(sched) == [(a, 8), (b, 8)]
    assert list(sched.waiting) == [e, f]
    assert _step(sched) == [(a, 8), (b, 8)]
    # the group catches up (mean 2): short requests first again, and the long
    # prompt in progress keeps its held tokens
    sched.prefill_delayer.group_running_decode = 16
    assert _step(sched) == [(e, 4), (f, 4), (a, 8)]
    assert sched._decode_balance_stats[:2] == [4, 2]


def test_long_first_step_fills_the_budget(monkeypatch, seq_factory):
    """Nothing is held back in a long-first step: what the long prompts leave
    goes to the short requests in the same step."""
    sched = _sched(monkeypatch, band=0, max_num_batched_tokens=32)
    c, d = _shorts(seq_factory, 0, n=2)
    sched.extend([c, d])
    assert _step(sched) == [(c, 4), (d, 4)]
    sched.prefill_delayer.group_running_decode = 2
    a = _prompt(seq_factory, 1000, 40)
    e, f, g, h = _shorts(seq_factory, 1, n=4)
    sched.extend([e, a, f, g, h])
    taken = _step(sched)
    assert taken == [(a, 8), (e, 4), (f, 4), (g, 4), (h, 4)]
    assert sum(tokens for _, tokens in taken) == 24


def test_waiting_long_prompt_goes_ahead_of_short_requests(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, band=0)
    c, d = _shorts(seq_factory, 0, n=2)
    sched.extend([c, d])
    _step(sched)
    sched.prefill_delayer.group_running_decode = 2
    s1, s2 = _shorts(seq_factory, 1, n=2)
    long_one = _prompt(seq_factory, 1000, 40)
    long_two = _prompt(seq_factory, 2000, 40)
    sched.extend([s1, long_one, s2, long_two])
    partials, late_partials, late_before = sched._long_first_order()
    assert (partials, late_partials, late_before) == ([], [], None)
    assert list(sched.waiting) == [long_one, long_two, s1, s2]
    assert _step(sched) == [(long_one, 8), (long_two, 8)]


def test_front_tier_stays_in_front(monkeypatch, seq_factory):
    """A preemption victim and a request passed over
    ATOM_PREFILL_SHORT_FIRST_MAX_SKIPS times lead a long-first step too."""
    sched = _sched(monkeypatch, band=0, max_skips=2)
    c, d = _shorts(seq_factory, 0, n=2)
    sched.extend([c, d])
    _step(sched)
    sched.prefill_delayer.group_running_decode = 2
    long_one = _prompt(seq_factory, 1000, 40)
    aged = _prompt(seq_factory, 2000, 4)
    aged.num_skipped_steps = 2
    sched.extend([long_one, aged])
    sched._long_first_order()
    assert list(sched.waiting) == [aged, long_one]
    assert _step(sched) == [(aged, 4), (long_one, 8)]


def test_request_that_cannot_be_admitted_is_not_overtaken(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, band=0)
    s1 = _prompt(seq_factory, 2000, 4)
    long_one = _prompt(seq_factory, 1000, 40)
    blocked = _prompt(seq_factory, 3000, 4)
    long_two = _prompt(seq_factory, 4000, 40)
    sched.extend([s1, long_one, blocked, long_two])
    probe = sched.block_manager.can_allocate_memoized
    monkeypatch.setattr(
        sched.block_manager,
        "can_allocate_memoized",
        lambda seq: -1 if seq is blocked else probe(seq),
    )
    sched._long_first_order()
    assert list(sched.waiting) == [long_one, s1, blocked, long_two]


def test_without_a_delayer_the_switch_does_nothing(monkeypatch, seq_factory):
    reference = _sched(monkeypatch, band=None, delayer=False)
    expected = _run(reference, seq_factory, [0] * 6)
    sched = _sched(monkeypatch, band=0, delayer=False)
    assert _run(sched, seq_factory, [0] * 6) == expected
    assert sched._decode_balance_stats == [0, 0, 0, 0]

