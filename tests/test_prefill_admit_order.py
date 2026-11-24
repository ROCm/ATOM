"""engine.prefill_admit_order: the order a rank admits the prefill work it holds.

"fifo" (and unset) is today's order: prompts mid-prefill first, then new
prompts as they arrived. "short_new" takes the new prompts with the fewest
tokens left first; "srpt" takes the fewest left first across both, a prompt
mid-prefill leaving room for the shorter new prompts ahead of it. A prompt
that has gone engine.prefill_admit_max_age_ms without a chunk goes ahead of
every younger one. Nothing else moves: the chunk rules, the batch budget, the
DP cost target and which tokens a request prefills are the same in every mode.

The tests drive a scheduler through prefill steps the way the engine does
between forwards: `schedule()`, then the bookkeeping `postprocess` applies to
a prefill batch (`_advance_prefill_on_schedule`), then a prompt that finished
leaves `running`.
"""

import itertools
import random
from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.model_engine.prefill_delayer import PrefillDecision
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus
from atom.plugin_recipe import EngineOptions


class _Delayer:
    """Cross-DP delayer stub: releases prefill on the ticks `allow` says and
    answers the cost exchange with this rank's cost and `others`."""

    max_queue_ms = None
    is_local = False

    def __init__(self, others=(), allow=None):
        self.others = list(others)
        self.allow = allow
        self.rank_prefill_costs = []
        self.tick = 0

    def protects_decode(self, running_decode_batch):
        return False

    def should_allow_prefill(self, prefillable, pending_tokens, prefill_cost=0.0, **_):
        self.rank_prefill_costs = [prefill_cost, *self.others]
        allow = True if self.allow is None else bool(self.allow[self.tick])
        self.tick += 1
        return PrefillDecision(allow=allow, prefillable_ranks=1 + len(self.others))


def _sched(
    monkeypatch,
    order=None,
    *,
    max_age_ms=None,
    max_tokens=32,
    long_cap=0,
    slack=0.1,
    min_tokens=8,
    blocks=4096,
    max_num_seqs=8,
    prefix_caching=False,
    weight=0,
    engine=None,
):
    # The recipe's engine options as the config carries them; unvalidated, so
    # the scheduler's own checks are the ones exercised.
    options = {
        "prefill_admit_order": order,
        "prefill_admit_max_age_ms": max_age_ms,
        **(engine or {}),
    }
    engine_options = EngineOptions(
        **{k: v for k, v in options.items() if v is not None}
    )
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT", str(weight))
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_SLACK", str(slack))
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_MIN_TOKENS", str(min_tokens))
    config = MockConfig(
        max_num_seqs=max_num_seqs,
        num_kvcache_blocks=blocks,
        kv_cache_block_size=4,
        max_model_len=4096,
        max_num_batched_tokens=max_tokens,
        long_prefill_token_threshold=long_cap,
        enable_prefix_caching=prefix_caching,
        enable_dp_attention=True,
        parallel_config=SimpleNamespace(data_parallel_size=8),
    )
    config.plugin_engine_options = engine_options
    sched = Scheduler(config)
    sched.clock = [1000.0]
    sched._admit_clock = lambda: sched.clock[0]
    return sched


class _Run:
    """One rank's scheduler stepped through prefill ticks, keeping what each
    step admitted: ``(name, start, tokens, final)`` per request, in batch
    order."""

    def __init__(self, sched, seq_factory, delayer=None):
        self.sched = sched
        self.make = seq_factory
        self.names = {}
        self.seqs = {}
        self.steps = []
        self.served = {}
        self.done = {}
        self.prefilled = {}
        sched.set_prefill_delayer(delayer or _Delayer())

    def add(self, name, tokens, base=100):
        seq = self.make(list(range(base, base + tokens)))
        seq.arrive_time = self.sched.clock[0]
        self.names[seq.id] = name
        self.seqs[name] = seq
        self.prefilled[name] = 0
        self.sched.add(seq)
        return seq

    def waiting(self):
        return [self.names[s.id] for s in self.sched.waiting]

    def running(self):
        return [self.names[s.id] for s in self.sched.running]

    def step(self, dt=0.0):
        """One scheduler tick, `dt` seconds after the previous one."""
        sched = self.sched
        sched.clock[0] += dt
        out = sched.schedule()
        admitted = []
        if out is not None and out[0].total_tokens_num_prefill > 0:
            batch, seqs = out
            final = list(batch.is_final_chunk)
            tokens = [int(n) for n in batch.num_scheduled_tokens]
            starts = [int(n) for n in batch.num_cached_tokens]
            for i, seq in enumerate(seqs.values()):
                name = self.names[seq.id]
                admitted.append((name, starts[i], tokens[i], bool(final[i])))
                self.prefilled[name] += tokens[i]
                self.served.setdefault(name, []).append(sched.clock[0])
            sched._advance_prefill_on_schedule(seqs, tokens, final)
            for i, seq in enumerate(list(seqs.values())):
                if final[i]:
                    sched.running.remove(seq)
                    sched.block_manager.deallocate(seq)
                    self.done[self.names[seq.id]] = len(self.steps)
        self.steps.append(admitted)
        return admitted


# ---------------------------------------------------------------------------
# fifo is today's order
# ---------------------------------------------------------------------------

# A recorded run of a rank's queue: (tick, name, tokens) arrivals, the ticks
# the delayer vetoed prefill, the ticks a waiting request was cancelled, and
# every prefill step the scheduler admitted for it BEFORE this switch existed
# -- (name, start, tokens, final) in batch order.
_RECORDED_ARRIVALS = (
    (0, "a", 100),
    (0, "b", 12),
    (0, "c", 40),
    (1, "d", 8),
    (2, "e", 64),
    (2, "f", 4),
    (4, "g", 20),
    (5, "h", 96),
    (5, "i", 16),
    (7, "j", 8),
    (9, "k", 28),
)
_RECORDED_VETOES = (3, 6)
_RECORDED_CANCELS = ((2, "c"), (5, "i"))
_RECORDED_OTHERS = [1500.0] * 7
_RECORDED_TICKS = 18
_RECORDED_STEPS = (
    [("a", 0, 24, False)],
    [("a", 24, 24, False)],
    [("a", 48, 24, False)],
    [],
    [("a", 72, 20, False), ("b", 0, 12, True)],
    [("a", 92, 8, True), ("d", 0, 8, True)],
    [],
    [("e", 0, 24, False), ("f", 0, 4, True)],
    [("e", 24, 24, False)],
    [("e", 48, 16, True)],
    [("g", 0, 20, True)],
    [("h", 0, 24, False), ("j", 0, 8, True)],
    [("h", 24, 24, False)],
    [("h", 48, 24, False)],
    [("h", 72, 20, False)],
    [("h", 92, 4, True), ("k", 0, 24, False)],
    [("k", 24, 4, True)],
    [],
)


def _replay_recorded(run):
    """Drive `run` through the recorded queue states; return its steps."""
    arrivals = {}
    for tick, name, tokens in _RECORDED_ARRIVALS:
        arrivals.setdefault(tick, []).append((name, tokens))
    cancels = dict(_RECORDED_CANCELS)
    for tick in range(_RECORDED_TICKS):
        for name, tokens in arrivals.get(tick, ()):
            run.add(name, tokens)
        if tick in cancels:
            run.seqs[cancels[tick]].status = SequenceStatus.ABORTED
        run.step(dt=0.01)
    return tuple(run.steps)


def _recorded_delayer():
    allow = [t not in _RECORDED_VETOES for t in range(_RECORDED_TICKS)]
    return _Delayer(others=_RECORDED_OTHERS, allow=allow)


@pytest.mark.parametrize("order", [None, "fifo"])
def test_fifo_admits_what_the_scheduler_admitted_before_the_switch(
    monkeypatch, seq_factory, order
):
    """Unset and "fifo" make, step for step, the decisions recorded
    from the scheduler before the switch: same requests, same order on the
    batch, same chunk starts and sizes, same final flags."""
    sched = _sched(monkeypatch, order, long_cap=24)
    run = _Run(sched, seq_factory, _recorded_delayer())
    assert _replay_recorded(run) == _RECORDED_STEPS
    assert not sched._admit_order_active()


def test_the_ordering_code_leaves_a_queue_in_arrival_order_untouched(
    monkeypatch, seq_factory
):
    """With the age bound at 0 every prompt ranks by when it arrived, so
    "short_new" runs its ordering on every prefill step and must move nothing:
    the recorded decisions again, through the other code path."""
    sched = _sched(monkeypatch, "short_new", max_age_ms=0, long_cap=24)
    run = _Run(sched, seq_factory, _recorded_delayer())
    assert sched._admit_order_active()
    assert _replay_recorded(run) == _RECORDED_STEPS


def test_an_unknown_order_is_refused_at_construction(monkeypatch):
    with pytest.raises(ValueError, match="engine.prefill_admit_order"):
        _sched(monkeypatch, "shortest")
    with pytest.raises(ValueError, match="engine.prefill_admit_max_age_ms"):
        _sched(monkeypatch, "srpt", max_age_ms=-1)


@pytest.mark.parametrize("order", ["short_new", "srpt"])
def test_short_first_and_an_admission_order_are_refused_together(monkeypatch, order):
    """engine.prefill_short_first_rows and an order other than "fifo" both reorder
    a rank's admission; no combination is defined, so the scheduler refuses to
    start with the two set. Either alone, and short-first with "fifo" spelled
    out, starts."""
    short_first = {
        "prefill_short_first_rows": 16,
        "prefill_short_first_long_rows": 8,
    }
    with pytest.raises(ValueError, match="set one, not both"):
        _sched(monkeypatch, order, engine=short_first)
    assert _sched(monkeypatch, "fifo", engine=short_first)._short_first_rows == 16
    assert _sched(monkeypatch, None, engine=short_first)._short_first_rows == 16
    # The age bound alone selects no order: it may stay set beside short-first.
    for fifo in ("fifo", None):
        sched = _sched(monkeypatch, fifo, max_age_ms=6000, engine=short_first)
        assert sched._short_first_rows == 16
        assert not sched._admit_order_active()
    assert _sched(monkeypatch, order)._admit_order_active()


def test_the_pair_is_refused_before_the_engine_builds_anything(monkeypatch):
    """`EngineCore.__init__` checks the two options first, so a server set up
    with both stops before its runners start and the model loads."""
    from atom.model_engine.engine_core import EngineCore

    config = SimpleNamespace(
        plugin_engine_options=EngineOptions(
            prefill_short_first_rows=8192, prefill_admit_order="srpt"
        )
    )
    with pytest.raises(ValueError, match="set one, not both"):
        EngineCore(config=config, input_address="", output_address="")


def test_a_kv_connector_or_pd_admission_keeps_arrival_order(monkeypatch):
    sched = _sched(monkeypatch, "srpt")
    assert sched._admit_order_active()
    sched.kv_connector = object()
    assert not sched._admit_order_active()
    sched.kv_connector = None
    sched.pd_admission = object()
    assert not sched._admit_order_active()


# ---------------------------------------------------------------------------
# short_new
# ---------------------------------------------------------------------------


def test_short_new_takes_the_shortest_new_prompt_first(monkeypatch, seq_factory):
    run = _Run(_sched(monkeypatch, "short_new", max_tokens=32), seq_factory)
    run.add("long", 40)
    run.add("short", 8)
    run.add("mid", 20)
    assert run.step() == [("short", 0, 8, True), ("mid", 0, 20, True)]
    assert run.waiting() == ["long"]
    assert run.step() == [("long", 0, 32, False)]
    assert run.step() == [("long", 32, 8, True)]


def test_fifo_takes_them_as_they_came(monkeypatch, seq_factory):
    run = _Run(_sched(monkeypatch, "fifo", max_tokens=32), seq_factory)
    run.add("long", 40)
    run.add("short", 8)
    run.add("mid", 20)
    assert run.step() == [("long", 0, 32, False)]
    assert run.step() == [("long", 32, 8, True), ("short", 0, 8, True)]
    assert run.step() == [("mid", 0, 20, True)]


def test_short_new_leaves_a_prompt_mid_prefill_first(monkeypatch, seq_factory):
    """Only the new prompts are ordered: the running one keeps the batch."""
    run = _Run(_sched(monkeypatch, "short_new", max_tokens=32), seq_factory)
    run.add("long", 80)
    assert run.step() == [("long", 0, 32, False)]
    run.add("b", 20)
    run.add("a", 8)
    assert run.step() == [("long", 32, 32, False)]
    assert run.waiting() == ["a", "b"]
    assert run.step() == [("long", 64, 16, True), ("a", 0, 8, True)]
    assert run.step() == [("b", 0, 20, True)]


def test_size_is_what_the_prefix_cache_leaves(monkeypatch, seq_factory):
    """A long prompt whose prefix is cached is a short one."""
    sched = _sched(monkeypatch, "short_new", max_tokens=32, prefix_caching=True)
    run = _Run(sched, seq_factory)
    prefix = list(range(500, 564))
    first = seq_factory(prefix)
    sched.block_manager.allocate(first, 0)
    sched.block_manager.hash_blocks(first, len(prefix))
    sched.block_manager.deallocate(first)
    run.add("plain", 16)
    hit = seq_factory(prefix + list(range(900, 908)))
    hit.arrive_time = sched.clock[0]
    run.names[hit.id], run.seqs["hit"], run.prefilled["hit"] = "hit", hit, 0
    sched.add(hit)
    assert [entry[0] for entry in run.step()] == ["hit", "plain"]
    # the hit is the one admission found, not the one the ordering probed
    assert hit.prefix_cache_hit_tokens == 64


# ---------------------------------------------------------------------------
# srpt
# ---------------------------------------------------------------------------


def test_srpt_leaves_room_for_a_shorter_new_prompt(monkeypatch, seq_factory):
    """The running prompt still leads the batch, on what the shorter new one
    leaves (down to the chunk grid); in arrival order it takes the whole
    budget and the new prompt waits."""
    run = _Run(_sched(monkeypatch, "fifo", max_tokens=256), seq_factory)
    run.add("long", 1000)
    run.step()
    run.add("short", 16)
    assert run.step() == [("long", 256, 256, False)]

    run = _Run(_sched(monkeypatch, "srpt", max_tokens=256), seq_factory)
    run.add("long", 1000)
    assert run.step() == [("long", 0, 256, False)]
    run.add("short", 16)
    assert run.step() == [("long", 256, 192, False), ("short", 0, 16, True)]
    assert run.step() == [("long", 448, 256, False)]


def test_srpt_takes_the_shortest_running_prompt_first(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, "srpt", max_tokens=256, long_cap=128)
    run = _Run(sched, seq_factory)
    run.add("a", 640)
    run.add("b", 320)
    # new prompts: b (320 left) ahead of a (640 left), 128 tokens a prompt
    assert run.step() == [("b", 0, 128, False), ("a", 0, 128, False)]
    # running: b (192 left) ahead of a (512 left)
    assert run.step() == [("b", 128, 128, False), ("a", 128, 128, False)]
    run.add("c", 16)
    # c (16) is ahead of both: b finishes, a and c share what is left
    assert run.step() == [
        ("b", 256, 64, True),
        ("a", 256, 128, False),
        ("c", 0, 16, True),
    ]


def test_srpt_keeps_the_running_prompt_ahead_of_a_longer_new_one(
    monkeypatch, seq_factory
):
    run = _Run(_sched(monkeypatch, "srpt", max_tokens=256), seq_factory)
    run.add("short", 320)
    assert run.step() == [("short", 0, 256, False)]
    run.add("long", 1000)
    assert run.step() == [("short", 256, 64, True), ("long", 0, 192, False)]


def test_srpt_makes_room_under_the_cost_target(monkeypatch, seq_factory):
    """A rank at the step's cost target admits no new prompt. In arrival
    order the running prompt spends the target and the new one waits; under
    srpt the running prompt is sized against the target less the new prompt's
    cost, and both are on the batch within it."""

    def second_step(order):
        sched = _sched(monkeypatch, order, max_tokens=1024, min_tokens=64, weight=256)
        run = _Run(sched, seq_factory, _Delayer(others=[200000.0] * 7))
        run.add("long", 2000)
        run.step()
        run.add("short", 64)
        step = run.step()
        assert sched._tick_cost <= sched._tick_cost_limit
        return step

    assert second_step("fifo") == [("long", 512, 384, False)]
    assert second_step("srpt") == [("long", 512, 320, False), ("short", 0, 64, True)]


def test_srpt_holds_nothing_for_a_prompt_it_cannot_admit(monkeypatch, seq_factory):
    """No slot in `running`: the new prompt cannot be admitted this step, so
    the running prompt keeps the whole budget."""
    sched = _sched(monkeypatch, "srpt", max_tokens=256, max_num_seqs=1)
    run = _Run(sched, seq_factory)
    run.add("long", 1000)
    assert run.step() == [("long", 0, 256, False)]
    run.add("short", 16)
    assert run.step() == [("long", 256, 256, False)]
    assert run.waiting() == ["short"]


def test_srpt_holds_only_what_the_step_can_take(monkeypatch, seq_factory):
    """Room is left for the new prompts that fit the batch budget, not for
    the whole queue: the rest wait their turn and nothing counts as unused."""
    sched = _sched(monkeypatch, "srpt", max_tokens=128, long_cap=64)
    run = _Run(sched, seq_factory)
    run.add("long", 1000)
    run.step()
    for name in ("s0", "s1", "s2", "s3"):
        run.add(name, 64)
    assert run.step() == [("s0", 0, 64, True), ("s1", 0, 64, True)]
    assert not sched._admit_hold_missed
    assert run.step() == [("s2", 0, 64, True), ("s3", 0, 64, True)]
    assert run.step() == [("long", 64, 64, False)]


def test_srpt_stops_holding_after_a_hold_went_unused(monkeypatch, seq_factory):
    """The room left for a new prompt that was then not admitted (here its
    allocation fails) is not left again on the next step."""
    sched = _sched(monkeypatch, "srpt", max_tokens=256)
    run = _Run(sched, seq_factory)
    run.add("long", 2000)
    assert run.step() == [("long", 0, 256, False)]
    run.add("short", 16)
    allocate = sched.block_manager.allocate
    monkeypatch.setattr(sched.block_manager, "allocate", lambda *a, **k: False)
    assert run.step() == [("long", 256, 192, False)]
    assert sched._admit_hold_missed
    assert run.step() == [("long", 448, 256, False)]
    assert not sched._admit_hold_missed
    monkeypatch.setattr(sched.block_manager, "allocate", allocate)
    assert run.step() == [("long", 704, 192, False), ("short", 0, 16, True)]


# ---------------------------------------------------------------------------
# the age bound
# ---------------------------------------------------------------------------


def test_a_prompt_past_the_age_bound_goes_first(monkeypatch, seq_factory):
    run = _Run(
        _sched(monkeypatch, "short_new", max_age_ms=3000, max_tokens=128), seq_factory
    )
    run.add("old", 400)
    run.sched.clock[0] += 3.5
    run.add("new", 16)
    assert run.step() == [("old", 0, 128, False)]


def test_within_the_age_bound_the_shorter_goes_first(monkeypatch, seq_factory):
    run = _Run(
        _sched(monkeypatch, "short_new", max_age_ms=3000, max_tokens=128), seq_factory
    )
    run.add("old", 400)
    run.sched.clock[0] += 2.5
    run.add("new", 16)
    assert run.step() == [("new", 0, 16, True), ("old", 0, 64, False)]


def test_aged_prompts_go_oldest_first(monkeypatch, seq_factory):
    run = _Run(
        _sched(monkeypatch, "short_new", max_age_ms=1000, max_tokens=64), seq_factory
    )
    run.add("first", 400)
    run.sched.clock[0] += 0.5
    run.add("second", 16)
    run.sched.clock[0] += 2.0
    run.add("fresh", 8)
    assert [entry[0] for entry in run.step()] == ["first"]
    assert run.waiting() == ["second", "fresh"]


def test_the_age_of_a_running_prompt_counts_from_its_last_chunk(
    monkeypatch, seq_factory
):
    """A long prompt served on every step is never "aged", however long ago
    it arrived: shorter new prompts keep going ahead of it. Once it has gone
    the bound without a chunk it leads the step, and nothing is held back
    from it."""
    sched = _sched(monkeypatch, "srpt", max_age_ms=1000, max_tokens=128, long_cap=64)
    run = _Run(sched, seq_factory)
    run.add("long", 4000)
    run.step(dt=0.4)
    for i in range(6):
        run.add(f"s{i}", 16)
        step = run.step(dt=0.4)
        assert step == [("long", 64 * (i + 1), 64, False), (f"s{i}", 0, 16, True)]
    # 2.8 s after it arrived, served on the last step: still ranked by size
    assert sched._admit_rank(run.seqs["long"], 3552, sched.clock[0] + 0.4) == (2, 3552)
    # new prompts that fill the budget ahead of it
    for i in range(4):
        run.add(f"x{i}", 64)
        run.add(f"y{i}", 64)
    assert [entry[0] for entry in run.step(dt=0.4)] == ["x0", "y0"]
    assert [entry[0] for entry in run.step(dt=0.4)] == ["x1", "y1"]
    # 1.2 s without a chunk: past the bound, it leads on a full chunk
    assert run.step(dt=0.4) == [("long", 448, 64, False), ("x2", 0, 64, True)]
    assert [entry[0] for entry in run.step(dt=0.4)] == ["y2", "x3"]


# ---------------------------------------------------------------------------
# what the order must not touch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["fifo", "short_new", "srpt"])
def test_a_cancelled_waiting_prompt_is_dropped_where_it_stands(
    monkeypatch, seq_factory, order
):
    sched = _sched(monkeypatch, order, max_tokens=64)
    run = _Run(sched, seq_factory)
    run.add("a", 40)
    gone = run.add("gone", 4)
    run.add("b", 8)
    gone.status = SequenceStatus.ABORTED
    admitted = [entry[0] for entry in run.step()]
    assert sorted(admitted) == ["a", "b"]
    assert gone not in sched.waiting and gone not in sched.running
    assert [seq.id for seq in sched._rejected] == [gone.id]


@pytest.mark.parametrize("order", ["short_new", "srpt"])
def test_the_per_prompt_cap_and_the_budget_size_every_chunk(
    monkeypatch, seq_factory, order
):
    run = _Run(_sched(monkeypatch, order, max_tokens=32, long_cap=12), seq_factory)
    run.add("a", 100)
    run.add("b", 60)
    run.add("c", 6)
    for _ in range(40):
        step = run.step()
        assert sum(entry[2] for entry in step) <= 32
        for name, start, tokens, final in step:
            assert tokens <= 12
            assert final == (start + tokens == run.seqs[name].num_tokens)
        if len(run.done) == 3:
            break
    assert run.prefilled == {"a": 100, "b": 60, "c": 6}


def _random_run(monkeypatch, seq_factory, order, seed, ticks=300, cancel=True):
    """A random queue: arrivals, delayer vetoes, peer costs and cancellations
    drawn from `seed` alone, so every order sees the same ones."""
    rng = random.Random(seed)
    sched = _sched(
        monkeypatch,
        order,
        max_age_ms=rng.choice([0, 500, 3000]),
        max_tokens=rng.choice([128, 256, 512]),
        long_cap=rng.choice([0, 64, 128]),
        slack=rng.choice([0.1, 1.0]),
        min_tokens=rng.choice([32, 64]),
        max_num_seqs=rng.choice([4, 8]),
    )
    drain = 3000
    allow = [rng.random() < 0.7 for _ in range(ticks + drain)]
    delayer = _Delayer(others=[rng.choice([3e3, 3e4, 3e5])] * 7, allow=allow)
    arrivals = [
        [
            rng.choice([4, 8, 16, 40, 100, 300, 700])
            for _ in range(rng.choice([0, 0, 0, 1, 1, 2]))
        ]
        for _ in range(ticks)
    ]
    cancels = [
        rng.random() if cancel and rng.random() < 0.05 else None for _ in range(ticks)
    ]
    run = _Run(sched, seq_factory, delayer)
    cancelled = set()
    limits = SimpleNamespace(
        budget=sched.max_num_batched_tokens,
        cap=sched.long_prefill_token_threshold,
        floor=sched._balance_min_tokens,
    )
    count = 0
    for tick in range(ticks + drain):
        if tick < ticks:
            for size in arrivals[tick]:
                run.add(f"r{count}", size)
                count += 1
            waiting = [s for s in sched.waiting if s.status == SequenceStatus.WAITING]
            if waiting and cancels[tick] is not None:
                victim = waiting[int(cancels[tick] * len(waiting))]
                victim.status = SequenceStatus.ABORTED
                cancelled.add(run.names[victim.id])
        step = run.step(dt=0.05)
        yield tick, step, run, sched, limits, cancelled
        if tick >= ticks and not sched.waiting and not sched.running:
            break
    yield None, None, run, sched, limits, cancelled


@pytest.mark.parametrize("order", ["fifo", "short_new", "srpt"])
@pytest.mark.parametrize("seed", range(12))
def test_random_queues_admit_everything_once_within_every_cap(
    monkeypatch, seq_factory, order, seed
):
    """Whatever the order: every request that was not cancelled prefills
    exactly its tokens, contiguously from 0, and is admitted to `running`
    once; a cancelled one prefills nothing more; and every step stays within
    the batch budget, the per-prompt cap and the step's cost target."""
    position = {}
    for tick, step, run, sched, limits, cancelled in _random_run(
        monkeypatch, seq_factory, order, seed
    ):
        if tick is None:
            break
        assert sum(entry[2] for entry in step) <= limits.budget
        assert len({entry[0] for entry in step}) == len(step)
        for name, start, tokens, final in step:
            seq = run.seqs[name]
            assert tokens > 0
            assert start == position.get(name, 0), "a chunk starts where the last ended"
            position[name] = start + tokens
            assert position[name] <= seq.num_tokens
            assert final == (position[name] == seq.num_tokens)
            if limits.cap:
                assert tokens <= limits.cap
        if step and sched._tick_cost_limit < float("inf"):
            # within the target, or on the floor a trimmed rank may always take
            assert (
                sched._tick_cost <= sched._tick_cost_limit
                or sum(entry[2] for entry in step) <= limits.floor
            )
        partial = sum(1 for seq in sched.running if seq.is_partial_prefill)
        assert sched._partial_prefill_count == partial
        ids = [seq.id for seq in (*sched.waiting, *sched.running)]
        assert len(ids) == len(set(ids))
    assert not sched.waiting and not sched.running
    for name, seq in run.seqs.items():
        if name in cancelled:
            assert name not in run.done
        else:
            assert run.prefilled[name] == seq.num_tokens, name
            assert name in run.done


@pytest.mark.parametrize("seed", range(6))
def test_every_order_admits_the_same_requests_and_tokens(
    monkeypatch, seq_factory, seed
):
    """The order only reorders: the requests that finish, and the tokens each
    prefilled, are the same under every mode."""
    totals = {}
    for order in ("fifo", "short_new", "srpt"):
        for _, _, run, sched, _, _ in _random_run(
            monkeypatch, seq_factory, order, seed, cancel=False
        ):
            pass
        assert not sched.waiting and not sched.running
        totals[order] = (sorted(run.done), dict(run.prefilled))
    assert totals["short_new"] == totals["fifo"]
    assert totals["srpt"] == totals["fifo"]


@pytest.mark.parametrize("order", ["short_new", "srpt"])
def test_no_prompt_waits_past_the_age_bound_behind_shorter_ones(
    monkeypatch, seq_factory, order
):
    """A stream of short prompts that fills every step, so the queue only
    grows and every prompt in it ages: the long prompt still gets a chunk at
    least every bound plus a step, and finishes."""
    sched = _sched(monkeypatch, order, max_age_ms=1000, max_tokens=128, long_cap=64)
    run = _Run(sched, seq_factory)
    run.add("long", 1920)
    dt, count = 0.25, 0
    for _ in range(400):
        if "long" in run.done:
            break
        for _ in range(2):
            run.add(f"s{count}", 64)
            count += 1
        run.step(dt=dt)
    assert "long" in run.done
    assert run.prefilled["long"] == 1920
    served = [run.seqs["long"].arrive_time, *run.served["long"]]
    gaps = [b - a for a, b in itertools.pairwise(served)]
    assert max(gaps) <= 1.0 + dt + 1e-9
