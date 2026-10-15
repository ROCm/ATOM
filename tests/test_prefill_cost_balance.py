"""DP prefill cost balancing (on under DP attention with more than one rank).

Under DP attention every rank prefills in the same lockstep step, so the step
lasts as long as the costliest rank's chunk. With balancing on, the ranks
exchange their estimated chunk costs on the PrefillDelayer's per-tick
all_reduce (`prefill_cost` in, `rank_prefill_costs` out) and aim at
min(costliest, mean x (1 + slack)): a costlier rank shrinks its chunk.

With weight 0 a chunk of k tokens from position 0 costs k * k / 2.
"""

import math
from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.model_engine import block_manager as bm_module
from atom.model_engine.scheduler import Scheduler


class _Delayer:
    """Delayer stub that releases prefill every tick and answers the cost
    exchange with this rank's cost followed by `others`."""

    max_queue_ms = None

    def __init__(self, others=(), is_local=False):
        self.others = list(others)
        self.is_local = is_local
        self.reported = []
        self.pending = []
        self.rank_prefill_costs = []

    def protects_decode(self, running_decode_batch: int) -> bool:
        return False

    def should_allow_prefill(
        self, prefillable, pending_tokens, prefill_cost=0.0, **kwargs
    ):
        self.reported.append(prefill_cost)
        self.pending.append(pending_tokens)
        self.rank_prefill_costs = [prefill_cost, *self.others]
        return True


def _sched(
    monkeypatch,
    *,
    dp_attention=True,
    weight=0,
    slack=0.1,
    min_tokens=8,
    max_tokens=64,
    dp_size=8,
    prefix_caching=False,
):
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT", str(weight))
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_SLACK", str(slack))
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_MIN_TOKENS", str(min_tokens))
    return Scheduler(
        MockConfig(
            max_num_seqs=8,
            num_kvcache_blocks=64,
            kv_cache_block_size=4,
            max_model_len=256,
            max_num_batched_tokens=max_tokens,
            enable_prefix_caching=prefix_caching,
            enable_dp_attention=dp_attention,
            parallel_config=SimpleNamespace(data_parallel_size=dp_size),
        )
    )


def _cached_prefix(sched, seq_factory, tokens):
    """Publish `tokens` in the prefix cache, release them, and return the
    blocks that hold them."""
    bm = sched.block_manager
    first = seq_factory(tokens)
    bm.allocate(first, 0)
    bm.hash_blocks(first, len(tokens))
    resident = list(first.block_table)
    bm.deallocate(first)
    return resident


def _prefill_tokens(sched):
    batch, _ = sched.schedule()
    return int(batch.total_tokens_num)


def test_off_without_dp_attention(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, dp_attention=False)
    delayer = _Delayer(others=[1.0] * 7)
    sched.set_prefill_delayer(delayer)
    sched.add(seq_factory(list(range(100, 164))))
    assert not sched.cost_balance
    assert _prefill_tokens(sched) == 64
    assert delayer.reported == [0.0]


def test_off_with_a_single_dp_rank(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, dp_size=1)
    delayer = _Delayer()
    sched.set_prefill_delayer(delayer)
    sched.add(seq_factory(list(range(100, 164))))
    assert not sched.cost_balance
    assert _prefill_tokens(sched) == 64
    assert delayer.reported == [0.0]


def test_per_tick_probes_keep_the_head_walk(monkeypatch, seq_factory):
    """The estimate and the delayer's head check probe the waiting head on
    every tick with prefill work. Its prompt-block hashes and its prefix walk
    are kept between ticks (`can_allocate_memoized`), so a repeat probe walks
    and hashes nothing and costs the same."""
    sched = _sched(monkeypatch, prefix_caching=True)
    prefix = list(range(100, 132))
    _cached_prefix(sched, seq_factory, prefix)
    sched.add(seq_factory(prefix + list(range(500, 532))))
    bm = sched.block_manager
    hashed = []
    real = bm.compute_hash
    monkeypatch.setattr(bm, "compute_hash", lambda *a: hashed.append(a) or real(*a))
    # The fast prefix walk hashes a missing chain in bulk instead.
    bulk = bm_module.chained_block_hashes
    monkeypatch.setattr(
        bm_module, "chained_block_hashes", lambda *a: hashed.append(a) or bulk(*a)
    )
    walks = []
    walk = bm._walk_prefix
    monkeypatch.setattr(bm, "_walk_prefix", lambda s: walks.append(s) or walk(s))
    # 8 cached blocks of 4 tokens: the chunk is the 32 new tokens after them.
    cost = sched._estimate_prefill_chunk_cost()
    assert cost == sched._chunk_cost(32, 32)
    assert hashed and len(walks) == 1
    first_probe = len(hashed)
    assert sched._estimate_prefill_chunk_cost() == cost
    assert sched._can_admit_head_prefill() and sched._can_admit_head_prefill()
    assert len(hashed) == first_probe and len(walks) == 1


def test_balanced_admission_takes_exactly_the_cached_prefix(monkeypatch, seq_factory):
    """The estimate only sizes the chunk. Admission still reuses exactly the
    cached prefix blocks, and the shrunk chunk starts right after them."""
    sched = _sched(monkeypatch, prefix_caching=True)
    sched.set_prefill_delayer(_Delayer(others=[512.0] * 7))
    prefix = list(range(100, 132))
    resident = _cached_prefix(sched, seq_factory, prefix)
    seq = seq_factory(prefix + list(range(500, 564)))
    sched.add(seq)
    sched._estimate_prefill_chunk_cost()  # leave kept hashes behind
    # Own 64-token chunk after 32 cached costs 4096, seven ranks at 512:
    # target = 1.1 * 960 = 1056, i.e. 24 tokens.
    batch, _ = sched.schedule()
    assert seq.num_cached_tokens == 32
    assert list(seq.block_table[:8]) == resident
    assert list(batch.num_scheduled_tokens) == [24]


@pytest.mark.parametrize("start", [0, 1000, 250_000])
def test_room_is_the_largest_chunk_within_the_cost_left(monkeypatch, start):
    sched = _sched(monkeypatch, weight=4096, min_tokens=0, max_tokens=1 << 20)
    sched._tick_cost_limit = 5e8
    sched._tick_cost = 1e8
    room = sched._prefill_room(0, start)
    left = sched._tick_cost_limit - sched._tick_cost
    assert sched._chunk_cost(start, room) <= left
    assert sched._chunk_cost(start, room + 1) > left


def test_costliest_rank_shrinks_to_the_target(monkeypatch, seq_factory):
    sched = _sched(monkeypatch)
    # Own 64-token chunk costs 2048; seven ranks at 512: target =
    # min(2048, 1.1 * 704) = 774.4, i.e. 39 tokens.
    delayer = _Delayer(others=[512.0] * 7)
    sched.set_prefill_delayer(delayer)
    sched.add(seq_factory(list(range(100, 164))))
    assert _prefill_tokens(sched) == 39
    assert delayer.reported == [2048.0]


def test_shrinking_never_goes_below_the_floor(monkeypatch, seq_factory):
    # Seven near-idle ranks: target = 1.1 * (2048 + 3.5) / 8 = 282, i.e. 23
    # tokens, under the 32-token floor.
    sched = _sched(monkeypatch, min_tokens=32)
    sched.set_prefill_delayer(_Delayer(others=[0.5] * 7))
    sched.add(seq_factory(list(range(100, 164))))
    assert _prefill_tokens(sched) == 32


def test_cheaper_rank_keeps_its_chunk_without_room_to_grow(monkeypatch, seq_factory):
    sched = _sched(monkeypatch)
    sched.set_prefill_delayer(_Delayer(others=[1e9] * 7))
    sched.add(seq_factory(list(range(100, 164))))
    assert _prefill_tokens(sched) == 64


def test_idle_rank_reports_no_cost(monkeypatch):
    sched = _sched(monkeypatch)
    delayer = _Delayer(others=[512.0] * 7)
    sched.set_prefill_delayer(delayer)
    assert sched.schedule() is None
    assert delayer.reported == [0.0]


def test_no_rank_with_prefill_work_balances_nothing(monkeypatch):
    sched = _sched(monkeypatch)
    sched.set_prefill_delayer(_Delayer(others=[0.0] * 7))
    assert sched.schedule() is None
    assert sched._tick_cost_limit == math.inf
    assert sched._balance_stats[0] == 0


def test_local_pending_probe_ignores_the_previous_step_limit(monkeypatch, seq_factory):
    """Nothing that sizes a chunk ahead of this tick's balancing decision (here
    the local coalescer's pending probe, `_partial_prefill_chunk`) may see the
    previous step's cost limit, or it cuts the probe down to the floor."""
    sched = _sched(monkeypatch)
    delayer = _Delayer(others=[512.0] * 7, is_local=True)
    sched.set_prefill_delayer(delayer)
    s_partial = seq_factory(list(range(100, 196)))
    s_decode = seq_factory([1, 2, 3, 4])
    sched.add(s_partial)
    sched.add(s_decode)
    # Own 64-token chunk costs 2048, seven ranks at 512: target = 774.4, i.e.
    # 39 tokens of the 96-token prompt, then the 4-token prompt in what is left.
    batch, _ = sched.schedule()
    assert list(batch.num_scheduled_tokens) == [39, 4]
    assert sched._tick_cost_limit < math.inf

    s_partial.num_cached_tokens = 39
    s_partial.is_partial_prefill = True
    sched._partial_prefill_count = 1
    s_decode.num_cached_tokens = s_decode.num_prompt_tokens
    s_decode.append_token(99)
    s_decode.is_partial_prefill = False

    sched.schedule()
    # The 57 tokens left of the prompt, not the 8-token floor.
    assert delayer.pending[-1] == 57


def test_balances_a_prefill_released_inside_the_decode_window(monkeypatch, seq_factory):
    """With no decode on any rank the real delayer fires through its
    post-prefill decode-protection window (its usual exit under DP attention).
    The rank must have estimated its cost for that tick, or nothing balances."""
    import torch.distributed as dist

    from atom.model_engine.prefill_delayer import PrefillDelayer

    sched = _sched(monkeypatch)
    group = dist.ProcessGroupGloo(dist.HashStore(), 0, 1)
    delayer = PrefillDelayer(
        dp_size=1,
        cpu_group=group,
        max_num_batched_tokens=sched.max_num_batched_tokens,
        prefill_decode_interval=5,
        exchange_prefill_costs=True,
    )
    sched.set_prefill_delayer(delayer)
    delayer._first = False
    delayer._decode_interval_remaining = 3
    sched.add(seq_factory(list(range(100, 164))))
    assert _prefill_tokens(sched) == 64
    assert delayer.rank_prefill_costs == [2048.0]
    assert sched._balance_stats[0] == 1


def test_costs_ride_the_delayer_reduce_on_a_stateless_gloo_group():
    """The engine's DP group is created without a default process group, so the
    delayer takes its rank from the group, not torch.distributed.get_rank()."""
    import torch.distributed as dist

    from atom.model_engine.prefill_delayer import PrefillDelayer

    assert not dist.is_initialized()
    group = dist.ProcessGroupGloo(dist.HashStore(), 0, 1)
    delayer = PrefillDelayer(
        dp_size=1,
        cpu_group=group,
        max_num_batched_tokens=64,
        exchange_prefill_costs=True,
    )
    delayer.should_allow_prefill(prefillable=True, pending_tokens=8, prefill_cost=3.5e9)
    assert delayer.rank_prefill_costs == [3.5e9]
    # Without the exchange the buffer keeps its 8 decision slots only.
    plain = PrefillDelayer(dp_size=1, cpu_group=group, max_num_batched_tokens=64)
    plain.should_allow_prefill(prefillable=True, pending_tokens=8, prefill_cost=3.5e9)
    assert plain.rank_prefill_costs == []
