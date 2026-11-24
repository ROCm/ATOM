"""ATOM_PREFILL_ORDER=cached_first: `Scheduler._order_waiting`."""

from collections import deque
from types import SimpleNamespace

from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus

BLOCK = 128


def _seq(
    name,
    prompt,
    cached_blocks,
    arrive=99.0,
    tier=0,
    status=SequenceStatus.WAITING,
    resume=False,
):
    return SimpleNamespace(
        name=name,
        num_prompt_tokens=prompt,
        cached_blocks=cached_blocks,
        arrive_time=arrive,
        offload_joint=SimpleNamespace(kv_prefix_tokens=tier),
        status=status,
        resume=resume,
    )


def _sched(seqs, max_wait=60.0):
    s = Scheduler.__new__(Scheduler)
    s.waiting = deque(seqs)
    s.block_manager = SimpleNamespace(
        hash_block_size=BLOCK, can_allocate_memoized=lambda seq: seq.cached_blocks
    )
    s._is_offload_prefill_resume = lambda seq: seq.resume
    s._prefill_order_max_wait_s = max_wait
    return s


def _order(s, now=100.0):
    s._order_waiting(now)
    return [q.name for q in s.waiting]


def test_fewest_uncached_first_ties_by_arrival():
    cold = _seq("cold", 100_000, 0, arrive=90)
    hot = _seq("hot", 100_000, 760, arrive=95)  # 2,720 left
    warm = _seq("warm", 50_000, 300, arrive=91)  # 11,600 left
    hot2 = _seq("hot2", 100_000, 760, arrive=96)
    assert _order(_sched([cold, hot, warm, hot2])) == ["hot", "hot2", "warm", "cold"]


def test_tier_prefix_counts_when_longer():
    a = _seq("a", 100_000, 10, tier=99_000)  # 1,000 left through the tier
    b = _seq("b", 100_000, 700)  # 10,400 left
    assert _order(_sched([b, a])) == ["a", "b"]


def test_aged_requests_go_first_in_arrival_order():
    old_cold = _seq("old_cold", 500_000, 0, arrive=10)  # waited 90 s >= 60
    older_cold = _seq("older_cold", 400_000, 0, arrive=5)
    hot = _seq("hot", 100_000, 780, arrive=99)
    assert _order(_sched([hot, old_cold, older_cold])) == [
        "old_cold",
        "older_cold",
        "hot",
    ]
    # Equal ages keep their queue position among aged requests.


def test_resumes_and_aborted_keep_the_front_and_misfits_go_last():
    resume = _seq("resume", 100_000, 0, resume=True)
    aborted = _seq("aborted", 1, 0, status=SequenceStatus.ABORTED)
    misfit = _seq("misfit", 10, -1)
    hot = _seq("hot", 100_000, 780)
    assert _order(_sched([misfit, hot, resume, aborted])) == [
        "resume",
        "aborted",
        "hot",
        "misfit",
    ]


def test_order_keeps_every_request_once():
    seqs = [_seq(f"s{i}", 1000 * (i + 1), i % 3, arrive=float(i)) for i in range(50)]
    s = _sched(seqs)
    s._order_waiting(100.0)
    assert sorted(q.name for q in s.waiting) == sorted(q.name for q in seqs)
