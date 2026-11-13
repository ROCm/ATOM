from concurrent.futures import ThreadPoolExecutor

import pytest

from atom.kv_transfer.offload.metadata_lease import (
    MetadataLeasePool,
    MetadataPoolExhausted,
)


class Event:
    def __init__(self, *, done=False, error=False):
        self.done = done
        self.error = error

    def query(self):
        if self.error:
            raise RuntimeError("completion unknown")
        return self.done


def test_reuse_waits_for_completion_and_advances_generation():
    pool = MetadataLeasePool(lambda n: object())
    first = pool.acquire(3)
    generation = first.generation
    event = Event()
    pool.retire(first, completion=event, safe=True)
    second = pool.acquire(3)
    assert second is not first
    pool.retire(second, safe=False)
    event.done = True
    reused = pool.acquire(4)
    assert reused is first and reused.generation > generation
    assert second.state == "quarantined"


def test_unknown_completion_never_bypasses_hard_limits():
    allocations = []

    def factory(n):
        allocations.append(n)
        return object()

    pool = MetadataLeasePool(factory, max_leases=2, max_elements=16)
    first, second = pool.acquire(8), pool.acquire(8)
    pool.retire(first, completion=Event(error=True), safe=True)
    with pytest.raises(MetadataPoolExhausted):
        pool.acquire(1)
    assert first.state == "quarantined"
    assert second.state == "leased"
    assert allocations == [8, 8]


def test_retirement_is_exactly_once_and_thread_owned():
    pool = MetadataLeasePool(lambda n: object())
    lease = pool.acquire(8)
    with (
        ThreadPoolExecutor(1) as executor,
        pytest.raises(RuntimeError, match="construction thread"),
    ):
        executor.submit(pool.retire, lease, safe=False).result()
    assert lease.state == "leased"
    pool.retire(lease, completion=Event(done=True), safe=True)
    with pytest.raises(ValueError, match="owned/live"):
        pool.retire(lease, safe=False)


def test_repeated_transfers_reuse_one_bounded_allocation():
    allocations = []
    pool = MetadataLeasePool(lambda n: allocations.append(n) or object())
    for _ in range(100):
        lease = pool.acquire(25)
        pool.retire(lease, completion=Event(done=True), safe=True)
    assert allocations == [32]
    larger = pool.acquire(65)
    assert larger.capacity == 128
    assert len(pool.leases) == 1
