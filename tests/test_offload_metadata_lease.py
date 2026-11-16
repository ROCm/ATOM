import gc
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from atom.kv_transfer.offload import metadata_lease as ml
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


class Storage:
    """Stand-in for a pinned CPU + GPU ID buffer."""


def live_storage():
    # gc clears weakrefs to cyclic garbage even when a finalizer resurrects it,
    # so count the objects the collector still tracks instead.
    gc.collect()
    return sum(isinstance(x, Storage) for x in gc.get_objects())


class ThreadPools(threading.local):
    pool = None


def _worker_with_pending_lease_exits(event):
    """A worker's ID upload fails (no owner returned), then the worker exits."""
    tls = ThreadPools()

    def worker():
        tls.pool = MetadataLeasePool(lambda n: Storage())
        lease = tls.pool.acquire(8)
        tls.pool.retire(lease, completion=event, safe=True)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()


def test_pending_storage_outlives_its_exited_worker_until_completion():
    assert live_storage() == 0
    event = Event()
    _worker_with_pending_lease_exits(event)
    assert live_storage() == 1, "storage freed while its upload may be in flight"
    # Another worker's next acquire polls the orphan; still incomplete: kept.
    other = MetadataLeasePool(lambda n: Storage())
    other.acquire(1)
    assert live_storage() == 2 and len(ml._ORPHANED_PENDING) == 1
    event.done = True
    other.acquire(1)  # the orphan completed: released (other now holds 2 buffers)
    assert live_storage() == 2 and not ml._ORPHANED_PENDING


def test_orphaned_pending_storage_is_quarantined_if_completion_is_unknown():
    ml._QUARANTINED_STORAGE.clear()
    assert live_storage() == 0
    _worker_with_pending_lease_exits(Event(error=True))
    gc.collect()
    MetadataLeasePool(lambda n: Storage()).acquire(1)
    assert live_storage() == 1  # the orphan; the polling pool's buffer is gone
    assert not ml._ORPHANED_PENDING and len(ml._QUARANTINED_STORAGE) == 1
    ml._QUARANTINED_STORAGE.clear()


def test_sweep_releases_completed_orphans_behind_an_incomplete_one():
    done, waiting = Event(done=True), Event()
    leases = [
        ml.MetadataLease(Storage(), 1, "pending", completion=event)
        for event in (done, waiting)
    ]
    ml._ORPHANED_PENDING[:] = leases  # [completed, incomplete]
    del leases
    assert live_storage() == 2
    ml._sweep_orphaned_pending()
    assert len(ml._ORPHANED_PENDING) == 1
    assert ml._ORPHANED_PENDING[0].completion is waiting
    assert live_storage() == 1
    ml._ORPHANED_PENDING.clear()
