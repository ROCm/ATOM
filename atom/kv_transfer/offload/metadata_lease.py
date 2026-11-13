"""Thread-owned bounded reuse of metadata; uncertain GPU work stays retained."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_METADATA_ELEMENTS = 1 << 18
METADATA_GPU_RESERVE_BYTES = MAX_METADATA_ELEMENTS * 8  # int64 IDs
# Executor thread-local state can disappear when the worker closes. Unsafe
# storage must survive that teardown, not merely remain in its thread's pool.
_QUARANTINED_STORAGE: list[Any] = []


class MetadataPoolExhausted(RuntimeError):
    """The bounded pool cannot allocate without reusing unsafe metadata."""


@dataclass
class MetadataLease:
    buffer: Any
    capacity: int
    state: str = "leased"
    completion: Any = None
    pool: Any = None
    generation: int = 1


class MetadataLeasePool:
    def __init__(
        self,
        factory: Callable[[int], Any],
        *,
        max_leases=4,
        max_elements=MAX_METADATA_ELEMENTS,
    ):
        if max_leases <= 0 or max_elements <= 0:
            raise ValueError("positive metadata pool limits required")
        self.factory = factory
        self.leases: list[MetadataLease] = []
        self.max_leases, self.max_elements = max_leases, max_elements
        self.thread_id = threading.get_ident()

    def assert_thread(self):
        if threading.get_ident() != self.thread_id:
            raise RuntimeError("metadata pool belongs to its construction thread")

    def acquire(self, count: int) -> MetadataLease:
        self.assert_thread()
        if count <= 0:
            raise ValueError("positive metadata count required")
        for lease in self.leases:
            if lease.state == "pending":
                try:
                    if lease.completion.query():
                        lease.state = "free"
                except Exception:  # noqa: BLE001
                    # Any uncertain GPU work stays retained.
                    lease.state = "quarantined"
                    _QUARANTINED_STORAGE.append(lease.buffer)
        reusable = [x for x in self.leases if x.state == "free" and x.capacity >= count]
        if reusable:
            lease = min(reusable, key=lambda x: x.capacity)
            self.leases[:] = [x for x in self.leases if x is lease or x.state != "free"]
            lease.state, lease.completion = "leased", None
            lease.generation += 1
            return lease
        capacity = 1 << (count - 1).bit_length()
        # Retain live/pending/quarantined storage. At most one idle reusable
        # buffer remains after growth; dropping an idle buffer is device-safe.
        self.leases[:] = [x for x in self.leases if x.state != "free"]
        if (
            len(self.leases) >= self.max_leases
            or sum(x.capacity for x in self.leases) + capacity > self.max_elements
        ):
            raise MetadataPoolExhausted(
                "metadata limit reached; live/quarantined buffers cannot be reused"
            )
        lease = MetadataLease(self.factory(capacity), capacity, pool=self)
        self.leases.append(lease)
        return lease

    def retire(self, lease: MetadataLease, *, completion: Any = None, safe: bool):
        self.assert_thread()
        if not any(x is lease for x in self.leases) or lease.state != "leased":
            raise ValueError("metadata lease is not owned/live")
        lease.completion = completion
        lease.state = "pending" if safe and completion is not None else "quarantined"
        if lease.state == "quarantined":
            _QUARANTINED_STORAGE.append(lease.buffer)
