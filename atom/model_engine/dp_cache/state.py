# SPDX-License-Identifier: MIT
"""Request-local cache state carried by Sequence across the engine boundary."""

from dataclasses import dataclass

CacheAdmissionEvent = tuple[object, int, int]
CacheWorkEvent = tuple[object, int, int, int]


@dataclass(slots=True)
class DPCacheState:
    """Router initializes the request; the receiving scheduler updates its copy.

    Coordinator Session/Placement records remain authoritative for ownership.
    Only admission and work events reconcile those records with this state.
    """

    # Routing contract. Zero migration_id means no prepared migration.
    migration_id: int = 0
    minimum_hit: int = 0
    prepare_timeout: float = 5.0

    # Scheduler-owned preparation/confirmation, using its local monotonic clock.
    prepare_deadline: float = 0.0
    admission_sent: bool = False
    alignment_boundary: int = 0
    retry_reason: str | None = None
    recoverable_hit: int | None = None

    # Scheduler work reporting. Revisions order events at the coordinator;
    # epochs fence forwards issued before a local preemption.
    last_work: int | None = None
    work_revision: int = 0
    work_epoch: int = 0
