"""Scheduler-thread gate for native Store renewal.

Only immutable keys enter the existing renewal executor. Native read leases
expire by TTL; retiring a ticket is not a native unpin/remove operation.
"""

from dataclasses import dataclass
from enum import Enum
from time import monotonic


class RenewalState(Enum):
    PENDING = "pending"
    PINNED = "pinned"
    FAILED = "failed"


@dataclass
class _Ticket:
    owner: object
    identity: tuple
    future: object
    deadline: float


class RenewalGate:
    """No own thread. All ticket operations run on the scheduler thread."""

    def __init__(self, client, capacity=None, timeout_s=1.0, clock=monotonic):
        self.client = client
        self.capacity = client.capacity if capacity is None else capacity
        self.timeout_s = timeout_s
        self.clock = clock
        self.tickets = {}
        self.closed = False

    def _call(self, keys):
        with self.client.backend_slot.take(renewal=True):
            started = self.clock()
            held = self.client.backend.renew_keys(keys)
            return held, started, self.clock()

    def _retire(self, sid):
        ticket = self.tickets.pop(sid, None)
        if ticket is not None:
            ticket.future.cancel()
        # Running futures hold immutable keys only; no callbacks own Sequence.
        return ticket

    def cancel(self, sid, owner=None):
        ticket = self.tickets.get(sid)
        if ticket is not None and (owner is None or ticket.owner is owner):
            self._retire(sid)

    def request(self, owner, sid, keys, physical_identity):
        if self.closed:
            return RenewalState.FAILED
        identity = (tuple(keys), tuple(physical_identity))
        ticket = self.tickets.get(sid)
        if ticket is not None and (
            ticket.owner is not owner or ticket.identity != identity
        ):
            self._retire(sid)
            ticket = None
        if ticket is None:
            if len(self.tickets) >= self.capacity:
                return RenewalState.FAILED
            future = self.client.renew_executor.submit(self._call, identity[0])
            if future is None or future.cancelled():
                return RenewalState.FAILED
            ticket = _Ticket(
                owner,
                identity,
                future,
                self.clock() + self.timeout_s,
            )
            self.tickets[sid] = ticket
        if self.clock() >= ticket.deadline:
            self._retire(sid)
            return RenewalState.FAILED
        if not ticket.future.done():
            return RenewalState.PENDING
        try:
            _held, started, _completed = ticket.future.result()
        except Exception:  # noqa: BLE001 - failed renewal must wake the parked load
            self._retire(sid)
            return RenewalState.FAILED
        self._retire(sid)
        if self.clock() - started >= self.timeout_s:
            return RenewalState.FAILED
        # Preserve completed held=0: dispatch GET and report exact native failure.
        return RenewalState.PINNED

    def close(self):
        self.closed = True
        for sid in list(self.tickets):
            self._retire(sid)
