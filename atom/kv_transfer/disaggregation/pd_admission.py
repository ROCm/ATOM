# SPDX-License-Identifier: MIT
"""CPU-only D-ready quorum and phase timeout configuration."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PDTimeouts:
    admission: float = 300.0
    compute: float = 300.0
    transfer: float = 300.0

    @classmethod
    def from_config(cls, config):
        values = {}
        for phase in ("admission", "compute", "transfer"):
            key = f"{phase}_timeout_s"
            value = config.get(key, 300.0)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be a positive finite number")
            values[phase] = float(value)
        return cls(**values)


@dataclass
class DReadyAdmission:
    created: float = field(default_factory=lambda: time.monotonic())
    req_id: int | str | None = None
    prompt_digest: str | None = None
    consumer_count: int = 0
    consumers: dict[int, tuple] = field(default_factory=dict)
    ready_at: float | None = None
    error: str | None = None
    reported: bool = False

    def register(self, req_id, digest):
        if self.req_id is not None and self.req_id != req_id:
            self.error = "phase=admission_wait transfer ID reused by another request"
        if (self.req_id is not None or self.consumers) and self.prompt_digest != digest:
            self.error = "phase=admission_wait P/D prompt token IDs differ"
        self.req_id = req_id
        self.prompt_digest = digest

    def observe(self, data):
        if self.error:
            return
        count, rank = data.get("consumer_tp_size"), data.get("consumer_tp_rank")
        if (
            type(count) is not int
            or count <= 0
            or type(rank) is not int
            or not 0 <= rank < count
        ):
            self.error = "phase=admission_wait invalid D-ready rank/count"
            return
        digest = data.get("prompt_digest")
        identity = tuple(
            data.get(k)
            for k in ("notify_host", "notify_port", "request_id", "write_nonce")
        )
        if self.consumer_count and self.consumer_count != count:
            self.error = "phase=admission_wait inconsistent D-ready topology"
        elif (
            self.req_id is not None or self.consumers
        ) and self.prompt_digest != digest:
            self.error = "phase=admission_wait P/D prompt token IDs differ"
        elif rank in self.consumers and self.consumers[rank] != identity:
            self.error = "phase=admission_wait conflicting D-ready receive attempt"
        else:
            self.consumer_count = count
            self.prompt_digest = digest
            self.consumers[rank] = identity
            if len(self.consumers) == count and self.ready_at is None:
                self.ready_at = time.monotonic()

    def expire(self, timeouts):
        if self.error is None and self.ready_at is None:
            elapsed = time.monotonic() - self.created
            if elapsed >= timeouts.admission:
                self.error = (
                    f"phase=admission_wait D-ready timed out elapsed_s={elapsed:.3f} "
                    f"limit_s={timeouts.admission:.3f} ready_ranks={len(self.consumers)}/{self.consumer_count}"
                )
        return self.error
