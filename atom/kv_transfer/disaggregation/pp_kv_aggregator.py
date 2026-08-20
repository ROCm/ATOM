# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""PP-aware offload KV status aggregator.

Each PP stage holds different layers, so a request's offload load/save is
complete only when all stages report done. A single stage failure fails the
whole request, but only once every stage has reached a terminal state.
"""

from __future__ import annotations

from atom.kv_transfer.disaggregation.types import KVConnectorOutput, ReqId


class PPKVAggregator:
    """Aggregate offload ``finished_loading / failed_loading / finished_saving``
    across PP stages.

    Call :meth:`ingest` once per (pp_rank, output) pair.  The method returns a
    :class:`KVConnectorOutput` containing only the request IDs that have
    reached a terminal state across all stages.

    A load fails the request if any stage reports a failure, but the failure is
    only emitted once every stage has reported either success or failure.

    Only offload-specific fields are tracked.  Mooncake P/D fields
    (``finished_sending``, ``finished_recving``) have their own PP-aware
    side-channel and must NOT flow through this aggregator.
    """

    def __init__(self, pp_size: int) -> None:
        if pp_size <= 0:
            raise ValueError(f"pp_size must be positive, got {pp_size}")
        self._pp_size = pp_size
        self._loading: dict[ReqId, set[int]] = {}
        self._saving: dict[ReqId, set[int]] = {}
        self._failed_loading: dict[ReqId, set[int]] = {}

    def ingest(self, pp_rank: int, output: KVConnectorOutput) -> KVConnectorOutput:
        for rid in output.finished_loading:
            self._loading.setdefault(rid, set()).add(pp_rank)
        for rid in output.failed_loading:
            self._failed_loading.setdefault(rid, set()).add(pp_rank)
        for rid in output.finished_saving:
            self._saving.setdefault(rid, set()).add(pp_rank)

        # A load is only terminal once every stage has reported one way or the
        # other. Reporting the failure at the first failing stage would wake
        # the request for recompute into blocks the remaining stages are still
        # loading into, and would drop the tally that suppresses their reports.
        failed = set()
        for rid in set(self._loading) | set(self._failed_loading):
            bad = self._failed_loading.get(rid, set())
            reported = self._loading.get(rid, set()) | bad
            if bad and len(reported) >= self._pp_size:
                failed.add(rid)
        done_loading = {
            rid for rid, stages in self._loading.items() if len(stages) >= self._pp_size
        } - failed
        done_saving = {
            rid for rid, stages in self._saving.items() if len(stages) >= self._pp_size
        }

        for rid in done_loading | failed:
            self._loading.pop(rid, None)
            self._failed_loading.pop(rid, None)
        for rid in done_saving:
            self._saving.pop(rid, None)

        return KVConnectorOutput(
            finished_loading=done_loading,
            failed_loading=failed,
            finished_saving=done_saving,
        )

    def has_pending(self) -> bool:
        """True while any request is still short of its per-stage quorum.

        The head's busy loop keeps polling downstream stages while this holds;
        the tallies only drain when the missing stages report in.
        """
        return bool(self._loading or self._saving or self._failed_loading)

    def reset(self) -> None:
        self._loading.clear()
        self._saving.clear()
        self._failed_loading.clear()
