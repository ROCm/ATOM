# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""PP-aware offload KV status aggregator.

Each PP stage holds different layers, so a request's offload load/save is
complete only when all stages report done. Any single stage failure fails
the entire request.
"""

from __future__ import annotations

from atom.kv_transfer.disaggregation.types import KVConnectorOutput, ReqId


class PPKVAggregator:
    """Aggregate offload ``finished_loading / failed_loading / finished_saving``
    across PP stages.

    Call :meth:`ingest` once per (pp_rank, output) pair.  The method returns a
    :class:`KVConnectorOutput` containing only the request IDs that have
    reached a terminal state across all stages.

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

        failed = set(self._failed_loading.keys())
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
