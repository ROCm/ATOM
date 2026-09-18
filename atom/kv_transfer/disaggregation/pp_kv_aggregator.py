# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""PP-aware offload KV status aggregator.

Each PP stage holds different layers, so a request's offload load/save is
complete only when all stages report done. A single stage failure fails the
whole request, but only once every stage has reached a terminal state.
"""

from __future__ import annotations

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    ConnectorCompletionKey,
    KVConnectorOutput,
    ReqId,
)


class PPKVAggregator:
    """Aggregate offload ``finished_loading / failed_loading / finished_saving``
    and ``connector_completions`` across PP stages.

    Call :meth:`ingest` once per (pp_rank, output) pair.  The method returns a
    :class:`KVConnectorOutput` containing only the request IDs (and connector
    completion keys) that have reached a terminal state across all stages.

    A load fails the request if any stage reports a failure, but the failure is
    only emitted once every stage has reported either success or failure. A
    connector completion is failure-dominant the same way: each stage holds a
    slice of the layers a checkpoint/state boundary spans, so the boundary
    commits (``succeeded=True``) only if every stage succeeded, and any stage's
    failure sinks it -- but, like a load, only once every stage has reported.

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
        # Per connector-completion key: which stages have reported it, and
        # whether any stage reported it failed. The head's own TP aggregation
        # already collapsed each stage's TP ranks to one succeeded/failed
        # verdict per key (aggregator.py), so a stage reports a key at most once
        # per generation and PP quorum is just stage coverage.
        self._connector: dict[ConnectorCompletionKey, set[int]] = {}
        self._connector_failed: set[ConnectorCompletionKey] = set()

    def ingest(self, pp_rank: int, output: KVConnectorOutput) -> KVConnectorOutput:
        for rid in output.finished_loading:
            self._loading.setdefault(rid, set()).add(pp_rank)
        for rid in output.failed_loading:
            self._failed_loading.setdefault(rid, set()).add(pp_rank)
        for rid in output.finished_saving:
            self._saving.setdefault(rid, set()).add(pp_rank)
        for completion in output.connector_completions:
            self._connector.setdefault(completion.key, set()).add(pp_rank)
            if not completion.succeeded:
                self._connector_failed.add(completion.key)

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
        # A connector completion is terminal once every stage has reported it
        # (same all-stage quorum as a save); its verdict is the AND across
        # stages -- failed if any stage failed, else succeeded.
        done_connector = {
            key
            for key, stages in self._connector.items()
            if len(stages) >= self._pp_size
        }
        connector_completions = {
            ConnectorCompletion(
                channel=key[0],
                operation_id=key[1],
                succeeded=key not in self._connector_failed,
            )
            for key in done_connector
        }

        for rid in done_loading | failed:
            self._loading.pop(rid, None)
            self._failed_loading.pop(rid, None)
        for rid in done_saving:
            self._saving.pop(rid, None)
        for key in done_connector:
            self._connector.pop(key, None)
            self._connector_failed.discard(key)

        return KVConnectorOutput(
            finished_loading=done_loading,
            failed_loading=failed,
            finished_saving=done_saving,
            connector_completions=connector_completions,
        )

    def forget(self, rid: ReqId) -> None:
        """Drop a request's partial tallies: no further stage will report.

        A tally only drains on full quorum, so a stage whose completion is lost
        for good leaves its set here forever. Nothing else empties it -- the
        scheduler's own reclaim (`_reconcile_stalled_deferred_saves`) frees the
        blocks but cannot reach this dict -- and `has_pending()` then holds
        `has_pending_kv_work()` True for the life of the process: the head
        wakes every drain interval with nothing to do, and every shutdown burns
        the full `KV_SHUTDOWN_DRAIN_TIMEOUT_S`. Bounded, but permanent, and it
        accumulates per lost report.

        Called for exactly the requests the scheduler has already abandoned, so
        the two terminals share one trigger and cannot drift.

        The caller names a request; these dicts are keyed by whatever the
        worker reported, which for a save is a `SaveOperationId` as often as a
        bare id, and by `(channel, operation_id)` for a connector completion.
        So each key is collapsed onto its request first -- and through `str`,
        because the scheduler counts in ints and the connectors in strings.
        """

        def _owned_by(completion) -> bool:
            return str(getattr(completion, "req_id", completion)) == str(rid)

        for tally in (self._loading, self._failed_loading, self._saving):
            for key in [k for k in tally if _owned_by(k)]:
                tally.pop(key, None)
        # A connector completion whose operation names no request (a state-store
        # boundary) belongs to none and is left alone.
        for key in [k for k in self._connector if _owned_by(k[1])]:
            self._connector.pop(key, None)
            self._connector_failed.discard(key)

    def has_pending(self) -> bool:
        """True while any request is still short of its per-stage quorum.

        The head's busy loop keeps polling downstream stages while this holds;
        the tallies only drain when the missing stages report in.
        """
        return bool(
            self._loading or self._saving or self._failed_loading or self._connector
        )

    def reset(self) -> None:
        self._loading.clear()
        self._saving.clear()
        self._failed_loading.clear()
        self._connector.clear()
        self._connector_failed.clear()
