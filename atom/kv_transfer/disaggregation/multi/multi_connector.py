# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Composite KV connector — run several sub-connectors behind one interface.

The canonical use case is a prefill node that must do two things with the same
KV at once:

* **moriio** (``kv_role: kv_producer``) — RDMA-send the KV to a remote decode
  node for P/D disaggregation;
* **lmcache_offload** (``kv_role: offload``) — save the KV to CPU/NVMe so a
  future request that shares the prefix can skip recompute.

A single engine selects exactly one connector (``KVConnectorFactory`` reads one
``kv_connector`` name). ``MultiConnector`` is that one connector; it owns a list
of real sub-connectors and merges their results so the engine, scheduler, and
output aggregator stay unchanged.

Config::

    --kv-transfer-config '{
      "kv_connector": "multi",
      "connectors": [
        {"kv_connector": "moriio", "kv_role": "kv_producer", "proxy_ip": "...", ...},
        {"kv_connector": "lmcache_offload", "kv_role": "offload"}
      ]
    }'

Merge strategy mirrors vLLM's ``MultiConnector``, adapted to ATOM's
``base.py`` interface:

* ``get_num_new_matched_tokens`` — **first-hit-wins**: the first sub-connector
  that reports a prefix match owns the load for that request.
* ``update_state_after_alloc`` / ``request_finished`` — fan out to **all** subs
  (moriio sets up its send, offload sets up its save; both must run).
* ``build_connector_meta`` — returns :class:`MultiConnectorMetadata` carrying one
  sub-metadata per connector, in connector order. The worker de-multiplexes by
  index in ``start_load_kv``.
* ``get_finished`` — union the completion sets, **but** see the send/save
  pairing below.

Send/save pairing (the one tricky correctness point)
----------------------------------------------------
On a producer node the scheduler frees a finished request's blocks as soon as it
sees ``finished_sending`` (``scheduler.py``: producer path), and it can *also*
free on ``finished_saving`` when the connector does not defer. If offload is
still reading those blocks for its save when the moriio send completes (or vice
versa), the free would corrupt the in-flight transfer. So when a request needs
**both** a send and a save, ``MultiConnector`` withholds *both* completion
signals until the pair is done, then emits them together. The scheduler's
``finished_sending`` handler frees first; the ``finished_saving`` handler then
finds nothing to free and no-ops. This is the analogue of vLLM's
``_extra_async_saves`` refcount.

SLOT sidecar success/failure is carried in separate completion sets and bypasses
this pairing. PAGE and SLOT notifications use the same exact
``SaveOperationId`` generation, so delayed TP reports cannot cross-complete
overlapping saves. Those sets update scheduler-side commit state only; the
paired ``finished_saving`` signal remains responsible for block-lifetime release.
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import logging
from typing import Any

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    KVConnectorOutput,
    LoadOperationId,
    SaveOperationId,
    SendOperationId,
)

logger = logging.getLogger("atom")
_PARTIAL_SAVE = object()


def _save_req_id(value: Any) -> Any:
    return value.req_id if isinstance(value, SaveOperationId) else value


def _send_req_id(value: Any) -> Any:
    return value.req_id if isinstance(value, SendOperationId) else value


@dataclass(frozen=True)
class _LegacySaveOperation:
    """Internal serial identity for one raw-ReqId metadata registration."""

    req_id: Any
    nonce: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_subconnectors(config: Any, role: str) -> list:
    """Instantiate each sub-connector listed in ``kv_transfer_config.connectors``.

    Each entry is a full ``kv_transfer_config`` dict (with its own
    ``kv_connector`` name). We shallow-copy the engine config, swap in the
    sub-dict, and route through the normal factory — no recursion, since each
    sub names a concrete backend (moriio / lmcache_offload / ...), not ``multi``.
    """
    # Imported lazily: the factory module registers backends at import time and
    # we must not create an import cycle with it.
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    kvc = getattr(config, "kv_transfer_config", None) or {}
    subs = kvc.get("connectors")
    if not subs:
        raise ValueError(
            "multi connector requires a non-empty 'connectors' list in "
            "kv_transfer_config"
        )

    connectors = []
    for i, sub in enumerate(subs):
        if not isinstance(sub, dict) or "kv_connector" not in sub:
            raise ValueError(
                f"connectors[{i}] must be a dict with a 'kv_connector' key, "
                f"got {sub!r}"
            )
        if sub["kv_connector"] == "multi":
            raise ValueError("multi connector cannot nest another 'multi'")
        cfg_i = copy.copy(config)
        cfg_i.kv_transfer_config = sub
        connectors.append(KVConnectorFactory.create_connector(cfg_i, role=role))
        logger.debug(
            "multi: built sub-connector[%d] backend=%s role=%s",
            i,
            sub["kv_connector"],
            role,
        )
    return connectors


def _normalize_finished(finished: Any) -> KVConnectorOutput:
    """Coerce a sub-connector's ``get_finished()`` result to KVConnectorOutput.

    Legacy P/D connectors (moriio/mooncake) return a ``(done_sending,
    done_recving)`` tuple; the offload connector already returns a full
    :class:`KVConnectorOutput`.
    """
    if isinstance(finished, KVConnectorOutput):
        return finished
    done_sending, done_recving = finished
    return KVConnectorOutput(
        finished_sending=set(done_sending or ()),
        finished_recving=set(done_recving or ()),
    )


def _first_with(connectors: list, name: str):
    """Return the first sub-connector exposing attribute/method *name*, or None."""
    for c in connectors:
        if hasattr(c, name):
            return c
    return None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class MultiConnectorMetadata(ConnectorMetadata):
    """Carries one sub-connector metadata per connector, in connector order.

    Subclasses :class:`ConnectorMetadata` so existing ``isinstance`` checks and
    the worker dispatch path accept it unchanged. The worker reads ``metas`` and
    routes ``metas[i]`` to ``connectors[i].start_load_kv``.
    """

    def __init__(self, metas: list) -> None:
        super().__init__()
        self.metas = list(metas)

    @property
    def requests(self):
        """Aggregate of sub-metas' ``requests`` (offload uses this attribute).

        ``EngineCore._dispatch_idle_offload_work`` gates its idle dispatch on a
        truthy ``meta.requests``; exposing it here keeps offload's idle
        save/load flowing when offload runs inside a ``multi`` connector.
        """
        agg: list = []
        for m in self.metas:
            sub = getattr(m, "requests", None)
            if sub:
                agg.extend(sub)
        return agg


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


class MultiConnector(KVConnectorBase):
    """Worker-side composite connector (one instance per TP rank)."""

    def __init__(self, config: Any) -> None:
        self._connectors = _build_subconnectors(config, role="worker")
        # Producer if any sub is a producer (moriio kv_producer drives the
        # scheduler's producer-side deferred-free path).
        self.is_producer = any(
            getattr(c, "is_producer", False) for c in self._connectors
        )

        # Send/save pairing state (see module docstring).
        self._pending_save: dict[Any, set[int]] = {}
        self._operation_output: dict[Any, Any] = {}
        self._req_operations: dict[str, deque[Any]] = {}
        self._sent: dict[str, Any] = {}
        self._legacy_save_nonce = 0
        self._pairing_tombstone_limit = 4096
        self._terminal_save_order: deque[SaveOperationId] = deque()
        self._terminal_save: set[SaveOperationId] = set()
        # A native state-checkpoint destination may be read by more than one
        # child connector on this rank.  Do not forward a child's staging
        # terminal until every child registered for that exact copy generation
        # is done; the outer TP aggregator cannot provide this local fan-in.
        self._checkpoint_staging_expected: dict[int, set[int]] = {}
        self._checkpoint_staging_terminal: dict[int, dict[int, bool]] = {}
        self._terminal_checkpoint_staging_order: deque[int] = deque()
        self._terminal_checkpoint_staging: set[int] = set()

    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        for c in self._connectors:
            c.register_kv_caches(kv_caches, transfer_tensors, num_blocks)

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        self._ensure_pairing_state()
        metas = getattr(metadata, "metas", None)
        if metas is None:
            logger.warning(
                "multi: start_load_kv got %s, expected MultiConnectorMetadata",
                type(metadata).__name__,
            )
            return
        self._register_checkpoint_staging(metas, state_checkpoint_copies)
        legacy_operations: dict[str, _LegacySaveOperation] = {}
        for connector_idx, (c, m) in enumerate(zip(self._connectors, metas)):
            if m is None:
                continue
            # Remember which requests offload is about to save, so get_finished
            # can hold their send completion until the save also finishes.
            reqs = getattr(m, "requests", None)
            if reqs:
                for req in reqs:
                    if (
                        getattr(req, "save_spec", None) is not None
                        or getattr(req, "slot_save_spec", None) is not None
                    ):
                        req_id = getattr(req, "req_id")
                        output = getattr(req, "save_operation", None) or req_id
                        req_key = str(req_id)
                        if isinstance(output, SaveOperationId):
                            if output in self._terminal_save:
                                continue
                            operation = output
                        else:
                            operation = legacy_operations.get(req_key)
                            if operation is None:
                                operation = _LegacySaveOperation(
                                    req_id,
                                    self._legacy_save_nonce,
                                )
                                self._legacy_save_nonce += 1
                                legacy_operations[req_key] = operation
                        if operation not in self._pending_save:
                            self._pending_save[operation] = set()
                            self._operation_output[operation] = output
                            self._req_operations.setdefault(req_key, deque()).append(
                                operation
                            )
                        self._pending_save[operation].add(connector_idx)
            c.start_load_kv(m)

    def get_finished(self) -> KVConnectorOutput:
        self._ensure_pairing_state()
        recv: set = set()
        failed: set = set()
        loaded: set = set()
        load_failed: set = set()
        sidecar_saved: set = set()
        sidecar_failed: set = set()
        checkpoint_staged: set[int] = set()
        checkpoint_aborted: set[int] = set()
        send_now: list = []
        completed_save_now: list = []
        unregistered_save_now: list = []
        for connector_idx, c in enumerate(self._connectors):
            o = _normalize_finished(c.get_finished())
            recv |= o.finished_recving
            failed |= o.failed_recving
            loaded |= o.finished_loading
            load_failed |= o.failed_loading
            sidecar_saved |= o.finished_sidecar_saving
            sidecar_failed |= o.failed_sidecar_saving
            checkpoint_terminals = set(o.finished_checkpoint_staging) | set(
                o.aborted_checkpoint_staging
            )
            for copy_id in checkpoint_terminals:
                terminal = self._consume_checkpoint_staging_terminal(
                    connector_idx,
                    int(copy_id),
                    aborted=copy_id in o.aborted_checkpoint_staging,
                )
                if terminal is True:
                    checkpoint_aborted.add(int(copy_id))
                elif terminal is False:
                    checkpoint_staged.add(int(copy_id))
            send_now.extend(o.finished_sending)
            for completion in o.finished_saving:
                completed = self._consume_save_completion(connector_idx, completion)
                if completed is _PARTIAL_SAVE:
                    continue
                if completed is None:
                    if (
                        isinstance(completion, SaveOperationId)
                        and completion in self._terminal_save
                    ):
                        continue
                    unregistered_save_now.append(completion)
                else:
                    completed_save_now.append(completed)

        out = KVConnectorOutput(
            finished_recving=recv,
            failed_recving=failed,
            finished_loading=loaded,
            failed_loading=load_failed,
            finished_sidecar_saving=sidecar_saved,
            failed_sidecar_saving=sidecar_failed,
            finished_checkpoint_staging=checkpoint_staged,
            aborted_checkpoint_staging=checkpoint_aborted,
        )

        out.finished_saving = set(completed_save_now) | set(unregistered_save_now)
        if not self.is_producer:
            out.finished_sending = set(send_now)
            return out

        rel_send: set = set()
        for r in send_now:
            req_key = str(_send_req_id(r))
            if self._req_has_pending_save(req_key):
                self._sent[req_key] = r
            else:
                rel_send.add(r)

        for key, raw in list(self._sent.items()):
            if self._req_has_pending_save(key):
                continue
            rel_send.add(raw)
            del self._sent[key]

        out.finished_sending = rel_send
        return out

    def _ensure_pairing_state(self) -> None:
        """Initialize pairing state for lightweight ``__new__`` test doubles."""
        if not hasattr(self, "_operation_output"):
            self._operation_output = {}
            self._req_operations = {}
            self._legacy_save_nonce = 0
            self._pairing_tombstone_limit = 4096
            self._terminal_save_order = deque()
            self._terminal_save = set()
        if not hasattr(self, "_checkpoint_staging_expected"):
            self._checkpoint_staging_expected = {}
            self._checkpoint_staging_terminal = {}
            self._terminal_checkpoint_staging_order = deque()
            self._terminal_checkpoint_staging = set()

    @staticmethod
    def _checkpoint_identity(copy_record: Any) -> tuple[Any, int, int, int]:
        return (
            copy_record.request_id,
            int(copy_record.boundary_tokens),
            int(copy_record.boundary_block_hash),
            int(copy_record.source_group),
        )

    @staticmethod
    def _slot_save_identity(req: Any) -> tuple[Any, int, int, int] | None:
        spec = getattr(req, "slot_save_spec", None)
        if spec is None:
            return None
        return (
            req.req_id,
            int(spec.boundary_tokens),
            int(spec.boundary_block_hash),
            int(spec.source_group),
        )

    def _register_checkpoint_staging(self, metas, state_checkpoint_copies) -> None:
        """Record every child that will gather one leased checkpoint copy."""

        self._ensure_pairing_state()
        copies_by_identity = {}
        for copy_record in state_checkpoint_copies or ():
            try:
                identity = self._checkpoint_identity(copy_record)
                copy_id = int(copy_record.copy_id)
            except (AttributeError, TypeError, ValueError):
                # ConnectorMetadata deliberately types these records as Any for
                # compatibility with older connectors. They still receive the
                # opaque records; only typed checkpoint copies join this fan-in.
                continue
            copies_by_identity[identity] = copy_id
        if not copies_by_identity:
            return

        for connector_idx, meta in enumerate(metas):
            checkpoint_callback = getattr(
                self._connectors[connector_idx],
                "start_load_kv_with_state_checkpoints",
                None,
            )
            if not callable(checkpoint_callback):
                # This child receives only its ordinary sub-metadata and cannot
                # read the leased destination through the extended hook.
                continue
            for req in getattr(meta, "requests", ()) or ():
                try:
                    identity = self._slot_save_identity(req)
                except (AttributeError, TypeError, ValueError):
                    continue
                copy_id = copies_by_identity.get(identity)
                if copy_id is None or copy_id in self._terminal_checkpoint_staging:
                    continue
                self._checkpoint_staging_expected.setdefault(copy_id, set()).add(
                    connector_idx
                )

    def _consume_checkpoint_staging_terminal(
        self,
        connector_idx: int,
        copy_id: int,
        *,
        aborted: bool,
    ) -> bool | None:
        """Return the aggregate abort flag once every local reader is terminal.

        ``True`` means at least one child aborted, ``False`` means every child
        completed, and ``None`` means the copy is still pending or tombstoned.
        Unregistered terminals retain the historical pass-through behaviour;
        leased copies are always registered before their child hooks can run.
        """

        self._ensure_pairing_state()
        if copy_id in self._terminal_checkpoint_staging:
            return None

        expected = self._checkpoint_staging_expected.get(copy_id)
        if expected is None:
            self._remember_terminal_checkpoint_staging(copy_id)
            return bool(aborted)
        if connector_idx not in expected:
            logger.warning(
                "multi: ignoring checkpoint staging terminal from unregistered "
                "child=%d copy_id=%d",
                connector_idx,
                copy_id,
            )
            return None

        terminals = self._checkpoint_staging_terminal.setdefault(copy_id, {})
        # A contradictory duplicate fails closed while the local fan-in is
        # pending. Child contracts should emit exactly one terminal.
        terminals[connector_idx] = bool(aborted) or terminals.get(connector_idx, False)
        if not expected.issubset(terminals):
            return None

        aggregate_aborted = any(terminals[index] for index in expected)
        self._checkpoint_staging_expected.pop(copy_id, None)
        self._checkpoint_staging_terminal.pop(copy_id, None)
        self._remember_terminal_checkpoint_staging(copy_id)
        return aggregate_aborted

    def _remember_terminal_checkpoint_staging(self, copy_id: int) -> None:
        if copy_id in self._terminal_checkpoint_staging:
            return
        self._terminal_checkpoint_staging.add(copy_id)
        self._terminal_checkpoint_staging_order.append(copy_id)
        while (
            len(self._terminal_checkpoint_staging_order) > self._pairing_tombstone_limit
        ):
            self._terminal_checkpoint_staging.discard(
                self._terminal_checkpoint_staging_order.popleft()
            )

    def _consume_save_completion(
        self, connector_idx: int, completion: Any
    ) -> Any | None:
        operation = None
        if isinstance(completion, SaveOperationId):
            pending = self._pending_save.get(completion)
            if pending is not None and connector_idx not in pending:
                return _PARTIAL_SAVE
            if connector_idx in (pending or ()):
                operation = completion
        else:
            req_key = str(completion)
            # Legacy child notifications are serial: map only to this
            # connector's oldest outstanding operation for the request.
            for candidate in self._req_operations.get(req_key, ()):
                if connector_idx in self._pending_save.get(candidate, set()):
                    operation = candidate
                    break
        if operation is None:
            return None

        contributors = self._pending_save[operation]
        contributors.discard(connector_idx)
        if contributors:
            return _PARTIAL_SAVE

        self._pending_save.pop(operation, None)
        output = self._operation_output.pop(operation)
        req_key = str(_save_req_id(output))
        queue = self._req_operations.get(req_key)
        if queue is not None:
            try:
                queue.remove(operation)
            except ValueError:
                pass
            if not queue:
                self._req_operations.pop(req_key, None)
        if isinstance(output, SaveOperationId):
            self._remember_terminal_save(output)
        return output

    def _req_has_pending_save(self, req_key: str) -> bool:
        return bool(self._req_operations.get(req_key))

    def _remember_terminal_save(self, operation: SaveOperationId) -> None:
        if operation in self._terminal_save:
            return
        self._terminal_save.add(operation)
        self._terminal_save_order.append(operation)
        while len(self._terminal_save_order) > self._pairing_tombstone_limit:
            self._terminal_save.discard(self._terminal_save_order.popleft())

    @property
    def completed_save_tombstone_count(self) -> int:
        return len(self._terminal_save)

    @property
    def pairing_state_count(self) -> tuple[int, int]:
        return len(self._pending_save), len(self._sent)

    def get_finished_recv_blocks(self) -> list[int]:
        blocks: list[int] = []
        for c in self._connectors:
            blocks.extend(c.get_finished_recv_blocks())
        return blocks


# ---------------------------------------------------------------------------
# Scheduler side
# ---------------------------------------------------------------------------


class MultiConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler-side composite connector."""

    def __init__(self, config: Any) -> None:
        self._connectors = _build_subconnectors(config, role="scheduler")
        self.is_producer = any(
            getattr(c, "is_producer", False) for c in self._connectors
        )
        # Opt into the scheduler's offload suffix-prefill path if any sub is the
        # offload backend (Scheduler._is_offload_connector reads this).
        self.is_offload = any(getattr(c, "is_offload", False) for c in self._connectors)
        self.recap_prefill_after_finalize = any(
            getattr(c, "recap_prefill_after_finalize", False) for c in self._connectors
        )
        self._load_owner_by_req: dict[str, tuple[object, int]] = {}
        self._load_operation_owner: dict[LoadOperationId, tuple[int, object]] = {}

    def _ensure_load_ownership_state(self) -> None:
        if not hasattr(self, "_load_owner_by_req"):
            self._load_owner_by_req = {}
            self._load_operation_owner = {}

    def _cancel_connector_load(self, connector_idx: int, seq: Any) -> None:
        callback = getattr(self._connectors[connector_idx], "cancel_pending_load", None)
        if callback is not None:
            callback(seq)

    @staticmethod
    def _load_lifecycle_key(seq: Any) -> str:
        return str(getattr(seq, "id", f"@object:{id(seq)}"))

    def _clear_load_owner(self, seq: Any) -> None:
        self._ensure_load_ownership_state()
        sid = self._load_lifecycle_key(seq)
        owner = self._load_owner_by_req.get(sid)
        if owner is not None and owner[0] is seq:
            self._load_owner_by_req.pop(sid, None)
        for operation, (_idx, lifecycle) in list(self._load_operation_owner.items()):
            if lifecycle is seq:
                self._load_operation_owner.pop(operation, None)

    def _owner_for_seq(self, seq: Any) -> tuple[int, Any] | None:
        self._ensure_load_ownership_state()
        owner = self._load_owner_by_req.get(self._load_lifecycle_key(seq))
        if owner is None or owner[0] is not seq:
            return None
        return owner[1], self._connectors[owner[1]]

    # -- base interface -----------------------------------------------------

    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        """First-hit-wins: the first sub that reports a match owns the load."""
        self._ensure_load_ownership_state()
        sid = self._load_lifecycle_key(seq)
        previous = self._load_owner_by_req.get(sid)
        if previous is not None and previous[0] is not seq:
            self._cancel_connector_load(previous[1], previous[0])
            if hasattr(previous[0], "_remote_load_is_offload"):
                delattr(previous[0], "_remote_load_is_offload")
            self._clear_load_owner(previous[0])

        result = (0, False)
        probed: list[int] = []
        winner = None
        for connector_idx, c in enumerate(self._connectors):
            toks, needs_load = c.get_num_new_matched_tokens(seq)
            probed.append(connector_idx)
            if toks > 0:
                result = (toks, needs_load)
                winner = connector_idx
                break
        if winner is not None:
            self._load_owner_by_req[sid] = (seq, winner)
            try:
                seq._remote_load_is_offload = bool(
                    getattr(self._connectors[winner], "is_offload", False)
                )
            except (AttributeError, TypeError):
                pass
            losers = (idx for idx in range(len(self._connectors)) if idx != winner)
            for connector_idx in losers:
                self._cancel_connector_load(connector_idx, seq)
        else:
            for connector_idx in probed:
                self._cancel_connector_load(connector_idx, seq)
            self._load_owner_by_req.pop(sid, None)
            if hasattr(seq, "_remote_load_is_offload"):
                delattr(seq, "_remote_load_is_offload")
        return result

    def build_connector_meta(self) -> MultiConnectorMetadata:
        self._ensure_load_ownership_state()
        metas = [c.build_connector_meta() for c in self._connectors]
        for connector_idx, meta in enumerate(metas):
            stripped_req_ids: set[Any] = set()
            for field in (
                "reqs_to_recv",
                "reqs_to_load",
                "load_requests",
                "loads",
            ):
                load_map = getattr(meta, field, None)
                if not isinstance(load_map, dict):
                    continue
                for req_id in list(load_map):
                    owner = self._load_owner_by_req.get(str(req_id))
                    if owner is not None and owner[1] == connector_idx:
                        continue
                    load_map.pop(req_id, None)
                    stripped_req_ids.add(req_id)
                    if owner is not None:
                        self._cancel_connector_load(connector_idx, owner[0])
            for field in ("reqs_not_processed",):
                values = getattr(meta, field, None)
                if isinstance(values, set):
                    values.difference_update(stripped_req_ids)
            for req in getattr(meta, "requests", ()) or ():
                if (
                    getattr(req, "load_spec", None) is None
                    and getattr(req, "slot_load_spec", None) is None
                ):
                    continue
                owner = self._load_owner_by_req.get(str(req.req_id))
                if owner is None or owner[1] != connector_idx:
                    if owner is not None:
                        self._cancel_connector_load(connector_idx, owner[0])
                        loser_operation = getattr(req, "load_operation", None)
                        if (
                            loser_operation is not None
                            and getattr(owner[0], "_active_load_operation", None)
                            == loser_operation
                        ):
                            for marker in (
                                "_active_load_operation",
                                "_consumed_load_operation",
                                "_load_operation",
                            ):
                                if hasattr(owner[0], marker):
                                    delattr(owner[0], marker)
                    req.load_spec = None
                    req.slot_load_spec = None
                    req.load_operation = None
                    continue
                operation = getattr(req, "load_operation", None)
                if isinstance(operation, LoadOperationId):
                    self._load_operation_owner[operation] = (
                        connector_idx,
                        owner[0],
                    )
        for operation, (
            _connector_idx,
            lifecycle,
        ) in self._load_operation_owner.items():
            owner = self._load_owner_by_req.get(str(operation.req_id))
            if owner is not None and owner[0] is lifecycle:
                lifecycle._load_operation = operation
        return MultiConnectorMetadata(metas=metas)

    def connector_meta_dispatched(self, meta: MultiConnectorMetadata) -> None:
        for connector, sub_meta in zip(self._connectors, meta.metas):
            callback = getattr(connector, "connector_meta_dispatched", None)
            if callback is not None:
                callback(sub_meta)

    def update_state_after_alloc(self, seq: Any) -> None:
        for c in self._connectors:
            c.update_state_after_alloc(seq)

    def request_finished(self, seq: Any) -> None:
        for c in self._connectors:
            if hasattr(c, "request_finished"):
                c.request_finished(seq)
        self._clear_load_owner(seq)
        for marker in (
            "_remote_load_is_offload",
            "_active_load_operation",
            "_consumed_load_operation",
            "_load_operation",
        ):
            if hasattr(seq, marker):
                delattr(seq, marker)

    # -- offload-specific methods, forwarded to the owning sub --------------
    # The scheduler guards every one of these with hasattr(), so MultiConnector
    # only needs to expose them when a sub-connector implements them.

    def should_park_for_load_after_alloc(self, seq: Any) -> bool:
        owner = self._owner_for_seq(seq)
        if owner is None:
            return False
        c = owner[1]
        callback = getattr(c, "should_park_for_load_after_alloc", None)
        return callback(seq) if callback is not None else False

    def adjust_prefill_chunk_after_alloc(self, seq: Any, chunk: int) -> int:
        owner = self._owner_for_seq(seq)
        if owner is None:
            return chunk
        callback = getattr(owner[1], "adjust_prefill_chunk_after_alloc", None)
        return callback(seq, chunk) if callback is not None else chunk

    def should_park_partial_prefill_for_load(self, seq: Any) -> bool:
        owner = self._owner_for_seq(seq)
        if owner is None:
            return False
        callback = getattr(owner[1], "should_park_partial_prefill_for_load", None)
        return callback(seq) if callback is not None else False

    def should_pause_partial_prefill_for_save(self, seq: Any) -> bool:
        return any(
            getattr(
                connector, "should_pause_partial_prefill_for_save", lambda _seq: False
            )(seq)
            for connector in self._connectors
        )

    def cancel_pending_load(self, seq: Any) -> None:
        owner = self._owner_for_seq(seq)
        if owner is not None:
            self._cancel_connector_load(owner[0], seq)
        self._clear_load_owner(seq)
        if hasattr(seq, "_remote_load_is_offload"):
            delattr(seq, "_remote_load_is_offload")

    def should_defer_free(self, seq: Any) -> bool:
        # Defer if ANY sub wants to defer (so neither a pending save nor a
        # pending send loses its blocks early).
        return any(
            hasattr(c, "should_defer_free") and c.should_defer_free(seq)
            for c in self._connectors
        )

    def save_finished(self, req_id: Any) -> None:
        for c in self._connectors:
            if hasattr(c, "save_finished"):
                c.save_finished(req_id)

    def sidecar_save_finished(self, req_id: Any) -> None:
        for c in self._connectors:
            if hasattr(c, "sidecar_save_finished"):
                c.sidecar_save_finished(req_id)

    def sidecar_save_failed(self, req_id: Any) -> None:
        for c in self._connectors:
            if hasattr(c, "sidecar_save_failed"):
                c.sidecar_save_failed(req_id)

    def load_finished(self, req_id: Any) -> bool:
        return self._finish_load(req_id, "load_finished")

    def load_failed(self, req_id: Any) -> bool:
        return self._finish_load(req_id, "load_failed")

    def _finish_load(self, completion: Any, callback_name: str) -> bool:
        self._ensure_load_ownership_state()
        lifecycle = None
        if isinstance(completion, LoadOperationId):
            owned = self._load_operation_owner.get(completion)
            if owned is None:
                return False
            connector_idx, lifecycle = owned
        else:
            owner = self._load_owner_by_req.get(str(completion))
            if owner is None:
                return False
            lifecycle, connector_idx = owner
        callback = getattr(self._connectors[connector_idx], callback_name, None)
        if callback is None:
            return False
        result = callback(completion)
        handled = (
            result is True
            if isinstance(completion, LoadOperationId)
            else (result is not False)
        )
        if handled:
            if isinstance(completion, LoadOperationId):
                self._load_operation_owner.pop(completion, None)
            if lifecycle is not None:
                self._clear_load_owner(lifecycle)
        return handled
