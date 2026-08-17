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

Connector-owned terminal events bypass this pairing and are routed by opaque,
uniquely owned completion channels. The paired ``finished_saving`` signal
remains responsible for block-lifetime release; a backend-specific completion
may update its own scheduler state but cannot release PAGE blocks through Multi.
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
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


@dataclass(frozen=True)
class MultiSaveOperationId(SaveOperationId):
    """Wire identity for one exact save owned by a Multi child.

    Child schedulers allocate generations independently, so the child index is
    part of the composite identity.  As a ``SaveOperationId`` subclass it keeps
    the generic aggregator and scheduler request-ID handling compatible.
    """

    connector_idx: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.connector_idx < 0:
            raise ValueError("save operation connector index must be nonnegative")

    @property
    def child_operation(self) -> SaveOperationId:
        return SaveOperationId(self.req_id, self.generation)


def _namespace_save_operation(
    connector_idx: int,
    operation: SaveOperationId,
) -> MultiSaveOperationId:
    return MultiSaveOperationId(
        req_id=operation.req_id,
        generation=operation.generation,
        connector_idx=connector_idx,
    )


class _CompletedOperationWindow:
    """Bounded exact-operation memory with O(1) lookup and FIFO eviction."""

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("completed operation window limit must be positive")
        self.limit = int(limit)
        self._order: deque[Any] = deque()
        self._items: set[Any] = set()

    def __contains__(self, operation: Any) -> bool:
        return operation in self._items

    def __len__(self) -> int:
        return len(self._items)

    def remember(self, operation: Any) -> None:
        if operation in self._items:
            return
        self._items.add(operation)
        self._order.append(operation)
        while len(self._order) > self.limit:
            self._items.discard(self._order.popleft())


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
        backend_name = KVConnectorFactory.canonical_name(
            sub["kv_connector"],
            path=f"kv_transfer_config.connectors[{i}]",
        )
        if backend_name == "multi":
            raise ValueError("multi connector cannot nest another 'multi'")
        cfg_i = copy.copy(config)
        cfg_i.kv_transfer_config = sub
        connectors.append(KVConnectorFactory.create_connector(cfg_i, role=role))
        logger.debug(
            "multi: built sub-connector[%d] backend=%s role=%s",
            i,
            backend_name,
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


def _completion_channel_owners(connectors: list) -> dict[str, int]:
    """Return the unique child owner of every declared completion channel."""

    owners: dict[str, int] = {}
    for connector_idx, connector in enumerate(connectors):
        channels = getattr(connector, "completion_channels", ()) or ()
        for channel in channels:
            if not isinstance(channel, str) or not channel:
                raise ValueError(
                    "connector completion channels must be non-empty strings"
                )
            previous = owners.get(channel)
            if previous is not None:
                raise ValueError(
                    "multi connector completion channel must have one owner: "
                    f"channel={channel!r} children={previous},{connector_idx}"
                )
            owners[channel] = connector_idx
    return owners


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class MultiConnectorMetadata(ConnectorMetadata):
    """Carries one sub-connector metadata per connector, in connector order.

    Subclasses :class:`ConnectorMetadata` so existing ``isinstance`` checks and
    the worker dispatch path accept it unchanged. The worker reads ``metas`` and
    routes ``metas[i]`` to ``connectors[i].start_load_kv``.
    """

    def __init__(
        self,
        metas: list,
        completion_channel_owners: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.metas = list(metas)
        self.completion_channel_owners = dict(completion_channel_owners or {})

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

    def requests_for_completion_channel(self, channel: str) -> list:
        """Return only requests owned by one opaque completion channel."""

        owner = self.completion_channel_owners.get(channel)
        if owner is None or not 0 <= owner < len(self.metas):
            return []
        return list(getattr(self.metas[owner], "requests", ()) or ())


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
        # A request ID can be reused while an earlier send notification is
        # still in flight.  Keep every exact generation: a late notification
        # for the old lifecycle must not replace the current one.
        self._sent: dict[str, set[Any]] = {}
        self._legacy_save_nonce = 0
        self._completed_save_operations = _CompletedOperationWindow(4096)
        # Completion channels have exactly one child owner so opaque terminal
        # notifications are routed to the backend that declared them.
        self._completion_channel_owners = _completion_channel_owners(self._connectors)

    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        for c in self._connectors:
            c.register_kv_caches(kv_caches, transfer_tensors, num_blocks)

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        metas = getattr(metadata, "metas", None)
        if metas is None:
            logger.warning(
                "multi: start_load_kv got %s, expected MultiConnectorMetadata",
                type(metadata).__name__,
            )
            return
        if len(metas) != len(self._connectors):
            raise RuntimeError(
                "multi connector metadata/child count mismatch: "
                f"metas={len(metas)} children={len(self._connectors)}"
            )
        declared_owners = getattr(metadata, "completion_channel_owners", None)
        if declared_owners is not None and (
            dict(declared_owners) != self._completion_channel_owners
        ):
            raise RuntimeError(
                "multi connector completion-channel ownership differs between "
                "scheduler and worker"
            )
        legacy_operations: dict[str, _LegacySaveOperation] = {}
        for connector_idx, (c, m) in enumerate(zip(self._connectors, metas)):
            if m is None:
                continue
            # Remember backend-declared async saves, so get_finished can hold
            # producer send completion until every reader of those blocks is
            # done. Multi never inspects backend-specific request descriptors.
            iter_saves = getattr(m, "iter_async_save_operations", None)
            save_operations = iter_saves() if callable(iter_saves) else ()
            for req_id, output in save_operations:
                req_key = str(req_id)
                if isinstance(output, SaveOperationId):
                    operation = _namespace_save_operation(connector_idx, output)
                    if operation in self._completed_save_operations:
                        continue
                    outer_output = operation
                else:
                    operation = legacy_operations.get(req_key)
                    if operation is None:
                        operation = _LegacySaveOperation(
                            req_id,
                            self._legacy_save_nonce,
                        )
                        self._legacy_save_nonce += 1
                        legacy_operations[req_key] = operation
                    outer_output = output
                if operation not in self._pending_save:
                    self._pending_save[operation] = set()
                    self._operation_output[operation] = outer_output
                    self._req_operations.setdefault(req_key, deque()).append(operation)
                self._pending_save[operation].add(connector_idx)
            c.start_load_kv(m)

    def has_pending_work(self) -> bool:
        """Compose child hook-based liveness at the composite boundary."""

        for connector_idx, connector in enumerate(self._connectors):
            callback = getattr(connector, "has_pending_work", None)
            if not callable(callback):
                continue
            try:
                if callback():
                    return True
            except Exception:
                # Keep the outer engine polling, matching ModelRunner's direct
                # connector behavior when a liveness hook itself fails.
                logger.exception(
                    "multi: pending-work hook failed child=%d backend=%s",
                    connector_idx,
                    type(connector).__name__,
                )
                return True
        return False

    def get_finished(self) -> KVConnectorOutput:
        recv: set = set()
        failed: set = set()
        loaded: set = set()
        load_failed: set = set()
        pending_work = False
        connector_completions: set[ConnectorCompletion] = set()
        send_now: list = []
        completed_save_now: list = []
        unregistered_save_now: list = []
        for connector_idx, c in enumerate(self._connectors):
            o = _normalize_finished(c.get_finished())
            recv |= o.finished_recving
            failed |= o.failed_recving
            loaded |= o.finished_loading
            load_failed |= o.failed_loading
            pending_work |= bool(getattr(o, "pending_work", False))
            for completion in o.connector_completions:
                owner = self._completion_channel_owners.get(completion.channel)
                if owner != connector_idx:
                    logger.error(
                        "multi: dropping completion from non-owner child=%d "
                        "channel=%s owner=%s",
                        connector_idx,
                        completion.channel,
                        owner,
                    )
                    continue
                connector_completions.add(completion)
            send_now.extend(o.finished_sending)
            for completion in o.finished_saving:
                outer_completion = (
                    _namespace_save_operation(connector_idx, completion)
                    if isinstance(completion, SaveOperationId)
                    else completion
                )
                completed = self._consume_save_completion(connector_idx, completion)
                if completed is _PARTIAL_SAVE:
                    continue
                if completed is None:
                    if (
                        isinstance(outer_completion, SaveOperationId)
                        and outer_completion in self._completed_save_operations
                    ):
                        continue
                    unregistered_save_now.append(outer_completion)
                else:
                    completed_save_now.append(completed)

        out = KVConnectorOutput(
            finished_recving=recv,
            failed_recving=failed,
            finished_loading=loaded,
            failed_loading=load_failed,
            connector_completions=connector_completions,
            pending_work=pending_work,
        )

        out.finished_saving = set(completed_save_now) | set(unregistered_save_now)
        if not self.is_producer:
            out.finished_sending = set(send_now)
            return out

        rel_send: set = set()
        for r in send_now:
            req_key = str(_send_req_id(r))
            if self._req_has_pending_save(req_key):
                held = self._sent.setdefault(req_key, set())
                if isinstance(r, SendOperationId):
                    # Exact lifecycle identity supersedes a legacy raw-ID
                    # notification, but never another exact generation.
                    held.difference_update(
                        {
                            completion
                            for completion in held
                            if not isinstance(completion, SendOperationId)
                        }
                    )
                    held.add(r)
                elif not any(
                    isinstance(completion, SendOperationId) for completion in held
                ):
                    # Raw IDs remain supported only when this request has no
                    # exact send lifecycle in the held state.
                    held.add(r)
            else:
                rel_send.add(r)

        for key, completions in list(self._sent.items()):
            if self._req_has_pending_save(key):
                continue
            rel_send.update(completions)
            del self._sent[key]

        out.finished_sending = rel_send
        return out

    def _consume_save_completion(
        self, connector_idx: int, completion: Any
    ) -> Any | None:
        operation = None
        if isinstance(completion, SaveOperationId):
            operation_id = _namespace_save_operation(connector_idx, completion)
            pending = self._pending_save.get(operation_id)
            if pending is not None and connector_idx not in pending:
                return _PARTIAL_SAVE
            if connector_idx in (pending or ()):
                operation = operation_id
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
            self._completed_save_operations.remember(output)
        return output

    def _req_has_pending_save(self, req_key: str) -> bool:
        return bool(self._req_operations.get(req_key))

    @property
    def completed_save_tombstone_count(self) -> int:
        return len(self._completed_save_operations)

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
        self._completion_channel_owners = _completion_channel_owners(self._connectors)
        self.completion_channels = frozenset(self._completion_channel_owners)
        self._load_owner_by_req: dict[str, tuple[object, int]] = {}
        self._load_operation_owner: dict[LoadOperationId, tuple[int, object]] = {}

    def _cancel_connector_load(self, connector_idx: int, seq: Any) -> None:
        callback = getattr(self._connectors[connector_idx], "cancel_pending_load", None)
        if callback is not None:
            callback(seq)

    @staticmethod
    def _load_lifecycle_key(seq: Any) -> str:
        return str(getattr(seq, "id", f"@object:{id(seq)}"))

    def _clear_load_owner(self, seq: Any) -> None:
        sid = self._load_lifecycle_key(seq)
        owner = self._load_owner_by_req.get(sid)
        if owner is not None and owner[0] is seq:
            self._load_owner_by_req.pop(sid, None)
        for operation, (_idx, lifecycle) in list(self._load_operation_owner.items()):
            if lifecycle is seq:
                self._load_operation_owner.pop(operation, None)

    def _owner_for_seq(self, seq: Any) -> tuple[int, Any] | None:
        owner = self._load_owner_by_req.get(self._load_lifecycle_key(seq))
        if owner is None or owner[0] is not seq:
            return None
        return owner[1], self._connectors[owner[1]]

    # -- base interface -----------------------------------------------------

    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        """First-hit-wins: the first sub that reports a match owns the load."""
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
        return MultiConnectorMetadata(
            metas=metas,
            completion_channel_owners=self._completion_channel_owners,
        )

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
        if isinstance(req_id, MultiSaveOperationId):
            connector_idx = req_id.connector_idx
            if not 0 <= connector_idx < len(self._connectors):
                return
            connector = self._connectors[connector_idx]
            if hasattr(connector, "save_finished"):
                connector.save_finished(req_id.child_operation)
            return
        for c in self._connectors:
            if hasattr(c, "save_finished"):
                c.save_finished(req_id)

    def connector_completion(self, completion: ConnectorCompletion) -> bool:
        """Route one opaque completion to its uniquely declared child owner."""

        owner = self._completion_channel_owners.get(completion.channel)
        if owner is None:
            return False
        callback = getattr(self._connectors[owner], "connector_completion", None)
        if not callable(callback):
            return False
        return callback(completion) is not False

    def load_finished(self, req_id: Any) -> bool:
        return self._finish_load(req_id, "load_finished")

    def load_failed(self, req_id: Any) -> bool:
        return self._finish_load(req_id, "load_failed")

    def _finish_load(self, completion: Any, callback_name: str) -> bool:
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
