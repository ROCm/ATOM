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
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata, KVConnectorOutput

logger = logging.getLogger("atom")


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
        # _pending_save: str(req_id) for requests offload will save this lifetime.
        self._pending_save: set[str] = set()
        # _sent / _saved: completed-but-unpaired transfers, str(req_id) -> raw id.
        self._sent: dict[str, Any] = {}
        self._saved: dict[str, Any] = {}

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
        for c, m in zip(self._connectors, metas):
            if m is None:
                continue
            # Remember which requests offload is about to save, so get_finished
            # can hold their send completion until the save also finishes.
            reqs = getattr(m, "requests", None)
            if reqs:
                for req in reqs:
                    if getattr(req, "save_spec", None) is not None:
                        self._pending_save.add(str(getattr(req, "req_id")))
            c.start_load_kv(m)

    def get_finished(self) -> KVConnectorOutput:
        recv: set = set()
        failed: set = set()
        send_now: list = []
        save_now: list = []
        for c in self._connectors:
            o = _normalize_finished(c.get_finished())
            recv |= o.finished_recving
            failed |= o.failed_recving
            send_now.extend(o.finished_sending)
            save_now.extend(o.finished_saving)

        out = KVConnectorOutput(finished_recving=recv, failed_recving=failed)

        if not self.is_producer:
            # No moriio send to pair with: offload save / recv pass straight
            # through (the scheduler frees consumer/offload requests on
            # finished_saving via should_defer_free).
            out.finished_sending = set(send_now)
            out.finished_saving = set(save_now)
            return out

        # Producer + offload: pair each request's send and save before
        # releasing either (see module docstring).
        for r in send_now:
            self._sent[str(r)] = r
        for r in save_now:
            self._saved[str(r)] = r

        rel_send: set = set()
        rel_save: set = set()
        for key, raw in list(self._sent.items()):
            needs_save = key in self._pending_save
            if needs_save and key not in self._saved:
                continue  # hold: save still in flight for this request
            rel_send.add(raw)
            del self._sent[key]
            self._pending_save.discard(key)
            if key in self._saved:
                rel_save.add(self._saved.pop(key))

        out.finished_sending = rel_send
        out.finished_saving = rel_save
        return out

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

    # -- base interface -----------------------------------------------------

    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        """First-hit-wins: the first sub that reports a match owns the load."""
        result = (0, False)
        for c in self._connectors:
            toks, needs_load = c.get_num_new_matched_tokens(seq)
            if result[0] == 0 and toks > 0:
                result = (toks, needs_load)
        return result

    def build_connector_meta(self) -> MultiConnectorMetadata:
        return MultiConnectorMetadata(
            metas=[c.build_connector_meta() for c in self._connectors]
        )

    def update_state_after_alloc(self, seq: Any) -> None:
        for c in self._connectors:
            c.update_state_after_alloc(seq)

    def request_finished(self, seq: Any) -> None:
        for c in self._connectors:
            if hasattr(c, "request_finished"):
                c.request_finished(seq)

    # -- offload-specific methods, forwarded to the owning sub --------------
    # The scheduler guards every one of these with hasattr(), so MultiConnector
    # only needs to expose them when a sub-connector implements them.

    def should_park_for_load_after_alloc(self, seq: Any) -> bool:
        c = _first_with(self._connectors, "should_park_for_load_after_alloc")
        return c.should_park_for_load_after_alloc(seq) if c is not None else False

    def adjust_prefill_chunk_after_alloc(self, seq: Any, chunk: int) -> int:
        c = _first_with(self._connectors, "adjust_prefill_chunk_after_alloc")
        return (
            c.adjust_prefill_chunk_after_alloc(seq, chunk) if c is not None else chunk
        )

    def should_park_partial_prefill_for_load(self, seq: Any) -> bool:
        c = _first_with(self._connectors, "should_park_partial_prefill_for_load")
        return c.should_park_partial_prefill_for_load(seq) if c is not None else False

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

    def load_failed(self, req_id: Any) -> None:
        for c in self._connectors:
            if hasattr(c, "load_failed"):
                c.load_failed(req_id)
