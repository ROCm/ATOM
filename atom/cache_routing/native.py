# SPDX-License-Identifier: MIT
"""Map actual LMCache native keys to content, then consume public CPU facts."""

from __future__ import annotations

import json
import logging
import threading
import uuid

from atom.cache_routing.config import CacheRoutingConfig
from atom.cache_routing.keys import content_keys, root_key
from atom.cache_routing.server import post_cpu_report

logger = logging.getLogger(__name__)


class NativeCPUReporter:
    def __init__(
        self,
        engine,
        metadata,
        config: CacheRoutingConfig,
        chunk_size: int,
        piece_manifest: list[dict] | None = None,
    ):
        # Public backend collection: never read hot_cache or allocator internals.
        backends = engine.storage_manager.storage_backends
        self.backend = backends.get("LocalCPUBackend")
        if self.backend is None or not callable(
            getattr(self.backend, "residency_snapshot", None)
        ):
            raise ValueError("CPU catalog requires LMCache native residency v1 hooks")
        if chunk_size % config.canonical_block_size:
            raise ValueError(
                "LMCache chunk size must be divisible by canonical block size"
            )
        self.engine = engine
        self.metadata = metadata
        self.config = config
        self.chunk_size = chunk_size
        self.piece_manifest = piece_manifest or []
        self.bindings = {}
        self.pending = set()
        self.readable = {}
        self.sent = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.epoch = None
        self.report_epoch = uuid.uuid4().hex
        self.backend_seq = 0
        self.seq = 0
        self.force_snapshot = True
        self.thread = threading.Thread(
            target=self._run, name="native-cpu-catalog", daemon=True
        )
        self.thread.start()

    def bind(self, token_ids) -> None:
        """Register identities using the engine's token database, before store.

        Registration grants no residency. A later public backend snapshot/event
        must independently establish that the native object is readable.
        """
        tokens = [int(token) for token in token_ids]
        keys = content_keys(
            self.config.content_namespace, tokens, self.config.canonical_block_size
        )
        root = root_key(
            self.config.content_namespace, self.config.canonical_block_size
        ).hex()
        with self.lock:
            for start, end, key in self.engine.token_database.process_tokens(tokens):
                if end - start != self.chunk_size:
                    continue
                if (
                    len(self.bindings) >= self.config.max_entries
                    and key not in self.bindings
                ):
                    continue
                lo, hi = (
                    start // self.config.canonical_block_size,
                    end // self.config.canonical_block_size,
                )
                if key in self.bindings:
                    continue
                self.bindings[key] = {
                    "chunk_id": keys[hi - 1],
                    "token_start": start,
                    "token_end": end,
                    "content_keys": keys[lo:hi],
                    "parent_key": keys[lo - 1] if lo else root,
                }
                self.pending.add(key)

    def poll(self) -> None:
        """Reconcile new bindings and mutations without rescanning resident KV.

        The transport cursor is independent of the backend cursor: binding an
        already readable object is itself a catalog change. Recovery first
        clears this worker's scope, then rebuilds it in bounded pages. Partial
        recovery can only omit chunks, never preserve an obsolete READY.
        """
        with self.lock:
            snapshot = self.force_snapshot
            pending, self.pending = self.pending, set()
            self.force_snapshot = False
        if snapshot:
            page = self.backend.residency_snapshot()
            self.readable = {event.key: event for event in page.entries}
            pending.update(self.readable)
            self.epoch = page.source_epoch
            self._send([], snapshot=True)
            self.sent.clear()
        else:
            page = self.backend.residency_events(self.epoch, self.backend_seq)
            for event in page.events:
                pending.add(event.key)
                if event.readable:
                    self.readable[event.key] = event
                else:
                    self.readable.pop(event.key, None)
        with self.lock:
            bindings = {
                key: self.bindings[key] for key in pending if key in self.bindings
            }
        report = []
        for key in pending:
            event = self.readable.get(key)
            binding = bindings.get(key)
            if event is not None and binding is not None:
                report.append(
                    {**binding, "readable": True, "size_bytes": event.size_bytes}
                )
                self.sent[key] = binding["chunk_id"]
            elif key in self.sent:
                report.append({"chunk_id": self.sent.pop(key), "readable": False})
        # Bound bytes as well as object count, including large chunk geometry.
        batch, size = [], 0
        for event in report:
            event_size = len(json.dumps(event))
            if event_size > 1024 * 1024:
                continue  # Unknown identity loses a benefit; never split a chunk.
            if size + event_size > 1024 * 1024 or len(batch) >= 256:
                self._send(batch)
                batch, size = [], 0
            batch.append(event)
            size += event_size
        self._send(batch)  # An empty page is a source heartbeat.
        self.backend_seq = page.cut_seq

    def _send(self, events, *, snapshot=False):
        post_cpu_report(
            self.config.catalog_url,
            {
                "rank": self.metadata.worker_id,
                "source_epoch": f"{self.report_epoch}:{self.epoch}",
                "seq": str(self.seq + 1),
                "after_seq": str(self.seq),
                "snapshot": snapshot,
                "content_namespace": self.config.content_namespace,
                "layout_id": self.metadata.model_name,
                "chunk_size": self.chunk_size,
                "piece_manifest": self.piece_manifest if snapshot else None,
                "events": events,
            },
        )
        self.seq += 1

    def _run(self):
        while not self.stop.is_set():
            try:
                self.poll()
            # Observation is optional; any backend failure requires resync.
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    self.force_snapshot = True
                logger.debug("CPU catalog observation will resync: %s", exc)
            self.stop.wait(0.1)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
