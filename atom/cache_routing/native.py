# SPDX-License-Identifier: MIT
"""Publish sampled CPU residency through connector-supplied read-only APIs."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable, Hashable, Iterable

from atom.cache_routing.config import CacheRoutingConfig
from atom.cache_routing.keys import content_keys, root_key
from atom.cache_routing.server import post_cpu_report

logger = logging.getLogger(__name__)


class NativeCPUReporter:
    """Map native keys to content without depending on a storage implementation.

    The connector supplies a point-in-time key snapshot and its native token
    mapping. Full immutable chunks have a fixed byte size in the codec. Sampling
    can miss intermediate mutations; native lookup/retrieve remain authoritative.
    """

    def __init__(
        self,
        config: CacheRoutingConfig,
        *,
        rank: int,
        layout_id: str,
        chunk_size: int,
        chunk_size_bytes: int,
        get_keys: Callable[[], Iterable[Hashable]],
        process_tokens: Callable[[list[int]], Iterable[tuple[int, int, Hashable]]],
        piece_manifest: list[dict] | None = None,
    ):
        if chunk_size <= 0 or chunk_size % config.canonical_block_size:
            raise ValueError(
                "native chunk size must be positive and divisible by canonical block size"
            )
        if chunk_size_bytes <= 0:
            raise ValueError("native chunk byte size must be positive")
        self.get_keys = get_keys
        self.process_tokens = process_tokens
        self.rank = rank
        self.layout_id = layout_id
        self.config = config
        self.chunk_size = chunk_size
        self.chunk_size_bytes = chunk_size_bytes
        self.piece_manifest = piece_manifest or []
        self.bindings = {}
        self.sent = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.report_epoch = uuid.uuid4().hex
        self.seq = 0
        self.force_snapshot = True
        self.thread = threading.Thread(
            target=self._run, name="native-cpu-catalog", daemon=True
        )
        self.thread.start()

    def bind(self, token_ids) -> None:
        """Register identities using the engine's token database, before store.

        Registration grants no residency. A later key snapshot must independently
        establish that the native object is readable.
        """
        tokens = [int(token) for token in token_ids]
        keys = content_keys(
            self.config.content_namespace, tokens, self.config.canonical_block_size
        )
        root = root_key(
            self.config.content_namespace, self.config.canonical_block_size
        ).hex()
        with self.lock:
            for start, end, key in self.process_tokens(tokens):
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

    def poll(self) -> None:
        """Diff a fresh key snapshot and publish only changed known chunks.

        A snapshot reads metadata only. Failed reads do not renew CPU freshness.
        Recovery clears the worker scope before bounded replay; the ATOM-owned
        transport cursor describes sampled state, not backend mutation history.
        """
        self.sample_started = time.monotonic()
        with self.lock:
            bindings = dict(self.bindings)
        # get_keys owns storage synchronization; never hold the binding lock
        # while reading the backend or retain unknown keys between samples.
        readable = set(self.get_keys()).intersection(bindings)
        try:
            snapshot = self.force_snapshot
            if snapshot:
                self._send([], snapshot=True)
                self.sent.clear()
            # Remove first so catalog capacity is available for new chunks.
            report = [
                {"chunk_id": self.sent.pop(key), "readable": False}
                for key in self.sent.keys() - readable
            ]
            for key in readable - self.sent.keys():
                report.append(
                    {
                        **bindings[key],
                        "readable": True,
                        "size_bytes": self.chunk_size_bytes,
                    }
                )
                self.sent[key] = bindings[key]["chunk_id"]
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
            self._send(batch)  # Even an idle heartbeat requires a fresh sample.
            self.force_snapshot = False
        except Exception:
            self.force_snapshot = True
            raise

    def _send(self, events, *, snapshot=False):
        post_cpu_report(
            self.config.catalog_url,
            {
                "rank": self.rank,
                "source_epoch": self.report_epoch,
                "seq": str(self.seq + 1),
                "after_seq": str(self.seq),
                "snapshot": snapshot,
                "content_namespace": self.config.content_namespace,
                "layout_id": self.layout_id,
                "chunk_size": self.chunk_size,
                "sample_age_seconds": time.monotonic() - self.sample_started,
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
                logger.debug("CPU catalog observation will resync: %s", exc)
            self.stop.wait(self.config.cpu_poll_interval_seconds)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
