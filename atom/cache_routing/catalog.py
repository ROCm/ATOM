# SPDX-License-Identifier: MIT
"""Bounded, versioned cache facts. Residency and content identity are separate."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

from atom.cache_routing.config import CacheRoutingConfig
from atom.cache_routing.keys import extend_keys, root_key


class SnapshotRequired(ValueError):
    pass


@dataclass
class Snapshot:
    entries: tuple[dict, ...]
    seq: int
    expires: float


class CacheCatalog:
    """Commit metadata and its cursor together; consumers never inspect GPU pages.

    CPU readiness requires every worker of this execution, including all PP
    stages and TP/DCP shards. Worker restarts, gaps and stale heartbeats revoke
    CPU residency. HBM and CPU scopes are independently removable.
    """

    def __init__(
        self, config: CacheRoutingConfig, layout_id: str, hash_span: int, world: int
    ):
        if hash_span % config.canonical_block_size:
            raise ValueError("canonical block size must divide the engine hash span")
        self.config = config
        self.layout_id = layout_id
        self.hash_span = hash_span
        self.world = world
        self.epoch = uuid.uuid4().hex
        self.seq = 0
        self.lock = threading.RLock()
        self.entries: dict[tuple[str, str], dict] = {}
        self.log: deque[tuple[dict, int]] = deque()
        self.log_bytes = 0
        self.snapshots: dict[str, Snapshot] = {}
        self.native: dict[int, tuple[list[str], int]] = {}
        self.cpu_sources: dict[int, dict] = {}
        self.retired_epochs: dict[int, set[str]] = {}
        self.load: dict = {}
        self.info: dict = {}
        self.accepted_dispatches: deque[str] = deque(maxlen=4096)
        self.cpu_ready = False

    def _commit(self, event: dict) -> None:
        self.seq += 1
        event = {**event, "object_version": self.seq}
        batch = {
            "protocol_version": 1,
            "source_id": self.config.execution_id,
            "source_epoch": self.epoch,
            "seq": str(self.seq),
            "ts_unix_ns": str(time.time_ns()),
            "events": [event],
        }
        size = len(json.dumps(batch))
        self.log.append((batch, size))
        self.log_bytes += size
        while self.log_bytes > self.config.max_log_bytes and self.log:
            _, old_size = self.log.popleft()
            self.log_bytes -= old_size
        # An over-budget recovery is explicitly invalidated, never silently
        # continued across a missing eviction.
        oldest = int(self.log[0][0]["seq"]) if self.log else self.seq + 1
        self.snapshots = {
            k: s
            for k, s in self.snapshots.items()
            if s.expires > time.monotonic() and s.seq >= oldest - 1
        }

    def _upsert(self, tier: str, chunk_id: str, entry: dict) -> None:
        key = (tier, chunk_id)
        old = self.entries.get(key)
        if (
            old is not None
            and {k: v for k, v in old.items() if k != "object_version"} == entry
        ):
            return
        if key not in self.entries and len(self.entries) >= self.config.max_entries:
            return  # Unknown cache only loses a benefit, never adds one.
        self.entries[key] = {**entry, "object_version": self.seq + 1}
        self._commit({"type": "residency_upsert", **entry})

    def _remove(self, tier: str, chunk_id: str) -> None:
        old = self.entries.pop((tier, chunk_id), None)
        if old is not None:
            self._commit({"type": "residency_remove", **old})

    def hbm_events(self, events: list) -> None:
        """Consume committed BlockManager events before asynchronous transport."""
        from atom.distributed.kv_events import (
            AllBlocksCleared,
            BlockRemoved,
            BlockStored,
        )

        with self.lock:
            for event in events:
                if isinstance(event, AllBlocksCleared):
                    for tier, chunk in list(self.entries):
                        if tier == "HBM":
                            self._remove(tier, chunk)
                    self.native.clear()
                elif isinstance(event, BlockRemoved):
                    for native_hash in event.block_hashes:
                        self._remove("HBM", str(native_hash))
                    # Keep parent chain metadata: a nonresident ancestor can
                    # still be needed to define descendants received later.
                elif isinstance(event, BlockStored):
                    if (
                        event.medium != "GPU"
                        or event.lora_id is not None
                        or event.lora_name is not None
                        or any(event.extra_keys or [])
                    ):
                        continue  # First-version content identity covers plain text only.
                    if event.block_size != self.hash_span:
                        continue
                    parent_info = self.native.get(event.parent_block_hash)
                    if event.parent_block_hash is None:
                        parent = root_key(
                            self.config.content_namespace,
                            self.config.canonical_block_size,
                        )
                        start = 0
                    elif parent_info:
                        parent = bytes.fromhex(parent_info[0][-1])
                        start = parent_info[1]
                    else:
                        continue
                    for index, native_hash in enumerate(event.block_hashes):
                        tokens = event.token_ids[
                            index * self.hash_span : (index + 1) * self.hash_span
                        ]
                        if len(tokens) != self.hash_span:
                            break
                        keys = extend_keys(
                            parent, tokens, self.config.canonical_block_size
                        )
                        if (
                            len(self.native) >= self.config.max_entries
                            and native_hash not in self.native
                        ):
                            break
                        self.native[native_hash] = (keys, start + self.hash_span)
                        self._upsert(
                            "HBM",
                            str(native_hash),
                            {
                                "tier": "HBM",
                                "chunk_id": str(native_hash),
                                "execution_id": self.config.execution_id,
                                "content_namespace": self.config.content_namespace,
                                "layout_id": self.layout_id,
                                "content_keys": keys,
                                "parent_key": parent.hex(),
                                "token_start": start,
                                "token_end": start + self.hash_span,
                                "state": "READY",
                            },
                        )
                        parent = bytes.fromhex(keys[-1])
                        start += self.hash_span

    def cpu_update(self, update: dict) -> None:
        """Apply an ordered worker report, with snapshot recovery after a gap."""
        rank, epoch = update["rank"], update["source_epoch"]
        seq, after = int(update["seq"]), int(update["after_seq"])
        sample_age = float(update.get("sample_age_seconds", 0))
        if (
            type(rank) is not int
            or not 0 <= rank < self.world
            or not isinstance(epoch, str)
            or not epoch
            or after < 0
            or seq < after
            or type(update["chunk_size"]) is not int
            or update["chunk_size"] <= 0
            or not isinstance(update["layout_id"], str)
            or not update["layout_id"]
            or not math.isfinite(sample_age)
            or sample_age < 0
        ):
            raise ValueError("invalid CPU source identity/cursor")
        with self.lock:
            old = self.cpu_sources.get(rank)
            if old and old["epoch"] == epoch and seq < old["seq"]:
                return
            if (
                old
                and old["epoch"] == epoch
                and seq == old["seq"]
                and not update.get("snapshot")
            ):
                old["seen"] = time.monotonic() - sample_age
                self._refresh_cpu()
                return
            retired = self.retired_epochs.setdefault(rank, set())
            if epoch in retired:
                raise SnapshotRequired("retired CPU source epoch")
            if old and old["epoch"] != epoch:
                retired.add(old["epoch"])
                old = None
            changed = (
                {chunk for tier, chunk in self.entries if tier == "CPU"}
                if update.get("snapshot")
                else set()
            )
            if update.get("snapshot"):
                entries = {}
            elif old is None or old["seq"] != after:
                self.cpu_sources.pop(rank, None)
                self._refresh_cpu()
                raise SnapshotRequired("CPU report gap")
            else:
                if (old["layout_id"], old["chunk_size"]) != (
                    update["layout_id"],
                    update["chunk_size"],
                ):
                    raise SnapshotRequired("CPU geometry changed without a snapshot")
                entries = old["entries"]
            # Validate the entire page before mutating any committed entry.
            for event in update["events"]:
                if event["readable"]:
                    start, end = event["token_start"], event["token_end"]
                    keys = event["content_keys"]
                    if (
                        start < 0
                        or end <= start
                        or start % update["chunk_size"]
                        or end - start != update["chunk_size"]
                        or len(keys) * self.config.canonical_block_size != end - start
                        or any(len(bytes.fromhex(key)) != 32 for key in keys)
                        or event["chunk_id"] != keys[-1]
                        or len(bytes.fromhex(event["parent_key"])) != 32
                        or event["size_bytes"] < 0
                    ):
                        raise ValueError("invalid CPU chunk binding")
            for event in update["events"]:
                chunk = event["chunk_id"]
                changed.add(chunk)
                if event["readable"]:
                    if len(entries) < self.config.max_entries or chunk in entries:
                        entries[chunk] = event
                else:
                    entries.pop(chunk, None)
            self.cpu_sources[rank] = {
                "epoch": epoch,
                "seq": seq,
                "entries": entries,
                "seen": time.monotonic() - sample_age,
                "layout_id": update["layout_id"],
                "chunk_size": update["chunk_size"],
                "piece_manifest": update.get("piece_manifest")
                or (old or {}).get("piece_manifest", []),
            }
            self._refresh_cpu(changed)

    def _refresh_cpu(self, changed: set[str] | None = None) -> None:
        sources = list(self.cpu_sources.values())
        ready = (
            len(sources) == self.world
            and all(
                time.monotonic() - s["seen"] < self.config.stale_seconds
                for s in sources
            )
            and len({(s["layout_id"], s["chunk_size"]) for s in sources}) == 1
        )
        if not ready:
            if self.cpu_ready:
                for tier, chunk in list(self.entries):
                    if tier == "CPU":
                        self._remove(tier, chunk)
            self.cpu_ready = False
            return
        chunks = changed or set()
        if not self.cpu_ready:
            chunks = set.intersection(*(set(s["entries"]) for s in sources))
        self.cpu_ready = True
        self.info["cpu_layout_id"] = sources[0]["layout_id"]
        self.info["lmcache_chunk_size_tokens"] = sources[0]["chunk_size"]
        manifests = {
            str(rank): source["piece_manifest"]
            for rank, source in sorted(self.cpu_sources.items())
        }
        self.info["cpu_piece_manifests"] = manifests
        manifest_id = hashlib.sha256(
            json.dumps(manifests, sort_keys=True).encode()
        ).hexdigest()
        for chunk in chunks:
            if any(chunk not in s["entries"] for s in sources):
                self._remove("CPU", chunk)
                continue
            parts = [s["entries"][chunk] for s in sources]
            first = parts[0]
            identity = (first["token_start"], first["token_end"], first["content_keys"])
            if any(
                (p["token_start"], p["token_end"], p["content_keys"]) != identity
                for p in parts
            ):
                self._remove("CPU", chunk)
                continue
            self._upsert(
                "CPU",
                chunk,
                {
                    "tier": "CPU",
                    "chunk_id": chunk,
                    "storage_domain_id": self.config.storage_domain_id
                    or self.config.execution_id,
                    "content_namespace": self.config.content_namespace,
                    "layout_id": sources[0]["layout_id"],
                    "content_keys": first["content_keys"],
                    "parent_key": first["parent_key"],
                    "token_start": first["token_start"],
                    "token_end": first["token_end"],
                    "state": "READY",
                    "size_bytes": sum(p["size_bytes"] for p in parts),
                    "required_manifest_id": manifest_id,
                },
            )

    def snapshot(
        self, snapshot_id: str | None = None, offset: int = 0, page_size: int = 1024
    ) -> dict:
        """Page one immutable cut; leases are bounded by time, count and log bytes."""
        if offset < 0 or not 0 < page_size <= 1024:
            raise ValueError("invalid snapshot page")
        with self.lock:
            self._refresh_cpu()
            now = time.monotonic()
            self.snapshots = {
                k: s for k, s in self.snapshots.items() if s.expires > now
            }
            if snapshot_id is None:
                if len(self.snapshots) >= 2:
                    raise SnapshotRequired("snapshot budget exhausted")
                snapshot_id = uuid.uuid4().hex
                self.snapshots[snapshot_id] = Snapshot(
                    tuple(self.entries.values()), self.seq, now + 10
                )
            snap = self.snapshots.get(snapshot_id)
            if snap is None:
                raise SnapshotRequired("snapshot lease expired")
            page = snap.entries[offset : offset + page_size]
            next_offset = offset + len(page)
            if next_offset >= len(snap.entries):
                del self.snapshots[snapshot_id]
            return {
                "protocol_version": 1,
                "source_id": self.config.execution_id,
                "source_epoch": self.epoch,
                "snapshot_id": snapshot_id,
                "cut_seq": str(snap.seq),
                "entries": page,
                "next_page_token": (
                    str(next_offset) if next_offset < len(snap.entries) else None
                ),
                "complete": True,
            }

    def events(self, epoch: str, after_seq: int) -> dict:
        """Return ordered deltas; the cursor also detects a lost final eviction."""
        with self.lock:
            self._refresh_cpu()
            oldest = int(self.log[0][0]["seq"]) if self.log else self.seq + 1
            if epoch != self.epoch or not oldest - 1 <= after_seq <= self.seq:
                raise SnapshotRequired("catalog replay window expired")
            return {
                "protocol_version": 1,
                "source_id": self.config.execution_id,
                "source_epoch": self.epoch,
                "cut_seq": str(self.seq),
                "events": [
                    batch for batch, _ in self.log if int(batch["seq"]) > after_seq
                ],
                "complete": True,
            }
