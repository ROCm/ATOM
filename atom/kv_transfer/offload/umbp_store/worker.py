# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Worker side of the UMBP offload.

The windowed put/get loop through the transfer pool and its completion
protocol are the store-agnostic :class:`KVStoreOffloadConnector`'s. A packed
chunk moves through:

* save: KV blocks --pack, copy--> a slot of the registered transfer pool
  --``batch_put_from_ptr``--> the standalone server;
* load: server --``batch_get_into_ptr``--> a pool slot --copy, unpack--> KV
  blocks.

The pool is in the worker GPU's HBM, registered with the server, which copies
to and from it through HIP IPC before it answers each call. What is UMBP's own:

* A slot returns to the pool once its call returned, whatever it answered:
  the server copied before answering. Only a call that raised leaves its slots
  quarantined, and the pool's memory then outlives the worker (``pool.close``).
* A put that raised may also leave the server copying from a slot into an
  object; this worker then loads nothing more from the server and marks it in
  this process (``_distrust_store``, :mod:`.taint`). The mark is this
  process's only: other worker processes and other engines are not told.
* No groups: each rank's object of a chunk is put, and evicted, on its own.
* UMBP has no remove by key: the startup probe's object stays on the server
  until it is evicted.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import torch

from atom.kv_transfer.offload.kv_store import pool as _store_pool
from atom.kv_transfer.offload.kv_store.client import CallRefused
from atom.kv_transfer.offload.kv_store.keys import probe_key
from atom.kv_transfer.offload.kv_store.pool import Slot
from atom.kv_transfer.offload.kv_store.worker import (
    KVStoreOffloadConnector,
    _synchronize,
)
from atom.kv_transfer.offload.metadata import LMCacheReqMeta
from atom.kv_transfer.offload.umbp_store import taint
from atom.kv_transfer.offload.umbp_store.client import UMBPStoreClient
from atom.kv_transfer.offload.umbp_store.config import (
    UMBP_LAYOUTS,
    check_engine_compatibility,
    parse_umbp_store_config,
    store_namespace,
)
from atom.kv_transfer.offload.umbp_store.scheduler import require_monotonic_clock_id

logger = logging.getLogger("atom")


class TransferSlotPool(_store_pool.TransferSlotPool):
    """The transfer slots of one UMBP worker, registered with its server.

    A standalone client deregisters all of its memory at once, so the pool is
    the only region the worker registers.
    """

    LABEL = "UMBP offload"
    SETTING_PREFIX = "umbp_store."


class UMBPStoreOffloadConnector(KVStoreOffloadConnector):
    """Dense or M3 PAGE offload worker backed by a UMBP standalone server."""

    LABEL = "UMBP offload"
    POOL_CLASS = TransferSlotPool
    # Side streams of the highest priority (`_copy_stream_priority`), not the
    # forward's stream. A UMBP put or get is a synchronous RPC on the save or
    # load thread, after which the server has copied the slots; a window's
    # copy then ends in host waits on its stream (the block-id upload, the
    # final fence). On the forward's stream each of those waits for every
    # forward kernel queued ahead of it, so during long prefills a save took
    # about a forward per window and the save queue fell past the source read
    # deadline. UMBP runs (dense and M3 layouts; V4.1 is refused) have shown
    # no hang on these side streams, and a much shorter warm-up.
    STAGES_ON_FORWARD_STREAM = False

    # -- the backend ------------------------------------------------------
    @classmethod
    def _parse_store_config(cls, kv_transfer_config: Any) -> Any:
        return parse_umbp_store_config(kv_transfer_config)

    @staticmethod
    def _check_engine_compatibility(store_cfg: Any, config: Any) -> None:
        check_engine_compatibility(store_cfg, config)

    @staticmethod
    def _require_monotonic_clock_id() -> str:
        return require_monotonic_clock_id()

    def _build_codec(
        self, kv_caches: Any, transfer_tensors: Any, num_blocks: int | None
    ) -> Any:
        # The dense codec reads kv_caches; the M3 codec the PAGE region map,
        # which is where the NSA index cache is published.
        return UMBP_LAYOUTS.build_codec(
            self._config,
            kv_caches,
            transfer_tensors,
            num_blocks,
            permit_per_request_state=self._permit_per_request_state,
        )

    def _store_namespace(self) -> str:
        return store_namespace(self._config, self.chunk_size)

    def _select_endpoint(self, device: torch.device) -> str:
        """The server's address: one per node, whatever the GPU."""
        return self._store_cfg.address

    def _endpoint_log(self, endpoint: str) -> str:
        return f"server={endpoint}"

    def _open_client(self, endpoint: str) -> UMBPStoreClient:
        return UMBPStoreClient(
            address=endpoint, startup_timeout_ms=self._store_cfg.startup_timeout_ms
        )

    def _copy_stream_priority(self, device: torch.device) -> int:
        """The highest the device offers.

        A GPU has few hardware queues and the runtime spreads streams over
        them by priority; a copy stream of the default priority may land on
        the queue of a forward stream and then runs only behind every kernel
        queued ahead of it there. Streams of the highest priority take queues
        of their own, still ordered after the forward that wrote the KV by the
        producer event. The default off the GPU: a CPU-only torch has no
        range to ask.
        """
        if device.type != "cuda":
            return 0
        return int(torch.cuda.Stream.priority_range()[1])

    def _store_marked(self, client: UMBPStoreClient, endpoint: str) -> bool:
        return taint.reason(endpoint) is not None

    def _refuse_loads_from_startup_mark(self, endpoint: str) -> None:
        self._refuse_loads_from_marked_server(endpoint, taint.reason(endpoint) or "")

    def _probe_store(
        self,
        client: UMBPStoreClient,
        pool: TransferSlotPool,
        namespace: str,
        rank: int,
    ) -> None:
        """Send one chunk of random bytes through the server and compare it back.

        An unregistered pool or an unreachable server otherwise shows up only
        as every lookup missing. A slot whose call raised stays quarantined
        when startup gives up. UMBP cannot remove by key, so the probe's object
        stays on the server until it is evicted: one chunk per worker start.
        """
        key = probe_key(namespace, rank, uuid.uuid4().hex)
        nbytes = pool.chunk_bytes
        [source] = pool.acquire("save", 1)
        [target] = pool.acquire("load", 1)
        # The probe's slots no copy may still reach.
        leased = [source, target]
        try:
            source.tensor.copy_(
                torch.randint(
                    0, 256, (nbytes,), dtype=torch.uint8, device=source.tensor.device
                )
            )
            target.tensor.zero_()
            _synchronize(pool.device)
            [stored] = self._probe_call(
                pool,
                leased,
                source,
                lambda: client.put([key], [source.ptr], [nbytes]),
                # A put that raised may still be read by the server.
                on_raise=self._distrust_store,
            )
            if not stored:
                raise RuntimeError(
                    f"UMBP startup probe: the server at {client.address} did not "
                    f"store a {nbytes}-byte chunk (full, or the pool is not "
                    "registered?)"
                )
            [present] = client.exists([key])
            if not present:
                raise RuntimeError(
                    "UMBP startup probe: a chunk just put is not on the server "
                    f"at {client.address}"
                )
            [read] = self._probe_call(
                pool,
                leased,
                target,
                lambda: client.get([key], [target.ptr], [nbytes]),
            )
            if not read:
                raise RuntimeError(
                    f"UMBP startup probe: the server at {client.address} did not "
                    f"return the {nbytes}-byte chunk just put"
                )
            _synchronize(pool.device)
            if not torch.equal(source.tensor, target.tensor):
                raise RuntimeError(
                    "UMBP startup probe: the chunk read back differs from the "
                    "one written"
                )
        except BaseException:
            self._release_once_device_idle(pool, leased)
            raise
        pool.release(leased)
        logger.warning(
            "UMBP startup probe: UMBP cannot remove by key; the probe object %s "
            "stays on the server until it is evicted",
            key,
        )

    @staticmethod
    def _probe_call(
        pool: TransferSlotPool,
        leased: list[Slot],
        slot: Slot,
        call: Any,
        *,
        on_raise: Any = None,
    ) -> list[bool]:
        """Run the probe's put or get on ``slot``, quarantining it if it raised.

        A call that returned is settled whatever it answered. A quarantined
        slot leaves ``leased``, the probe's slots still to release, and is
        reported to ``on_raise``, with why, when one is given.
        """
        try:
            return call()
        except CallRefused:
            raise  # nothing was sent: the slot is released with the rest
        except BaseException as exc:
            leased.remove(slot)
            pool.quarantine([slot], reason="a startup probe call raised")
            if on_raise is not None:
                on_raise(f"the startup probe's put raised {exc!r}")
            raise

    # -- transfers -----------------------------------------------------------
    def _put_chunks(
        self,
        client: UMBPStoreClient,
        req: LMCacheReqMeta,
        run: range,
        keys: list[str],
        slots: list[Slot],
    ) -> list[bool]:
        return client.put(
            keys, [slot.ptr for slot in slots], [self._chunk_bytes] * len(slots)
        )

    def _settle_put(
        self, pool: TransferSlotPool, slots: list[Slot], answer: list[bool]
    ) -> str | None:
        # The server copied before it answered: the slots are idle.
        pool.release(slots)
        not_stored = answer.count(False)
        if not_stored:
            return (
                f"the server did not store {not_stored} of {len(answer)} chunks "
                "(UMBP gives no reason: full, or a failed RPC)"
            )
        return None

    def _get_chunks(
        self, client: UMBPStoreClient, keys: list[str], slots: list[Slot]
    ) -> list[bool]:
        return client.get(
            keys, [slot.ptr for slot in slots], [self._chunk_bytes] * len(slots)
        )

    def _get_failure(
        self, pool: TransferSlotPool, slots: list[Slot], answer: list[bool]
    ) -> str | None:
        # A miss, an eviction or an object of another size (another layout)
        # all answer False.
        not_read = answer.count(False)
        if not not_read:
            return None
        pool.release(slots)
        return (
            f"the server did not return {not_read} of {len(answer)} chunks "
            "(UMBP gives no reason: evicted, or a failed RPC)"
        )

    # -- distrust ------------------------------------------------------------
    def _distrust_store(self, why: str) -> None:
        """Load nothing more from the server, and mark it in this process.

        ``why`` is a put that raised: the server may still be copying from the
        quarantined slot into an object, and whether that copy can land in an
        object the server has since given to another key is not something
        UMBP documents. This worker stops loading, for good, and marks the
        server in this process (:mod:`.taint`). The mark is this process's
        only: other worker processes and other engines on the server go on
        loading from it, and a restarted engine does too. A load needs every
        rank's chunks, so every load of this engine fails and its request
        prefills. Saves go on.
        """
        address = self._store_cfg.address
        taint.mark(address, why)
        if self._refuse_loads(why):
            logger.error(
                "UMBP offload: %s, and the server at %s may still be copying it "
                "into an object. This worker loads nothing more from that "
                "server and marks it in this process only; other processes and "
                "engines are not told. Every load of this engine prefills "
                "instead; restart the server, then the engine, to load again",
                why,
                address,
            )

    def _refuse_loads_from_marked_server(self, address: str, why: str) -> str:
        """Load nothing more from a server this process has marked."""
        if self._refuse_loads(why):
            logger.error(
                "UMBP offload: this process marked the server at %s (%s); this "
                "worker loads nothing more from it, and every load of this "
                "engine prefills instead",
                address,
                why,
            )
        return self._store_distrusted or why

    def _load_refusal(self, client: UMBPStoreClient) -> str | None:
        """Why this worker may not load from its server now; None when it may.

        Its own verdict holds for good (`_distrust_store`), and so does this
        process's mark of the server, which another worker of the process may
        have placed a moment ago.
        """
        if self._store_distrusted is not None:
            return self._store_distrusted
        address = self._store_cfg.address
        marked = taint.reason(address)
        if marked is not None:
            return self._refuse_loads_from_marked_server(address, marked)
        return None


__all__ = ["UMBPStoreOffloadConnector"]
