# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Worker side of the native Mooncake Store offload.

The windowed put/get loop through the transfer pool and its completion
protocol are the store-agnostic :class:`KVStoreOffloadConnector`'s. A packed
chunk moves through:

* save: KV blocks --pack, copy--> a slot of the registered transfer pool
  --``batch_put_from``--> the Store owners;
* load: owners --``batch_get_into``--> a pool slot --copy, unpack--> KV blocks.

The pool is in the worker GPU's HBM: the NIC reads and writes GPU memory
directly, and no byte passes through host memory. This module adds the
Mooncake Store's own part:

* one PCI-local RDMA device per worker, and the Mooncake settings that would
  undo it (``_require_rdma_environment``);
* every rank's object of a chunk is put in the chunk's Mooncake group;
* a slot returns to the pool only once no NIC access can still reach it:
  Mooncake gives up on a transfer without cancelling its RDMA work, so a slot
  whose call outlived its batch wait is quarantined for good (``_settle``);
* a put that leaves its slot quarantined may also still write the owner
  memory the Store took back, inside another key's object: from then on the
  worker loads nothing from the Store, and marks it tainted so that no client
  of it does (``_distrust_store``, :mod:`.taint`). Before every get, and
  after a load's last one, it looks for that mark (``_load_refusal``);
* a save asks the Store which of its chunks it already holds, one master
  RPC per window (``KVStoreOffloadConnector._save_segments``). The master
  grants each key it finds a read lease, as a scheduler lookup does: only
  chunks already held, for a few seconds, which keeps them from eviction that
  long and never stops a new chunk's put, since a put of a missing key takes
  no lease;
* the startup probe removes its object again.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import torch

from atom.kv_transfer.offload.kv_store.save_pipeline import SettlementTransaction
from atom.kv_transfer.offload.kv_store.worker import (
    KVStoreOffloadConnector,
    _synchronize,
)
from atom.kv_transfer.offload.metadata import LMCacheReqMeta
from atom.kv_transfer.offload.mooncake_store import client as store_client
from atom.kv_transfer.offload.mooncake_store.config import (
    StorePool,
    check_engine_compatibility,
    parse_mooncake_store_config,
    store_state_tier,
)
from atom.kv_transfer.offload.mooncake_store.keys import (
    STORE_TAINT_KEY,
    chunk_group_ids,
    probe_key,
    store_namespace,
)
from atom.kv_transfer.offload.mooncake_store.layouts import build_mooncake_codec
from atom.kv_transfer.offload.mooncake_store.nic import requester_rdma_device
from atom.kv_transfer.offload.mooncake_store.pool import Slot, TransferSlotPool
from atom.kv_transfer.offload.mooncake_store.scheduler import (
    require_monotonic_clock_id,
)
from atom.kv_transfer.offload.mooncake_store.taint import TaintMarker

logger = logging.getLogger("atom")


def _require_rdma_environment() -> None:
    """Refuse Mooncake settings that silently undo one-NIC-per-worker reads.

    ``MC_NUM_QP_PER_EP`` is read by every Mooncake transfer engine of the
    process, the P/D one included. With Mooncake's default of 2, concurrent
    reads from several workers stalled for 30-60 s and then failed.
    ``MC_MS_AUTO_DISC=1`` makes the Store client ignore its RDMA device and
    take every NIC.
    """
    qp = os.environ.get("MC_NUM_QP_PER_EP", "").strip()
    if qp != "1":
        raise ValueError(
            "the Mooncake Store offload needs MC_NUM_QP_PER_EP=1 in every "
            f"Mooncake process, the decode's P/D engine included (got {qp or 'unset'})"
        )
    if os.environ.get("MC_MS_AUTO_DISC", "").strip() == "1":
        raise ValueError(
            "MC_MS_AUTO_DISC=1 makes the Store client use every NIC; unset it, "
            "the offload picks one NIC per worker"
        )


def _device_index(device: torch.device) -> int:
    if device.type == "cuda" and device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


class MooncakeStoreOffloadConnector(KVStoreOffloadConnector):
    """Offload worker (dense KV, M3 PAGE, or V4.1 replay PAGEs) whose tier is a
    Mooncake Store, reached directly."""

    LABEL = "Mooncake Store offload"
    POOL_CLASS = TransferSlotPool

    def __new__(cls, config: Any = None, *args: Any, **kwargs: Any):
        # Registered once for every layout; one with state adds its tier.
        # No config: `copy` and `pickle` rebuilding an instance.
        if (
            cls is MooncakeStoreOffloadConnector
            and config is not None
            and store_state_tier(config)
        ):
            from atom.kv_transfer.offload.mooncake_store.state_tier import (
                MooncakeStoreStateConnector,
            )

            return super().__new__(MooncakeStoreStateConnector)
        return super().__new__(cls)

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # Places the taint marker in this worker's Store; and when a load
        # next looks for one there (`_load_refusal`).
        self._taint_marker: TaintMarker | None = None

    # -- the backend ------------------------------------------------------
    @classmethod
    def _parse_store_config(cls, kv_transfer_config: Any) -> Any:
        return parse_mooncake_store_config(kv_transfer_config)

    @staticmethod
    def _check_engine_compatibility(store_cfg: Any, config: Any) -> None:
        check_engine_compatibility(store_cfg, config)

    @staticmethod
    def _require_monotonic_clock_id() -> str:
        return require_monotonic_clock_id()

    def _build_codec(
        self, kv_caches: Any, transfer_tensors: Any, num_blocks: int | None
    ) -> Any:
        return build_mooncake_codec(
            self._config,
            kv_caches,
            transfer_tensors,
            num_blocks,
            permit_per_request_state=self._permit_per_request_state,
        )

    def _store_namespace(self) -> str:
        return store_namespace(self._config, self.chunk_size)

    def _select_endpoint(self, device: torch.device) -> tuple[str | None, StorePool]:
        """This worker's RDMA device (None over tcp) and its Store pool."""
        cfg = self._store_cfg
        nic = None
        if cfg.protocol == "rdma":
            _require_rdma_environment()
            nic = requester_rdma_device(_device_index(device), cfg)
        return nic, cfg.pool_of(nic)

    def _endpoint_log(self, endpoint: tuple[str | None, StorePool]) -> str:
        nic, store_pool = endpoint
        return f"nic={nic or self._store_cfg.protocol} master={store_pool.master}"

    def _open_client(
        self, endpoint: tuple[str | None, StorePool]
    ) -> store_client.MooncakeStoreClient:
        nic, store_pool = endpoint
        cfg = self._store_cfg
        client = store_client.MooncakeStoreClient(
            local_hostname=cfg.local_hostname,
            metadata_server=store_pool.metadata,
            master_server_addr=store_pool.master,
            protocol=cfg.protocol,
            rdma_devices=nic or "",
        )
        # Before the probe, whose put may leave a write behind.
        self._taint_marker = TaintMarker(client)
        return client

    def _stop_client_users(self) -> bool:
        # A retry of the taint marker still inside a Store call keeps the
        # client open.
        marker, self._taint_marker = self._taint_marker, None
        return marker is None or marker.close()

    def _client_kept_open(self) -> None:
        logger.warning(
            "Mooncake Store offload: the taint marker is still being placed; "
            "the Store client stays open"
        )

    def _store_marked(
        self,
        client: store_client.MooncakeStoreClient,
        endpoint: tuple[str | None, StorePool],
    ) -> bool:
        tainted, detail = self._look_for_taint(client)
        if tainted is None:
            raise RuntimeError(
                "Mooncake Store startup: cannot tell whether the Store at "
                f"{endpoint[1].master} carries the taint marker {STORE_TAINT_KEY} "
                f"({detail})"
            )
        return tainted

    def _probe_store(
        self,
        client: store_client.MooncakeStoreClient,
        pool: TransferSlotPool,
        namespace: str,
        rank: int,
    ) -> None:
        """Send one chunk of random bytes through the Store and compare it back.

        An unregistered pool, an unreachable owner or a misrouted NIC
        otherwise shows up only as every lookup missing. The probe's slots are
        settled as a save's and a load's are, so one its failed put or get
        may still reach stays quarantined when startup gives up.
        """
        key = probe_key(namespace, rank, uuid.uuid4().hex)
        nbytes = pool.chunk_bytes
        [source] = pool.acquire("save", 1)
        [target] = pool.acquire("load", 1)
        # The probe's slots no transfer may still reach.
        leased = [source, target]
        try:
            source.tensor.copy_(
                torch.randint(
                    0, 256, (nbytes,), dtype=torch.uint8, device=source.tensor.device
                )
            )
            target.tensor.zero_()
            _synchronize(pool.device)
            put_rc = self._probe_transfer(
                pool,
                leased,
                source,
                lambda: client.put([key], [source.ptr], [nbytes]),
                # A put that may still write marks the Store before startup
                # gives up.
                on_unsettled=self._distrust_store,
            )
            if put_rc != 0:
                raise RuntimeError(
                    f"Mooncake Store startup probe: put of {nbytes} bytes to "
                    f"{client.master_server_addr} failed "
                    f"({store_client.describe(put_rc)})"
                )
            try:
                [present] = client.exists([key])
                if present != 1:
                    raise RuntimeError(
                        "Mooncake Store startup probe: a chunk just put is not in "
                        f"the Store ({store_client.describe(present)}); is an "
                        "owner mounted and is the pool registered?"
                    )
                get_rc = self._probe_transfer(
                    pool,
                    leased,
                    target,
                    lambda: client.get([key], [target.ptr], [nbytes]),
                )
                if get_rc != nbytes:
                    raise RuntimeError(
                        "Mooncake Store startup probe: get returned "
                        f"{store_client.describe(get_rc)} for a {nbytes}-byte chunk"
                    )
                _synchronize(pool.device)
                if not torch.equal(source.tensor, target.tensor):
                    raise RuntimeError(
                        "Mooncake Store startup probe: the chunk read back differs "
                        "from the one written"
                    )
            finally:
                removed = client.remove(key, force=True)
                if removed != 0:
                    logger.warning(
                        "Mooncake Store startup probe: could not remove %s (%s); "
                        "it stays until evicted",
                        key,
                        store_client.describe(removed),
                    )
        except BaseException:
            self._release_once_device_idle(pool, leased)
            raise
        pool.release(leased)

    @staticmethod
    def _probe_transfer(
        pool: TransferSlotPool,
        leased: list[Slot],
        slot: Slot,
        call: Any,
        *,
        on_unsettled: Any = None,
    ) -> int:
        """Run the probe's put or get on ``slot``, quarantining it if unsettled.

        A quarantined slot leaves ``leased``, the probe's slots still to release,
        and is reported to ``on_unsettled``, with why, when one is given.
        """
        clock = store_client.CallClock()
        try:
            [code] = call()
        except store_client.CallRefused:
            raise  # nothing was posted: the slot is released with the rest
        except BaseException as exc:
            leased.remove(slot)
            pool.quarantine([slot], reason="a startup probe transfer raised")
            if on_unsettled is not None:
                on_unsettled(f"the startup probe's transfer raised {exc!r}")
            raise
        if not store_client.buffer_settled(code, clock.seconds()):
            why = (
                f"a startup probe transfer gave up after {clock.seconds():.0f}s "
                "with its RDMA work possibly still posted"
            )
            leased.remove(slot)
            pool.quarantine([slot], reason=why)
            if on_unsettled is not None:
                on_unsettled(why)
        return code

    # -- transfers -----------------------------------------------------------
    def _put_chunks(
        self,
        client: store_client.MooncakeStoreClient,
        req: LMCacheReqMeta,
        run: range,
        keys: list[str],
        slots: list[Slot],
    ) -> tuple[list[int], float]:
        """Per key Mooncake's code, and how long the call took."""
        group_ids = chunk_group_ids(
            self._namespace, req.chunk_hashes, run.start, run.stop
        )
        clock = store_client.CallClock()
        codes = client.put(
            keys,
            [slot.ptr for slot in slots],
            [self._chunk_bytes] * len(slots),
            group_ids=group_ids,
        )
        return codes, clock.seconds()

    def _settle_put(
        self,
        pool: TransferSlotPool | SettlementTransaction,
        slots: list[Slot],
        answer: tuple[list[int], float],
    ) -> str | None:
        codes, call_seconds = answer
        unsettled = self._settle(pool, slots, codes, call_seconds)
        if unsettled:
            self._distrust_store(
                f"a put gave up on {unsettled} of {len(slots)} chunks after "
                f"{call_seconds:.0f}s"
            )
        bad = [code for code in codes if code != 0]
        if bad:
            return (
                f"{len(bad)} of {len(codes)} chunks, first "
                f"{store_client.describe(bad[0])}"
            )
        return None

    def _get_chunks(
        self,
        client: store_client.MooncakeStoreClient,
        keys: list[str],
        slots: list[Slot],
    ) -> tuple[list[int], float]:
        """Per key the bytes read or Mooncake's code, and how long it took."""
        clock = store_client.CallClock()
        codes = client.get(
            keys,
            [slot.ptr for slot in slots],
            [self._chunk_bytes] * len(slots),
        )
        return codes, clock.seconds()

    def _get_failure(
        self,
        pool: TransferSlotPool,
        slots: list[Slot],
        answer: tuple[list[int], float],
    ) -> str | None:
        codes, call_seconds = answer
        # An object of another size is another layout.
        bad = [code for code in codes if code != self._chunk_bytes]
        if not bad:
            return None
        self._settle(pool, slots, codes, call_seconds)
        return f"{len(bad)} of {len(codes)} chunks, first " + (
            store_client.describe(bad[0])
            if bad[0] < 0
            else f"{bad[0]} bytes where {self._chunk_bytes} were expected"
        )

    # -- slot settlement -----------------------------------------------------
    @staticmethod
    def _settle(
        pool: TransferSlotPool | SettlementTransaction,
        slots: list[Slot],
        codes: list[int],
        call_seconds: float,
    ) -> int:
        """Release each slot whose Store transfer is over; quarantine the rest.

        ``call_seconds`` is how long the call that returned ``codes`` took: a
        failure is unsettled only after Mooncake's batch wait ran out. Returns
        how many slots were quarantined.
        """
        settled: list[Slot] = []
        unsettled: list[Slot] = []
        for slot, code in zip(slots, codes, strict=True):
            if store_client.buffer_settled(code, call_seconds):
                settled.append(slot)
            else:
                unsettled.append(slot)
        pool.release(settled)
        if unsettled:
            pool.quarantine(
                unsettled,
                reason=(
                    f"a Store transfer gave up after {call_seconds:.0f}s with "
                    "its RDMA work possibly still posted"
                ),
            )
        return len(unsettled)

    # -- distrust ------------------------------------------------------------
    def _distrust_store(self, why: str) -> None:
        """Load nothing more from the Store, and mark it so no client does.

        ``why`` is a put that may still write owner memory the Store has since
        given to other objects (see :mod:`.taint`). Quarantining the slot keeps
        this worker's memory out of that write's reach; the owners' memory is
        out of every client's, so this worker stops loading, for good, and
        places the Store's taint marker, which every other client of it checks.
        A load needs every rank's chunks, so every load of the engine fails and
        its request prefills. Saves go on.
        """
        if self._refuse_loads(why):
            logger.error(
                "Mooncake Store offload: %s, and its RDMA writes may still land "
                "in owner memory the Store has since given to other objects. "
                "This worker loads nothing more from the Store and marks it "
                "tainted (%s), so no client of it loads from it until it "
                "restarts; every load of this engine prefills instead",
                why,
                STORE_TAINT_KEY,
            )
        marker = self._taint_marker
        if marker is not None:
            marker.place()

    def _refuse_loads_from_startup_mark(
        self, endpoint: tuple[str | None, StorePool]
    ) -> None:
        self._refuse_loads_from_marked_store(endpoint[1].master)

    def _refuse_loads_from_marked_store(self, master: str) -> str:
        """Load nothing more from a Store that carries the taint marker."""
        why = f"the Store carries the taint marker {STORE_TAINT_KEY}"
        if self._refuse_loads(why):
            logger.error(
                "Mooncake Store offload: the Store at %s carries the taint "
                "marker %s: a put another client gave up on may still write "
                "memory the Store has since given to other objects. This worker "
                "loads nothing more from it, and every load of this engine "
                "prefills instead; restart the Store, then this engine, to load "
                "from it again",
                master,
                STORE_TAINT_KEY,
            )
        return self._store_distrusted or why

    def _load_refusal(self, client: store_client.MooncakeStoreClient) -> str | None:
        """Why this worker may not load from its Store now; None when it may.

        Its own verdict holds for good (`_distrust_store`). The Store's taint
        marker is looked for right before every get and after a load's last
        get, never trusted from an earlier look: another client may have
        marked the Store a moment ago. Once seen it holds for good too, and a
        look that gets no answer refuses this load only.
        """
        if self._store_distrusted is not None:
            return self._store_distrusted
        tainted, detail = self._look_for_taint(client)
        if tainted is None:
            return f"cannot tell whether the Store carries the taint marker ({detail})"
        if tainted:
            return self._refuse_loads_from_marked_store(client.master_server_addr)
        return None

    @staticmethod
    def _look_for_taint(
        client: store_client.MooncakeStoreClient,
    ) -> tuple[bool | None, str]:
        """Whether the Store carries the taint marker; None, and why, if unsaid."""
        try:
            [code] = client.exists([STORE_TAINT_KEY])
        except Exception as exc:  # noqa: BLE001  # an unanswered look refuses
            return None, repr(exc)
        if code < 0:
            return None, store_client.describe(code)
        return code == 1, ""


__all__ = ["MooncakeStoreOffloadConnector"]
