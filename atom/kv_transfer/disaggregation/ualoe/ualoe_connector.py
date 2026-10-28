# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""P/D KV transfer over the scale-up fabric: the decode rank maps the prefill
rank's KV pool and copies the request's blocks out of it with one kernel.

Pull, as MoRI-IO's default read mode: the router hands the prefill response
to decode, and decode starts the transfer.

* The prefill rank allocates its paged KV pool in exportable memory
  (:meth:`UALoEConnector.allocate_kv_pool`, ``shared_pool``).
* Per request, decode rank ``i`` sends ``READ_BEGIN`` to prefill rank ``i``.
  The reply claims the source blocks and, when decode's cached epoch is not
  the pool's, carries the pool descriptor and region layout.
* Decode maps the pool once (``peer_table``), launches the copy on a side
  stream ordered after its own compute (``fabric_copy``), reports the request
  received when the copy's event completes, and sends ``READ_END``.
* ``READ_END`` (success or not) releases the source blocks on prefill. Blocks
  no decode claimed within ``ualoe_send_timeout_s``, or claimed and not
  released within it (a decode that died mid-read), are released too.

Everything on the decode side runs on the worker's own loop: messages are
non-blocking and progress is polled from ``get_finished``.

Scope: equal TP on both sides (any DP), no PP, no DCP, no chunked transfer,
and plain paged KV only -- a model with slot, SWA ring or staging regions is
refused. The two ranks must share a fabric domain (``fabric`` handles, ROCm
>= 7.14) or a host (``posix_fd``).

``kv_transfer_config`` keys:

* ``handshake_port``        -- base of the prefill side-channel ports (6301).
* ``ualoe_handle``          -- ``auto`` (``fabric`` when the runtime exports
  one, else ``posix_fd``), ``fabric`` or ``posix_fd``. Prefill only.
* ``ualoe_domain``          -- fabric domain (rack / virtual pod); defaults to
  ``ATOM_UALOE_FABRIC_DOMAIN``. A pool from another named domain is refused.
* ``ualoe_send_timeout_s``  -- prefill: release blocks no decode claimed, or
  claimed and never released, after this long (120).
* ``ualoe_read_timeout_s``  -- decode: fail a read whose claim got no reply (30).
* ``ualoe_peer_idle_s``     -- decode: unmap a pool unused this long (600).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
import torch
import zmq
from aiter.dist.parallel_state import get_dp_group, get_tp_group

from atom.config import Config
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.port_offset import side_channel_port_offset
from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    KVConnectorOutput,
    ReqId,
    ReqMeta,
    TransferId,
)
from atom.kv_transfer.disaggregation.ualoe import shared_pool
from atom.kv_transfer.disaggregation.ualoe.fabric_copy import launch_copy
from atom.kv_transfer.disaggregation.ualoe.hip_vmm import HipError
from atom.kv_transfer.disaggregation.ualoe.peer_table import (
    PeerMapping,
    PeerTable,
    PeerUnreachable,
)
from atom.model_engine.sequence import Sequence
from atom.utils import envs, make_zmq_path, zmq_socket_ctx
from atom.utils.network import get_ip

logger = logging.getLogger("atom")

DEFAULT_HANDSHAKE_PORT = 6301
MSG_READ_BEGIN = b"read_begin"
MSG_READ_END = b"read_end"

HANDLE_AUTO = "auto"
HANDLES = (HANDLE_AUTO, *shared_pool.KINDS)

# What VMM rounding can add to the KV pool past its torch-sized bytes.
_POOL_ROUNDING_RESERVE = 2 << 20
# Released or expired transfer ids remembered, so a late message about one
# neither re-arms its deadline nor claims its (freed) blocks.
_RETIRED_REMEMBERED = 65536


def _option(kv_cfg: dict, key: str, default: Any, choices: tuple) -> Any:
    value = kv_cfg.get(key, default)
    if value not in choices:
        raise ValueError(
            f"kv_transfer_config[{key!r}] must be one of {choices}, got {value!r}"
        )
    return value


def _is_producer(kv_cfg: dict) -> bool:
    return kv_cfg.get("kv_role", "kv_producer") == "kv_producer"


def _domain(kv_cfg: dict) -> str:
    return str(kv_cfg.get("ualoe_domain") or envs.ATOM_UALOE_FABRIC_DOMAIN)


def check_scope(config: Config) -> None:
    """Refuse the topologies this connector does not handle."""
    if getattr(config, "pipeline_parallel_size", 1) > 1:
        raise NotImplementedError("the ualoe connector does not support PP")
    if getattr(config, "decode_context_parallel_size", 1) > 1:
        raise NotImplementedError("the ualoe connector does not support DCP")
    if config.kv_transfer_config.get("enable_chunked_transfer"):
        raise NotImplementedError(
            "the ualoe connector does not support chunked transfer"
        )


def refuse_extra_regions(tt: Any) -> None:
    """Refuse transfer tensors with anything besides paged block regions."""
    extra = [
        name
        for name in (
            "slot_regions",
            "swa_block_regions",
            "staging_region",
            "index_staging_region",
        )
        if getattr(tt, name, None)
    ]
    if extra:
        raise NotImplementedError(
            "the ualoe connector moves paged KV blocks only; this model also "
            f"publishes {', '.join(extra)}. Use the mooncake connector."
        )


def block_addresses(
    bases: list[int], unit_bytes: list[int], block_ids: list[int]
) -> tuple[list[int], list[int]]:
    """Every region's address of every block, region-major, and their sizes."""
    if not bases:
        return [], []
    ids = np.asarray(block_ids, dtype=np.int64)
    addrs = np.concatenate([base + ids * unit for base, unit in zip(bases, unit_bytes)])
    sizes = np.repeat(np.asarray(unit_bytes, dtype=np.int64), ids.size)
    return addrs.tolist(), sizes.tolist()


# ===========================================================================
# Scheduler side
# ===========================================================================


class UALoEConnectorScheduler(KVConnectorSchedulerBase):
    """Prefill hands its block ids out and keeps the blocks until decode is
    done reading them; decode queues one read per request."""

    def __init__(self, config: Config) -> None:
        check_scope(config)
        kv_cfg = config.kv_transfer_config
        self.is_producer = _is_producer(kv_cfg)
        self.handshake_port = int(kv_cfg.get("handshake_port", DEFAULT_HANDSHAKE_PORT))
        self.send_timeout_s = float(kv_cfg.get("ualoe_send_timeout_s", 120.0))
        self.engine_id = uuid.uuid4().hex
        self.tp_size = config.tensor_parallel_size
        self.dp_rank = config.parallel_config.data_parallel_rank
        self.block_size = config.kv_cache_block_size
        self.host_ip = get_ip()

        self._reqs_need_recv: dict[ReqId, tuple[Sequence, list[int]]] = {}
        # Producer: source blocks handed out, and when to stop holding them.
        self._reqs_to_send: dict[ReqId, float] = {}
        self._awaiting_send: set[str] = set()

    def get_num_new_matched_tokens(self, seq: Sequence) -> tuple[int, bool]:
        params = seq.kv_transfer_params or {}
        if params.get("do_remote_prefill") and not hasattr(seq, "kv_async_tagged"):
            seq.kv_async_tagged = True
            return len(seq.prompt_token_ids), True
        return 0, False

    def build_connector_meta(self) -> ConnectorMetadata:
        meta = ConnectorMetadata()
        for req_id, (seq, block_ids) in self._reqs_need_recv.items():
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=seq.kv_transfer_params,
            )
        meta.reqs_to_send = self._reqs_to_send
        self._reqs_need_recv = {}
        self._reqs_to_send = {}
        return meta

    def update_state_after_alloc(self, seq: Sequence) -> None:
        params = seq.kv_transfer_params or {}
        if not params.get("do_remote_prefill"):
            return
        assert not self.is_producer, "only decode handles do_remote_prefill"
        params["do_remote_prefill"] = False
        # Blocks decode already holds from its own prefix cache are not read.
        # Only meaningful when both sides cut the prompt into the same blocks.
        params["num_computed_blocks"] = (
            seq.num_cached_tokens // self.block_size
            if params.get("block_size") == self.block_size
            else 0
        )
        self._reqs_need_recv[seq.id] = (seq, list(seq.block_table))

    def request_finished(self, seq: Sequence) -> None:
        if not self.is_producer:
            return
        if getattr(seq, "leave_reason", None) == "aborted":
            # Nothing claims an abort's blocks: never hand their ids out.
            seq.kv_transfer_params_output = None
            return
        if not (seq.kv_transfer_params or {}).get("do_remote_decode"):
            return  # served here, not routed to a decode: nothing to hold
        self._awaiting_send.add(str(seq.id))
        self._reqs_to_send[seq.id] = time.monotonic() + self.send_timeout_s
        drafts = getattr(seq, "spec_token_ids", None)
        seq.kv_transfer_params_output = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": list(seq.block_table),
            "remote_engine_id": self.engine_id,
            "remote_host": self.host_ip,
            "remote_handshake_port": self.handshake_port,
            "tp_size": self.tp_size,
            "dp_rank": self.dp_rank,
            "block_size": self.block_size,
            "transfer_id": seq.id,
            "first_token_id": seq.output_tokens[0] if seq.output_tokens else None,
            "draft_token_ids": (
                [int(x) for x in drafts] if drafts is not None and len(drafts) else []
            ),
            "prefix_cache_hit_tokens": getattr(seq, "prefix_cache_hit_tokens", 0),
        }

    def should_defer_free(self, seq: Sequence) -> bool:
        return str(seq.id) in self._awaiting_send

    def send_finished(self, req_id) -> None:
        self._awaiting_send.discard(str(req_id))

    def source_blocks_released(self, seq: Sequence) -> None:
        """No block-lifetime state remains after the send claim is retired."""


# ===========================================================================
# Worker side
# ===========================================================================


@dataclass
class _Read:
    """One request's read, from READ_BEGIN to its copy's completion."""

    req_id: ReqId
    target: str
    transfer_id: TransferId
    src_blocks: list[int]
    dst_blocks: list[int]
    sent_at: float
    claimed: bool = False
    mapping: PeerMapping | None = None
    tables: tuple[torch.Tensor, torch.Tensor] | None = None
    done: torch.cuda.Event | None = None


class UALoEConnector(KVConnectorBase):
    """Prefill: export the KV pool and serve read claims. Decode: map the
    prefill pool and copy each request's blocks out of it."""

    @classmethod
    def kv_budget_reserve_bytes(cls, config) -> int:
        return _POOL_ROUNDING_RESERVE if _is_producer(config.kv_transfer_config) else 0

    @classmethod
    def allocate_kv_pool(cls, config, nbytes: int, device) -> Any:
        """The prefill KV pool, in memory decode can map."""
        kv_cfg = config.kv_transfer_config
        if not _is_producer(kv_cfg):
            return None
        handle = _option(kv_cfg, "ualoe_handle", HANDLE_AUTO, HANDLES)
        kinds = (
            [shared_pool.KIND_FABRIC, shared_pool.KIND_POSIX_FD]
            if handle == HANDLE_AUTO
            else [handle]
        )
        errors = []
        for kind in kinds:
            try:
                pool = shared_pool.allocate(nbytes, device, kind, _domain(kv_cfg))
                return pool.tensor
            except (HipError, OSError) as exc:
                logger.info("UALoE: no %s KV pool on this runtime (%s)", kind, exc)
                errors.append(f"{kind}: {exc}")
        raise RuntimeError(f"UALoE: cannot export the KV pool ({'; '.join(errors)})")

    def __init__(self, config: Config) -> None:
        check_scope(config)
        kv_cfg = config.kv_transfer_config
        self.is_producer = _is_producer(kv_cfg)
        self.tp_rank = get_tp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        self.dp_rank = get_dp_group().rank_in_group
        self.dp_size = get_dp_group().world_size
        self._device = torch.cuda.current_device()
        self._handshake_port = int(kv_cfg.get("handshake_port", DEFAULT_HANDSHAKE_PORT))
        self._send_timeout_s = float(kv_cfg.get("ualoe_send_timeout_s", 120.0))
        self._bases: list[int] = []
        self._unit_bytes: list[int] = []
        self._roles: list[str | None] = []

        # Producer: the listener thread and the engine loop share these.
        self._lock = threading.Lock()
        self._layout: dict[str, Any] | None = None
        self._deadlines: dict[TransferId, float] = {}
        self._claimed: set[TransferId] = set()
        self._retired: OrderedDict[TransferId, None] = OrderedDict()
        self._done_sending: set[TransferId] = set()
        self._listening = threading.Event()

        # Consumer: touched by the engine loop only.
        self._read_timeout_s = float(kv_cfg.get("ualoe_read_timeout_s", 30.0))
        self._peers = PeerTable(
            local_domain=_domain(kv_cfg),
            idle_s=float(kv_cfg.get("ualoe_peer_idle_s", 600.0)),
        )
        self._zmq: zmq.Context | None = None
        self._sockets: dict[str, zmq.Socket] = {}
        self._layouts: dict[str, dict[str, Any]] = {}
        self._reads: dict[tuple[str, TransferId], _Read] = {}
        self._stream: torch.cuda.Stream | None = None
        self._done_recving: set[ReqId] = set()
        self._failed_recving: set[ReqId] = set()

    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        if transfer_tensors is None:
            raise RuntimeError("the ualoe connector needs the KV transfer regions")
        refuse_extra_regions(transfer_tensors)
        regions = transfer_tensors.block_regions
        self._bases = [r.base_addr for r in regions]
        self._unit_bytes = [r.unit_bytes for r in regions]
        self._roles = [r.semantic_role for r in regions]
        if not self.is_producer:
            return
        pool = shared_pool.find_exported(
            [(r.base_addr, r.total_bytes) for r in regions]
        )
        if pool is None:
            raise RuntimeError(
                "UALoE: the KV block regions are not in an exported pool; the "
                "model runner must allocate it through allocate_kv_pool"
            )
        self._layout = {
            "epoch": pool.epoch,
            "pool": pool.descriptor(),
            "bases": self._bases,
            "unit_bytes": self._unit_bytes,
            "roles": self._roles,
        }
        port = self._handshake_port + side_channel_port_offset(
            self.dp_rank, self.tp_rank, self.tp_size, 0, 1, self.dp_size
        )
        threading.Thread(
            target=self._serve, args=(port,), daemon=True, name="ualoe-serve"
        ).start()
        if not self._listening.wait(timeout=30):
            raise RuntimeError(f"UALoE: side channel on port {port} did not come up")
        logger.info(
            "UALoE: serving a %s KV pool (%.2f GiB, %d block regions) on port %d",
            pool.kind,
            pool.mapped_bytes / (1 << 30),
            len(regions),
            port,
        )

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        if metadata is None:
            return
        if self.is_producer:
            with self._lock:
                for transfer_id, deadline in metadata.reqs_to_send.items():
                    if (
                        transfer_id not in self._claimed
                        and transfer_id not in self._retired
                    ):
                        self._deadlines[transfer_id] = deadline
            return
        for req_id, meta in metadata.reqs_to_recv.items():
            self._begin_read(req_id, meta)

    def get_finished(self) -> KVConnectorOutput:
        if self.is_producer:
            return KVConnectorOutput(finished_sending=self._take_finished_sending())
        self._poll_replies()
        self._poll_copies()
        self._expire_unanswered()
        self._peers.evict_idle()
        done, failed = self._done_recving, self._failed_recving
        self._done_recving, self._failed_recving = set(), set()
        return KVConnectorOutput(finished_recving=done, failed_recving=failed)

    def close(self) -> None:
        for sock in self._sockets.values():
            sock.close(linger=0)
        self._sockets.clear()
        self._peers.close()

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    def _serve(self, port: int) -> None:
        path = make_zmq_path("tcp", "*", port)
        with zmq_socket_ctx(path, zmq.ROUTER, bind=True) as sock:
            self._listening.set()
            while True:
                frames = sock.recv_multipart()
                msg_type = frames[1] if len(frames) > 1 else b"?"
                try:
                    identity, msg_type, payload = frames
                    data = msgpack.loads(payload)
                    if msg_type == MSG_READ_BEGIN:
                        reply = self.claim(data)
                        sock.send_multipart(
                            [identity, MSG_READ_BEGIN, msgpack.dumps(reply)]
                        )
                    elif msg_type == MSG_READ_END:
                        self.release(data["transfer_id"])
                    else:
                        logger.error("UALoE: unknown message type %r", msg_type)
                except Exception:
                    logger.exception("UALoE: handling a %r message failed", msg_type)

    def claim(self, data: dict) -> dict[str, Any]:
        """READ_BEGIN: hold the transfer's blocks until READ_END. The layout
        goes along unless the reader already has this pool's."""
        transfer_id = data["transfer_id"]
        with self._lock:
            if transfer_id in self._retired:
                return {
                    "transfer_id": transfer_id,
                    "ok": False,
                    "error": "the source blocks were already released",
                }
            self._claimed.add(transfer_id)
            # A copy takes milliseconds; a claim this old has a dead reader.
            self._deadlines[transfer_id] = time.monotonic() + self._send_timeout_s
        reply = {"transfer_id": transfer_id, "ok": True}
        if data.get("epoch") != self._layout["epoch"]:
            reply["layout"] = self._layout
        return reply

    def release(self, transfer_id: TransferId) -> None:
        """READ_END: the reader is done with the transfer's blocks."""
        with self._lock:
            if transfer_id in self._retired:
                return
            self._claimed.discard(transfer_id)
            self._deadlines.pop(transfer_id, None)
            self._retire_locked(transfer_id)

    def _take_finished_sending(self, now: float | None = None) -> set[TransferId]:
        now = time.monotonic() if now is None else now
        with self._lock:
            for transfer_id, deadline in list(self._deadlines.items()):
                if now > deadline:
                    del self._deadlines[transfer_id]
                    claimed = transfer_id in self._claimed
                    self._claimed.discard(transfer_id)
                    self._retire_locked(transfer_id)
                    logger.warning(
                        "UALoE: %s transfer %s; releasing its blocks",
                        "no READ_END for claimed" if claimed else "no decode claimed",
                        transfer_id,
                    )
            done, self._done_sending = self._done_sending, set()
        return done

    def _retire_locked(self, transfer_id: TransferId) -> None:
        self._retired[transfer_id] = None
        if len(self._retired) > _RETIRED_REMEMBERED:
            self._retired.popitem(last=False)
        self._done_sending.add(transfer_id)

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    def _socket(self, target: str) -> zmq.Socket:
        sock = self._sockets.get(target)
        if sock is None:
            if self._zmq is None:
                self._zmq = zmq.Context()
            sock = self._zmq.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            host, port = target.rsplit(":", 1)
            sock.connect(make_zmq_path("tcp", host, int(port)))
            self._sockets[target] = sock
        return sock

    def _begin_read(self, req_id: ReqId, meta: ReqMeta) -> None:
        if meta.tp_size != self.tp_size:
            logger.error(
                "UALoE: req %s comes from prefill TP %s but decode TP is %d; "
                "the ualoe connector needs equal TP",
                req_id,
                meta.tp_size,
                self.tp_size,
            )
            self._failed_recving.add(req_id)
            return
        port = int(meta.remote_handshake_port or DEFAULT_HANDSHAKE_PORT)
        port += side_channel_port_offset(
            meta.remote_dp_rank, self.tp_rank, self.tp_size, 0, 1, meta.remote_dp_size
        )
        target = f"{meta.remote_host}:{port}"
        count = min(len(meta.local_block_ids), len(meta.remote_block_ids))
        # Never past ``count``: a block decode's prefix cache holds may be
        # read by other requests right now, so it is not written even when
        # the cache covers the whole prompt (then nothing is copied).
        skip = min(max(meta.num_computed_blocks, 0), count)
        self._reads[(target, meta.transfer_id)] = _Read(
            req_id=req_id,
            target=target,
            transfer_id=meta.transfer_id,
            src_blocks=list(meta.remote_block_ids[skip:count]),
            dst_blocks=list(meta.local_block_ids[skip:count]),
            sent_at=time.monotonic(),
        )
        layout = self._layouts.get(target)
        request = {
            "transfer_id": meta.transfer_id,
            "epoch": layout["epoch"] if layout else 0,
        }
        self._socket(target).send_multipart([MSG_READ_BEGIN, msgpack.dumps(request)])

    def _poll_replies(self) -> None:
        for target, sock in list(self._sockets.items()):
            while True:
                try:
                    frames = sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                try:
                    msg_type, payload = frames
                    reply = msgpack.loads(payload)
                    if not isinstance(reply, dict) or "transfer_id" not in reply:
                        raise ValueError(f"no transfer_id in {reply!r}")
                except Exception:
                    logger.exception("UALoE: malformed reply from %s", target)
                    continue
                if msg_type == MSG_READ_BEGIN:
                    self._on_claimed(target, reply)
                else:
                    logger.error("UALoE: unexpected reply %r from %s", msg_type, target)

    def _on_claimed(self, target: str, reply: dict) -> None:
        read = self._reads.get((target, reply["transfer_id"]))
        if read is None or read.claimed:
            return  # timed out already, or a duplicate
        read.claimed = True
        if not reply.get("ok"):
            self._fail(read, reply.get("error", "refused"), notify=False)
            return
        if "layout" in reply:
            self._layouts[target] = reply["layout"]
            self._peers.attach(target, reply["layout"]["pool"])
        try:
            self._launch(read, self._layouts[target])
        except Exception as exc:
            # This request falls back to the scheduler's failed-receive path;
            # the engine and every other read carry on.
            if not isinstance(exc, (PeerUnreachable, HipError, ValueError)):
                logger.exception("UALoE: launching the copy failed")
            self._fail(read, str(exc), notify=True)

    def _launch(self, read: _Read, layout: dict[str, Any]) -> None:
        if layout["unit_bytes"] != self._unit_bytes or layout["roles"] != self._roles:
            raise ValueError(
                f"the prefill KV layout ({len(layout['unit_bytes'])} regions) "
                f"does not match decode's ({len(self._unit_bytes)} regions); "
                "both sides must run the same model, KV dtype and TP"
            )
        read.mapping = self._peers.acquire(read.target, self._device)
        src, sizes = block_addresses(
            layout["bases"], layout["unit_bytes"], read.src_blocks
        )
        local_src = read.mapping.translate(src, sizes)
        if local_src is None:
            raise ValueError("source blocks fall outside the prefill pool")
        dst, _ = block_addresses(self._bases, self._unit_bytes, read.dst_blocks)
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self._device)
        # After everything already launched here: a destination block may
        # have been freed by a request whose last forward is still running.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self._device))
        self._stream.wait_event(ready)
        read.tables = launch_copy(local_src, dst, sizes, self._stream)
        read.done = torch.cuda.Event()
        read.done.record(self._stream)

    def _poll_copies(self) -> None:
        for key, read in list(self._reads.items()):
            if read.done is None or not read.done.query():
                continue
            del self._reads[key]
            self._peers.release(read.mapping)
            self._send_end(read)
            self._done_recving.add(read.req_id)

    def _expire_unanswered(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        for read in list(self._reads.values()):
            if not read.claimed and now - read.sent_at > self._read_timeout_s:
                # The claim may still land: tell the producer to let go.
                self._fail(read, "no reply to READ_BEGIN", notify=True)

    def _fail(self, read: _Read, reason: str, *, notify: bool) -> None:
        logger.error(
            "UALoE: reading req %s (transfer %s) from %s failed: %s",
            read.req_id,
            read.transfer_id,
            read.target,
            reason,
        )
        self._reads.pop((read.target, read.transfer_id), None)
        if read.tables is not None:
            # A launched copy still reads the source and writes the
            # destination: both are released below, so let it finish first.
            self._stream.synchronize()
        if read.mapping is not None:
            self._peers.release(read.mapping)
        if notify:
            self._send_end(read)
        self._failed_recving.add(read.req_id)

    def _send_end(self, read: _Read) -> None:
        request = {"transfer_id": read.transfer_id}
        self._socket(read.target).send_multipart([MSG_READ_END, msgpack.dumps(request)])
