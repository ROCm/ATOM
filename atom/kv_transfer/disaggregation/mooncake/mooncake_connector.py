# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Worker-side and scheduler-side KV cache connectors for disaggregated P/D.

Uses Mooncake TransferEngine for TCP- or RDMA-based push (WRITE) transfers of
KV cache data from producer (prefill) to consumer (decode) nodes.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import msgpack
import msgspec
import numpy as np
import torch
import zmq
from aiter.dist.parallel_state import get_dp_group, get_tp_group

from atom.config import Config
from atom.distributed.dcp_utils import get_dcp_group
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.mooncake.mla_landing import (
    MLA_LANDING_CREDIT_WAIT_S,
    MLA_LANDING_MIN_SLOTS,
    MLA_LANDING_POOL_BYTES,
    MLA_LANDING_SLOT_BYTES,
    MSG_LANDING_CREDIT,
    MSG_LANDING_READY,
    LandingCredits,
    LandingReceiver,
    mla_landing_reserve_bytes,
)
from atom.kv_transfer.disaggregation.mooncake.rail_engine_pool import RailEnginePool
from atom.kv_transfer.disaggregation.pd_producer import (
    MLA_STAGING_SLOT_BYTES,
    mla_staging_reserve_bytes,
    mla_staging_slot_count,
    send_worker_count,
)
from atom.kv_transfer.disaggregation.port_offset import (
    consumer_region_indices,
)
from atom.kv_transfer.disaggregation.port_offset import (
    side_channel_port_offset as _port_offset,
)
from atom.kv_transfer.disaggregation.sharded_transfer import (
    build_dcp_shard_plan,
    coalesce_contiguous,
    pack_slots,
)
from atom.kv_transfer.disaggregation.types import (
    INDEX_CACHE_FP4_PREFIX,
    INDEX_CACHE_ROLE,
    MLA_KV_ROLE,
    ConnectorMetadata,
    KVConnectorOutput,
    KVTransferRegion,
    ReqId,
    TransferId,
)
from atom.model_engine.sequence import Sequence
from atom.models.utils import get_pp_indices
from atom.utils import (
    envs,
    get_open_port,
    make_zmq_path,
    make_zmq_socket,
    zmq_socket_ctx,
)
from atom.utils.network import get_ip

logger = logging.getLogger("atom")

# ---------------------------------------------------------------------------
# Mooncake availability check
# ---------------------------------------------------------------------------

_MOONCAKE_AVAILABLE = False
try:
    from mooncake.engine import TransferEngine

    _MOONCAKE_AVAILABLE = True
    logger.info("Mooncake TransferEngine loaded successfully")
except ImportError:
    logger.warning(
        "Mooncake is not available — KV cache disaggregation via mooncake "
        "will not work. Install the mooncake package to enable push-mode "
        "RDMA transfers."
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOONCAKE_DEFAULT_PROTOCOL = "rdma"
PREFILL_LOOKUP_TIMEOUT = 60
PREFILL_LOOKUP_POLL_INTERVAL = 0.01
_IB_SYSFS_ROOT = Path("/sys/class/infiniband")


_NOTIFY_BIND_ATTEMPTS = 16


def _bind_router_on_open_port(ctx: zmq.Context) -> tuple[zmq.Socket, int]:
    """Bind a ROUTER socket on a free TCP port now; return it with the port.

    Choosing a port with ``get_open_port()`` and binding it later leaves the
    port free in between. Under ``--network host`` any outgoing connection on
    the node can take it as its ephemeral port, and the later bind fails with
    EADDRINUSE. Binding right after choosing closes that window; the retry
    covers a port taken in the instant between the probe and the bind.
    """
    for _ in range(_NOTIFY_BIND_ATTEMPTS):
        port = get_open_port()
        try:
            sock = make_zmq_socket(
                ctx, make_zmq_path("tcp", "*", port), zmq.ROUTER, bind=True
            )
        except zmq.ZMQError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            continue
        return sock, port
    raise RuntimeError(
        f"could not bind a notification listener after {_NOTIFY_BIND_ATTEMPTS} "
        "attempts: every probed port was taken before it could be bound"
    )


@contextlib.contextmanager
def _owned_zmq_socket(ctx: zmq.Context, sock: zmq.Socket, linger: int = 0):
    """Yield an already-bound socket and destroy its context on exit, the way
    ``zmq_socket_ctx`` does for a socket it creates itself."""
    try:
        yield sock
    except KeyboardInterrupt:
        logger.debug("Got Keyboard Interrupt.")
    finally:
        ctx.destroy(linger=linger)


def _swa_ring_ids(seq) -> list[int]:
    """The ids the SWA region transfer is keyed by: this request's ring slot.

    A sliding window used to be its own content-addressed block pool, so the
    key was a list of block ids and window-freeing left only the live tail
    non-sentinel. It is a per-request ring now: one slot, whose whole
    `win_with_spec` rows are the transfer unit (see the backend's
    `swa_block_regions`, whose `unit_bytes` is a full ring).

    Returns `[]` when no slot is assigned, which is every request on a backend
    with no per-request state and is what makes the SWA transfer inert there.
    Whether an empty list is legitimate depends on whether the backend emitted
    any SWA regions at all, which only the connector knows -- see the guard at
    the transfer site.
    """
    slot = getattr(seq, "state_slot", -1)
    return [int(slot)] if slot is not None and slot >= 0 else []


def _ib_device_exists(device_name: str) -> bool:
    return (_IB_SYSFS_ROOT / device_name).exists()


def _auto_select_ib_device(phys_idx: int) -> str:
    # Older environments expose paired HCAs as rdmaN. Spur MI350 fabric exposes
    # them as ionic_N, so try ionic_N only when rdmaN is not present.
    rdma_device = f"rdma{phys_idx}"
    if _ib_device_exists(rdma_device):
        return rdma_device
    ionic_device = f"ionic_{phys_idx}"
    if _ib_device_exists(ionic_device):
        return ionic_device
    return rdma_device


def _parse_ib_devices(configured_devices: str) -> list[str]:
    """Normalize a comma-separated Mooncake RDMA device filter."""
    return list(
        dict.fromkeys(
            device.strip() for device in configured_devices.split(",") if device.strip()
        )
    )


def _select_ib_devices(
    protocol: str,
    configured_devices: str,
    phys_idx: int | None,
    *,
    enable_alternate_hca: bool = False,
    hca_count: int = 8,
) -> list[str]:
    """Resolve the HCAs on which Mooncake registers this rank's GPU memory."""
    if protocol.strip().lower() == "tcp":
        return []
    if configured_devices:
        return _parse_ib_devices(configured_devices)
    if phys_idx is None:
        raise ValueError("physical GPU index is required for RDMA device selection")

    primary = _auto_select_ib_device(phys_idx)
    devices = [primary]
    if enable_alternate_hca:
        if hca_count <= 0:
            raise ValueError("ib_hca_count must be a positive integer")
        for idx in range(hca_count):
            device = _auto_select_ib_device(idx)
            if device not in devices and _ib_device_exists(device):
                devices.append(device)
    return devices


def _discover_active_matched_rails(primary_device: str) -> list[str]:
    """Discover active HCAs in the primary's numbered device-name family."""
    family = re.fullmatch(r"(.*\D)(\d+)", primary_device)
    if family is None:
        raise ValueError(
            f"Cannot auto-discover matched rails for HCA {primary_device!r}; "
            "set ATOM_MOONCAKE_MATCHED_RAILS to an explicit HCA list"
        )
    pattern = re.compile(re.escape(family[1]) + r"(\d+)")
    try:
        devices = list(_IB_SYSFS_ROOT.iterdir())
    except OSError as exc:
        raise ValueError(
            f"Cannot discover RDMA HCAs in {_IB_SYSFS_ROOT}; "
            "set ATOM_MOONCAKE_MATCHED_RAILS to an explicit HCA list"
        ) from exc

    active = []
    for device in devices:
        match = pattern.fullmatch(device.name)
        if match is None:
            continue
        for state_file in device.glob("ports/*/state"):
            try:
                state = state_file.read_text().partition(":")[0].strip()
            except OSError:
                continue
            if state == "4":  # IB_PORT_ACTIVE, also used by RoCE HCAs.
                active.append((int(match[1]), device.name))
                break
    rails = [name for _, name in sorted(active)]
    if primary_device not in rails:
        raise ValueError(
            f"Primary HCA {primary_device!r} has no readable ACTIVE RDMA port; "
            "check the link and sysfs visibility before using matched rails"
        )
    logger.info(
        "Auto-discovered Mooncake matched rails: primary=%s rails=%s",
        primary_device,
        rails,
    )
    return rails


def _resolve_matched_rails(
    protocol: str, ib_devices: list[str], configured_rails: str
) -> list[str]:
    value = configured_rails.strip()
    if value.lower() == "auto":
        # Check transport and primary shape before reading RDMA sysfs.
        if protocol.strip().lower() != "rdma":
            raise ValueError("ATOM_MOONCAKE_MATCHED_RAILS requires protocol=rdma")
        if len(ib_devices) != 1:
            raise ValueError(
                "Matched rails require a single primary HCA per engine; "
                "disable ib_enable_alternate_hca and use at most one ib_device"
            )
        rails = _discover_active_matched_rails(ib_devices[0])
    else:
        rails = _parse_ib_devices(value)
    _validate_matched_rails(protocol, ib_devices, rails)
    return rails


def _validate_matched_rails(
    protocol: str, ib_devices: list[str], matched_rails: list[str]
) -> None:
    if not matched_rails:
        return
    if protocol.strip().lower() != "rdma":
        raise ValueError("ATOM_MOONCAKE_MATCHED_RAILS requires protocol=rdma")
    if len(ib_devices) != 1:
        raise ValueError(
            "Matched rails require a single primary HCA per engine; "
            "disable ib_enable_alternate_hca and use at most one ib_device"
        )
    if ib_devices[0] not in matched_rails or any(
        not _ib_device_exists(device) for device in matched_rails
    ):
        raise ValueError(
            "Matched rails must name existing local HCAs including the primary HCA"
        )


def _select_ib_device(
    protocol: str, configured_device: str, phys_idx: int | None
) -> str:
    """Resolve the Mooncake device filter without enabling RDMA for TCP.

    Mooncake's TCP transport requires an empty device list. Passing a usable
    HCA alongside ``protocol=tcp`` allows the transfer engine to activate RDMA
    as an alternate path, which violates the caller's explicit transport
    choice. RDMA-family transports retain the existing configured/automatic
    device selection.
    """
    return ",".join(_select_ib_devices(protocol, configured_device, phys_idx))


def _configure_mooncake_transport(protocol: str) -> None:
    """Make Mooncake honor ATOM's explicit transport selection.

    Legacy TransferEngine builds auto-discover installed HCAs independently
    of the device filter. ``MC_FORCE_TCP`` is Mooncake's supported override
    for preventing that implicit RDMA transport from being installed.
    """
    if protocol.strip().lower() == "tcp":
        os.environ["MC_FORCE_TCP"] = "true"


# ZMQ side-channel message types
MSG_WRITE_REQUEST = b"write_request"
MSG_WRITE_DONE = b"write_done"
MSG_GET_META = b"get_meta"
# PP-prefill only: consumer tells stage-0 a request's KV is fully received from
# every stage, so stage-0 may reuse the shared page table (see _record_release).
MSG_RELEASE = b"release"


# ---------------------------------------------------------------------------
# Metadata struct for bootstrap handshake
# ---------------------------------------------------------------------------


class MooncakeAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
    kw_only=True,
):
    """Serializable metadata exchanged during the mooncake bootstrap."""

    engine_id: str
    rpc_port: int
    kv_caches_base_addr: list[int] | None = None
    num_blocks: int = 0
    block_len: int = 0
    has_slot_regions: bool = False
    block_base_addrs: list[int] | None = None
    block_bpb: list[int] | None = None
    slot_base_addrs: list[int] | None = None
    slot_bps: list[int] | None = None
    num_slots: int = 0


# ===================================================================
# MooncakeConnectorScheduler — scheduler-side connector
# ===================================================================


class MooncakeConnectorScheduler(KVConnectorSchedulerBase):
    def __init__(self, config: Config) -> None:
        kv_transfer_config = config.kv_transfer_config
        self.is_producer = (
            kv_transfer_config.get("kv_role", "kv_producer") == "kv_producer"
        )
        self.handshake_port = get_open_port()
        self.base_handshake_port = kv_transfer_config.get("handshake_port", 6301)
        self.engine_id = "None"
        self.tp_size = config.tensor_parallel_size
        self.dp_size = config.parallel_config.data_parallel_size
        self.dp_rank = config.parallel_config.data_parallel_rank
        self.pp_size = config.pipeline_parallel_size
        self.block_size = config.kv_cache_block_size
        self.dcp_size = config.decode_context_parallel_size
        self.hash_block_size = self.block_size * self.dcp_size
        self.host_ip = get_ip()

        # Pending requests: req_id -> (Sequence, block_table)
        self._reqs_need_recv: dict[ReqId, tuple[Any, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Any, list[int]]] = {}

        # Source-block ownership: the scheduler frees a finished request's HBM
        # only once every connector stops claiming it. Claimed in
        # `request_finished` -- the call that hands the peer these addresses --
        # and NOT at alloc: `Scheduler._is_preemptable` negates this same
        # predicate, so an earlier claim would pin every running request.
        self._awaiting_send: set[str] = set()

        # Bidirectional transfer_id <-> request_id mapping
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}

    def _remote_page_geometry(self, params: dict[str, Any]) -> tuple[int, int] | None:
        """Return ``(producer_block_size, producer_dcp_size)`` when incremental
        prefix reuse is safe.

        Prefers the explicit ``block_size`` / ``dcp_size`` wire fields. Legacy
        producers only send ``hash_block_size = block_size * dcp_size``; that
        product matches this consumer's page size (unsharded producer) or its
        virtual hash block (symmetric DCP), and nothing else.
        """

        remote_block_size = params.get("block_size")
        remote_dcp_size = params.get("dcp_size")
        if remote_block_size is not None and remote_dcp_size is not None:
            remote_block_size = int(remote_block_size)
            remote_dcp_size = int(remote_dcp_size)
            if remote_block_size != self.block_size:
                return None
            if remote_dcp_size not in (1, self.dcp_size):
                return None
            return remote_block_size, remote_dcp_size

        remote_hash_size = params.get("hash_block_size")
        if remote_hash_size == self.block_size:
            return self.block_size, 1
        if remote_hash_size == self.hash_block_size:
            return self.block_size, self.dcp_size
        return None

    def get_num_new_matched_tokens(self, seq: Sequence) -> tuple[int, bool]:
        params = seq.kv_transfer_params or {}

        if params.get("do_remote_prefill") and not getattr(
            seq, "kv_async_tagged", False
        ):
            return len(seq.prompt_token_ids), True

        return 0, False

    def build_connector_meta(self) -> ConnectorMetadata:
        meta = ConnectorMetadata()
        meta.request_id_to_transfer_id = self.request_id_to_transfer_id

        for req_id, (req, block_ids, slot_idx) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            req.kv_transfer_params["local_slot_index"] = slot_idx
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                local_swa_block_ids=_swa_ring_ids(req),
            )

        # Producer side: pass completed prefill block_ids to worker
        for req_id, (req, block_ids, slot_idx) in self._reqs_need_save.items():
            assert req.kv_transfer_params is not None
            req.kv_transfer_params["local_slot_index"] = slot_idx
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                local_swa_block_ids=_swa_ring_ids(req),
            )

        if self._reqs_need_recv or self._reqs_need_save:
            logger.debug(
                "[SCHEDULER] build_connector_meta: %d recv, %d save, " "id_map=%s",
                len(self._reqs_need_recv),
                len(self._reqs_need_save),
                meta.request_id_to_transfer_id,
            )
        self._reqs_need_recv.clear()
        self._reqs_need_save.clear()
        return meta

    def update_state_after_alloc(self, seq: Sequence) -> None:
        params = seq.kv_transfer_params or {}

        if not self.is_producer:
            transfer_id = params.get("transfer_id")
            if transfer_id is not None:
                self.transfer_id_to_request_id[transfer_id] = seq.id
                self.request_id_to_transfer_id[seq.id] = transfer_id

        slot_index = getattr(seq, "state_slot", -1)

        # Consumer side: queue for remote KV loading
        if params.get("do_remote_prefill"):
            assert (
                not self.is_producer
            ), "Only the decode (consumer) side handles do_remote_prefill"
            self._reqs_need_recv[seq.id] = (seq, list(seq.block_table), slot_index)
            params["do_remote_prefill"] = False
            params["local_slot_index"] = slot_index
            # PD incremental: skip leading blocks already in the decode node's
            # prefix cache. Per-request state (including the SWA ring slot) is
            # not covered by a block-only delta, so it takes a full transfer.
            num_computed_blocks = 0
            # Number of producer source blocks represented by one consumer block.
            src_block_skip_factor = 1
            remote_geometry = self._remote_page_geometry(params)
            if remote_geometry is None:
                logger.warning(
                    "PD incremental transfer disabled for req %s: producer "
                    "block_size=%r dcp_size=%r hash_block_size=%r, consumer "
                    "block_size=%d dcp=%d; falling back to full transfer",
                    seq.id,
                    params.get("block_size"),
                    params.get("dcp_size"),
                    params.get("hash_block_size"),
                    self.block_size,
                    self.dcp_size,
                )
            elif not seq.has_per_req_cache and self.hash_block_size > 0:
                _, remote_dcp_size = remote_geometry
                num_computed_blocks = seq.num_cached_tokens // self.hash_block_size
                src_block_skip_factor = self.dcp_size // remote_dcp_size
            params["num_computed_blocks"] = num_computed_blocks
            params["src_block_skip_factor"] = src_block_skip_factor
            logger.debug(
                "[SCHEDULER-CONSUMER] Queued req %s for remote KV recv "
                "(%d blocks, %d locally cached, slot=%d), transfer_id=%s, "
                "remote_host=%s, remote_handshake_port=%s",
                seq.id,
                len(seq.block_table),
                num_computed_blocks,
                slot_index,
                params.get("transfer_id"),
                params.get("remote_host"),
                params.get("remote_handshake_port"),
            )

        # Producer side: queue block_ids for the write listener to look up
        if params.get("do_remote_decode"):
            assert self.is_producer, "Only the producer side handles do_remote_decode"
            self._reqs_need_save[seq.id] = (seq, list(seq.block_table), slot_index)
            logger.debug(
                "Queued req %s for KV save (%d blocks, slot=%d)",
                seq.id,
                len(seq.block_table),
                slot_index,
            )

    def request_finished(self, seq: Sequence) -> None:
        if self.is_producer and getattr(seq, "leave_reason", None) == "aborted":
            # No send claim protects an abort's blocks from reuse. Never
            # advertise their addresses or dispatch a queued producer save.
            self._reqs_need_save.pop(seq.id, None)
            seq.kv_transfer_params_output = None
            return

        # Claim iff we will send: the same `do_remote_decode` gate that fills
        # `_reqs_need_save`.
        if self.is_producer and (getattr(seq, "kv_transfer_params", None) or {}).get(
            "do_remote_decode"
        ):
            self._awaiting_send.add(str(seq.id))

        first_token_id = seq.output_tokens[0] if seq.output_tokens else None
        drafts = getattr(seq, "spec_token_ids", None)
        draft_token_ids = (
            [int(x) for x in drafts] if drafts is not None and len(drafts) else []
        )
        seq.kv_transfer_params_output = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": list(seq.block_table),
            # The consumer's SWA ring slot; the producer keys the SWA region
            # transfer by it. Empty for backends with no SWA state.
            "remote_swa_block_ids": _swa_ring_ids(seq),
            "remote_engine_id": self.engine_id,
            "remote_host": self.host_ip,
            "remote_port": self.handshake_port,
            "remote_handshake_port": self.base_handshake_port,
            "tp_size": self.tp_size,
            "dp_rank": self.dp_rank,
            "remote_pp_size": self.pp_size,
            "hash_block_size": self.hash_block_size,
            "block_size": self.block_size,
            "dcp_size": self.dcp_size,
            "transfer_id": seq.id,
            "first_token_id": first_token_id,
            "draft_token_ids": draft_token_ids,
            "local_slot_index": getattr(seq, "state_slot", -1),
            "prefix_cache_hit_tokens": getattr(seq, "prefix_cache_hit_tokens", 0),
        }

        if not self.is_producer:
            transfer_id = self.request_id_to_transfer_id.pop(seq.id, None)
            if transfer_id is not None:
                self.transfer_id_to_request_id.pop(transfer_id, None)

    def should_defer_free(self, seq: Sequence) -> bool:
        return str(seq.id) in self._awaiting_send

    def send_finished(self, req_id) -> None:
        self._awaiting_send.discard(str(req_id))

    def source_blocks_released(self, seq: Sequence) -> None:
        """No block-lifetime state remains after the send claim is retired."""


# ===================================================================
# MooncakeConnector — worker-side connector (runs inside each TP rank)
# ===================================================================


class MooncakeConnector(KVConnectorBase):
    """Worker-side KV cache connector using Mooncake push-mode RDMA.

    Mooncake uses a push/WRITE model: the prefill (producer) node writes
    KV cache data directly into the decode (consumer) node's registered
    GPU memory via ``batch_transfer_sync_write``.
    """

    # Class-level default so every construction path has a rail pool to read.
    # ``__init__`` overwrites it, but the transfer path also runs on instances
    # built without it (tests use ``object.__new__``) and on instances whose
    # ``__init__`` aborted before the matched-rail block; both must fall back to
    # the single shared ``transfer_engine`` instead of raising AttributeError.
    _rail_pool: RailEnginePool | None = None

    def __init__(self, config: Config) -> None:
        self.tp_rank = get_tp_group().rank_in_group
        self.dp_rank = get_dp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        self.dp_size = get_dp_group().world_size
        self.pp_rank = config.parallel_config.pipeline_parallel_rank
        self.pp_size = config.pipeline_parallel_size
        self.num_hidden_layers = config.hf_config.num_hidden_layers
        self.block_size = config.kv_cache_block_size
        # The consumer ships its DCP topology in the write_request; the
        # producer relayouts on the way out and keeps dcp_size == 1 of its own.
        self.dcp_size = config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0
        self.dcp_interleave_size = config.dcp_config.interleave_size
        # Global index of this stage's first layer; consumer regions are ordered
        # over all layers, so a producer stage writes at this layer offset.
        self._start_layer = 0
        self._num_local_layers = 0

        kv_transfer_config = config.kv_transfer_config
        # Mooncake's P2P handshake and ZMQ notifications need a routable TCP
        # address. Honor ATOM_HOST_IP via get_ip(); an HCA's RoCE address may
        # only support RDMA traffic and must not replace the control address.
        self.local_ip = get_ip()
        self._local_ping_port = get_open_port()

        self.is_producer = (
            kv_transfer_config.get("kv_role", "kv_producer") == "kv_producer"
        )
        self.is_consumer = not self.is_producer

        # Networking config
        self.http_port = kv_transfer_config.get("http_port", 8000)
        self.request_address = f"{self.local_ip}:{self.http_port}"
        self.protocol = kv_transfer_config.get("protocol", MOONCAKE_DEFAULT_PROTOCOL)

        # Side channel port (ZMQ) — deterministic from config for proxy relay
        self.base_handshake_port = kv_transfer_config.get("handshake_port", 6301)
        self._side_channel_port = self.base_handshake_port + _port_offset(
            self.dp_rank,
            self.tp_rank,
            self.tp_size,
            self.pp_rank,
            self.pp_size,
            self.dp_size,
        )

        # --- Mooncake TransferEngine initialization ---
        if not _MOONCAKE_AVAILABLE:
            raise RuntimeError(
                "Mooncake is not installed but kv_connector='mooncake' was requested. "
                "Install the mooncake package to use push-mode transfers."
            )

        # Determine which RDMA device this TP rank should use. TCP is
        # intentionally initialized with an empty device filter so Mooncake
        # cannot activate an available HCA as an alternate path.
        # AMD GPU nodes pair GPU N with NIC N, but the HCA name is cluster
        # dependent: Spur MI350 exposes ionic_N while older setups used rdmaN.
        # By default, register only with the local NIC. Alternate-HCA mode
        # registers a single engine on multiple NICs. On rail-isolated fabrics,
        # matched-rail mode instead selects a single-HCA engine per request.
        _configure_mooncake_transport(self.protocol)
        configured_ib_device = kv_transfer_config.get(
            "ib_device", ""
        ) or os.environ.get("ATOM_MOONCAKE_IB_DEVICE", "")
        enable_alternate_hca = bool(
            kv_transfer_config.get("ib_enable_alternate_hca", False)
        )
        hca_count = int(kv_transfer_config.get("ib_hca_count", 8))
        phys_idx: int | None = None
        if self.protocol.strip().lower() != "tcp" and not configured_ib_device:
            visible_idx = torch.cuda.current_device()
            # ROCR first, for the reason in numa_utils._physical_index: it
            # filters before HIP, so a ROCR-only mask must not fall through to
            # the identity mapping.
            visible_env = (
                os.environ.get("ROCR_VISIBLE_DEVICES")
                or os.environ.get("HIP_VISIBLE_DEVICES")
                or os.environ.get("CUDA_VISIBLE_DEVICES")
            )
            if visible_env:
                visible_list = [d for d in visible_env.split(",") if d != ""]
                phys_idx = int(visible_list[visible_idx])
            else:
                phys_idx = visible_idx
        ib_devices = _select_ib_devices(
            self.protocol,
            configured_ib_device,
            phys_idx,
            enable_alternate_hca=enable_alternate_hca,
            hca_count=hca_count,
        )
        ib_device = ",".join(ib_devices)
        self.ib_devices = ib_devices
        matched_rails = _resolve_matched_rails(
            self.protocol, ib_devices, envs.ATOM_MOONCAKE_MATCHED_RAILS
        )
        # Advertise only an unambiguous, single-HCA destination. An engine
        # registered on multiple NICs can still choose an unreachable rail.
        self.ib_device = ib_devices[0] if len(ib_devices) == 1 else None
        self._rail_pool = None
        if self.protocol.strip().lower() == "tcp":
            logger.info("Mooncake TCP selected; RDMA device selection is disabled")
        elif not configured_ib_device:
            logger.info(
                "Auto-selecting RDMA devices %s for physical GPU %d "
                "(visible_idx=%d, tp_rank=%d)",
                ib_device,
                phys_idx,
                visible_idx,
                self.tp_rank,
            )

        # The device filter independently selects the RDMA data path; the
        # server name below is also advertised in consumer/notification metadata.
        self.transfer_engine = TransferEngine()
        ret = self.transfer_engine.initialize(
            self.local_ip,
            "P2PHANDSHAKE",
            self.protocol,
            ib_device,
        )
        if ret != 0:
            raise RuntimeError(
                f"Mooncake TransferEngine.initialize() failed (ret={ret}) "
                f"on ip={self.local_ip}, protocol={self.protocol}, "
                f"ib_device={ib_device}"
            )
        self.rpc_port = self.transfer_engine.get_rpc_port()
        self.engine_id = f"{self.local_ip}:{self.rpc_port}"
        logger.info(
            "Mooncake TransferEngine initialized: ip=%s, protocol=%s, "
            "ib_device=%s, rpc_port=%d",
            self.local_ip,
            self.protocol,
            ib_device,
            self.rpc_port,
        )

        if matched_rails and self.is_producer:
            # _resolve_matched_rails requires exactly one primary HCA.
            primary_ib_device = ib_devices[0]
            self._rail_pool = RailEnginePool(
                TransferEngine,
                self.transfer_engine,
                primary_ib_device,
                matched_rails,
                # Every engine uses the reachable control address. Each
                # engine's device filter independently selects its RDMA rail.
                local_ip_for_device=lambda _device: self.local_ip,
            )
            logger.info(
                "Mooncake matched rails enabled: primary=%s rails=%s",
                primary_ib_device,
                matched_rails,
            )

        # --- KV cache state (populated in register_kv_caches) ---
        self.kv_caches: dict[str, Any] | None = None
        self.kv_caches_base_addr: list[int] = []
        self._per_block_bytes_list: list[int] = []
        self._block_region_roles: list[str | None] = []
        self._fp4_index_layout: bool = False
        self.kv_cache_shape: tuple[int, ...] | None = None
        self.block_len: int = config.kv_cache_block_size
        self.num_blocks: int = 0
        self._per_block_bytes: int = 0

        # --- region-based transfer state (populated in register_kv_caches) ---
        self._has_slot_regions: bool = False
        # (base_addr, bytes_per_block) per region
        self._block_regions: list[tuple[int, int]] = []
        self._block_region_consumer_indices: list[int] | None = None
        # Sliding-window regions, keyed by the request's state slot (not by the
        # compressed block_table above). Kept whole rather than as
        # `(base, unit)` because a window region may be reverse-indexed, and
        # `KVTransferRegion.unit_addr` is the only place that knows.
        self._swa_block_regions: list[KVTransferRegion] = []
        # (base_addr, bytes_per_slot) per region
        self._slot_regions: list[tuple[int, int]] = []
        self._gather_slot = None
        self._scatter_slot = None
        self._staging_base_addr: int = 0
        self._staging_slot_bytes: int = 0
        self._staging_pool_size: int = 0
        self._staging_free: list[int] = []
        self._staging_lock = threading.Lock()
        self._index_staging_pool_size: int = 0
        self._index_staging_chunk_pages: int = 0
        self._index_staging_free: list[int] = []
        self._index_staging_lock = threading.Lock()
        self._prepare_sharded_index = None
        self._gather_sharded_index = None
        # MLA staging for DCP consumers (see _execute_staged_mla_regions).
        self._mla_staging: torch.Tensor | None = None
        self._mla_staging_free: list[int] = []
        self._mla_staging_cv = threading.Condition()
        # Each send worker gathers on its own stream (see _send_worker_stream).
        self._send_worker_streams = threading.local()
        # region_idx -> uint8 [num_blocks * block_size, token_bytes] alias
        self._mla_token_views: dict[int, torch.Tensor] = {}
        # MLA landing (see mla_landing.py): the consumer's pool and scatter
        # thread, and the producer's per-decode-rank slot credits.
        self._mla_landing: LandingReceiver | None = None
        self._landing_credits = LandingCredits() if self.is_producer else None

        # --- Producer: completed prefill block_ids cache ---
        # Populated from ConnectorMetadata.reqs_to_save each step.
        # The write listener looks up block_ids here when consumer requests a write.
        self._completed_prefills: dict[ReqId, dict] = {}
        self._kv_cache_ready_events: dict[ReqId, torch.cuda.Event] = {}
        self._completed_prefills_lock = threading.Lock()
        self._completed_prefills_cv = threading.Condition(self._completed_prefills_lock)
        self._transfer_refcount: dict[ReqId, int] = {}
        self._transfer_refcount_lock = threading.Lock()

        # --- Consumer: pending receive tracking ---
        self._pending_recv: set[ReqId] = set()
        self._pending_recv_blocks: dict[ReqId, list[int]] = {}
        self._pending_recv_slots: dict[ReqId, tuple[int, int]] = {}
        # Write-done notifications still expected per request. Under PP-prefill a
        # request is served by one producer stage per port, so the consumer must
        # collect ``remote_pp_size`` notifications before the receive is complete.
        self._pending_recv_expected: dict[ReqId, int] = {}
        # Distinct producer ranks whose write-done has arrived, per request.
        # Write-done is deduped by (pp_rank, tp_rank) — the producer may send a
        # notification more than once for reliability, so counting messages would
        # finalize early; we count distinct producer ranks instead.
        self._pending_recv_stages: dict[ReqId, set[tuple[int, int]]] = {}
        # Requests one of whose producer ranks reported a failure. The failure
        # is published only with the last rank's write-done, as success is:
        # producers are never cancelled, so a rank still running (or not yet
        # dispatched) may write the request's pages after the scheduler reuses
        # them.
        self._pending_recv_failed: set[ReqId] = set()
        # Per-request nonce for write-done corruption detection.
        self._pending_recv_nonce: dict[ReqId, int] = {}
        # PP-prefill: consumer stashes stage-0's release address per request, and
        # the producer (stage-0) counts releases to defer freeing the shared page
        # table until every stage has written the KV out.
        self._release_targets: dict[ReqId, tuple[str, int, int]] = {}
        self._release_count: dict[TransferId, int] = {}
        self._released_transfers: set[TransferId] = set()
        # The consumer's write-done listener is bound here, not when its thread
        # starts after model load and KV registration: a port that sits unbound
        # for minutes gets taken by other traffic on the host network (see
        # `_bind_router_on_open_port`). Producers never listen on it.
        self._notification_ctx: zmq.Context | None = None
        self._notification_sock: zmq.Socket | None = None
        if self.is_producer:
            self._notification_port = get_open_port()
        else:
            self._notification_ctx = zmq.Context()
            self._notification_sock, self._notification_port = (
                _bind_router_on_open_port(self._notification_ctx)
            )

        # --- Completion tracking ---
        self.done_sending: set[str] = set()
        self.done_recving: set[str] = set()
        self.failed_recving: set[str] = set()
        self._completion_lock = threading.Lock()

        # --- GPU memory fence: blocks pending coherence enforcement ---
        self._blocks_pending_fence: list[int] = []
        self._fence_lock = threading.Lock()

        # --- Transfer ID mapping (worker side) ---
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}

        # --- Producer: thread pool for RDMA writes ---
        self._cuda_device = torch.cuda.current_device()
        self._num_send_workers = (
            send_worker_count(kv_transfer_config) if self.is_producer else 0
        )
        if self.is_producer:
            self._send_executor = ThreadPoolExecutor(
                max_workers=self._num_send_workers,
                thread_name_prefix="mooncake-send-worker",
                initializer=torch.cuda.set_device,
                initargs=(self._cuda_device,),
            )

        # --- ZMQ for metadata exchange ---
        self.zmq_context = zmq.Context()

        # --- Producer: persistent socket cache for write-done notifications ---
        self._notify_sockets: dict[str, zmq.Socket] = {}
        self._notify_sockets_lock = threading.Lock()

        # --- Msgspec encoder/decoder for bootstrap metadata ---
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)

    # -----------------------------------------------------------------
    # KVConnectorBase: register_kv_caches
    # -----------------------------------------------------------------
    _MAX_RDMA_CHUNK_BYTES = 2 * 1024 * 1024 * 1024 - 64 * 1024
    _MAX_RDMA_ENTRIES_PER_BATCH = 4096
    # Class-level so a connector built without __init__ keeps the per-token
    # paths; register_kv_caches sets the instance values.
    _mla_staging: torch.Tensor | None = None
    _mla_landing: LandingReceiver | None = None

    def _rdma_chunk_units(self, unit_bytes: int) -> int:
        """Units per MR chunk of a region registered by ``_rdma_chunk_sizes``.

        Every internal chunk boundary of such a region, on either side of the
        transfer, falls at a multiple of this many units from its base.
        """
        return max(1, self._MAX_RDMA_CHUNK_BYTES // unit_bytes)

    def _rdma_chunk_sizes(self, total_bytes: int, unit_bytes: int) -> list[int]:
        """Split a region into MR chunks, each <= _MAX_RDMA_CHUNK_BYTES and, apart
        from the final remainder, aligned down to a whole multiple of unit_bytes.

        Aligning every internal boundary to a unit (block/slot) means no unit ever
        crosses an MR boundary, so its single RDMA op never spans two registered
        MRs. Only a unit larger than the max chunk can still straddle, which no KV
        block/slot reaches; that degenerate case is logged and left unaligned.
        """
        mc = self._MAX_RDMA_CHUNK_BYTES
        sizes: list[int] = []
        offset = 0
        while offset < total_bytes:
            remaining = total_bytes - offset
            chunk = min(mc, remaining)
            if chunk < remaining and unit_bytes > 0:
                aligned = chunk - (chunk % unit_bytes)
                if aligned > 0:
                    chunk = aligned
                else:
                    logger.warning(
                        "RDMA unit_bytes=%d exceeds max chunk %d; a block will "
                        "straddle an MR boundary and its transfer may fail",
                        unit_bytes,
                        mc,
                    )
            sizes.append(chunk)
            offset += chunk
        return sizes

    @classmethod
    def kv_budget_reserve_bytes(cls, config) -> int:
        """The MLA staging pool (producer) and landing pool (DCP consumer).

        ``register_kv_caches`` allocates both after the KV cache is sized.
        """
        return mla_staging_reserve_bytes(config) + mla_landing_reserve_bytes(config)

    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        """Register KV cache tensors with the Mooncake TransferEngine."""
        self.kv_caches = kv_caches

        if transfer_tensors is None:
            logger.warning(
                "register_kv_caches called without transfer_tensors; "
                "RDMA transfers will not be available."
            )
            return

        from atom.kv_transfer.disaggregation.types import KVTransferTensors

        tt: KVTransferTensors = transfer_tensors

        self._has_slot_regions = (
            len(tt.slot_regions) > 0 or tt.staging_region is not None
        )
        self.num_blocks = tt.num_blocks
        self._gather_slot = tt.gather_slot
        self._scatter_slot = tt.scatter_slot

        if tt.staging_region is not None:
            self._staging_base_addr = tt.staging_region.base_addr
            self._staging_slot_bytes = tt.staging_region.unit_bytes
            self._staging_pool_size = tt.staging_pool_size
            self._staging_free = list(range(tt.staging_pool_size))
        if tt.index_staging_region is not None:
            self._index_staging_pool_size = tt.index_staging_pool_size
            self._index_staging_chunk_pages = tt.index_staging_chunk_pages
            self._index_staging_free = list(range(tt.index_staging_pool_size))
            self._prepare_sharded_index = tt.prepare_sharded_index
            self._gather_sharded_index = tt.gather_sharded_index

        # Populate block/slot region lists for transfer offset computation
        self._block_regions = [(r.base_addr, r.unit_bytes) for r in tt.block_regions]
        self._block_region_consumer_indices = getattr(
            tt, "block_region_consumer_indices", None
        )
        if self._block_region_consumer_indices is not None and len(
            self._block_region_consumer_indices
        ) != len(self._block_regions):
            raise ValueError(
                "block_region_consumer_indices must match block_regions: "
                f"{len(self._block_region_consumer_indices)} != "
                f"{len(self._block_regions)}"
            )
        # Window regions, transferred one whole entry per state slot.
        self._swa_block_regions = list(tt.swa_block_regions)
        self._slot_regions = [(r.base_addr, r.unit_bytes) for r in tt.slot_regions]

        self.kv_caches_base_addr = [r.base_addr for r in tt.block_regions]
        self._per_block_bytes_list = [r.unit_bytes for r in tt.block_regions]
        self._block_region_roles = [r.semantic_role for r in tt.block_regions]
        self._fp4_index_layout = any(
            role is not None and role.startswith(INDEX_CACHE_FP4_PREFIX)
            for role in self._block_region_roles
        )
        if (
            not self.is_producer
            and self.dcp_size > 1
            and self.dcp_interleave_size != 1
            and INDEX_CACHE_ROLE in self._block_region_roles
        ):
            raise RuntimeError(
                "Sharded preshuffled DSA index P/D requires "
                "dcp interleave_size=1, got "
                f"{self.dcp_interleave_size}"
            )

        # Under pipeline parallelism this stage holds only layers
        # [start_layer, end_layer); its local regions map onto the consumer's
        # full-layer region list starting at start_layer (see
        # _consumer_region_map).
        self._num_local_layers = len(kv_caches)
        if self.pp_size > 1:
            self._start_layer = get_pp_indices(
                self.num_hidden_layers, self.pp_rank, self.pp_size
            )[0]
            if self.is_producer and self._has_slot_regions:
                # Per-request slot/state regions are only routed for stage 0 in
                # this path: the consumer sends one dst_slot/staging address and
                # downstream stages run with src_slot=-1 (slot phase skipped).
                # Fine when all slot regions live on stage 0 (e.g. MLA has none);
                # a real per-layer slot backend (V4/DSA sparse) under PP would
                # drop downstream slot state — not yet supported here.
                logger.warning(
                    "PP-prefill with per-request slot regions: only stage-0 slot "
                    "state is transferred (pp_rank=%d, %d slot regions). Verify "
                    "the backend keeps slot state off downstream stages.",
                    self.pp_rank,
                    len(self._slot_regions),
                )
        else:
            self._start_layer = 0

        # Chunk all regions for RDMA memory registration
        reg_ptrs: list[int] = []
        reg_sizes: list[int] = []

        all_regions = (
            list(tt.block_regions) + list(tt.swa_block_regions) + list(tt.slot_regions)
        )
        if tt.staging_region is not None:
            all_regions.append(tt.staging_region)
        if tt.index_staging_region is not None:
            all_regions.append(tt.index_staging_region)
        mla_staging_region = self._build_mla_staging(tt)
        if mla_staging_region is not None:
            all_regions.append(mla_staging_region)
        self._mla_landing = self._build_mla_landing()
        if self._mla_landing is not None:
            all_regions.append(self._mla_landing.region())
        for r in all_regions:
            offset = 0
            for chunk in self._rdma_chunk_sizes(r.total_bytes, r.unit_bytes):
                reg_ptrs.append(r.base_addr + offset)
                reg_sizes.append(chunk)
                offset += chunk

        logger.info(
            "Registering %d RDMA chunks (%d block regions, %d slot regions, "
            "max_chunk=%.2f GiB, ib_devices=%s)",
            len(reg_ptrs),
            len(tt.block_regions),
            len(tt.slot_regions),
            self._MAX_RDMA_CHUNK_BYTES / (1024**3),
            ",".join(self.ib_devices) or "<none>",
        )

        ret = self.transfer_engine.batch_register_memory(reg_ptrs, reg_sizes)
        if ret != 0:
            logger.error(
                "batch_register_memory FAILED (ret=%d), "
                "trying individual registration as fallback...",
                ret,
            )
            # A later chunk can fail after earlier ones registered. Leaving those
            # behind strands Mooncake MRs for memory the caller is about to tear
            # down, so a retry or a second connector in this process would leak
            # registration resources. Roll back before propagating.
            registered: list[int] = []
            for ptr, sz_bytes in zip(reg_ptrs, reg_sizes):
                r = self.transfer_engine.register_memory(ptr, sz_bytes)
                if r != 0:
                    for done_ptr in reversed(registered):
                        try:
                            self.transfer_engine.unregister_memory(done_ptr)
                        except Exception:
                            logger.exception(
                                "Rollback of Mooncake registration failed for "
                                "ptr=%#x; continuing to unwind",
                                done_ptr,
                            )
                    raise RuntimeError(
                        f"Mooncake register_memory failed: "
                        f"ptr={ptr:#x} size={sz_bytes} ret={r}"
                    )
                registered.append(ptr)
        else:
            logger.info("batch_register_memory OK (%d chunks)", len(reg_ptrs))

        if self._rail_pool is not None:
            self._rail_pool.set_regions(reg_ptrs, reg_sizes)
        if self._mla_landing is not None:
            # Registered and warmed up before any producer can learn its
            # address, and before CUDA graph capture.
            self._mla_landing.start()

        # Build metadata for bootstrap exchange
        if self._has_slot_regions:
            self._local_metadata = MooncakeAgentMetadata(
                engine_id=self.engine_id,
                rpc_port=self.rpc_port,
                num_blocks=tt.num_blocks,
                block_len=self.block_len,
                has_slot_regions=True,
                block_base_addrs=[b for b, _ in self._block_regions],
                block_bpb=[bpb for _, bpb in self._block_regions],
                slot_base_addrs=[b for b, _ in self._slot_regions],
                slot_bps=[bps for _, bps in self._slot_regions],
                num_slots=tt.num_slots,
            )
        else:
            self._local_metadata = MooncakeAgentMetadata(
                engine_id=self.engine_id,
                rpc_port=self.rpc_port,
                kv_caches_base_addr=self.kv_caches_base_addr,
                num_blocks=tt.num_blocks,
                block_len=self.block_len,
            )

        logger.info(
            "Mooncake KV registration complete: role=%s, engine_id=%s, "
            "has_slot_regions=%s, num_blocks=%d, block_regions=%d, slot_regions=%d",
            "PRODUCER" if self.is_producer else "CONSUMER",
            self.engine_id,
            self._has_slot_regions,
            tt.num_blocks,
            len(self._block_regions),
            len(self._slot_regions),
        )

        # Start side channel threads
        if self.is_producer:
            self._write_listener_thread = threading.Thread(
                target=self._write_listener,
                daemon=True,
                name="mooncake-write-listener",
            )
            self._write_listener_thread.start()
        else:
            self._notification_listener_thread = threading.Thread(
                target=self._notification_listener,
                daemon=True,
                name="mooncake-notify-listener",
            )
            self._notification_listener_thread.start()

    def _build_mla_landing(self) -> LandingReceiver | None:
        """The consumer's MLA landing pool; None keeps staged per-page pulls.

        Only a DCP consumer with MLA regions and no per-request slot regions
        lands. ``mla_landing_reserve_bytes`` holds the pool back from the KV
        cache budget before the regions are known, so it reserves the pool
        for a DCP consumer of an MLA model with slot regions too.
        """
        mla_regions = [
            idx
            for idx, role in enumerate(self._block_region_roles)
            if role == MLA_KV_ROLE
        ]
        if (
            self.is_producer
            or self.dcp_size <= 1
            or not envs.ATOM_PD_MLA_LANDING
            or self._has_slot_regions
            or not mla_regions
        ):
            return None
        slots = MLA_LANDING_POOL_BYTES // MLA_LANDING_SLOT_BYTES
        receiver = LandingReceiver(
            device=self._cuda_device,
            pool_slots=slots,
            slot_bytes=MLA_LANDING_SLOT_BYTES,
            block_size=self.block_size,
            consumer_key=f"{self.local_ip}:{self.rpc_port}",
            region_bases=self.kv_caches_base_addr,
            region_block_bytes=self._per_block_bytes_list,
            mla_regions=mla_regions,
            send=self._send_on_socket,
            finish=self._complete_recv,
        )
        logger.info(
            "PD MLA landing: %d slots x %.1f MiB for %d MLA regions",
            slots,
            MLA_LANDING_SLOT_BYTES / (1 << 20),
            len(mla_regions),
        )
        return receiver

    def _build_mla_staging(self, tt) -> KVTransferRegion | None:
        """Allocate the producer's MLA staging pool; None keeps per-token sends.

        A DCP consumer rank owns every ``dcp_size``-th token of a source block,
        so its MLA bytes cannot be sent as whole blocks. The pool lets a send
        worker gather those tokens into destination page order on the GPU and
        send them page-contiguous. One slot per send worker, capped by
        ``MLA_STAGING_POOL_BYTES``; ``mla_staging_reserve_bytes`` holds
        the pool back from the KV cache budget.
        """
        if (
            not self.is_producer
            or self.dcp_size > 1
            or self._has_slot_regions
            or not envs.ATOM_PD_MLA_STAGING
        ):
            return None
        views = tt.block_tensor_views
        if len(views) != len(tt.block_regions):
            return None
        token_views: dict[int, torch.Tensor] = {}
        for region_idx, (region, view) in enumerate(zip(tt.block_regions, views)):
            if (
                region.semantic_role != MLA_KV_ROLE
                or region.unit_bytes % self.block_size
                or not view.is_cuda
            ):
                continue
            token_views[region_idx] = view.view(
                -1, region.unit_bytes // self.block_size
            )
        if not token_views:
            return None
        widest = max(tt.block_regions[idx].unit_bytes for idx in token_views)
        slot_bytes = max(widest, MLA_STAGING_SLOT_BYTES // widest * widest)
        pool_size = mla_staging_slot_count(self._num_send_workers, slot_bytes)
        device = next(iter(token_views.values())).device
        self._mla_staging = torch.empty(
            (pool_size, slot_bytes), dtype=torch.uint8, device=device
        )
        self._mla_staging_free = list(range(pool_size))
        self._mla_token_views = token_views
        logger.info(
            "PD MLA staging: %d slots x %.1f MiB for %d MLA regions, "
            "%d send workers",
            pool_size,
            slot_bytes / (1 << 20),
            len(token_views),
            self._num_send_workers,
        )
        return KVTransferRegion(
            base_addr=self._mla_staging.data_ptr(),
            total_bytes=pool_size * slot_bytes,
            unit_bytes=slot_bytes,
            semantic_role="mla.kv_staging",
        )

    # -----------------------------------------------------------------
    # KVConnectorBase: start_load_kv
    # -----------------------------------------------------------------

    def record_kv_cache_ready(self, req_ids: list[ReqId]) -> None:
        """Record when this prefill batch's KV writes become visible to staging."""
        if (
            not self.is_producer
            or (self._index_staging_pool_size == 0 and self._mla_staging is None)
            or not req_ids
        ):
            return
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(self._cuda_device))
        with self._completed_prefills_lock:
            for req_id in req_ids:
                self._kv_cache_ready_events[req_id] = ready_event

    def _get_kv_cache_ready_event(self, req_id: ReqId) -> torch.cuda.Event | None:
        with self._completed_prefills_lock:
            return self._kv_cache_ready_events.get(req_id)

    def _discard_kv_cache_ready_event(self, req_id: ReqId) -> None:
        with self._completed_prefills_lock:
            self._kv_cache_ready_events.pop(req_id, None)

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        """Initiate KV transfers for pending requests.

        **Producer side**: Cache completed prefill block_ids from
        ``metadata.reqs_to_save`` so the write listener can look them up.

        **Consumer side**: For each pending recv request, connect to the
        producer's ZMQ side channel and send a write request with our
        memory addresses and block allocation.
        """
        if metadata is None:
            return

        self.request_id_to_transfer_id = metadata.request_id_to_transfer_id

        # Producer: cache block_ids + slot_index from completed prefills
        if self.is_producer:
            for req_id, meta in metadata.reqs_to_save.items():
                with self._completed_prefills_cv:
                    self._completed_prefills[req_id] = {
                        "block_ids": meta.local_block_ids,
                        "swa_block_ids": meta.local_swa_block_ids,
                        "slot_index": meta.local_slot_index,
                    }
                    self._completed_prefills_cv.notify_all()
                logger.debug(
                    "[PRODUCER] Cached %d prefill blocks (slot=%d) for req %s",
                    len(meta.local_block_ids),
                    meta.local_slot_index,
                    req_id,
                )
            return

        # Consumer: send write requests to producer
        if not metadata.reqs_to_recv:
            return

        logger.debug(
            "[CONSUMER] start_load_kv: %d reqs_to_recv, id_map=%s",
            len(metadata.reqs_to_recv),
            metadata.request_id_to_transfer_id,
        )

        for req_id, meta in metadata.reqs_to_recv.items():
            remote_tp_size = meta.tp_size
            if remote_tp_size != self.tp_size:
                remote_tp_rank = self.tp_rank % remote_tp_size
            else:
                remote_tp_rank = self.tp_rank

            # Under PP-prefill the producer is one stage process per pipeline
            # rank, each owning a contiguous slice of layers on its own port.
            # The consumer sends the same write_request to every stage; each
            # stage writes only its layer window (see _consumer_region_map).
            remote_pp_size = max(1, meta.remote_pp_size)
            expected_responses = remote_pp_size
            write_nonce = int.from_bytes(os.urandom(8), "big")
            with self._completion_lock:
                self._pending_recv_expected[req_id] = expected_responses
                self._pending_recv_nonce[req_id] = write_nonce

            # PD incremental: slice off locally cached prefix blocks; invalid
            # offset falls back to full transfer. Under DCP the same prefix
            # costs dcp_size times as many source blocks.
            remote_block_ids = meta.remote_block_ids or []
            off = meta.num_computed_blocks
            src_block_skip_factor = max(1, meta.src_block_skip_factor)
            if (
                off < 0
                or off >= len(meta.local_block_ids)
                or off * src_block_skip_factor >= len(remote_block_ids)
            ):
                off = 0
            dst_block_ids = meta.local_block_ids[off:]
            src_block_ids = remote_block_ids[off * src_block_skip_factor :]

            # Build the (stage-independent) write_request payload once.
            request_body = {
                "request_id": req_id,
                "transfer_id": meta.transfer_id,
                "consumer_host": self.local_ip,
                "consumer_rpc_port": self.rpc_port,
                "consumer_ib_device": self.ib_device,
                "consumer_dp_rank": self.dp_rank,
                # Consumer's layer count per group for producer stride validation.
                "consumer_num_layers": self._num_local_layers,
                # Role of each of this side's block regions, so the producer can
                # check its per-region plan lands on the same kind of region.
                "consumer_region_roles": self._block_region_roles,
                "consumer_block_bpb": [bpb for _, bpb in self._block_regions],
                "dst_block_ids": dst_block_ids,
                # Source block_ids so downstream stages (no scheduler, no
                # _completed_prefills) can transfer without a local lookup.
                "src_block_ids": src_block_ids,
                # Producer slices its local prefill block_ids by this offset in
                # the TP-TP path (where it uses its own cache, not src above).
                "num_computed_blocks": off,
                "src_block_skip_factor": src_block_skip_factor,
                "notify_host": self.local_ip,
                "notify_port": self._notification_port,
                "consumer_tp_size": self.tp_size,
                "write_nonce": write_nonce,
                # DCP relayout: which shard of each block this rank owns.
                "consumer_dcp_size": self.dcp_size,
                "consumer_dcp_rank": self.dcp_rank,
                "consumer_dcp_interleave": self.dcp_interleave_size,
            }

            consumer_staging_pool_idx = -1
            if self._has_slot_regions:
                # Acquire one staging pool slot for this request's state RDMA.
                consumer_staging_addr = 0
                if self._staging_pool_size > 0:
                    consumer_staging_pool_idx = self._acquire_staging_slot()
                    consumer_staging_addr = (
                        self._staging_base_addr
                        + consumer_staging_pool_idx * self._staging_slot_bytes
                    )
                request_body.update(
                    {
                        "has_slot_regions": True,
                        "dst_slot_index": meta.local_slot_index,
                        "consumer_block_base_addrs": [
                            b for b, _ in self._block_regions
                        ],
                        "consumer_block_bpb": [bpb for _, bpb in self._block_regions],
                        "consumer_region_roles": self._block_region_roles,
                        # SWA ring, keyed by state slot. The whole region
                        # travels, not just its base: a reverse-indexed one
                        # needs its extent to place slot 0.
                        "dst_swa_block_ids": meta.local_swa_block_ids,
                        "consumer_swa_block_regions": [
                            asdict(r) for r in self._swa_block_regions
                        ],
                        "consumer_slot_base_addrs": [b for b, _ in self._slot_regions],
                        "consumer_slot_bps": [bps for _, bps in self._slot_regions],
                        "consumer_staging_addr": consumer_staging_addr,
                        "consumer_staging_bytes": self._staging_slot_bytes,
                    }
                )
                if self._fp4_index_layout:
                    # Version fence. A producer predating the two-region FP4
                    # layout looks up "consumer_block_base_addrs"
                    # unconditionally, so publishing under an FP4-only key
                    # makes it raise KeyError, which _execute_transfer turns
                    # into a transfer failure. The mixed-version pair fails
                    # closed instead of writing FP8-shaped bytes into FP4
                    # regions. Current producers accept either key.
                    request_body["consumer_block_base_addrs_fp4"] = request_body.pop(
                        "consumer_block_base_addrs"
                    )
            else:
                request_body["consumer_base_addrs"] = self.kv_caches_base_addr

            write_request = msgpack.dumps(request_body)

            stage_addrs: dict[int, str] = {}
            for stage in range(remote_pp_size):
                remote_port = meta.remote_handshake_port + _port_offset(
                    meta.remote_dp_rank,
                    remote_tp_rank,
                    remote_tp_size,
                    stage,
                    remote_pp_size,
                    meta.remote_dp_size,
                )
                stage_addrs[stage] = make_zmq_path("tcp", meta.remote_host, remote_port)
            # MLA landing: each stage gets its own write request carrying the
            # landing partition it may write (see mla_landing.py).
            stage_requests: dict[int, bytes] = {}
            landing = self._mla_landing
            if landing is not None:
                for stage, addr in stage_addrs.items():
                    advertised = landing.advertise(addr, remote_pp_size)
                    if advertised is not None:
                        stage_requests[stage] = msgpack.dumps(
                            {**request_body, "mla_landing": advertised}
                        )

            # Registered before the first send: a producer failure can be
            # notified while this loop is still running, and the handler needs
            # the slot and block records to already be there or it cannot
            # reclaim them.
            if stage_requests:
                landing.begin(req_id, write_nonce, dst_block_ids, stage_addrs)
            self._pending_recv.add(req_id)
            # Only delta blocks need fencing; reused prefix blocks are coherent.
            self._pending_recv_blocks[req_id] = list(dst_block_ids)
            if meta.local_slot_index >= 0:
                self._pending_recv_slots[req_id] = (
                    meta.local_slot_index,
                    consumer_staging_pool_idx,
                )

            for stage, remote_addr in stage_addrs.items():
                if stage == 0 and remote_pp_size > 1:
                    # stage-0 owns the block manager; it must not reuse the shared
                    # page table until all stages finished writing (see
                    # _record_release / _record_write_done).
                    with self._completion_lock:
                        self._release_targets[req_id] = (
                            remote_addr,
                            meta.transfer_id,
                            self.tp_size,
                        )
                self._send_on_socket(
                    remote_addr,
                    [MSG_WRITE_REQUEST, stage_requests.get(stage, write_request)],
                )
                logger.debug(
                    "[CONSUMER] write_request sent for req %s (transfer_id=%s) "
                    "to stage %d/%d at %s, off=%d, dst_block_ids=%s",
                    req_id,
                    meta.transfer_id,
                    stage,
                    remote_pp_size,
                    remote_addr,
                    off,
                    dst_block_ids[:10],
                )

    # -----------------------------------------------------------------
    # Staging pool management
    # -----------------------------------------------------------------

    def _acquire_staging_slot(self) -> int:
        with self._staging_lock:
            if self._staging_free:
                return self._staging_free.pop()
        logger.warning(
            "Staging pool exhausted (size=%d), blocking until a slot is freed. "
            "Increase ATOM_PD_STAGING_POOL if this happens frequently.",
            self._staging_pool_size,
        )
        while True:
            time.sleep(0.001)
            with self._staging_lock:
                if self._staging_free:
                    return self._staging_free.pop()

    def _release_staging_slot(self, idx: int) -> None:
        with self._staging_lock:
            self._staging_free.append(idx)

    def _acquire_index_staging_slot(self) -> int:
        with self._index_staging_lock:
            if self._index_staging_free:
                return self._index_staging_free.pop()
        logger.warning(
            "Index staging pool exhausted (size=%d), waiting for a slot",
            self._index_staging_pool_size,
        )
        while True:
            time.sleep(0.001)
            with self._index_staging_lock:
                if self._index_staging_free:
                    return self._index_staging_free.pop()

    def _release_index_staging_slot(self, idx: int) -> None:
        with self._index_staging_lock:
            self._index_staging_free.append(idx)

    def _acquire_mla_staging_slot(self) -> int:
        """A free MLA staging slot, waiting for one if the pool is drained.

        A worker holds at most one slot and releases it once its RDMA write
        returns, so the wait always ends even with fewer slots than workers.
        """
        with self._mla_staging_cv:
            self._mla_staging_cv.wait_for(lambda: self._mla_staging_free)
            return self._mla_staging_free.pop()

    def _release_mla_staging_slot(self, idx: int) -> None:
        with self._mla_staging_cv:
            self._mla_staging_free.append(idx)
            self._mla_staging_cv.notify()

    def _send_worker_stream(self) -> torch.cuda.Stream:
        """The calling send worker's staging stream, created on first use.

        A private stream per worker lets ``synchronize`` wait for this
        worker's gathers and ready event only, not every other worker's.
        """
        streams = self._send_worker_streams
        stream = getattr(streams, "stream", None)
        if stream is None:
            stream = streams.stream = torch.cuda.Stream(device=self._cuda_device)
        return stream

    # -----------------------------------------------------------------
    # KVConnectorBase: get_finished
    # -----------------------------------------------------------------

    def get_finished(self) -> KVConnectorOutput:
        """Return send/recv completion status and clear internal sets."""
        if self._mla_landing is not None:
            self._mla_landing.sweep()
        with self._completion_lock:
            ds = self.done_sending.copy()
            dr = self.done_recving.copy()
            failed = self.failed_recving.copy()
            self.done_sending.clear()
            self.done_recving.clear()
            self.failed_recving.clear()
        if ds or dr or failed:
            logger.debug(
                "[%s] get_finished: sending=%s, recving=%s, failed_recving=%s",
                "PRODUCER" if self.is_producer else "CONSUMER",
                ds,
                dr,
                failed,
            )
        return KVConnectorOutput(
            finished_sending=ds,
            finished_recving=dr,
            failed_recving=failed,
        )

    def get_finished_recv_blocks(self) -> list[int]:
        """Return block IDs from recently completed RDMA receives."""
        with self._fence_lock:
            blocks = self._blocks_pending_fence
            self._blocks_pending_fence = []
        return blocks

    # -----------------------------------------------------------------
    # Producer: write listener (ZMQ ROUTER)
    # -----------------------------------------------------------------

    def _write_listener(self) -> None:
        """Accept write requests from consumers and dispatch RDMA writes."""
        path = make_zmq_path("tcp", "*", self._side_channel_port)
        logger.info("Mooncake write listener bound to %s", path)

        with zmq_socket_ctx(path, zmq.ROUTER, bind=True) as sock:
            while True:
                parts = sock.recv_multipart()
                identity, msg_type = parts[0], parts[1]

                if msg_type == MSG_GET_META:
                    encoded = self._encoder.encode(self._local_metadata)
                    sock.send_multipart([identity, b"", encoded])
                    logger.debug("Sent metadata to peer")

                elif msg_type == MSG_WRITE_REQUEST:
                    request_data = msgpack.loads(parts[2])
                    logger.debug(
                        "[PRODUCER] Received write_request for req %s "
                        "(transfer_id=%s, consumer=%s:%s)",
                        request_data["request_id"],
                        request_data.get("transfer_id"),
                        request_data.get("consumer_host"),
                        request_data.get("consumer_rpc_port"),
                    )
                    landing = request_data.get("mla_landing")
                    if landing is not None:
                        # Adopt the partition before any worker needs it.
                        try:
                            self._landing_credits.sync(
                                f"{request_data['consumer_host']}:"
                                f"{request_data['consumer_rpc_port']}",
                                landing,
                            )
                        except Exception:
                            # Not adopted: the transfer finds no credit and
                            # stages its MLA rows, or fails on the same field.
                            logger.exception(
                                "[PRODUCER] adopting a landing partition failed"
                            )
                    self._send_executor.submit(self._execute_transfer, request_data)

                elif msg_type == MSG_LANDING_CREDIT:
                    try:
                        data = msgpack.loads(parts[2])
                        self._landing_credits.release(
                            data["consumer"], data["epoch"], data["slots"]
                        )
                    except Exception:
                        # Write requests arrive on this thread too.
                        logger.exception("[PRODUCER] handling a landing credit failed")

                elif msg_type == MSG_RELEASE:
                    data = msgpack.loads(parts[2])
                    self._record_release(
                        data["transfer_id"], data.get("consumer_tp_size", 1)
                    )

                else:
                    logger.error("Unknown message type: %s", msg_type)

    def _record_release(self, transfer_id: TransferId, consumer_tp_size: int) -> None:
        """Count a consumer-rank release; free the shared page after all ranks.

        PP-prefill only. Each decode rank sends one release once it has received
        the KV from every stage, so ``consumer_tp_size`` releases mean all
        stage×rank writes for this request are done and stage-0 may reuse the
        page table. Marks the request in ``done_sending`` for the scheduler.
        """
        with self._completion_lock:
            if transfer_id in self._released_transfers:
                # Already released once; a duplicate/late release must not
                # re-add to done_sending (the block was already freed → the
                # scheduler would assert on a missing deferred block).
                return
            count = self._release_count.get(transfer_id, 0) + 1
            if count < consumer_tp_size:
                self._release_count[transfer_id] = count
                return
            self._release_count.pop(transfer_id, None)
            self._released_transfers.add(transfer_id)
            self.done_sending.add(transfer_id)
        with self._completed_prefills_lock:
            self._completed_prefills.pop(transfer_id, None)
            self._kv_cache_ready_events.pop(transfer_id, None)
        logger.debug(
            "[PRODUCER] All %d decode ranks released transfer_id=%s; page freed",
            consumer_tp_size,
            transfer_id,
        )

    # -----------------------------------------------------------------
    # Producer: execute RDMA write
    # -----------------------------------------------------------------

    def _execute_transfer(self, request_data: dict) -> None:
        """Compute offsets and perform RDMA write for a single request."""
        try:
            req_id = request_data["request_id"]
            transfer_id = request_data.get("transfer_id", req_id)
            consumer_host = request_data["consumer_host"]
            consumer_rpc_port = request_data["consumer_rpc_port"]
            dst_block_ids = request_data["dst_block_ids"]
            consumer_tp_size = request_data.get("consumer_tp_size", self.tp_size)
            consumers_per_rank = max(1, consumer_tp_size // self.tp_size)
            consumer_dcp_size = max(1, request_data.get("consumer_dcp_size", 1))
            has_slot_data = request_data.get("has_slot_regions", False)

            logger.debug(
                "[PRODUCER] _execute_transfer: req_id=%s, transfer_id=%s, "
                "consumer=%s:%s, dst_blocks=%d, has_slot_data=%s",
                req_id,
                transfer_id,
                consumer_host,
                consumer_rpc_port,
                len(dst_block_ids),
                has_slot_data,
            )

            request_src_block_ids = request_data.get("src_block_ids")
            if self.pp_size == 1:
                # TP-TP: authoritative block_ids come from the local prefill
                # cache (populated by the scheduler). Wait for it.
                prefill_data = self._wait_for_prefill_data(transfer_id)
                if prefill_data is None:
                    logger.error(
                        "[PRODUCER] Timed out waiting for prefill data for "
                        "transfer_id=%s (req_id=%s). Available keys: %s",
                        transfer_id,
                        req_id,
                        list(self._completed_prefills.keys()),
                    )
                    self._notify_transfer_result(request_data, success=False)
                    return
            else:
                # PP: the consumer supplies src_block_ids for EVERY stage (all
                # stages share the head's page table). Never block on the local
                # cache — under the PP engine loop even stage-0's may be empty,
                # so waiting would burn the full PREFILL_LOOKUP_TIMEOUT per
                # request. Peek non-blocking for slot_index only (slot path).
                if request_src_block_ids is None:
                    logger.error(
                        "[PRODUCER] PP stage %d got no src_block_ids for "
                        "transfer_id=%s (req_id=%s); cannot transfer.",
                        self.pp_rank,
                        transfer_id,
                        req_id,
                    )
                    self._notify_transfer_result(request_data, success=False)
                    return
                with self._completed_prefills_lock:
                    cached = self._completed_prefills.get(transfer_id)
                prefill_data = {
                    "block_ids": request_src_block_ids,
                    "slot_index": (
                        cached["slot_index"]
                        if cached
                        else request_data.get("src_slot_index", -1)
                    ),
                }

            src_block_ids = prefill_data["block_ids"]
            kv_cache_ready_event = self._get_kv_cache_ready_event(transfer_id)
            src_block_skip_factor = max(
                1, request_data.get("src_block_skip_factor", consumer_dcp_size)
            )
            # PD incremental (TP-TP only): consumer already sliced dst; slice
            # producer's src by the same offset. PP src arrives pre-sliced.
            if self.pp_size == 1:
                # Destination offset counts consumer blocks; unsharded producers
                # store dcp_size source blocks per destination block.
                off = request_data.get("num_computed_blocks", 0) * src_block_skip_factor
                if 0 < off < len(src_block_ids):
                    src_block_ids = src_block_ids[off:]
            expected_dst_blocks = -(-len(src_block_ids) // src_block_skip_factor)
            if len(dst_block_ids) != expected_dst_blocks:
                logger.error(
                    "[PRODUCER] src/dst block count mismatch for req %s "
                    "(src=%d, dst=%d, expected dst=%d at skip=%d dcp_size=%d); "
                    "aborting transfer to avoid misaligned KV.",
                    req_id,
                    len(src_block_ids),
                    len(dst_block_ids),
                    expected_dst_blocks,
                    src_block_skip_factor,
                    consumer_dcp_size,
                )
                self._notify_transfer_result(request_data, success=False)
                return
            if has_slot_data and consumer_dcp_size > 1:
                raise RuntimeError(
                    "P/D slot-region transfer does not support a DCP consumer "
                    f"(consumer_dcp_size={consumer_dcp_size})"
                )
            target = f"{consumer_host}:{consumer_rpc_port}"

            # Keep this engine local to the request: concurrent requests can
            # target different decode rails, including during retries/staging.
            engine = self.transfer_engine
            if self._rail_pool is not None:
                engine = self._rail_pool.get(request_data.get("consumer_ib_device"))
                logger.debug(
                    "Mooncake matched rail: req=%s p_dp=%d d_dp=%s device=%s",
                    req_id,
                    self.dp_rank,
                    request_data.get("consumer_dp_rank"),
                    request_data.get("consumer_ib_device"),
                )

            if hasattr(engine, "get_first_buffer_address"):
                remote_buf = engine.get_first_buffer_address(target)
                if remote_buf == 0:
                    logger.error(
                        "[PRODUCER] Consumer %s has NO registered buffers.",
                        target,
                    )

            if has_slot_data:
                transfer_ok = self._execute_block_slot_transfer(
                    request_data,
                    target,
                    src_block_ids,
                    dst_block_ids,
                    prefill_data,
                    req_id,
                    engine=engine,
                )
            else:
                transfer_ok = self._execute_block_transfer(
                    request_data,
                    target,
                    src_block_ids,
                    dst_block_ids,
                    req_id,
                    kv_cache_ready_event,
                    engine=engine,
                )

            if not transfer_ok:
                logger.error(
                    "[PRODUCER] transfer failed for req %s (transfer_id=%s); "
                    "not sending write-done",
                    req_id,
                    transfer_id,
                )
                self._notify_transfer_result(request_data, success=False)
                return

            # Notify consumer — all data (blocks + slot state) is written.
            self._notify_transfer_result(request_data, success=True)

            # Track refcount for multi-consumer TP fan-out.
            all_done = False
            with self._transfer_refcount_lock:
                if transfer_id not in self._transfer_refcount:
                    self._transfer_refcount[transfer_id] = consumers_per_rank
                self._transfer_refcount[transfer_id] -= 1
                if self._transfer_refcount[transfer_id] <= 0:
                    self._transfer_refcount.pop(transfer_id)
                    all_done = True

            if all_done:
                if self.pp_size > 1:
                    if self.pp_rank != 0:
                        self._discard_kv_cache_ready_event(transfer_id)
                    # PP-prefill: this stage's write is done, but stage-0 must not
                    # reuse the shared page until ALL stages finish. Freeing is
                    # deferred to _record_release (driven by consumer releases).
                    logger.debug(
                        "[PRODUCER] stage pp_rank=%d served %d consumers for "
                        "transfer_id=%s; awaiting release",
                        self.pp_rank,
                        consumers_per_rank,
                        transfer_id,
                    )
                else:
                    with self._completion_lock:
                        self.done_sending.add(transfer_id)
                    with self._completed_prefills_lock:
                        self._completed_prefills.pop(transfer_id, None)
                        self._kv_cache_ready_events.pop(transfer_id, None)
                    logger.debug(
                        "[PRODUCER] All %d consumers served for transfer_id=%s",
                        consumers_per_rank,
                        transfer_id,
                    )
        except Exception:
            logger.exception(
                "[PRODUCER] transfer FAILED for req %s (transfer_id=%s); "
                "notifying consumer so the request does not hang.",
                request_data.get("request_id"),
                request_data.get("transfer_id"),
            )
            try:
                self._notify_transfer_result(request_data, success=False)
            except Exception:
                logger.exception(
                    "[PRODUCER] failed to notify consumer of transfer failure"
                )

    def _consumer_region_map(
        self,
        num_local_regions: int,
        num_consumer_regions: int,
        consumer_num_layers: int | None = None,
        explicit_indices: list[int] | None = None,
    ) -> list[int]:
        """Map this stage's local RDMA regions onto the consumer's region list.

        Group-major layout: region ``i`` maps to
        ``(i // L) * stride + start_layer + (i % L)``.
        Identity for pp_size == 1. Raises on layout mismatch.
        """
        if explicit_indices is not None:
            if len(explicit_indices) != num_local_regions:
                raise ValueError(
                    "Explicit consumer region map length does not match local "
                    f"regions: {len(explicit_indices)} != {num_local_regions}"
                )
            return explicit_indices
        if (
            consumer_num_layers is not None
            and self.pp_size > 1
            and num_local_regions
            and self._num_local_layers
            and num_local_regions % self._num_local_layers == 0
        ):
            groups = num_local_regions // self._num_local_layers
            if consumer_num_layers * groups != num_consumer_regions:
                raise RuntimeError(
                    f"Region group mismatch: this stage has {groups} group(s) of "
                    f"{self._num_local_layers} layers, but the consumer reports "
                    f"{num_consumer_regions} regions over {consumer_num_layers} "
                    "layers per group. Producer and consumer must register the "
                    "same region groups."
                )
        cmap = consumer_region_indices(
            num_local_regions,
            self._num_local_layers,
            self._start_layer,
            num_consumer_regions,
            self.pp_size,
        )
        if cmap is None:
            raise RuntimeError(
                f"Cannot layer-map transfer: {num_local_regions} local regions / "
                f"{self._num_local_layers} local layers (start_layer="
                f"{self._start_layer}) do not map onto {num_consumer_regions} "
                "consumer regions as uniform group-major. Producer and consumer "
                "must register the same region groups — check that both sides "
                "agree on speculative decode (a draft KV layer widens every group)."
            )
        return cmap

    def _execute_block_transfer(
        self,
        request_data: dict,
        target: str,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        req_id: str,
        kv_cache_ready_event: torch.cuda.Event | None = None,
        *,
        engine=None,
    ) -> bool:
        """Block-only RDMA transfer (MHA, MLA, and other block-indexed backends)."""
        consumer_base_addrs = request_data["consumer_base_addrs"]

        src_addrs: list[int] = []
        dst_addrs: list[int] = []
        sizes: list[int] = []
        block_descriptor_count = 0
        debug_block_transfer = logger.isEnabledFor(logging.DEBUG)
        total_block_bytes = 0

        def flush_block_descriptors() -> bool:
            if not src_addrs:
                return True
            if not self._rdma_write_with_retry(
                target, src_addrs, dst_addrs, sizes, req_id, "block", engine=engine
            ):
                return False
            src_addrs.clear()
            dst_addrs.clear()
            sizes.clear()
            return True

        num_regions = len(self.kv_caches_base_addr)
        cmap = self._consumer_region_map(
            num_regions,
            len(consumer_base_addrs),
            request_data.get("consumer_num_layers"),
            self._block_region_consumer_indices,
        )
        # Under DCP the consumer rank owns only part of each block, so
        # whole-block descriptors no longer line up and the push becomes the
        # per-region relayout described at the top of this file. Safe in token
        # units because DCP only runs on MLA, which stores a token contiguously.
        dcp_size = max(1, request_data.get("consumer_dcp_size", 1))
        interleave = request_data.get("consumer_dcp_interleave", 1)
        sharded_plan = None
        sharded_runs = None
        stages_sharded_index = False
        stages_sharded_mla = False
        if self.dcp_size > 1:
            if dcp_size != self.dcp_size:
                raise RuntimeError(
                    "Asymmetric DCP P/D is unsupported: producer "
                    f"dcp={self.dcp_size}, consumer dcp={dcp_size}"
                )
        elif dcp_size > 1:
            sharded_plan = build_dcp_shard_plan(
                src_block_ids,
                block_size=self.block_size,
                dcp_size=dcp_size,
                dcp_rank=request_data["consumer_dcp_rank"],
                interleave_size=interleave,
                dst_pages=len(dst_block_ids),
            )
            sharded_runs = sharded_plan.token_runs(dst_block_ids)
            # Without the ready event the gather could read KV still being
            # written, so such a request keeps the per-token path.
            stages_sharded_mla = (
                self._mla_staging is not None and kv_cache_ready_event is not None
            )
            stages_sharded_index = (
                interleave < self.block_size
                and INDEX_CACHE_ROLE in self._block_region_roles
            )
            if stages_sharded_index and (
                interleave != 1
                or self.block_size % 16
                or self._prepare_sharded_index is None
                or self._gather_sharded_index is None
                or self._index_staging_chunk_pages <= 0
            ):
                raise RuntimeError(
                    "Sharded preshuffled DSA index transfer requires "
                    "interleave=1, a block size divisible by 16, and producer "
                    "index staging; got "
                    f"{interleave=}, block_size={self.block_size}, "
                    f"has_staging={self._gather_sharded_index is not None and self._prepare_sharded_index is not None}."
                )

        staged_regions: list[tuple[int, int, int]] = []
        staged_mla_regions: list[tuple[int, int, int]] = []
        # The plan comes from this stage's region order but the bytes land at
        # cmap[region_idx], and equal region counts do not make the two orders
        # match. Validate both semantic role and physical width so incompatible
        # producer/consumer layouts fail instead of silently corrupting KV.
        consumer_roles = request_data.get("consumer_region_roles")
        consumer_bpb = request_data.get("consumer_block_bpb")
        for region_idx in range(num_regions):
            src_base = self.kv_caches_base_addr[region_idx]
            dst_base = consumer_base_addrs[cmap[region_idx]]
            bpb = self._per_block_bytes_list[region_idx]
            role = self._block_region_roles[region_idx]
            if consumer_roles is not None and consumer_roles[cmap[region_idx]] != role:
                raise RuntimeError(
                    f"Region role mismatch for req {req_id}: local region "
                    f"{region_idx} is {role!r}, but consumer region "
                    f"{cmap[region_idx]} is {consumer_roles[cmap[region_idx]]!r}"
                )
            if consumer_bpb is not None and consumer_bpb[cmap[region_idx]] != bpb:
                raise RuntimeError(
                    f"Region byte-size mismatch for req {req_id}: producer "
                    f"region {region_idx} has {bpb}, consumer region "
                    f"{cmap[region_idx]} has {consumer_bpb[cmap[region_idx]]}"
                )
            if stages_sharded_index and role == INDEX_CACHE_ROLE:
                staged_regions.append((region_idx, dst_base, bpb))
                continue
            if sharded_runs is None:
                for sb, db in zip(src_block_ids, dst_block_ids):
                    src_addrs.append(src_base + sb * bpb)
                    dst_addrs.append(dst_base + db * bpb)
                    sizes.append(bpb)
                    block_descriptor_count += 1
                    if debug_block_transfer:
                        total_block_bytes += bpb
                    if (
                        len(src_addrs) == self._MAX_RDMA_ENTRIES_PER_BATCH
                        and not flush_block_descriptors()
                    ):
                        logger.error(
                            "[PRODUCER] block transfer failed for req %s", req_id
                        )
                        return False
                continue
            if role != MLA_KV_ROLE:
                raise RuntimeError(
                    f"DCP token relayout refuses region {region_idx} with "
                    f"semantic_role={role!r}; declare {MLA_KV_ROLE} for "
                    "token-contiguous MLA pages or "
                    f"{INDEX_CACHE_ROLE} for staged index pages"
                )
            # The destination page is wider only in whole tokens and the plan
            # already counts in its token space, so both ends scale by the
            # source's per-token width.
            if bpb % self.block_size:
                raise RuntimeError(
                    f"Region {region_idx} stores {bpb} bytes per block, which "
                    f"block_size {self.block_size} does not divide. Addressing "
                    "a single token needs its bytes contiguous, which holds for "
                    "the MLA layout the token-unit relayout is built on."
                )
            if stages_sharded_mla and region_idx in self._mla_token_views:
                staged_mla_regions.append((region_idx, dst_base, bpb))
                continue
            unit = bpb // self.block_size
            run_src, run_dst, run_len = sharded_runs
            run_start = 0
            while run_start < len(run_src):
                remaining = self._MAX_RDMA_ENTRIES_PER_BATCH - len(src_addrs)
                run_stop = min(run_start + remaining, len(run_src))
                run_slice = slice(run_start, run_stop)
                src_addrs.extend((src_base + run_src[run_slice] * unit).tolist())
                dst_addrs.extend((dst_base + run_dst[run_slice] * unit).tolist())
                sizes.extend((run_len[run_slice] * unit).tolist())
                added = run_stop - run_start
                block_descriptor_count += added
                if debug_block_transfer:
                    total_block_bytes += int(run_len[run_slice].sum()) * unit
                run_start = run_stop
                if (
                    len(src_addrs) == self._MAX_RDMA_ENTRIES_PER_BATCH
                    and not flush_block_descriptors()
                ):
                    logger.error("[PRODUCER] block transfer failed for req %s", req_id)
                    return False

        if not flush_block_descriptors():
            logger.error("[PRODUCER] block transfer failed for req %s", req_id)
            return False
        if debug_block_transfer and block_descriptor_count:
            logger.debug(
                "[PRODUCER] block RDMA write: req=%s, regions=%d, "
                "source_blocks=%d, descriptors=%d, total_bytes=%d",
                req_id,
                num_regions - len(staged_regions) - len(staged_mla_regions),
                len(src_block_ids),
                block_descriptor_count,
                total_block_bytes,
            )
        if staged_mla_regions and not self._execute_mla_regions(
            request_data,
            target,
            sharded_plan,
            dst_block_ids,
            staged_mla_regions,
            cmap,
            req_id,
            kv_cache_ready_event,
            engine=engine,
        ):
            return False
        if staged_regions:
            if kv_cache_ready_event is None:
                raise RuntimeError(
                    "Missing prefill KV-cache ready event for staged DSA index transfer"
                )
            # Wait only for this request's prefill writes; unrelated GPU work
            # can continue on other streams while the staging stream is blocked.
            self._send_worker_stream().wait_event(kv_cache_ready_event)
            for dst_start in range(
                0, len(dst_block_ids), self._index_staging_chunk_pages
            ):
                dst_chunk = dst_block_ids[
                    dst_start : dst_start + self._index_staging_chunk_pages
                ]
                chunk_plan = sharded_plan.slice_pages(
                    dst_start, dst_start + len(dst_chunk)
                )
                gather_indices = self._prepare_sharded_index(chunk_plan)

                for region_idx, dst_base, bpb in staged_regions:
                    if not self._execute_staged_index_layer_chunk(
                        target,
                        region_idx,
                        dst_base,
                        bpb,
                        dst_chunk,
                        req_id,
                        gather_indices,
                        engine=engine,
                    ):
                        return False
        return True

    def _execute_staged_mla_regions(
        self,
        target: str,
        plan,
        dst_block_ids: list[int],
        regions: list[tuple[int, int, int]],
        req_id: str,
        kv_cache_ready_event: torch.cuda.Event,
        *,
        engine=None,
        first_pages: list[int] | None = None,
    ) -> bool:
        """Gather a DCP rank's MLA tokens into destination pages, then RDMA them.

        ``regions`` holds ``(region_idx, dst_base, bytes_per_block)``. Each
        staging slot is filled page-major with as many (region, page range)
        items as fit, so the NIC sees one descriptor per run of adjacent
        destination pages instead of one per token. The bytes written match
        the per-token path exactly. ``first_pages`` skips each region's pages
        before that index (already landed, see _execute_mla_regions).
        """
        block_size = self.block_size
        stream, src_token_index = self._upload_mla_gather_index(
            plan.source_token_per_dst_token(), kv_cache_ready_event
        )
        slots = pack_slots(
            [bpb for _, _, bpb in regions],
            len(dst_block_ids),
            self._mla_staging.shape[1],
            1,
            first_pages,
        )
        for items in slots:
            pool_idx = self._acquire_mla_staging_slot()
            try:
                staging = self._mla_staging[pool_idx]
                self._gather_mla_rows(
                    stream,
                    staging,
                    [
                        (
                            regions[pos][0],
                            src_token_index[start * block_size : stop * block_size],
                            off,
                        )
                        for pos, start, stop, off in items
                    ],
                )
                # Coalesced runs stop at the destination's MR chunk boundaries,
                # which the per-token descriptors never cross either.
                runs = []
                for region_pos, page_start, page_stop, offset in items:
                    _, dst_base, bpb = regions[region_pos]
                    runs.append(
                        plan.slice_pages(page_start, page_stop).staged_page_runs(
                            dst_block_ids[page_start:page_stop],
                            staging.data_ptr() + offset,
                            dst_base,
                            bpb // block_size,
                            self._rdma_chunk_units(bpb),
                        )
                    )
                src_addrs, dst_addrs, sizes = (
                    np.concatenate(parts) for parts in zip(*runs)
                )
                if not self._rdma_write_with_retry(
                    target,
                    src_addrs.tolist(),
                    dst_addrs.tolist(),
                    sizes.tolist(),
                    req_id,
                    "staged-mla",
                    engine=engine,
                ):
                    logger.error(
                        "[PRODUCER] staged MLA transfer failed for req %s", req_id
                    )
                    return False
            finally:
                self._release_mla_staging_slot(pool_idx)
        return True

    def _execute_mla_regions(
        self,
        request_data: dict,
        target: str,
        plan,
        dst_block_ids: list[int],
        regions: list[tuple[int, int, int]],
        cmap: list[int],
        req_id: str,
        kv_cache_ready_event: torch.cuda.Event,
        *,
        engine=None,
    ) -> bool:
        """Land a DCP rank's MLA rows in the consumer's slots; stage the rest.

        With a landing offer (``mla_landing``) the rows go out in rank order
        without page padding (``landing_source_tokens``), packed across
        regions into slots of the partition the consumer advertised. Each
        slot is one RDMA descriptor, followed by ``MSG_LANDING_READY`` on the
        write-done socket, so the consumer sees every READY before this
        stage's write-done unless a reconnect reorders them. Rows that do not
        land go out through the staged per-page path: all of them without an
        offer or below ``MLA_LANDING_MIN_SLOTS`` slots, the rest once no slot
        frees up within ``MLA_LANDING_CREDIT_WAIT_S``. Slots are split only at
        page boundaries, so the rest is whole pages. See ``mla_landing.py``.
        """
        first_pages = None
        landing = request_data.get("mla_landing")
        if landing is not None:
            epoch = int(landing["epoch"])
            landing_slot_bytes = int(landing["slot_bytes"])
            rows = plan.landing_source_tokens()
            slots = pack_slots(
                [bpb // self.block_size for _, _, bpb in regions],
                rows.size,
                min(self._mla_staging.shape[1], landing_slot_bytes),
                self.block_size,
            )
            if len(slots) >= MLA_LANDING_MIN_SLOTS:
                stream, src_rows = self._upload_mla_gather_index(
                    rows, kv_cache_ready_event
                )
                notify_path = make_zmq_path(
                    "tcp", request_data["notify_host"], request_data["notify_port"]
                )
                credits = self._landing_credits
                for seq, items in enumerate(slots):
                    slot = credits.acquire(target, epoch, MLA_LANDING_CREDIT_WAIT_S)
                    if slot is None:
                        break
                    pool_idx = self._acquire_mla_staging_slot()
                    written = False
                    try:
                        staging = self._mla_staging[pool_idx]
                        used = self._gather_mla_rows(
                            stream,
                            staging,
                            [
                                (regions[pos][0], src_rows[start:stop], off)
                                for pos, start, stop, off in items
                            ],
                        )
                        written = self._rdma_write_with_retry(
                            target,
                            [staging.data_ptr()],
                            [int(landing["base"]) + slot * landing_slot_bytes],
                            [used],
                            req_id,
                            "landed-mla",
                            engine=engine,
                        )
                        if written:
                            # Recorded before anything else can raise: the
                            # failure write-done then still lists the slot (at
                            # index seq), and the consumer returns it as a
                            # lost READY.
                            request_data.setdefault("_mla_landed", []).append(slot)
                    finally:
                        self._release_mla_staging_slot(pool_idx)
                        if not written:
                            # Never announced, so the consumer will not return it.
                            credits.release(target, epoch, [slot])
                    if not written:
                        logger.error(
                            "[PRODUCER] landed MLA transfer failed for req %s", req_id
                        )
                        return False
                    ready = {
                        "request_id": req_id,
                        "write_nonce": request_data.get("write_nonce", 0),
                        "pp_rank": self.pp_rank,
                        "seq": seq,
                        "slot": slot,
                        "items": [
                            [cmap[regions[pos][0]], start, stop - start, off]
                            for pos, start, stop, off in items
                        ],
                    }
                    self._send_on_socket(
                        notify_path, [MSG_LANDING_READY, msgpack.dumps(ready)]
                    )
                else:
                    return True
                # No credit came: stage each region from its first page not landed.
                first_pages = [len(dst_block_ids)] * len(regions)
                for rest in slots[seq:]:
                    for region_pos, row_start, _, _ in rest:
                        first_pages[region_pos] = min(
                            first_pages[region_pos], row_start // self.block_size
                        )
        return self._execute_staged_mla_regions(
            target,
            plan,
            dst_block_ids,
            regions,
            req_id,
            kv_cache_ready_event,
            engine=engine,
            first_pages=first_pages,
        )

    def _upload_mla_gather_index(
        self, host_index: np.ndarray, kv_cache_ready_event: torch.cuda.Event
    ) -> tuple[torch.cuda.Stream, torch.Tensor]:
        """Upload a request's MLA gather index on this send worker's stream.

        Returns the stream, which first waits for this request's prefill
        writes only (as the index path does), and the device index.
        """
        stream = self._send_worker_stream()
        stream.wait_event(kv_cache_ready_event)
        host = torch.from_numpy(host_index)
        if self._mla_staging.is_cuda:
            # Pinned, so the upload is queued on the worker stream instead of
            # blocking the host until that stream drains.
            host = host.pin_memory()
        with torch.cuda.stream(stream):
            index = host.to(self._mla_staging.device, non_blocking=True)
        return stream, index

    def _gather_mla_rows(
        self,
        stream: torch.cuda.Stream,
        staging: torch.Tensor,
        pieces: list[tuple[int, torch.Tensor, int]],
    ) -> int:
        """Gather MLA token rows into a staging slot; return the bytes filled.

        Each piece is ``(region_idx, token_index, slot_offset)``: that
        region's tokens at ``token_index`` go back to back from
        ``slot_offset``. The stream is synchronized even when a gather
        raises: the NIC reads the slot directly, and the slot must not go
        back to the pool while a gather may still write it.
        """
        used = 0
        try:
            with torch.cuda.stream(stream):
                for region_idx, token_index, offset in pieces:
                    source = self._mla_token_views[region_idx]
                    end = offset + token_index.numel() * source.shape[1]
                    torch.index_select(
                        source,
                        0,
                        token_index,
                        out=staging[offset:end].view(-1, source.shape[1]),
                    )
                    used = max(used, end)
        finally:
            stream.synchronize()
        return used

    def _execute_staged_index_layer_chunk(
        self,
        target: str,
        region_idx: int,
        dst_base: int,
        bytes_per_page: int,
        dst_block_ids: list[int],
        req_id: str,
        gather_indices,
        *,
        engine=None,
    ) -> bool:
        """GPU-repack one index layer/chunk, then RDMA its local pages."""

        pool_idx = self._acquire_index_staging_slot()
        try:
            stream = self._send_worker_stream()
            try:
                with torch.cuda.stream(stream):
                    staging_base, staged_pages = self._gather_sharded_index(
                        region_idx,
                        gather_indices,
                        pool_idx,
                    )
            finally:
                # As in _gather_mla_rows: the slot goes back to the pool below,
                # so even a gather that raised must have finished writing it.
                stream.synchronize()
            if staged_pages != len(dst_block_ids):
                raise RuntimeError(
                    f"Index staging produced {staged_pages} pages for "
                    f"{len(dst_block_ids)} destinations"
                )

            src_page = np.arange(staged_pages, dtype=np.int64)
            dst_page = np.asarray(dst_block_ids, dtype=np.int64)
            length = np.full(staged_pages, bytes_per_page, dtype=np.int64)
            # Like the MLA path, never merge across a destination MR chunk.
            src_addrs, dst_addrs, sizes = coalesce_contiguous(
                staging_base + src_page * bytes_per_page,
                dst_base + dst_page * bytes_per_page,
                length,
                dst_page % self._rdma_chunk_units(bytes_per_page) == 0,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[PRODUCER] staged index RDMA write: req=%s, region=%d, "
                    "pages=%d, descriptors=%d, total_bytes=%d",
                    req_id,
                    region_idx,
                    staged_pages,
                    len(src_addrs),
                    sum(sizes),
                )
            if not self._rdma_write_with_retry(
                target,
                src_addrs.tolist(),
                dst_addrs.tolist(),
                sizes.tolist(),
                req_id,
                "staged-index",
                engine=engine,
            ):
                logger.error(
                    "[PRODUCER] staged index transfer failed for req %s region %d",
                    req_id,
                    region_idx,
                )
                return False
            return True
        finally:
            self._release_index_staging_slot(pool_idx)

    def _execute_block_slot_transfer(
        self,
        request_data: dict,
        target: str,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        prefill_data: dict,
        req_id: str,
        *,
        engine=None,
    ) -> bool:
        """Two-phase RDMA for backends with per-request state: block regions first, then slot regions."""
        consumer_is_fp4 = "consumer_block_base_addrs_fp4" in request_data
        addr_key = (
            "consumer_block_base_addrs_fp4"
            if consumer_is_fp4
            else "consumer_block_base_addrs"
        )
        consumer_block_addrs = request_data[addr_key]
        consumer_block_bpb = request_data["consumer_block_bpb"]
        consumer_slot_addrs = request_data["consumer_slot_base_addrs"]
        consumer_slot_bps = request_data["consumer_slot_bps"]
        dst_slot = request_data["dst_slot_index"]
        src_slot = prefill_data["slot_index"]
        # SWA ring, keyed by state slot.
        consumer_swa_regions = [
            KVTransferRegion(**r)
            for r in request_data.get("consumer_swa_block_regions", [])
        ]
        dst_swa_block_ids = request_data.get("dst_swa_block_ids", [])
        src_swa_block_ids = prefill_data.get("swa_block_ids", [])

        # ---- Phase 1: Block transfer ----
        block_src: list[int] = []
        block_dst: list[int] = []
        block_sizes: list[int] = []

        block_cmap = self._consumer_region_map(
            len(self._block_regions),
            len(consumer_block_addrs),
            request_data.get("consumer_num_layers"),
            self._block_region_consumer_indices,
        )
        # The plan comes from this stage's region order but the bytes land at
        # block_cmap[region_idx], and equal region counts do not make the two
        # orders match. Validate both semantic role and physical width so an
        # incompatible layout fails before any RDMA write instead of silently
        # corrupting KV: FP4 splits the indexer into separate packed-data and
        # e8m0 scale regions, so a mismapped plan is otherwise invisible.
        consumer_roles = request_data.get("consumer_region_roles")
        n_consumer = len(consumer_block_addrs)
        if consumer_is_fp4 != self._fp4_index_layout:
            raise RuntimeError(
                f"Index cache dtype mismatch for req {req_id}: producer "
                f"fp4={self._fp4_index_layout}, consumer fp4={consumer_is_fp4}"
            )
        if len(block_cmap) != len(self._block_regions) or any(
            not 0 <= cidx < n_consumer for cidx in block_cmap
        ):
            raise RuntimeError(
                f"Region map out of range for req {req_id}: {len(block_cmap)} "
                f"mapped indices for {len(self._block_regions)} local regions "
                f"onto {n_consumer} consumer regions"
            )
        if len(consumer_block_bpb) != n_consumer or (
            consumer_roles is not None and len(consumer_roles) != n_consumer
        ):
            raise RuntimeError(
                f"Consumer region arrays disagree for req {req_id}: "
                f"{n_consumer} base addresses, {len(consumer_block_bpb)} byte "
                f"widths, "
                f"{len(consumer_roles) if consumer_roles is not None else 'no'} "
                "roles"
            )
        if (
            self._fp4_index_layout
            and self.pp_size == 1
            and sorted(block_cmap) != list(range(n_consumer))
        ):
            # Bounds only prove each producer region lands somewhere valid,
            # not that the map is a permutation. A consumer with an extra
            # trailing data/scale pair keeps a stale PAGE, and a duplicated
            # destination silently overwrites one region with another -- which
            # the role check cannot catch, because semantic_role is optional
            # and two None-role regions of equal width compare equal. Under
            # pp_size > 1 the map is legitimately partial and
            # _consumer_region_map's group check enforces total coverage.
            raise RuntimeError(
                f"Region map is not one-to-one for req {req_id}: "
                f"{sorted(block_cmap)} over {n_consumer} consumer regions; "
                "an FP4 layout must map each region exactly once"
            )
        if self._fp4_index_layout and consumer_roles is None:
            # Both ends of a PD pair run the same source tree, so this is a
            # deployment error rather than a version to negotiate with: a peer
            # that predates the two-region indexer layout advertises no roles,
            # and there is nothing to map its single FP8 region onto. The
            # reverse pairing -- an older producer receiving FP4 requests -- is
            # fenced on the consumer side by the FP4-only base-address key,
            # which such a producer cannot look up.
            raise RuntimeError(
                f"FP4 index layout requires the consumer to advertise region "
                f"roles, but req {req_id} carries none; the peer predates the "
                "two-region indexer layout"
            )
        for region_idx, (src_base, bpb) in enumerate(self._block_regions):
            cidx = block_cmap[region_idx]
            dst_base = consumer_block_addrs[cidx]
            role = self._block_region_roles[region_idx]
            if consumer_roles is not None and consumer_roles[cidx] != role:
                raise RuntimeError(
                    f"Region role mismatch for req {req_id}: local region "
                    f"{region_idx} is {role!r}, but consumer region "
                    f"{cidx} is {consumer_roles[cidx]!r}"
                )
            if consumer_block_bpb[cidx] != bpb:
                raise RuntimeError(
                    f"Region byte-size mismatch for req {req_id}: producer "
                    f"region {region_idx} has {bpb}, consumer region "
                    f"{cidx} has {consumer_block_bpb[cidx]}"
                )
            for sb, db in zip(src_block_ids, dst_block_ids):
                block_src.append(src_base + sb * bpb)
                block_dst.append(dst_base + db * consumer_block_bpb[cidx])
                block_sizes.append(bpb)

        # SWA ring transfer: every row of a live slot crosses the wire.
        # Catch the case where regions exist but no slot was sent.
        if self._swa_block_regions and not (src_swa_block_ids and dst_swa_block_ids):
            raise RuntimeError(
                f"backend registered {len(self._swa_block_regions)} SWA ring "
                f"regions but the transfer carries no state slot "
                f"(src={src_swa_block_ids}, dst={dst_swa_block_ids}); the "
                "resuming request would read an empty sliding window"
            )
        swa_cmap = self._consumer_region_map(
            len(self._swa_block_regions), len(consumer_swa_regions)
        )
        for region_idx, src_region in enumerate(self._swa_block_regions):
            dst_region = consumer_swa_regions[swa_cmap[region_idx]]
            for sb, db in zip(src_swa_block_ids, dst_swa_block_ids):
                if sb < 0 or db < 0:
                    continue
                block_src.append(src_region.unit_addr(sb))
                block_dst.append(dst_region.unit_addr(db))
                block_sizes.append(src_region.unit_bytes)

        logger.debug(
            "[PRODUCER] block RDMA: req=%s, %d regions × %d blocks, " "total_bytes=%d",
            req_id,
            len(self._block_regions),
            len(src_block_ids),
            sum(block_sizes),
        )

        if not self._rdma_write_with_retry(
            target, block_src, block_dst, block_sizes, req_id, "block", engine=engine
        ):
            logger.error("[PRODUCER] block transfer failed for req %s", req_id)
            return False

        # ---- Phase 2: Slot transfer ----
        if src_slot < 0 or dst_slot < 0:
            logger.debug(
                "[PRODUCER] slot transfer skipped (src_slot=%d, dst_slot=%d)",
                src_slot,
                dst_slot,
            )
            return True

        slot_src: list[int] = []
        slot_dst: list[int] = []
        slot_sizes: list[int] = []

        # Phase 2a: SWA slot regions (direct, no staging)
        slot_cmap = self._consumer_region_map(
            len(self._slot_regions), len(consumer_slot_addrs)
        )
        for region_idx, (src_base, bps) in enumerate(self._slot_regions):
            cidx = slot_cmap[region_idx]
            dst_base = consumer_slot_addrs[cidx]
            slot_src.append(src_base + src_slot * bps)
            slot_dst.append(dst_base + dst_slot * consumer_slot_bps[cidx])
            slot_sizes.append(bps)

        # Phase 2b: compressor states via staging buffer (182 → 1)
        producer_pool_idx = -1
        consumer_staging_addr = request_data.get("consumer_staging_addr", 0)
        if self._gather_slot is not None and consumer_staging_addr:
            producer_pool_idx = self._acquire_staging_slot()
            self._gather_slot(src_slot, producer_pool_idx)
            # Synchronize on the gather kernel before NIC starts reading the
            # staging buffer. Without this, the RDMA can race the still-in-flight
            # gather kernel on TBO prefill (page fault under high concurrency).
            torch.cuda.current_stream().synchronize()
            slot_src.append(
                self._staging_base_addr + producer_pool_idx * self._staging_slot_bytes
            )
            slot_dst.append(consumer_staging_addr)
            slot_sizes.append(self._staging_slot_bytes)

        logger.debug(
            "[PRODUCER] slot RDMA: req=%s, %d entries, "
            "src_slot=%d → dst_slot=%d, total_bytes=%d",
            req_id,
            len(slot_src),
            src_slot,
            dst_slot,
            sum(slot_sizes),
        )

        slot_ok = self._rdma_write_with_retry(
            target, slot_src, slot_dst, slot_sizes, req_id, "slot", engine=engine
        )
        if not slot_ok:
            logger.error("[PRODUCER] slot transfer failed for req %s", req_id)

        if producer_pool_idx >= 0:
            self._release_staging_slot(producer_pool_idx)
        return slot_ok

    def _wait_for_prefill_data(self, req_id: str) -> dict | None:
        """Wait until prefill data is available for this request.

        Returns dict with "block_ids" and "slot_index" keys, or None on timeout.
        """
        with self._completed_prefills_cv:
            ready = self._completed_prefills_cv.wait_for(
                lambda: req_id in self._completed_prefills,
                timeout=PREFILL_LOOKUP_TIMEOUT,
            )
            if ready:
                return self._completed_prefills[req_id]
            return None

    def _rdma_write_with_retry(
        self,
        target: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        sizes: list[int],
        req_id: str,
        label: str,
        *,
        engine=None,
    ) -> bool:
        """Chunked writes and retries on the same request-local engine."""
        if engine is None:
            engine = self.transfer_engine
        max_entries_per_batch = self._MAX_RDMA_ENTRIES_PER_BATCH
        total_entries = len(src_addrs)
        max_retries = 3

        for chunk_start in range(0, total_entries, max_entries_per_batch):
            chunk_end = min(chunk_start + max_entries_per_batch, total_entries)
            chunk_src = src_addrs[chunk_start:chunk_end]
            chunk_dst = dst_addrs[chunk_start:chunk_end]
            chunk_sizes = sizes[chunk_start:chunk_end]

            retry_delay = 2.0
            for attempt in range(max_retries):
                try:
                    ret = engine.batch_transfer_sync_write(
                        target, chunk_src, chunk_dst, chunk_sizes
                    )
                    if ret == 0:
                        break
                    logger.error(
                        "[PRODUCER] %s RDMA chunk error %d for req %s → %s "
                        "(entries %d-%d/%d, attempt %d/%d)",
                        label,
                        ret,
                        req_id,
                        target,
                        chunk_start,
                        chunk_end,
                        total_entries,
                        attempt + 1,
                        max_retries,
                    )
                except Exception:
                    logger.exception(
                        "[PRODUCER] %s RDMA chunk FAILED for req %s "
                        "(entries %d-%d/%d, attempt %d/%d)",
                        label,
                        req_id,
                        chunk_start,
                        chunk_end,
                        total_entries,
                        attempt + 1,
                        max_retries,
                    )
                    ret = -1
                if ret == 0:
                    break
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return False
        return True

    def _send_on_socket(self, addr: str, parts: list, repeat: int = 1) -> None:
        """Send ``parts`` on a cached DEALER socket to ``addr``.

        The socket is created and connected on first use. All sends share
        ``_notify_sockets`` across the listener and executor threads, so the
        whole get-or-create-and-send runs under ``_notify_sockets_lock``
        because ZMQ sockets are not thread-safe.
        """
        with self._notify_sockets_lock:
            sock = self._notify_sockets.get(addr)
            if sock is None:
                sock = self.zmq_context.socket(zmq.DEALER)
                sock.setsockopt(zmq.LINGER, 5000)
                sock.setsockopt(zmq.SNDHWM, 0)
                sock.connect(addr)
                self._notify_sockets[addr] = sock
            for _ in range(repeat):
                sock.send_multipart(parts)

    def _notify_transfer_result(self, request_data: dict, *, success: bool) -> None:
        host = request_data.get("notify_host")
        port = request_data.get("notify_port")
        req_id = request_data.get("request_id")
        if host is None or port is None or req_id is None:
            return
        self._send_write_done(
            host,
            port,
            req_id,
            self.pp_rank,
            request_data.get("write_nonce", 0),
            success=success,
            # None unless the consumer offered landing slots, so it can tell
            # "landed none" from "producer does not land".
            landed_slots=(
                request_data.get("_mla_landed", [])
                if request_data.get("mla_landing") is not None
                else None
            ),
        )

    def _send_write_done(
        self,
        host: str,
        port: int,
        req_id: str,
        pp_rank: int,
        write_nonce: int = 0,
        *,
        success: bool = True,
        landed_slots: list[int] | None = None,
    ) -> None:
        """Send write-done notification to consumer via persistent socket.

        Sends the notification multiple times for reliability. The message
        carries this stage's ``(pp_rank, tp_rank)`` and a ``write_nonce``
        echoed from the write request. The consumer dedups by distinct
        producer rank and validates the nonce, so duplicates are harmless
        (see _record_write_done). ``success=False`` unblocks the consumer
        without treating the KV as ready.
        """
        path = make_zmq_path("tcp", host, port)
        notification = msgpack.dumps(
            {
                "request_id": req_id,
                "pp_rank": pp_rank,
                "tp_rank": self.tp_rank,
                "write_nonce": write_nonce,
                "success": success,
                "landed_slots": landed_slots,
            }
        )
        self._send_on_socket(path, [MSG_WRITE_DONE, notification], repeat=3)
        logger.debug(
            "[PRODUCER] write-done sent for req %s success=%s", req_id, success
        )

    # -----------------------------------------------------------------
    # Consumer: notification listener (ZMQ ROUTER)
    # -----------------------------------------------------------------

    def _notification_listener(self) -> None:
        """Receive write-done notifications from producers."""
        logger.info(
            "Mooncake notification listener bound to tcp://*:%d",
            self._notification_port,
        )

        with _owned_zmq_socket(self._notification_ctx, self._notification_sock) as sock:
            while True:
                parts = sock.recv_multipart()
                msg_type = parts[1]

                if msg_type == MSG_WRITE_DONE:
                    try:
                        data = msgpack.loads(parts[2])
                        self._record_write_done(
                            data["request_id"],
                            data.get("pp_rank", 0),
                            data.get("tp_rank", 0),
                            data.get("write_nonce", 0),
                            success=data.get("success", True),
                            landed_slots=data.get("landed_slots"),
                        )
                    except Exception:
                        # Every later receive on this rank needs this thread.
                        logger.exception("[CONSUMER] handling a write-done failed")
                elif msg_type == MSG_LANDING_READY and self._mla_landing is not None:
                    try:
                        self._mla_landing.on_ready(msgpack.loads(parts[2]))
                    except Exception:
                        # A lost READY: write-done fails the request, returns the slot.
                        logger.exception("[CONSUMER] unreadable landing READY dropped")
                else:
                    logger.error("Unknown notification type: %s", msg_type)

    def _send_release(self, req_id: str) -> None:
        """Tell stage-0 this request's KV is fully received from every stage.

        PP-prefill only. stage-0 defers reusing the shared page table until it
        has one release per decode rank (all stage×rank writes complete).
        """
        with self._completion_lock:
            target = self._release_targets.pop(req_id, None)
        if target is None:
            return
        remote_addr, transfer_id, consumer_tp_size = target
        payload = msgpack.dumps(
            {"transfer_id": transfer_id, "consumer_tp_size": consumer_tp_size}
        )
        self._send_on_socket(remote_addr, [MSG_RELEASE, payload])

    def _record_write_done(
        self,
        req_id: str,
        pp_rank: int,
        tp_rank: int = 0,
        write_nonce: int = 0,
        *,
        success: bool = True,
        landed_slots: list[int] | None = None,
    ) -> bool:
        """Register a producer rank's write-done for ``req_id``.

        Under PP-prefill (and future TP-asymmetric PD) the receive spans
        multiple producer ranks, one write-done each. The producer may resend
        a notification for reliability, so we dedup by distinct
        ``(pp_rank, tp_rank)`` rather than counting messages — otherwise
        duplicates would finalize the receive before lagging ranks have
        written their layers.  A ``write_nonce`` echoed from the write
        request is validated to catch corrupted or misrouted notifications.
        Only the message that completes the last distinct producer rank runs
        slot scatter / block fence and marks the request done, or failed if
        any rank reported a failure: an earlier failure would let the
        scheduler reuse pages a rank still running then writes. Returns True
        when this was that final message.

        A request with MLA landing slots settles in ``LandingReceiver``
        instead: it completes only once its landed slots are scattered too,
        and fails only once every stage ended (``_complete_recv``).
        """
        landing = self._mla_landing
        if landing is not None and landing.stage_done(
            req_id, pp_rank, write_nonce, success, landed_slots
        ):
            return False
        with self._completion_lock:
            expected = self._pending_recv_expected.get(req_id)
            if expected is None:
                return False
            expected_nonce = self._pending_recv_nonce.get(req_id, 0)
            if expected_nonce and write_nonce != expected_nonce:
                logger.error(
                    "[CONSUMER] Write-done nonce mismatch for req %s: "
                    "expected %d, got %d. Ignoring corrupted notification.",
                    req_id,
                    expected_nonce,
                    write_nonce,
                )
                return False
            if not success:
                self._pending_recv_failed.add(req_id)
            stages = self._pending_recv_stages.setdefault(req_id, set())
            stages.add((pp_rank, tp_rank))
            if len(stages) < expected:
                logger.debug(
                    "[CONSUMER] Write-done req %s rank (%d,%d) success=%s (%d/%d)",
                    req_id,
                    pp_rank,
                    tp_rank,
                    success,
                    len(stages),
                    expected,
                )
                return False
            del self._pending_recv_expected[req_id]
            self._pending_recv_stages.pop(req_id, None)
            self._pending_recv_nonce.pop(req_id, None)
            failed = req_id in self._pending_recv_failed
            self._pending_recv_failed.discard(req_id)

        self._complete_recv(req_id, failed)
        return True

    def _complete_recv(self, req_id: str, failed: bool) -> None:
        """Publish a finished receive and let stage-0 free its page table."""
        if self._mla_landing is not None:
            # Landing requests settle outside the write-done bookkeeping.
            with self._completion_lock:
                self._pending_recv_expected.pop(req_id, None)
                self._pending_recv_stages.pop(req_id, None)
                self._pending_recv_nonce.pop(req_id, None)
                self._pending_recv_failed.discard(req_id)
        if failed:
            # Return the staging row to the pool. The scatter is deliberately
            # skipped -- the bytes never landed -- but the row itself must not
            # leak, or _acquire_staging_slot() blocks forever once the pool is
            # exhausted by repeated failures.
            slot_info = self._pending_recv_slots.pop(req_id, None)
            if slot_info is not None:
                _, pool_idx = slot_info
                if pool_idx >= 0:
                    self._release_staging_slot(pool_idx)
            self._pending_recv_blocks.pop(req_id, None)
            with self._completion_lock:
                self.failed_recving.add(req_id)
                self._pending_recv.discard(req_id)
            logger.error(
                "[CONSUMER] Producer reported transfer failure for req %s", req_id
            )
            self._send_release(req_id)
            return

        slot_info = self._pending_recv_slots.pop(req_id, None)
        if slot_info is not None and self._scatter_slot is not None:
            compute_slot, pool_idx = slot_info
            if pool_idx >= 0:
                self._scatter_slot(compute_slot, pool_idx)
                self._release_staging_slot(pool_idx)
        dst_blocks = self._pending_recv_blocks.pop(req_id, None)
        if dst_blocks:
            with self._fence_lock:
                self._blocks_pending_fence.extend(dst_blocks)
        with self._completion_lock:
            self.done_recving.add(req_id)
            self._pending_recv.discard(req_id)
        logger.debug(
            "[CONSUMER] Write-done received for req %s (all stages), "
            "done_recving now: %s",
            req_id,
            self.done_recving,
        )
        # PP-prefill: signal stage-0 it may now reuse the shared page table.
        self._send_release(req_id)
