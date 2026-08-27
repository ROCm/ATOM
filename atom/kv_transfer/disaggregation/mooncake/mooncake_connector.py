# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Worker-side and scheduler-side KV cache connectors for disaggregated P/D.

Uses Mooncake TransferEngine for TCP- or RDMA-based push (WRITE) transfers of
KV cache data from producer (prefill) to consumer (decode) nodes.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

import msgpack
import msgspec
import torch
import zmq
from aiter.dist.parallel_state import get_dp_group, get_tp_group

from atom.config import Config
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.port_offset import (
    consumer_region_indices,
)
from atom.kv_transfer.disaggregation.port_offset import (
    side_channel_port_offset as _port_offset,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    KVTransferRegion,
    ReqId,
    TransferId,
)
from atom.model_engine.sequence import Sequence
from atom.models.utils import get_pp_indices
from atom.utils import get_open_port, make_zmq_path, zmq_socket_ctx
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
    return os.path.exists(f"/sys/class/infiniband/{device_name}")


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
    if protocol.strip().lower() == "tcp":
        return ""
    if configured_device:
        return configured_device
    if phys_idx is None:
        raise ValueError("physical GPU index is required for RDMA device selection")
    return _auto_select_ib_device(phys_idx)


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


def _ip_for_ib_device(ib_device: str, fallback: str) -> str:
    """Return the IPv4 address bound to the netdev backing an RDMA HCA."""
    net_root = f"/sys/class/infiniband/{ib_device}/device/net"
    netdevs: list[str] = []
    try:
        netdevs = sorted(os.listdir(net_root))
    except OSError:
        logger.info(
            "Could not list netdevs for ib_device=%s under %s; "
            "falling back to RDMA GID lookup",
            ib_device,
            net_root,
        )

    for netdev in netdevs:
        try:
            out = subprocess.check_output(
                ["ip", "-o", "-4", "addr", "show", "dev", netdev, "scope", "global"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        for line in out.splitlines():
            parts = line.split()
            if "inet" in parts:
                ip_cidr = parts[parts.index("inet") + 1]
                return ip_cidr.split("/", 1)[0]

    gid_ip = _ip_for_ib_device_from_gid(ib_device)
    if gid_ip:
        logger.info(
            "Using IPv4 %s parsed from RDMA GID for ib_device=%s", gid_ip, ib_device
        )
        return gid_ip

    logger.info(
        "Could not determine RDMA-local IPv4 for ib_device=%s (netdevs=%s); "
        "falling back to default IP %s",
        ib_device,
        ",".join(netdevs) if netdevs else "<none>",
        fallback,
    )
    return fallback


def _ip_for_ib_device_from_gid(ib_device: str) -> str | None:
    """Parse an IPv4-mapped RoCE GID from sysfs for containers without host netns."""
    ports_root = f"/sys/class/infiniband/{ib_device}/ports"
    try:
        ports = sorted(os.listdir(ports_root))
    except OSError:
        return None

    preferred_gid_indexes: list[str] = []
    for env_name in ("MC_IB_GID_INDEX", "MOONCAKE_IB_GID_INDEX"):
        env_value = os.environ.get(env_name)
        if env_value:
            preferred_gid_indexes.append(env_value)
    preferred_gid_indexes.extend(["1", "3", "0"])

    for port in ports:
        gids_root = os.path.join(ports_root, port, "gids")
        try:
            available_indexes = sorted(os.listdir(gids_root), key=lambda x: int(x))
        except (OSError, ValueError):
            available_indexes = []

        seen_indexes: set[str] = set()
        gid_indexes = []
        for idx in preferred_gid_indexes + available_indexes:
            if idx not in seen_indexes:
                seen_indexes.add(idx)
                gid_indexes.append(idx)

        for gid_index in gid_indexes:
            gid_path = os.path.join(gids_root, gid_index)
            try:
                with open(gid_path) as f:
                    gid = f.read().strip()
            except OSError:
                continue
            ip = _ipv4_from_gid(gid)
            if ip and ip != "0.0.0.0":
                logger.info(
                    "Parsed RDMA GID %s from %s for ib_device=%s as IPv4 %s",
                    gid,
                    gid_path,
                    ib_device,
                    ip,
                )
                return ip
    return None


def _ipv4_from_gid(gid: str) -> str | None:
    try:
        ip = ipaddress.ip_address(gid)
    except ValueError:
        return None

    if isinstance(ip, ipaddress.IPv4Address):
        return str(ip)
    mapped = ip.ipv4_mapped
    if mapped is not None:
        return str(mapped)
    return None


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
        self.hash_block_size = self.block_size * config.decode_context_parallel_size
        self.host_ip = get_ip()

        # Pending requests: req_id -> (Sequence, block_table)
        self._reqs_need_recv: dict[ReqId, tuple[Any, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Any, list[int]]] = {}

        # Bidirectional transfer_id <-> request_id mapping
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}

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
            # Preserve the producer's final, possibly checkpoint-moved source
            # slot before replacing local_slot_index with the consumer's slot.
            params["remote_slot_index"] = params.get(
                "remote_slot_index", params.get("local_slot_index", -1)
            )
            self._reqs_need_recv[seq.id] = (seq, list(seq.block_table), slot_index)
            params["do_remote_prefill"] = False
            params["local_slot_index"] = slot_index
            # PD incremental: skip leading blocks already in the decode node's
            # prefix cache. Per-request state (including the SWA ring slot) is
            # not covered by a block-only delta, so it takes a full transfer.
            num_computed_blocks = 0
            remote_hash_block_size = params.get("hash_block_size")
            if remote_hash_block_size != self.hash_block_size:
                logger.warning(
                    "PD incremental transfer disabled for req %s: producer "
                    "hash_block_size=%r, consumer hash_block_size=%d; "
                    "falling back to full transfer",
                    seq.id,
                    remote_hash_block_size,
                    self.hash_block_size,
                )
            elif not seq.has_per_req_cache and self.hash_block_size > 0:
                num_computed_blocks = seq.num_cached_tokens // self.hash_block_size
            params["num_computed_blocks"] = num_computed_blocks
            logger.debug(
                "[SCHEDULER-CONSUMER] Queued req %s for remote KV recv "
                "(%d blocks, %d locally cached, dst_slot=%d, src_slot=%d), "
                "transfer_id=%s, remote_host=%s, remote_handshake_port=%s",
                seq.id,
                len(seq.block_table),
                num_computed_blocks,
                slot_index,
                params["remote_slot_index"],
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
        first_token_id = seq.output_tokens[0] if seq.output_tokens else None
        drafts = getattr(seq, "spec_token_ids", None)
        draft_token_ids = (
            [int(x) for x in drafts] if drafts is not None and len(drafts) else []
        )
        final_state_slot = getattr(seq, "state_slot", -1)
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
            "transfer_id": seq.id,
            "first_token_id": first_token_id,
            "draft_token_ids": draft_token_ids,
            "local_slot_index": final_state_slot,
            # Keep source and destination namespaces separate on the consumer:
            # its allocation overwrites local_slot_index with the destination.
            "remote_slot_index": final_state_slot,
            "prefix_cache_hit_tokens": getattr(seq, "prefix_cache_hit_tokens", 0),
        }
        logger.debug(
            "[PD-STATE-SLOT] producer-finished transfer_id=%s final_src_slot=%d",
            seq.id,
            final_state_slot,
        )

        if not self.is_producer:
            transfer_id = self.request_id_to_transfer_id.pop(seq.id, None)
            if transfer_id is not None:
                self.transfer_id_to_request_id.pop(transfer_id, None)


# ===================================================================
# MooncakeConnector — worker-side connector (runs inside each TP rank)
# ===================================================================


class MooncakeConnector(KVConnectorBase):
    """Worker-side KV cache connector using Mooncake push-mode RDMA.

    Mooncake uses a push/WRITE model: the prefill (producer) node writes
    KV cache data directly into the decode (consumer) node's registered
    GPU memory via ``batch_transfer_sync_write``.
    """

    def __init__(self, config: Config) -> None:
        self.tp_rank = get_tp_group().rank_in_group
        self.dp_rank = get_dp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        self.dp_size = get_dp_group().world_size
        self.pp_rank = config.parallel_config.pipeline_parallel_rank
        self.pp_size = config.pipeline_parallel_size
        self.num_hidden_layers = config.hf_config.num_hidden_layers
        # Global index of this stage's first layer; consumer regions are ordered
        # over all layers, so a producer stage writes at this layer offset.
        self._start_layer = 0
        self._num_local_layers = 0

        kv_transfer_config = config.kv_transfer_config
        default_local_ip = get_ip()
        self.local_ip = default_local_ip
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
        # Registering GPU memory with a non-local RDMA NIC fails with
        # EINVAL.  Pass the device name as a filter so Mooncake only
        # creates a context for the local NIC.
        _configure_mooncake_transport(self.protocol)
        configured_ib_device = kv_transfer_config.get(
            "ib_device", ""
        ) or os.environ.get("ATOM_MOONCAKE_IB_DEVICE", "")
        phys_idx: int | None = None
        if self.protocol.strip().lower() != "tcp" and not configured_ib_device:
            visible_idx = torch.cuda.current_device()
            visible_env = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            )
            if visible_env:
                visible_list = [d for d in visible_env.split(",") if d != ""]
                phys_idx = int(visible_list[visible_idx])
            else:
                phys_idx = visible_idx
        ib_device = _select_ib_device(self.protocol, configured_ib_device, phys_idx)
        if self.protocol.strip().lower() == "tcp":
            logger.info("Mooncake TCP selected; RDMA device selection is disabled")
        elif not configured_ib_device:
            logger.info(
                "Auto-selecting RDMA device %s for physical GPU %d "
                "(visible_idx=%d, tp_rank=%d)",
                ib_device,
                phys_idx,
                visible_idx,
                self.tp_rank,
            )

        rdma_local_ip = (
            _ip_for_ib_device(ib_device, default_local_ip)
            if ib_device
            else default_local_ip
        )
        if rdma_local_ip != default_local_ip:
            logger.info(
                "Using RDMA-local IP %s for ib_device=%s instead of default IP %s",
                rdma_local_ip,
                ib_device,
                default_local_ip,
            )
        self.local_ip = rdma_local_ip
        self.request_address = f"{self.local_ip}:{self.http_port}"

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

        # --- KV cache state (populated in register_kv_caches) ---
        self.kv_caches: dict[str, Any] | None = None
        self.kv_caches_base_addr: list[int] = []
        self._per_block_bytes_list: list[int] = []
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

        # --- Producer: completed prefill block_ids cache ---
        # Populated from ConnectorMetadata.reqs_to_save each step.
        # The write listener looks up block_ids here when consumer requests a write.
        self._completed_prefills: dict[ReqId, dict] = {}
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
        # Per-request nonce for write-done corruption detection.
        self._pending_recv_nonce: dict[ReqId, int] = {}
        # PP-prefill: consumer stashes stage-0's release address per request, and
        # the producer (stage-0) counts releases to defer freeing the shared page
        # table until every stage has written the KV out.
        self._release_targets: dict[ReqId, tuple[str, int, int]] = {}
        self._release_count: dict[TransferId, int] = {}
        self._released_transfers: set[TransferId] = set()
        self._notification_port = get_open_port()

        # --- Completion tracking ---
        self.done_sending: set[str] = set()
        self.done_recving: set[str] = set()
        self._completion_lock = threading.Lock()

        # --- GPU memory fence: blocks pending coherence enforcement ---
        self._blocks_pending_fence: list[int] = []
        self._fence_lock = threading.Lock()

        # --- Transfer ID mapping (worker side) ---
        self.request_id_to_transfer_id: dict[ReqId, TransferId] = {}

        # --- Producer: thread pool for RDMA writes ---
        if self.is_producer:
            self._send_executor = ThreadPoolExecutor(
                max_workers=kv_transfer_config.get("num_worker_threads", 16),
                thread_name_prefix="mooncake-send-worker",
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
        for r in all_regions:
            offset = 0
            for chunk in self._rdma_chunk_sizes(r.total_bytes, r.unit_bytes):
                reg_ptrs.append(r.base_addr + offset)
                reg_sizes.append(chunk)
                offset += chunk

        logger.info(
            "Registering %d RDMA chunks (%d block regions, %d slot regions, "
            "max_chunk=%.2f GiB)",
            len(reg_ptrs),
            len(tt.block_regions),
            len(tt.slot_regions),
            self._MAX_RDMA_CHUNK_BYTES / (1024**3),
        )

        ret = self.transfer_engine.batch_register_memory(reg_ptrs, reg_sizes)
        if ret != 0:
            logger.error(
                "batch_register_memory FAILED (ret=%d), "
                "trying individual registration as fallback...",
                ret,
            )
            for ptr, sz_bytes in zip(reg_ptrs, reg_sizes):
                r = self.transfer_engine.register_memory(ptr, sz_bytes)
                if r != 0:
                    logger.error(
                        "  register_memory FAILED ptr=0x%x size=%d ret=%d",
                        ptr,
                        sz_bytes,
                        r,
                    )
        else:
            logger.info("batch_register_memory OK (%d chunks)", len(reg_ptrs))

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

    # -----------------------------------------------------------------
    # KVConnectorBase: start_load_kv
    # -----------------------------------------------------------------

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
            # offset falls back to full transfer.
            remote_block_ids = meta.remote_block_ids or []
            off = meta.num_computed_blocks
            if (
                off < 0
                or off >= len(meta.local_block_ids)
                or off >= len(remote_block_ids)
            ):
                off = 0
            dst_block_ids = meta.local_block_ids[off:]
            src_block_ids = remote_block_ids[off:]

            # Build the (stage-independent) write_request payload once.
            request_body = {
                "request_id": req_id,
                "transfer_id": meta.transfer_id,
                "consumer_host": self.local_ip,
                "consumer_rpc_port": self.rpc_port,
                # Consumer's layer count per group for producer stride validation.
                "consumer_num_layers": self._num_local_layers,
                "dst_block_ids": dst_block_ids,
                # Source block_ids so downstream stages (no scheduler, no
                # _completed_prefills) can transfer without a local lookup.
                "src_block_ids": src_block_ids,
                # Producer slices its local prefill block_ids by this offset in
                # the TP-TP path (where it uses its own cache, not src above).
                "num_computed_blocks": off,
                "notify_host": self.local_ip,
                "notify_port": self._notification_port,
                "consumer_tp_size": self.tp_size,
                "write_nonce": write_nonce,
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
                        # Recurrent state is slot-indexed, and prefix
                        # checkpointing may move the producer request after its
                        # initial allocation.
                        "src_slot_index": meta.remote_slot_index,
                        "src_swa_block_ids": meta.remote_swa_block_ids,
                        "dst_slot_index": meta.local_slot_index,
                        "consumer_block_base_addrs": [
                            b for b, _ in self._block_regions
                        ],
                        "consumer_block_bpb": [bpb for _, bpb in self._block_regions],
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
            else:
                request_body["consumer_base_addrs"] = self.kv_caches_base_addr

            write_request = msgpack.dumps(request_body)

            for stage in range(remote_pp_size):
                remote_port = meta.remote_handshake_port + _port_offset(
                    meta.remote_dp_rank,
                    remote_tp_rank,
                    remote_tp_size,
                    stage,
                    remote_pp_size,
                    meta.remote_dp_size,
                )
                remote_addr = make_zmq_path("tcp", meta.remote_host, remote_port)
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
                self._send_on_socket(remote_addr, [MSG_WRITE_REQUEST, write_request])
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

            self._pending_recv.add(req_id)
            # Only delta blocks need fencing; reused prefix blocks are coherent.
            self._pending_recv_blocks[req_id] = list(dst_block_ids)
            if meta.local_slot_index >= 0:
                self._pending_recv_slots[req_id] = (
                    meta.local_slot_index,
                    consumer_staging_pool_idx,
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

    # -----------------------------------------------------------------
    # KVConnectorBase: get_finished
    # -----------------------------------------------------------------

    def get_finished(self) -> tuple[set, set]:
        """Return ``(done_sending, done_recving)`` and clear internal sets."""
        with self._completion_lock:
            ds = self.done_sending.copy()
            dr = self.done_recving.copy()
            self.done_sending.clear()
            self.done_recving.clear()
        if ds or dr:
            logger.debug(
                "[%s] get_finished: sending=%s, recving=%s",
                "PRODUCER" if self.is_producer else "CONSUMER",
                ds,
                dr,
            )
        return ds, dr

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
                    self._send_executor.submit(self._execute_transfer, request_data)

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
            notify_host = request_data["notify_host"]
            notify_port = request_data["notify_port"]
            consumer_tp_size = request_data.get("consumer_tp_size", self.tp_size)
            consumers_per_rank = max(1, consumer_tp_size // self.tp_size)
            write_nonce = request_data.get("write_nonce", 0)
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
            final_src_slot = -1
            if has_slot_data:
                try:
                    final_src_slot = int(request_data["src_slot_index"])
                except (KeyError, TypeError, ValueError):
                    pass
                if final_src_slot < 0:
                    raise RuntimeError(
                        "stateful Mooncake transfer is missing the producer's "
                        f"final source slot (req_id={req_id}, "
                        f"transfer_id={transfer_id})"
                    )
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
                    return
                with self._completed_prefills_lock:
                    cached = self._completed_prefills.get(transfer_id)
                prefill_data = {
                    "block_ids": request_src_block_ids,
                    "slot_index": (
                        cached["slot_index"]
                        if cached
                        else final_src_slot
                    ),
                }

            src_block_ids = prefill_data["block_ids"]
            # PD incremental (TP-TP only): consumer already sliced dst; slice
            # producer's src by the same offset. PP src arrives pre-sliced.
            if self.pp_size == 1:
                off = request_data.get("num_computed_blocks", 0)
                if 0 < off < len(src_block_ids):
                    src_block_ids = src_block_ids[off:]
            if len(src_block_ids) != len(dst_block_ids):
                logger.error(
                    "[PRODUCER] src/dst block count mismatch for req %s "
                    "(src=%d, dst=%d); aborting transfer to avoid misaligned KV.",
                    req_id,
                    len(src_block_ids),
                    len(dst_block_ids),
                )
                return
            if has_slot_data:
                initial_src_slot = int(prefill_data.get("slot_index", -1))
                prefill_data = dict(prefill_data)
                prefill_data["slot_index"] = final_src_slot
                prefill_data["swa_block_ids"] = request_data.get(
                    "src_swa_block_ids", prefill_data.get("swa_block_ids", [])
                )
                logger.debug(
                    "[PD-STATE-SLOT] transfer req_id=%s transfer_id=%s "
                    "initial_src_slot=%d final_src_slot=%d dst_slot=%d",
                    req_id,
                    transfer_id,
                    initial_src_slot,
                    final_src_slot,
                    int(request_data.get("dst_slot_index", -1)),
                )
            target = f"{consumer_host}:{consumer_rpc_port}"

            if hasattr(self.transfer_engine, "get_first_buffer_address"):
                remote_buf = self.transfer_engine.get_first_buffer_address(target)
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
                )
            else:
                transfer_ok = self._execute_block_transfer(
                    request_data,
                    target,
                    src_block_ids,
                    dst_block_ids,
                    req_id,
                )

            if not transfer_ok:
                logger.error(
                    "[PRODUCER] transfer failed for req %s (transfer_id=%s); "
                    "not sending write-done",
                    req_id,
                    transfer_id,
                )
                return

            # Notify consumer — all data (blocks + slot state) is written.
            self._send_write_done(
                notify_host, notify_port, req_id, self.pp_rank, write_nonce
            )

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
                    logger.debug(
                        "[PRODUCER] All %d consumers served for transfer_id=%s",
                        consumers_per_rank,
                        transfer_id,
                    )
        except Exception:
            logger.exception(
                "[PRODUCER] transfer FAILED for req %s (transfer_id=%s); "
                "consumer will not receive write-done and will time out.",
                request_data.get("request_id"),
                request_data.get("transfer_id"),
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
    ) -> bool:
        """Block-only RDMA transfer (MHA, MLA, and other block-indexed backends)."""
        consumer_base_addrs = request_data["consumer_base_addrs"]

        src_addrs: list[int] = []
        dst_addrs: list[int] = []
        sizes: list[int] = []

        num_regions = len(self.kv_caches_base_addr)
        cmap = self._consumer_region_map(
            num_regions,
            len(consumer_base_addrs),
            request_data.get("consumer_num_layers"),
            self._block_region_consumer_indices,
        )
        for region_idx in range(num_regions):
            src_base = self.kv_caches_base_addr[region_idx]
            dst_base = consumer_base_addrs[cmap[region_idx]]
            bpb = self._per_block_bytes_list[region_idx]
            for sb, db in zip(src_block_ids, dst_block_ids):
                src_addrs.append(src_base + sb * bpb)
                dst_addrs.append(dst_base + db * bpb)
                sizes.append(bpb)

        logger.debug(
            "[PRODUCER] block RDMA write: req=%s, %d regions × %d blocks, "
            "total_bytes=%d",
            req_id,
            num_regions,
            len(src_block_ids),
            sum(sizes),
        )

        if not self._rdma_write_with_retry(
            target, src_addrs, dst_addrs, sizes, req_id, "block"
        ):
            logger.error("[PRODUCER] block transfer failed for req %s", req_id)
            return False
        return True

    def _execute_block_slot_transfer(
        self,
        request_data: dict,
        target: str,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        prefill_data: dict,
        req_id: str,
    ) -> bool:
        """Two-phase RDMA for backends with per-request state: block regions first, then slot regions."""
        consumer_block_addrs = request_data["consumer_block_base_addrs"]
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
        for region_idx, (src_base, bpb) in enumerate(self._block_regions):
            cidx = block_cmap[region_idx]
            dst_base = consumer_block_addrs[cidx]
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
            target, block_src, block_dst, block_sizes, req_id, "block"
        ):
            logger.error("[PRODUCER] block transfer failed for req %s", req_id)
            return False

        # ---- Phase 2: Slot transfer ----
        # Registered slot regions but no slot means no per-request state moves
        # and the resuming side decodes from a zeroed recurrent state -- 69 of
        # 93 layers on Kimi-K3, still fluent, so the skip below would report
        # success. Same shape as the SWA guard above.
        if self._slot_regions and (src_slot < 0 or dst_slot < 0):
            raise RuntimeError(
                f"backend registered {len(self._slot_regions)} state slot "
                f"regions but the transfer carries no state slot "
                f"(src={src_slot}, dst={dst_slot}); the resuming request would "
                "decode from a zeroed recurrent state"
            )
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

        # Phase 2a: slot-indexed state, written direct (no staging)
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
            target, slot_src, slot_dst, slot_sizes, req_id, "slot"
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
    ) -> bool:
        """Chunked RDMA write with retry. Returns True on success."""
        max_entries_per_batch = 4096
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
                    ret = self.transfer_engine.batch_transfer_sync_write(
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

    def _send_write_done(
        self,
        host: str,
        port: int,
        req_id: str,
        pp_rank: int,
        write_nonce: int = 0,
    ) -> None:
        """Send write-done notification to consumer via persistent socket.

        Sends the notification multiple times for reliability. The message
        carries this stage's ``(pp_rank, tp_rank)`` and a ``write_nonce``
        echoed from the write request. The consumer dedups by distinct
        producer rank and validates the nonce, so duplicates are harmless
        (see _record_write_done).
        """
        path = make_zmq_path("tcp", host, port)
        notification = msgpack.dumps(
            {
                "request_id": req_id,
                "pp_rank": pp_rank,
                "tp_rank": self.tp_rank,
                "write_nonce": write_nonce,
            }
        )
        self._send_on_socket(path, [MSG_WRITE_DONE, notification], repeat=3)
        logger.debug("[PRODUCER] write-done sent for req %s", req_id)

    # -----------------------------------------------------------------
    # Consumer: notification listener (ZMQ ROUTER)
    # -----------------------------------------------------------------

    def _notification_listener(self) -> None:
        """Receive write-done notifications from producers."""
        path = make_zmq_path("tcp", "*", self._notification_port)
        logger.info("Mooncake notification listener bound to %s", path)

        with zmq_socket_ctx(path, zmq.ROUTER, bind=True) as sock:
            while True:
                parts = sock.recv_multipart()
                msg_type = parts[1]

                if msg_type == MSG_WRITE_DONE:
                    data = msgpack.loads(parts[2])
                    self._record_write_done(
                        data["request_id"],
                        data.get("pp_rank", 0),
                        data.get("tp_rank", 0),
                        data.get("write_nonce", 0),
                    )
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
        slot scatter / block fence and marks the request done.  Returns True
        when this was that final message.
        """
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
            stages = self._pending_recv_stages.setdefault(req_id, set())
            stages.add((pp_rank, tp_rank))
            if len(stages) < expected:
                logger.debug(
                    "[CONSUMER] Write-done req %s rank (%d,%d) (%d/%d)",
                    req_id,
                    pp_rank,
                    tp_rank,
                    len(stages),
                    expected,
                )
                return False
            del self._pending_recv_expected[req_id]
            self._pending_recv_stages.pop(req_id, None)
            self._pending_recv_nonce.pop(req_id, None)

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
        return True
