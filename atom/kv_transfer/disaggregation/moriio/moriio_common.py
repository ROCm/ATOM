# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Shared types, constants, enums, and helpers for the MoRIIO KV connector.

This module has no heavy dependencies (no torch at import time, no RDMA
engine instances) so it can be imported freely by the other moriio
submodules.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import nullcontext
from enum import Enum
from typing import Any, Optional

import msgspec

from atom.kv_transfer.disaggregation.utils import (  # noqa: F401
    MAX_RDMA_CHUNK_BYTES,
    chunk_tensor_for_rdma,
)

logger = logging.getLogger("atom")

_chunk_tensor_for_rdma = chunk_tensor_for_rdma

# ---------------------------------------------------------------------------
# MoRIIO availability check
# ---------------------------------------------------------------------------

_MORIIO_AVAILABLE = False
try:
    import mori.io  # noqa: F401

    _MORIIO_AVAILABLE = True
    logger.info("MoRIIO RDMA library loaded successfully")
except ImportError:
    logger.warning(
        "MoRIIO is not available — KV cache disaggregation will not work. "
        "Install the mori package to enable RDMA transfers."
    )


# ---------------------------------------------------------------------------
# Msgspec metadata structs
# ---------------------------------------------------------------------------


class MoRIIOAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
    kw_only=True,
):
    """Serializable metadata exchanged during the RDMA handshake."""

    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: Optional[list[int]] = None
    num_blocks: int = 0
    block_len: int = 0
    attn_backend_name: str = "aiter"


class MoRIIOWriteRegion(
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
    kw_only=True,
):
    """One unit-addressed RDMA region advertised for write-mode transfer."""

    kind: str
    chunks: list[bytes]
    unit_bytes: int
    units_per_chunk: int
    total_units: int


class MoRIIOWriteRequest(
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
    kw_only=True,
):
    """Consumer-to-producer write-mode transfer request."""

    decode_req_id: str | int
    transfer_id: str | int
    consumer_engine_desc: bytes
    consumer_regions: list[MoRIIOWriteRegion]
    dst_block_ids: list[int]
    dst_slot_index: int = -1
    dst_staging_pool_idx: int = -1
    notify_host: str
    notify_port: int
    consumer_tp_size: int
    consumer_dp_rank: int = 0


class MoRIIOWriteDone(
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
    kw_only=True,
):
    """Producer-to-consumer write-mode completion notification."""

    decode_req_id: str | int
    status: str
    reason: str | None = None


# ---------------------------------------------------------------------------
# Enums & role management
# ---------------------------------------------------------------------------


class TransferMode(str, Enum):
    """MoRIIO transfer direction."""

    READ_PULL = "read"
    WRITE_PUSH = "write"


class Role(Enum):
    """Role of the current engine instance in the P/D architecture."""

    PRODUCER = "producer"
    CONSUMER = "consumer"
    NOT_INITIALIZED = "not_initialized"


def convert_virtual_to_physical_pages(
    virtual_pages: list[int],
    virtual_block_size: int = 16,
    physical_block_size: int = 1,
) -> list[int]:
    """Expand virtual (coarse) block IDs into physical (fine-grained) page IDs.

    In paged-attention the scheduler works with *virtual* blocks of
    ``virtual_block_size`` tokens, but the RDMA transfer operates at
    ``physical_block_size`` granularity.

    Args:
        virtual_pages: List of virtual block IDs.
        virtual_block_size: Tokens per virtual block.
        physical_block_size: Tokens per physical block.

    Returns:
        Expanded list of physical page IDs.
    """
    block_ratio = virtual_block_size // physical_block_size
    physical_pages: list[int] = []
    for vp in virtual_pages:
        start = vp * block_ratio
        physical_pages.extend(range(start, start + block_ratio))
    return physical_pages


class _RoleManager:
    """Thread-safe singleton that tracks the P/D role of this process.

    Use the module-level :func:`get_role` / :func:`set_role` helpers
    instead of accessing this class directly.
    """

    _instance: Optional[_RoleManager] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._role: Role = Role.NOT_INITIALIZED

    @classmethod
    def get_instance(cls) -> _RoleManager:
        """Return the singleton, creating it on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = object.__new__(cls)
                    instance.__init__()
                    cls._instance = instance
        return cls._instance

    def set_role(self, role: Role) -> None:
        with self._lock:
            self._role = role

    @property
    def role(self) -> Role:
        return self._role


def set_role(role: Role) -> None:
    """Set the global P/D role for this process."""
    _RoleManager.get_instance().set_role(role)


def get_role() -> Role:
    """Get the global P/D role for this process."""
    return _RoleManager.get_instance().role


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class MoRIIOConstants:
    """Protocol constants for the MoRIIO-based KV connector."""

    # ZMQ handshake message types
    GET_META_MSG = b"get_meta_msg"
    POP_DONE_RECV = b"pop_done_recv"
    WRITE_REQUEST = b"write_request"
    WRITE_DONE = b"write_done"
    OVER = b"OVER"
    COMPLETION_PREFIX = "cmpl"

    # Networking
    DEFAULT_HANDSHAKE_PORT = 6301
    DEFAULT_NOTIFY_PORT = "61005"

    # Timeouts
    ABORT_REQUEST_TIMEOUT = 3600


def get_port_offset(dp_rank: int, tp_rank: int, tp_size: int = 1) -> int:
    return (dp_rank) * tp_size + tp_rank


# ---------------------------------------------------------------------------
# Fabric KV allocation pool (mem-pool provider for KVConnectorFactory)
# ---------------------------------------------------------------------------

_FABRIC_KV_MEM_POOL = None


def _moriio_fabric_enabled(kv_transfer_config: Optional[dict]) -> bool:
    """Whether the MoRIIO FABRIC backend is selected (env or kv config)."""
    if os.environ.get("ATOM_MORIIO_FABRIC", "0") == "1":
        return True
    kv_cfg = kv_transfer_config or {}
    return str(kv_cfg.get("moriio_backend", "")).lower() == "fabric"


def maybe_fabric_kv_mem_pool_ctx(config: Any = None):
    """Mem-pool provider for the MoRIIO connector (see KVConnectorFactory).

    The mori-io FABRIC backend (UALink scale-up) can only register/transfer VMM
    memory created with a fabric handle, so KV tensors must be allocated inside
    a fabric-exportable MemPool. Returns ``torch.cuda.use_mem_pool(pool)`` when
    the fabric backend is selected, else a ``nullcontext`` (RDMA registers
    ordinary torch memory and needs no custom pool). The pool is created once
    and cached at module scope, which also keeps it alive for the process
    lifetime. Analogous to mooncake's ``init_mooncake_custom_mem_pool``.
    """
    kv_transfer_config = getattr(config, "kv_transfer_config", None)
    if not _moriio_fabric_enabled(kv_transfer_config):
        return nullcontext()

    import torch
    from mori.io import make_fabric_mem_pool

    global _FABRIC_KV_MEM_POOL
    if _FABRIC_KV_MEM_POOL is None:
        _FABRIC_KV_MEM_POOL = make_fabric_mem_pool()
    return torch.cuda.use_mem_pool(_FABRIC_KV_MEM_POOL)
