# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Type definitions for the local KV-offload subsystem.

OFFLOAD connectors maintain a second/third-tier KV store (CPU DRAM, NVMe)
behind the engine's HBM paged KV cache. They differ from PD-disaggregation
connectors in two key ways:

* No remote engine. Saves/loads are local D2H/H2D copies, so transfer
  metadata carries only local ``(block_id, hash)`` pairs — no host/port
  or remote-engine identifiers.
* No remote block-ownership transfer at request finish. The source GPU
  blocks remain owned by the BlockManager, but an async D2H save still
  needs those blocks to remain valid until the worker reports
  ``finished_sending``. The scheduler therefore defers freeing only for
  OFFLOAD requests with pending saves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atom.kv_transfer.disaggregation.types import ReqId


@dataclass
class OffloadReqMeta:
    """Per-request metadata for one offload operation (save or load).

    A *save* and a *load* use the same dataclass: for save, ``block_ids``
    are GPU paged-block slot ids holding KV the connector should copy out;
    for load, ``block_ids`` are GPU paged-block slot ids the connector
    should fill from the external store. ``block_hashes`` is the prefix-
    cache hash chain produced by ``BlockManager.compute_hash``, used as the
    cache key in the external store.
    """

    block_ids: list[int] = field(default_factory=list)
    block_hashes: list[int] = field(default_factory=list)


class OffloadConnectorMetadata:
    """Snapshot of pending offload operations, passed scheduler -> worker.

    Mirrors :class:`atom.kv_transfer.disaggregation.types.ConnectorMetadata`
    so the engine-core dispatch path (``process_kvconnector_output``) can
    handle both flavors interchangeably; only the field types differ.
    """

    def __init__(self) -> None:
        self.reqs_to_save: dict[ReqId, OffloadReqMeta] = {}
        self.reqs_to_load: dict[ReqId, OffloadReqMeta] = {}

    def is_empty(self) -> bool:
        return not self.reqs_to_save and not self.reqs_to_load

    def __repr__(self) -> str:
        return (
            f"OffloadConnectorMetadata(saves={len(self.reqs_to_save)}, "
            f"loads={len(self.reqs_to_load)})"
        )
