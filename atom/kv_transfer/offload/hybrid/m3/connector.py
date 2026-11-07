# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""PAGE-only multi-region LMCache offload for MiniMax-M3 (NSA sparse attention).

MiniMax-M3 is a pure-attention model: dense full attention on layers 0-2 and
NSA "lightning-indexer" sparse attention on layers 3-59. It has no recurrent /
SSM / linear-attention layers, hence no per-request state and no SLOT sidecar
(``transfer_tensors.slot_regions`` is always empty).

Why a dedicated variant instead of ``dense``:

* The dense connector builds its byte codec from the per-layer ``KVCacheTensor``
  objects in ``kv_caches`` (K / V / scales). For M3 the NSA per-sparse-layer
  ``index_cache`` is delivered ONLY through ``transfer_tensors.block_regions``;
  it is not an attribute of any ``KVCacheTensor``. Routing M3 through ``dense``
  would silently offload K / V but drop the index cache, corrupting restored
  prefixes on load.

Why not ``hybrid`` (dsv4):

* The dsv4 connector requires ``kv_caches`` to be empty and drives a compressed
  PAGE + stateful SLOT geometry via ``build_dsv4_profile``. M3 delivers a
  non-empty ``kv_caches`` and has no SLOT state, so dsv4 rejects it.

This variant reuses the model-agnostic dense scheduler / worker lifecycle
unchanged and swaps only the byte codec: it builds a ``DSV4PageSlotCodec`` from
``transfer_tensors.block_regions`` (the complete, ordered K / V / scale / index
region list) with ``slot_regions=()`` / ``num_slots=0`` -- i.e. pure PAGE. The
DSV4 codec's PAGE path is byte-exact and exposes the same ``BlockByteCodec``
contract (``bytes_per_block``, ``has_fused_chunk_major_staging``,
``gpu_to_chunk_major_device_buffer``, ``chunk_major_device_buffer_to_gpu``) that
:class:`BlockGPUConnector` requires. The model NEVER changes -- the model runner
already emits complete transfer_tensors; this is a serving-layer wiring only.
"""

from __future__ import annotations

import logging

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload._offload_common import (
    build_offload_engine,
    pp_aware_rank_and_world,
)
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.hybrid.m3.codec import build_m3_page_codec

logger = logging.getLogger("atom")


class M3OffloadConnector(DenseOffloadConnector):
    """PAGE-only, multi-region offload worker for MiniMax-M3.

    Identical lifecycle to :class:`DenseOffloadConnector`; only
    ``register_kv_caches`` differs -- it sources the byte codec from
    ``transfer_tensors.block_regions`` (which include the NSA index cache)
    instead of the per-layer ``KVCacheTensor`` K / V.
    """

    def register_kv_caches(
        self, kv_caches: dict, transfer_tensors=None, num_blocks: int | None = None
    ) -> None:
        from aiter.dist.parallel_state import get_tp_group

        tp = get_tp_group()
        rank, world = pp_aware_rank_and_world(self._config, tp)
        self._rank = rank

        self._codec = build_m3_page_codec(transfer_tensors, num_blocks)

        self._engine, cfg, meta = build_offload_engine(
            self._config,
            engine_id=f"{offcfg.lmcache_engine_id(self._config)}-{rank}",
            block_size=self.virtual_block_size,
            bytes_per_block=self._codec.bytes_per_block,
            gpu_connector_factory=lambda cfg, meta: BlockGPUConnector(
                self._codec,
                self.block_size,
                chunk_size=int(cfg.chunk_size),
                virtual_block_size=self.virtual_block_size,
                source_safe_callback=(
                    self._source_group_safe
                    if getattr(self, "_early_release", False)
                    else None
                ),
            ),
            world=world,
            rank=rank,
        )
        self.chunk_size = int(cfg.chunk_size)

        try:
            from lmcache.v1.lookup_client.factory import LookupClientFactory

            self._lookup_server = LookupClientFactory.create_lookup_server(
                self._engine, meta
            )
        except Exception as e:  # noqa: BLE001  # optional save-only dependency
            logger.warning("LMCache offload: lookup server not started: %s", e)

        gpu_connector = self._engine.gpu_connector
        logger.info(
            "LMCache M3 PAGE offload worker rank=%d: page_regions=%d "
            "bytes_per_block=%d chunk=%d gpu_staging_chunk_bytes=%d "
            "gpu_staging_buffer_chunks=%d gpu_staging_buffer_bytes=%d "
            "release_gpu_staging=%s save=%s load=%s",
            rank,
            len(self._codec.page_regions),
            self._codec.bytes_per_block,
            self.chunk_size,
            gpu_connector.gpu_staging_chunk_bytes,
            gpu_connector.gpu_staging_buffer_chunks,
            gpu_connector.gpu_staging_buffer_bytes,
            gpu_connector.release_gpu_staging_after_transfer,
            self._do_save,
            self._do_load,
        )


class M3OffloadScheduler(DenseOffloadScheduler):
    """M3 reuses the model-agnostic dense offload scheduler unchanged.

    The scheduler operates purely on token counts, block tables, and LMCache
    hit lookups -- no model-specific geometry -- so PAGE-only M3 needs no
    scheduler changes.
    """
