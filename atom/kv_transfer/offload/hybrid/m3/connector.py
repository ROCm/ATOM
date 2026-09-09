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

import torch

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
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec

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

        block_regions = getattr(transfer_tensors, "block_regions", None)
        if not block_regions:
            raise ValueError(
                "M3 offload requires transfer_tensors.block_regions "
                "(K/V/scale + NSA index_cache regions); got none. The M3 model "
                "runner must emit get_kv_transfer_tensors() block_regions."
            )
        slot_regions = getattr(transfer_tensors, "slot_regions", ()) or ()
        if slot_regions:
            # M3 has no per-request state; a non-empty SLOT list means the model
            # or config selected the wrong offload variant.
            raise ValueError(
                "M3 offload is PAGE-only but transfer_tensors carries "
                f"{len(slot_regions)} SLOT region(s); expected none"
            )

        page_num_blocks = (
            num_blocks
            if num_blocks is not None
            else getattr(transfer_tensors, "num_blocks", None)
        )
        if page_num_blocks is None:
            raise ValueError("M3 offload PAGE regions require a num_blocks value")

        # Build the codec from the ordered region list (NOT from kv_caches), so
        # the NSA index_cache regions are included. slot_regions empty ->
        # DSV4PageSlotCodec runs pure-PAGE (num_slots=0).
        self._codec = DSV4PageSlotCodec(
            page_regions=block_regions,
            slot_regions=(),
            num_blocks=int(page_num_blocks),
            num_slots=0,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        if not self._codec.has_fused_chunk_major_staging:
            raise RuntimeError(
                "M3 offload requires the Triton fused chunk-major PAGE staging "
                "kernels (DSV4PageSlotCodec reported them unavailable)"
            )

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
            len(block_regions),
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
