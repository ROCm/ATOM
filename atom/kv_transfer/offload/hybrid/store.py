# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache-backed :class:`LMCacheBundleStore` for hybrid offload bundles.

Unlike the MHA/MLA offload path (which drives ``engine.store``/``retrieve`` with
LMCache's token-major ChunkedTokenDatabase), a hybrid terminal checkpoint is *one*
opaque object addressed by an exact (prefix, B) key. We therefore go one level
below the token database and use LMCache's ``StorageManager`` directly:
``allocate`` a uint8 MemoryObj, fill it, ``put`` under a ``CacheEngineKey`` whose
``chunk_hash`` is our :func:`checkpoint_key` digest; ``get``/``contains`` mirror it.
This reuses LMCache's CPU/NVMe tiering + eviction for free.

NOTE: exercised via the LMCache C++/Python storage backend; it has no offline
unit test (needs a live engine). The pure region/lifecycle/policy logic it feeds
IS covered by ``test_dsv4_offload_*``. Serve validation is pending — see
``dsv4-lmcache-bundle-plan.md`` Phase 4.
"""

from __future__ import annotations

import hashlib
import logging

import torch

logger = logging.getLogger("atom")

_HASH_MASK = (1 << 63) - 1  # keep chunk_hash in a safe non-negative 63-bit range


def _key_to_chunk_hash(key_str: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(key_str.encode("utf-8"), digest_size=8).digest(), "little"
    ) & _HASH_MASK


class LMCacheBundleStore:
    """One-object-per-checkpoint store over an LMCache ``StorageManager``."""

    def __init__(
        self,
        engine,
        *,
        model_name: str,
        world_size: int,
        worker_id: int,
    ) -> None:
        from lmcache.v1.memory_management import MemoryFormat

        self._engine = engine
        self._sm = engine.storage_manager
        if self._sm is None:
            raise RuntimeError("LMCacheBundleStore: engine.storage_manager is None")
        self._model_name = str(model_name)
        self._world_size = int(world_size)
        self._worker_id = int(worker_id)
        # KV_2LTD is inert here (same allocator-compat hack as the MHA path); the
        # real shape/dtype are forced by the uint8 allocation.
        self._fmt = MemoryFormat.KV_2LTD

    def _ce_key(self, key_str: str):
        from lmcache.utils import CacheEngineKey

        return CacheEngineKey(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=self._worker_id,
            chunk_hash=_key_to_chunk_hash(key_str),
            dtype=torch.uint8,
        )

    def contains(self, key: str) -> bool:
        try:
            return self._sm.contains(self._ce_key(key)) is not None
        except Exception:
            logger.exception("LMCacheBundleStore.contains failed key=%s", key)
            return False

    def put(self, key: str, unit_host: torch.Tensor) -> bool:
        flat = unit_host.reshape(-1)
        nbytes = int(flat.numel())
        mem = self._sm.allocate(torch.Size((nbytes,)), torch.uint8, fmt=self._fmt)
        if mem is None:
            # Eviction couldn't free room; drop this offload opportunity.
            logger.debug("LMCacheBundleStore: allocate returned None key=%s", key)
            return False
        try:
            dst = mem.tensor
            if dst is None and hasattr(mem, "get_tensor"):
                dst = mem.get_tensor(0)
            dst.reshape(-1)[:nbytes].copy_(flat)
            # StorageManager.put is deprecated in LMCache 0.4.5; use batched_put.
            self._sm.batched_put([self._ce_key(key)], [mem])
            return True
        except Exception:
            logger.exception("LMCacheBundleStore.put failed key=%s", key)
            # Best-effort: release the object we failed to hand off.
            try:
                mem.ref_count_down()
            except Exception:
                pass
            return False

    def get(self, key: str) -> "torch.Tensor | None":
        try:
            mem = self._sm.get(self._ce_key(key))
        except Exception:
            logger.exception("LMCacheBundleStore.get failed key=%s", key)
            return None
        if mem is None:
            return None
        try:
            src = mem.tensor
            if src is None and hasattr(mem, "get_tensor"):
                src = mem.get_tensor(0)
            # Clone to decouple from LMCache's object lifetime; the caller
            # validates + scatters, then discards.
            return src.reshape(-1).clone()
        finally:
            try:
                mem.ref_count_down()
            except Exception:
                pass
