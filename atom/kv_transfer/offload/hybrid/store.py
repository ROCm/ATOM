# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opaque SLOT sidecar storage through an existing LMCache StorageManager."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import threading

import torch

from atom.kv_transfer.offload.hybrid.sidecar_format import SlotSidecarKey

logger = logging.getLogger("atom")


class SlotSidecarCorruptionError(RuntimeError):
    """Raised when LMCache returns a malformed SLOT sidecar object.

    This is deliberately distinct from a storage lookup/read failure: callers
    must invalidate a corrupt object before trying to publish the same
    content-addressed key again.
    """


class SlotSidecarStore:
    """Persist rank-local AOS1 bytes in the engine's configured LMCache tiers."""

    def __init__(
        self,
        engine,
        *,
        model_name: str,
        world_size: int,
        worker_id: int,
    ) -> None:
        storage_manager = getattr(engine, "storage_manager", None)
        if storage_manager is None:
            raise ValueError("engine.storage_manager must not be None")

        # LMCache is optional for unit-test collection and for ATOM deployments
        # that do not enable offload, so import its runtime types only when this
        # adapter is actually constructed.
        from lmcache.utils import CacheEngineKey
        from lmcache.v1.memory_management import MemoryFormat

        self._storage_manager = storage_manager
        self._store_location = getattr(engine, "store_location", None)
        retrieve_locations = getattr(engine, "retrieve_locations", None)
        self._retrieve_locations = (
            None if retrieve_locations is None else list(retrieve_locations)
        )
        self._model_name = model_name
        self._world_size = world_size
        self._worker_id = worker_id
        self._cache_engine_key_type = CacheEngineKey
        self._memory_format = MemoryFormat.KV_2LTD
        # A failed removal leaves LMCache's duplicate-key guard pointing at
        # the corrupt object. Keep the key fenced until a later removal
        # succeeds; otherwise batched_put may be accepted while silently
        # retaining the old bytes.
        self._corruption_lock = threading.Lock()
        self._unresolved_corrupt_keys: set[SlotSidecarKey] = set()

    def put(
        self,
        key: SlotSidecarKey,
        blob: torch.Tensor | bytes | bytearray | memoryview,
    ) -> bool:
        """Submit one opaque sidecar to the configured LMCache storage tiers.

        ``True`` means ``StorageManager.batched_put`` accepted/submitted the
        object. It does not mean an asynchronous backend has durably flushed it.
        """

        sidecar_key = self._require_key(key)
        payload = self._payload_tensor(blob)
        if not self._prepare_republication(sidecar_key):
            return False
        cache_key = self._cache_key(sidecar_key)
        memory_obj = None

        try:
            memory_obj = self._storage_manager.allocate(
                torch.Size((1, 1, payload.numel())),
                torch.uint8,
                fmt=self._memory_format,
                busy_loop=False,
            )
            if memory_obj is None:
                return False

            target = self._memory_tensor(memory_obj)
            if not isinstance(target, torch.Tensor):
                raise RuntimeError("LMCache allocation did not expose a tensor")
            if target.dtype is not torch.uint8:
                raise RuntimeError("LMCache allocation did not preserve uint8 dtype")
            if target.device.type != "cpu":
                raise RuntimeError("LMCache allocation is not on the CPU")
            if target.numel() != payload.numel():
                raise RuntimeError(
                    "LMCache allocation size does not match the sidecar payload"
                )

            target.reshape(-1).copy_(payload)
            self._storage_manager.batched_put(
                [cache_key],
                [memory_obj],
                location=self._store_location,
            )
        except Exception as exc:  # noqa: BLE001  # third-party storage boundary
            logger.warning(
                "LMCache SLOT sidecar put failed error_type=%s",
                type(exc).__name__,
            )
            if memory_obj is not None:
                self._ref_count_down(memory_obj)
            return False

        # StorageManager owns and decrements the MemoryObj once batched_put
        # returns normally. Releasing it here would double-decrement the object.
        return True

    def get(self, key: SlotSidecarKey) -> torch.Tensor | None:
        """Blocking-read one sidecar and return an ownership-independent clone."""

        cache_key = self._cache_key(self._require_key(key))
        try:
            location = self._locate(cache_key)
        except RuntimeError:
            return None
        if location is None:
            return None
        try:
            memory_obj = self._storage_manager.get(cache_key, location=location)
        except Exception as exc:  # noqa: BLE001  # third-party storage boundary
            logger.warning(
                "LMCache SLOT sidecar get failed error_type=%s",
                type(exc).__name__,
            )
            return None

        if memory_obj is None:
            return None

        result = None
        try:
            tensor = self._validated_sidecar_tensor(memory_obj)
            result = tensor.reshape(-1).clone()
        except SlotSidecarCorruptionError:
            raise
        except Exception as exc:  # noqa: BLE001  # corrupt/storage-owned object
            logger.warning(
                "LMCache SLOT sidecar decode failed error_type=%s",
                type(exc).__name__,
            )
        finally:
            if not self._ref_count_down(memory_obj):
                result = None
        return result

    @contextmanager
    def borrow(self, key: SlotSidecarKey):
        """Yield the storage-owned tensor until the caller's H2D copy completes."""

        cache_key = self._cache_key(self._require_key(key))
        location = self._locate(cache_key)
        try:
            memory_obj = (
                None
                if location is None
                else self._storage_manager.get(cache_key, location=location)
            )
        except Exception as exc:  # noqa: BLE001  # third-party storage boundary
            raise RuntimeError("LMCache SLOT sidecar get failed") from exc
        if memory_obj is None:
            yield None
            return

        active_exception = False
        try:
            tensor = self._validated_sidecar_tensor(memory_obj)
            yield tensor.reshape(-1)
        except BaseException:
            # Preserve corruption/format errors thrown before or through the
            # yield so the connector can invalidate them. A release failure is
            # secondary and must not hide the self-healing signal.
            active_exception = True
            raise
        finally:
            if not self._ref_count_down(memory_obj) and not active_exception:
                raise RuntimeError("LMCache SLOT sidecar release failed")

    def contains(self, key: SlotSidecarKey) -> bool:
        """Return whether any configured LMCache storage tier contains the key.

        Backend probe failures raise a sanitized exception so publication
        polling fails immediately instead of warning and retrying in a loop.
        """

        cache_key = self._cache_key(self._require_key(key))
        return self._locate(cache_key) is not None

    def invalidate(self, key: SlotSidecarKey) -> bool:
        """Remove every stored copy so a recompute can republish this key.

        LMCache deliberately skips ``batched_put`` for an existing key. A
        corrupt sidecar therefore has to be evicted before the scheduler can
        self-heal it with a freshly computed snapshot.
        """

        sidecar_key = self._require_key(key)
        with self._corruption_lock:
            return self._invalidate_locked(sidecar_key)

    def _prepare_republication(self, key: SlotSidecarKey) -> bool:
        """Clear a prior corruption fence before accepting replacement bytes."""

        with self._corruption_lock:
            if key not in self._unresolved_corrupt_keys:
                return True
            return self._invalidate_locked(key)

    def _invalidate_locked(self, key: SlotSidecarKey) -> bool:
        # Fence before calling into the backend so every failure path remains
        # fail-closed. _corruption_lock serializes this with put's preflight.
        self._unresolved_corrupt_keys.add(key)
        cache_key = self._cache_key(key)
        try:
            removed = bool(self._storage_manager.remove(cache_key, locations=None))
        except Exception as exc:  # noqa: BLE001  # third-party storage boundary
            logger.warning(
                "LMCache SLOT sidecar invalidation failed error_type=%s",
                type(exc).__name__,
            )
            return False
        if removed:
            self._unresolved_corrupt_keys.discard(key)
            return True
        logger.warning("LMCache SLOT sidecar invalidation removed no stored copy")
        return False

    def _locate(self, cache_key) -> str | None:
        try:
            return self._storage_manager.contains(
                cache_key,
                search_range=self._retrieve_locations,
                pin=False,
            )
        except Exception as exc:  # noqa: BLE001  # third-party storage boundary
            raise RuntimeError("LMCache SLOT sidecar visibility probe failed") from exc

    def _cache_key(self, key: SlotSidecarKey):
        return self._cache_engine_key_type(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=self._worker_id,
            chunk_hash=key.storage_hash(),
            dtype=torch.uint8,
        )

    @staticmethod
    def _require_key(key: object) -> SlotSidecarKey:
        if not isinstance(key, SlotSidecarKey):
            raise TypeError("key must be a SlotSidecarKey")
        return key

    @staticmethod
    def _payload_tensor(blob: object) -> torch.Tensor:
        if isinstance(blob, torch.Tensor):
            if blob.dtype is not torch.uint8:
                raise ValueError("blob tensor must have dtype torch.uint8")
            if blob.device.type != "cpu":
                raise ValueError("blob tensor must be on the CPU")
            if not blob.is_contiguous():
                raise ValueError("blob tensor must be contiguous")
            if blob.numel() == 0:
                raise ValueError("blob must be nonempty")
            return blob.reshape(-1)

        try:
            view = memoryview(blob)
        except (TypeError, ValueError) as exc:
            raise TypeError("blob must be a torch.Tensor or bytes-like object") from exc
        if not view.c_contiguous:
            raise ValueError("blob bytes-like object must be contiguous")
        try:
            byte_view = (
                view if view.format == "B" and view.ndim == 1 else view.cast("B")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "blob must expose a contiguous byte representation"
            ) from exc
        if len(byte_view) == 0:
            raise ValueError("blob must be nonempty")

        # A writable snapshot avoids torch.frombuffer's warning for immutable
        # ``bytes`` while preserving the caller's opaque byte representation.
        return torch.frombuffer(bytearray(byte_view), dtype=torch.uint8)

    @staticmethod
    def _memory_tensor(memory_obj):
        tensor = getattr(memory_obj, "tensor", None)
        if tensor is None:
            get_tensor = getattr(memory_obj, "get_tensor", None)
            if callable(get_tensor):
                tensor = get_tensor(0)
        return tensor

    @classmethod
    def _validated_sidecar_tensor(cls, memory_obj) -> torch.Tensor:
        try:
            tensor = cls._memory_tensor(memory_obj)
        except Exception as exc:  # noqa: BLE001  # malformed third-party object
            raise SlotSidecarCorruptionError(
                "LMCache sidecar object did not expose a readable tensor"
            ) from exc
        if not isinstance(tensor, torch.Tensor):
            raise SlotSidecarCorruptionError(
                "LMCache sidecar object did not expose a tensor"
            )
        if tensor.dtype is not torch.uint8:
            raise SlotSidecarCorruptionError(
                "LMCache sidecar object must have dtype torch.uint8"
            )
        if tensor.device.type != "cpu":
            raise SlotSidecarCorruptionError(
                "LMCache sidecar object must be on the CPU"
            )
        if tensor.numel() == 0:
            raise SlotSidecarCorruptionError("LMCache sidecar object must be nonempty")
        if not tensor.is_contiguous():
            raise SlotSidecarCorruptionError(
                "LMCache sidecar object must be contiguous"
            )
        return tensor

    @staticmethod
    def _ref_count_down(memory_obj) -> bool:
        try:
            memory_obj.ref_count_down()
        except Exception as exc:  # noqa: BLE001  # third-party object lifetime
            logger.warning("LMCache SLOT sidecar release failed: %s", exc)
            return False
        return True
