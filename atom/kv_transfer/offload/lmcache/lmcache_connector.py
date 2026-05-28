# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Per-rank CPU offload backend for ATOM standalone serving.

Owns a single contiguous pinned CPU pool, allocated via
``lmcache.c_ops.alloc_pinned_ptr`` — direct ``cudaHostRegister`` on raw
``malloc()``-ed memory, bypassing PyTorch's pinned host allocator and its
power-of-two bucket rounding. ATOM's per-layer K/V tensors use AITER
swizzled layouts (K: ``(NB, NH, HD/x, BS, x)``, V: ``(NB, NH, HD, BS)``)
that match none of LMCache's ``GPUKVFormat`` enums, so D2H/H2D copies stay
on byte-view ``dst.copy_(src, non_blocking=True)`` over a dedicated copy
stream rather than ``lmcache.c_ops.multi_layer_kv_transfer``.

* Worker side: pinned CPU pool of N slots × sizeof(one paged block × all
  layers, K and V). ``register_kv_caches`` learns the per-layer geometry
  and (re)sizes the pool. ``start_load_kv`` is the single per-step entry
  the EngineCore dispatches to (matches the existing PD path).
* Scheduler side: an *optimistic* mirror of the worker's index. Entries
  appear when the scheduler queues a save; when the worker can't satisfy
  a load (pool eviction), it reports the request via ``get_finished``'s
  third return value ``failed_load`` and the scheduler drops the mirror
  entry plus re-prefills the request.

NVMe (L3) tier, real LRU eviction (vs FIFO), and full LMCacheEngine
integration are deferred — see ``atom_lmcache_integration_plan.md``.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from atom.kv_transfer.offload.base import (
    OffloadConnectorBase,
    OffloadConnectorSchedulerBase,
)
from atom.kv_transfer.offload.types import (
    OffloadConnectorMetadata,
    OffloadReqMeta,
)

if TYPE_CHECKING:
    from atom.config import Config
    from atom.model_engine.sequence import Sequence

logger = logging.getLogger("atom")

# Sentinel returned by lookup when nothing is queued — keep distinct from
# integer slot ids (which are >= 0).
_NO_SLOT = -1


def _check_environment() -> None:
    """Fail fast on misconfigurations that silently corrupt cache keys or
    fall back to a slow Python backend."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "LMCacheOffloadConnector requires PYTHONHASHSEED=0. "
            "Without it, TP-rank Python dict hashing diverges and the "
            "scheduler-side cache index disagrees with workers."
        )
    try:
        import lmcache  # noqa: F401
        from lmcache import c_ops  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "lmcache or lmcache.c_ops failed to import. Build LMCache from "
            "source with BUILD_WITH_HIP=1 — the PyPI wheel ships CUDA "
            "binaries that do not work on ROCm. See "
            "atom/kv_transfer/offload/README.md."
        ) from e
    # `lmcache.c_ops` exists iff the HIP/CUDA native extension built; the
    # Python fallback (`lmcache.python_ops_fallback`) is orders of magnitude
    # slower and must not be used for serving.
    backend_mod = getattr(__import__("lmcache"), "_backend_module", None)
    if backend_mod is not None and "c_ops" not in str(backend_mod):
        raise RuntimeError(
            f"LMCache fell back to Python backend {backend_mod!r}. Rebuild "
            "with BUILD_WITH_HIP=1 — see offload README."
        )


def _kv_transfer_extra(config: "Config") -> dict[str, Any]:
    """Extract the ``kv_transfer_config`` dict regardless of how it was
    supplied (dataclass with ``.get_from_extra_config`` vs plain dict via
    --kv-transfer-config JSON)."""
    cfg = getattr(config, "kv_transfer_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    return getattr(cfg, "extra_config", None) or {}


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


class LMCacheOffloadConnector(OffloadConnectorBase):
    """Per-TP-rank LMCache offload backend."""

    def __init__(self, config: "Config") -> None:
        _check_environment()
        extra = _kv_transfer_extra(config)
        # Per-rank CPU pool size. ``cpu_bytes`` is server-wide and split
        # across TP ranks below; ``cpu_bytes_per_rank`` overrides if set
        # (matches the SimpleCPUOffloadConnector vocabulary).
        world = max(1, getattr(config, "tensor_parallel_size", 1))
        cpu_bytes = int(extra.get("cpu_bytes", 8 * (1024**3)))
        self.cpu_bytes_per_rank = int(
            extra.get("cpu_bytes_per_rank", cpu_bytes // world)
        )
        self.disk_path = extra.get("disk_path", None)  # M5
        if self.disk_path:
            logger.warning(
                "LMCacheOffloadConnector: disk_path=%r set but NVMe tier "
                "is not implemented yet (M5). Ignoring.",
                self.disk_path,
            )

        # KV cache geometry filled in by register_kv_caches.
        self.kv_caches: dict[str, Any] | None = None
        self.num_layers: int = 0
        self.block_size = int(getattr(config, "kv_cache_block_size", 1))
        # Per layer: K block size in bytes + V block size in bytes. Each
        # K (resp. V) block is contiguous in GPU memory at offset
        # `block_id * bytes_per_block_per_layer_k` (resp. _v) into the
        # tensor's storage; the AITER-specific swizzled layout doesn't
        # affect byte-level offsets within a block.
        self._k_bytes_per_layer: list[int] = []
        self._v_bytes_per_layer: list[int] = []
        self.bytes_per_block: int = 0  # K + V across all layers

        # Flat byte views used for the copies — keyed by layer index.
        # Built once at register_kv_caches and reused per copy.
        self._k_uint8_views: list[Any] = []  # torch.Tensor of dtype=uint8
        self._v_uint8_views: list[Any] = []

        # CPU pool — allocated lazily in register_kv_caches once geometry
        # is known. The pool is a single contiguous pinned byte buffer of
        # size `num_slots * bytes_per_block` allocated via
        # ``c_ops.alloc_pinned_ptr`` (cudaHostRegister on raw malloc);
        # ``self._cpu_pool`` is a ``torch.frombuffer`` uint8 view over the
        # same memory, so ``dst.copy_(src, non_blocking=True)`` and offset
        # slicing keep working unchanged. Slot ``s`` occupies bytes
        # [s * bytes_per_block : (s+1) * bytes_per_block].
        self._cpu_pool: Any = None  # torch.Tensor view, dtype=uint8
        self._cpu_pool_base_ptr: int = 0  # raw ptr from c_ops, freed in close()
        self._cpu_pool_total_bytes: int = 0
        self._num_slots: int = 0
        # Hash -> slot index. OrderedDict so we can FIFO-evict on overflow.
        # (Real LRU comes with the LMCacheEngine wrapper in M5.)
        self.hash_to_slot: OrderedDict[int, int] = OrderedDict()
        self._free_slots: list[int] = []

        # Dedicated CUDA stream for D2H/H2D transfers — overlaps with the
        # compute stream's next-step work. Lazily created on first use so
        # we don't bind to a specific device at __init__ (rank may not be
        # initialized yet on the scheduler process).
        self._copy_stream: Any = None  # torch.cuda.Stream
        # Per-step pending completions: list of (req_id, kind, event).
        # kind in {"save", "load"}. Drained in get_finished.
        self._pending: list[tuple[str, str, Any]] = []

        # Per-step completion tracking (request_id sets).
        self._done_saving: set[str] = set()
        self._done_loading: set[str] = set()
        # Per-step load failures (request had a pool miss between the
        # scheduler's optimistic lookup and the worker's hash_to_slot
        # check). Drained by get_failed_load and surfaced to the
        # scheduler in KVConnectorOutput.failed_recving so the request
        # can be re-prefilled instead of running attention against an
        # uninitialized GPU block.
        self._failed_load: set[str] = set()

        logger.info(
            "LMCacheOffloadConnector initialized (rank-local pool=%.2f GB, "
            "disk_path=%s)",
            self.cpu_bytes_per_rank / (1024**3),
            self.disk_path,
        )

    # ------------------------------------------------------------------ #
    # KVConnectorBase methods
    # ------------------------------------------------------------------ #

    def register_kv_caches(
        self, kv_caches: dict[str, Any], transfer_tensors: Any = None
    ) -> None:
        """Capture per-layer KV tensor geometry and allocate the CPU pool.

        ATOM's per-layer K and V tensors don't match any of LMCache's
        ``GPUKVFormat`` variants (AITER swizzles K as
        ``(NB, NH, HD/x, BS, x)`` and V as ``(NB, NH, HD, BS)``), so we
        bypass ``c_ops.multi_layer_block_kv_transfer`` and do byte-level
        async copies via ``torch.Tensor.copy_(non_blocking=True)``. The
        underlying storage of each block is contiguous (allocated by
        ``torch.zeros``), so byte views with reinterpret-cast strides
        work fine — only the within-block layout is unusual.
        """
        import torch

        self.kv_caches = kv_caches
        self.num_layers = len(kv_caches)
        if self.num_layers == 0:
            logger.warning(
                "register_kv_caches received empty kv_caches; offload disabled."
            )
            return

        self._k_bytes_per_layer = []
        self._v_bytes_per_layer = []
        self._k_uint8_views = []
        self._v_uint8_views = []
        bytes_per_block = 0
        for name, kvt in kv_caches.items():
            k = kvt.k_cache
            v = kvt.v_cache
            if not k.is_cuda or not k.is_contiguous():
                logger.warning(
                    "Layer %s K cache not on CUDA or non-contiguous "
                    "(device=%s, contig=%s); offload disabled.",
                    name,
                    k.device,
                    k.is_contiguous(),
                )
                self._num_slots = 0
                return
            if v is None:
                # MLA registers a token-major K-only cache shaped like
                # [num_blocks * block_size, 1, hidden]. A logical paged block
                # spans block_size consecutive token rows.
                k_bytes = self.block_size * k.stride(0) * k.element_size()
                v_bytes = 0
            else:
                k_bytes = (k.numel() // k.shape[0]) * k.element_size()
                v_bytes = (
                    (v.numel() // v.shape[0]) * v.element_size() if v.numel() > 0 else 0
                )
            self._k_bytes_per_layer.append(k_bytes)
            self._v_bytes_per_layer.append(v_bytes)
            # Flatten then reinterpret as bytes so block_id offsets are
            # byte-linear: `flat[bid * k_bytes : (bid+1) * k_bytes]`.
            # `.view(-1)` requires contiguity (checked above).
            self._k_uint8_views.append(k.view(-1).view(torch.uint8))
            self._v_uint8_views.append(
                v.view(-1).view(torch.uint8) if v_bytes > 0 else None
            )
            bytes_per_block += k_bytes + v_bytes

        self.bytes_per_block = bytes_per_block
        if self.bytes_per_block == 0:
            logger.warning("bytes_per_block == 0; offload disabled.")
            self._num_slots = 0
            return

        self._num_slots = max(1, self.cpu_bytes_per_rank // self.bytes_per_block)
        self._cpu_pool_total_bytes = self._num_slots * self.bytes_per_block
        # Allocate via lmcache.c_ops: a raw malloc() + cudaHostRegister on
        # exactly self._cpu_pool_total_bytes bytes. This avoids PyTorch's
        # pinned host allocator (which buckets large requests to the next
        # power of two and would lock ~128 GB to satisfy a ~100 GB pool).
        # We then wrap the raw pointer in a uint8 torch.Tensor view so
        # ``dst.copy_(src, non_blocking=True)`` works unchanged. Lifetime:
        # the buffer is freed in close()/__del__ via c_ops.free_pinned_ptr;
        # torch.frombuffer does not own the memory.
        import ctypes

        from lmcache import c_ops

        # c_ops wants an int device id for cudaHostRegister's flags lookup.
        # A real torch.Tensor.device is a torch.device with int .index; fake
        # tensors in tests use a string ("cuda:0"). Fall back to 0 if we
        # can't extract a usable int — the registration works fine on any
        # device, the id is only a hint to the driver.
        view_device = self._k_uint8_views[0].device if self._k_uint8_views else None
        device_id = getattr(view_device, "index", None)
        if not isinstance(device_id, int):
            device_id = 0
        self._cpu_pool_base_ptr = int(
            c_ops.alloc_pinned_ptr(self._cpu_pool_total_bytes, device_id)
        )
        if self._cpu_pool_base_ptr == 0:
            raise RuntimeError(
                "c_ops.alloc_pinned_ptr returned NULL — out of pinnable "
                "host memory? Try lowering kv_transfer_config.cpu_bytes / "
                "cpu_bytes_per_rank."
            )
        buf = (ctypes.c_uint8 * self._cpu_pool_total_bytes).from_address(
            self._cpu_pool_base_ptr
        )
        self._cpu_pool = torch.frombuffer(buf, dtype=torch.uint8)
        self._free_slots = list(range(self._num_slots))
        first_layer_name = next(iter(kv_caches.keys()))
        logger.info(
            "LMCacheOffload: %d slots × %d bytes (pool=%.2f GB, "
            "num_layers=%d, bytes/block=%d, first_layer=%s, k_bytes[0]=%d, "
            "v_bytes[0]=%d)",
            self._num_slots,
            self.bytes_per_block,
            self._cpu_pool_total_bytes / (1024**3),
            self.num_layers,
            self.bytes_per_block,
            first_layer_name,
            self._k_bytes_per_layer[0],
            self._v_bytes_per_layer[0],
        )

    def _ensure_copy_stream(self, device):
        """Lazy-create the copy stream on the GPU device of the first
        registered KV layer. Workers initialize cuda after engine setup,
        so we can't bind in __init__.
        """
        import torch

        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=device)

    def start_load_kv(self, metadata: OffloadConnectorMetadata) -> None:
        """Issue async D2H saves for this step.

        Per save block:
          1. Claim a CPU slot (FIFO-evict if pool full).
          2. For each layer, copy K bytes then V bytes from the GPU paged
             block into the slot's region of the pinned CPU pool, using
             ``torch.copy_(non_blocking=True)`` on a dedicated CUDA stream.
          3. After all per-req copies, record a CUDA event on the stream
             and stash ``(req_id, "save", event)``. ``get_finished`` polls
             events and reports completed req_ids.

        Loads remain stubbed (loads metadata always empty in this commit —
        ``LMCacheOffloadConnectorScheduler.get_num_new_matched_tokens``
        returns 0). Enabled in the follow-up alongside BlockManager
        binding so HBM hits aren't double-counted.
        """
        import torch

        if metadata is None or metadata.is_empty():
            return
        if self._num_slots == 0 or self._cpu_pool is None:
            logger.warning(
                "LMCacheOffload start_load_kv before pool init "
                "(saves=%d, loads=%d) — dropping.",
                len(metadata.reqs_to_save),
                len(metadata.reqs_to_load),
            )
            self._done_saving.update(metadata.reqs_to_save.keys())
            self._done_loading.update(metadata.reqs_to_load.keys())
            return

        # All layers live on the same device; sample the first.
        device = self._k_uint8_views[0].device
        self._ensure_copy_stream(device)
        # Make our copy stream wait for whatever produced the K/V (the
        # default compute stream that ran the forward pass). Without this
        # sync the D2H reads can race with the still-in-flight attention
        # write of the same paged block.
        self._copy_stream.wait_stream(torch.cuda.current_stream(device))

        total_save_blocks = sum(
            len(m.block_ids) for m in metadata.reqs_to_save.values()
        )
        if total_save_blocks:
            logger.debug(
                "LMCacheOffload start_load_kv: saves=%d blocks across %d reqs (pool %d/%d slots in use)",
                total_save_blocks,
                len(metadata.reqs_to_save),
                len(self.hash_to_slot),
                self._num_slots,
            )
        with torch.cuda.stream(self._copy_stream):
            for req_id, meta in metadata.reqs_to_save.items():
                for block_id, h in zip(meta.block_ids, meta.block_hashes):
                    # Slot claim (skip if already cached — same hash chain
                    # re-published is a no-op).
                    if h in self.hash_to_slot:
                        continue
                    if not self._free_slots:
                        evict_hash, evict_slot = self.hash_to_slot.popitem(last=False)
                        self._free_slots.append(evict_slot)
                    slot = self._free_slots.pop(0)
                    self.hash_to_slot[h] = slot
                    # Per-layer K then V copy into the slot's byte region.
                    cpu_offset = slot * self.bytes_per_block
                    for layer_idx in range(self.num_layers):
                        k_view = self._k_uint8_views[layer_idx]
                        k_bytes = self._k_bytes_per_layer[layer_idx]
                        v_view = self._v_uint8_views[layer_idx]
                        v_bytes = self._v_bytes_per_layer[layer_idx]
                        # K block: flat byte view of k_cache[block_id].
                        src_k = k_view[block_id * k_bytes : (block_id + 1) * k_bytes]
                        dst_k = self._cpu_pool[cpu_offset : cpu_offset + k_bytes]
                        dst_k.copy_(src_k, non_blocking=True)
                        cpu_offset += k_bytes
                        if v_bytes > 0:
                            src_v = v_view[
                                block_id * v_bytes : (block_id + 1) * v_bytes
                            ]
                            dst_v = self._cpu_pool[cpu_offset : cpu_offset + v_bytes]
                            dst_v.copy_(src_v, non_blocking=True)
                            cpu_offset += v_bytes
                # One event per request — coarser than per-block but the
                # scheduler granularity is per-request anyway.
                evt = torch.cuda.Event()
                evt.record(self._copy_stream)
                self._pending.append((req_id, "save", evt))

        # H2D loads — issued on the same stream after the saves above so
        # any save→load same-step ordering is respected naturally.
        if metadata.reqs_to_load:
            total_load_blocks = sum(
                len(m.block_ids) for m in metadata.reqs_to_load.values()
            )
            logger.debug(
                "LMCacheOffload start_load_kv: loads=%d blocks across %d reqs (pool %d/%d slots in use)",
                total_load_blocks,
                len(metadata.reqs_to_load),
                len(self.hash_to_slot),
                self._num_slots,
            )
            with torch.cuda.stream(self._copy_stream):
                for req_id, meta in metadata.reqs_to_load.items():
                    # Pre-scan: if ANY block in this req's chain is no longer
                    # in the worker pool (FIFO-evicted between the scheduler's
                    # optimistic lookup and now), the whole request fails the
                    # load. We do NOT enqueue partial H2D copies — the
                    # corresponding GPU blocks would still be uninitialized
                    # for the missed positions and attention would read
                    # garbage. Instead surface req_id in self._failed_load
                    # so the scheduler drops the mirror entry and re-prefills.
                    missing_hashes = [
                        h for h in meta.block_hashes if self.hash_to_slot.get(h, -1) < 0
                    ]
                    if missing_hashes:
                        logger.warning(
                            "LMCacheOffload: load FAILED for req=%s — "
                            "%d/%d blocks evicted from CPU pool; re-prefill "
                            "via scheduler failed_recving.",
                            req_id,
                            len(missing_hashes),
                            len(meta.block_hashes),
                        )
                        self._failed_load.add(req_id)
                        continue
                    for block_id, h in zip(meta.block_ids, meta.block_hashes):
                        slot = self.hash_to_slot[h]
                        cpu_offset = slot * self.bytes_per_block
                        for layer_idx in range(self.num_layers):
                            k_view = self._k_uint8_views[layer_idx]
                            k_bytes = self._k_bytes_per_layer[layer_idx]
                            v_view = self._v_uint8_views[layer_idx]
                            v_bytes = self._v_bytes_per_layer[layer_idx]
                            dst_k = k_view[
                                block_id * k_bytes : (block_id + 1) * k_bytes
                            ]
                            src_k = self._cpu_pool[cpu_offset : cpu_offset + k_bytes]
                            dst_k.copy_(src_k, non_blocking=True)
                            cpu_offset += k_bytes
                            if v_bytes > 0:
                                dst_v = v_view[
                                    block_id * v_bytes : (block_id + 1) * v_bytes
                                ]
                                src_v = self._cpu_pool[
                                    cpu_offset : cpu_offset + v_bytes
                                ]
                                dst_v.copy_(src_v, non_blocking=True)
                                cpu_offset += v_bytes
                    evt = torch.cuda.Event()
                    evt.record(self._copy_stream)
                    self._pending.append((req_id, "load", evt))
            # No reverse barrier here: the request that needs this H2D
            # load is parked in WAITING_FOR_REMOTE_KVS and only re-enters
            # the next forward batch after get_finished() reports its
            # load event done (via finished_recving). The current batch
            # never reads load-target GPU blocks, so a stream-wide fence
            # would only block unrelated forward work. Per-load CUDA
            # events + next-step scheduling is the correctness contract.

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Drain completed CUDA events into the done sets, then return
        them (and clear).
        """
        if self._pending:
            still_pending: list[tuple[str, str, Any]] = []
            for req_id, kind, evt in self._pending:
                if evt.query():
                    if kind == "save":
                        self._done_saving.add(req_id)
                    else:
                        self._done_loading.add(req_id)
                else:
                    still_pending.append((req_id, kind, evt))
            self._pending = still_pending
        done_save = self._done_saving
        done_load = self._done_loading
        self._done_saving = set()
        self._done_loading = set()
        return done_save, done_load

    def get_failed_load(self) -> set[str]:
        """Drain accumulated load-failures and return; called by the
        worker once per step right after ``get_finished``."""
        failed = self._failed_load
        self._failed_load = set()
        return failed

    def close(self) -> None:
        """Release the cudaHostRegister'd CPU pool. Safe to call twice.

        Drop the torch view first so the pool's refcount goes to zero
        before unpinning the underlying buffer — otherwise PyTorch could
        access freed pinned memory during gc.
        """
        if self._cpu_pool is not None:
            self._cpu_pool = None
        if self._cpu_pool_base_ptr:
            from lmcache import c_ops

            c_ops.free_pinned_ptr(self._cpu_pool_base_ptr)
            self._cpu_pool_base_ptr = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 — destructors must not raise
            pass


# ---------------------------------------------------------------------------
# Scheduler side
# ---------------------------------------------------------------------------


class LMCacheOffloadConnectorScheduler(OffloadConnectorSchedulerBase):
    """Scheduler-side companion of :class:`LMCacheOffloadConnector`.

    Maintains an *optimistic* set of block hashes that have been queued for
    save. A `queue_save` immediately marks the hash as available; if a
    subsequent load actually misses on the worker, the connector reports it
    via :meth:`OffloadConnectorBase.get_finished` and we evict from the
    mirror.
    """

    def __init__(self, config: "Config") -> None:
        _check_environment()
        extra = _kv_transfer_extra(config)
        self.chunk_size = int(extra.get("chunk_size", 256))
        # Optimistic mirror — set of block hashes the worker is believed
        # to have. Bounded only by total prefill block count across the
        # run; trimmed as worker reports load misses.
        self.saved_hashes: set[int] = set()
        # Per-step accumulator drained by build_connector_meta.
        self._pending_save: dict[str, OffloadReqMeta] = {}
        self._pending_load: dict[str, OffloadReqMeta] = {}
        # Set by Scheduler.__init__ via bind_block_manager.
        self.block_manager: Any = None
        # Per-seq stash between get_num_new_matched_tokens and
        # update_state_after_alloc, keyed by `id(seq)`.
        self._external_match_stash: dict[int, list[int]] = {}
        # Per-request memory of the external hashes that were queued for
        # load via update_state_after_alloc, keyed by str(seq.id). Used by
        # handle_failed_load to drop the corresponding mirror entries
        # when the worker reports the load could not be satisfied.
        # Cleared on successful finished_recving (in request_finished) or
        # on handle_failed_load.
        self._external_hashes_by_req: dict[str, list[int]] = {}
        logger.info(
            "LMCacheOffloadConnectorScheduler initialized (chunk_size=%d)",
            self.chunk_size,
        )

    # ------------------------------------------------------------------ #
    # KVConnectorSchedulerBase methods (PD-style names; OFFLOAD semantics)
    # ------------------------------------------------------------------ #

    def get_num_new_matched_tokens(self, seq: "Sequence") -> tuple[int, bool]:
        """Return ``(tokens_in_external_store, async_load_required)``.

        Walks the seq's prompt hash chain (same xxhash chain as
        ``BlockManager.compute_hash``) and finds the longest *contiguous*
        prefix where either:

        * The block is an HBM prefix-cache hit (already free, no work), OR
        * The block is in the external store (needs H2D load).

        Returns the count of *external-only* tokens (HBM hits subtracted)
        plus ``async=True`` so the scheduler parks the seq in
        ``WAITING_FOR_REMOTE_KVS`` until the worker confirms the load.

        Returns ``(0, False)`` if the connector isn't ready (no block
        manager bound, no external hits, etc.).
        """
        if self.block_manager is None or seq.num_blocks <= 1:
            return 0, False
        if not self.block_manager.enable_prefix_caching:
            # No prefix cache means no HBM dedup possible AND no hash
            # chain — fall through to plain prefill.
            return 0, False
        from atom.model_engine.block_manager import BlockManager

        h = -1
        external_hashes: list[int] = []
        n_hbm = 0
        n_ext = 0
        # Same loop bound as BlockManager.can_allocate: skip the last
        # block (always allocated fresh; prefill needs to forward at
        # least one block for sampler logits).
        state = "hbm"
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = BlockManager.compute_hash(token_ids, h)
            if state == "hbm":
                hit_bid = self.block_manager.hash_to_block_id.get(h, -1)
                if hit_bid != -1 and (
                    self.block_manager.blocks[hit_bid].token_ids == token_ids
                ):
                    n_hbm += 1
                    continue
                state = "ext"
            # state == "ext"
            if h in self.saved_hashes:
                n_ext += 1
                external_hashes.append(h)
            else:
                break  # chain broke; nothing after this can hit either

        if n_ext == 0:
            return 0, False
        # Stash for update_state_after_alloc to consume.
        self._external_match_stash[id(seq)] = external_hashes
        return n_ext * seq.block_size, True

    def update_state_after_alloc(self, seq: "Sequence") -> None:
        """Enqueue the H2D load tasks for the external-prefix slots
        BlockManager just allocated.

        BlockManager.allocate has already laid out ``seq.block_table``:
        first ``n_hbm`` entries are HBM-cached blocks (reused), the rest
        are fresh slots from the free pool. External-store loads target
        ``seq.block_table[n_hbm : n_hbm + n_ext]`` — the freshly-allocated
        slots that correspond to the external-only prefix prefix.
        """
        stashed = self._external_match_stash.pop(id(seq), None)
        if stashed is None:
            return
        external_hashes = stashed
        n_ext = len(external_hashes)
        n_hbm = seq.num_cached_blocks
        dest_block_ids = list(seq.block_table[n_hbm : n_hbm + n_ext])
        if len(dest_block_ids) != n_ext:
            # block_table is shorter than expected (allocation deviated
            # from what we predicted). Skip the load — seq will fall back
            # to full prefill of those blocks; correctness preserved.
            logger.warning(
                "LMCacheOffload: dest_block_ids len %d != n_ext %d "
                "(n_hbm=%d, num_blocks=%d) — skipping load",
                len(dest_block_ids),
                n_ext,
                n_hbm,
                seq.num_blocks,
            )
            return
        req_id = str(seq.id)
        self._pending_load[req_id] = OffloadReqMeta(
            block_ids=dest_block_ids, block_hashes=external_hashes
        )
        # Remember the hashes so handle_failed_load can drop them from
        # the mirror if the worker can't satisfy this load.
        self._external_hashes_by_req[req_id] = list(external_hashes)

    def lookup_external_hits(self, seq: "Sequence", block_hashes: list[int]) -> int:
        """Return the longest prefix of ``block_hashes`` known to the
        optimistic mirror. Bounded by the chain rule — stop at first miss.

        Currently unused: :meth:`get_num_new_matched_tokens` returns 0 in
        this first commit (load path disabled until BlockManager binding
        and real c_ops H2D land in the follow-up). Keeping the method
        implemented makes the class instantiable and primes the lookup
        logic for that commit.
        """
        n = 0
        for h in block_hashes:
            if h in self.saved_hashes:
                n += 1
            else:
                break
        return n

    def queue_save(self, request_id: str, published: list[tuple[int, int]]) -> None:
        if not published:
            return
        block_ids = [bid for bid, _ in published]
        block_hashes = [h for _, h in published]
        # Mark optimistically saved so subsequent lookups find it.
        self.saved_hashes.update(block_hashes)
        # Accumulate; one request may publish across multiple steps.
        meta = self._pending_save.setdefault(request_id, OffloadReqMeta())
        meta.block_ids.extend(block_ids)
        meta.block_hashes.extend(block_hashes)
        logger.debug(
            "LMCacheOffload queue_save req=%s n_new=%d total_mirror=%d",
            request_id,
            len(published),
            len(self.saved_hashes),
        )

    def queue_load(
        self, request_id: str, block_ids: list[int], block_hashes: list[int]
    ) -> None:
        self._pending_load[request_id] = OffloadReqMeta(
            block_ids=list(block_ids), block_hashes=list(block_hashes)
        )

    def build_connector_meta(self) -> OffloadConnectorMetadata | None:
        if not self._pending_save and not self._pending_load:
            return None
        meta = OffloadConnectorMetadata()
        meta.reqs_to_save = self._pending_save
        meta.reqs_to_load = self._pending_load
        self._pending_save = {}
        self._pending_load = {}
        return meta

    def request_finished(self, seq: "Sequence") -> None:
        # OFFLOAD: hash_blocks already queued saves block-by-block during
        # postprocess; the scheduler defers block free while any queued
        # async save for this request is still pending. Drop any
        # outstanding load-recovery stash for this req — the load either
        # succeeded (req moved out of WAITING_FOR_REMOTE_KVS) or failed
        # and was already handled via handle_failed_load.
        self._external_hashes_by_req.pop(str(seq.id), None)

    def handle_failed_load(self, request_id: str) -> list[int]:
        """Worker reported the H2D load for ``request_id`` could not be
        satisfied (pool eviction). Drop the corresponding mirror entries
        so a future ``get_num_new_matched_tokens`` does not re-hit the
        same stale hashes, and clear the per-request stash.

        Returns the evicted hashes for logging.
        """
        hashes = self._external_hashes_by_req.pop(request_id, [])
        for h in hashes:
            self.saved_hashes.discard(h)
        return hashes
