"""Supply the shuffled-KV ``_rocm_C::zero_kv_blocks`` op from Python.

The MiniMax-M3 gluon/ASM attention path needs the KV cache physically laid out
in the ROCm SHUFFLE layout, which ATOM's backends only publish when
``VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`` is set. vLLM's ``KVBlockZeroer`` reacts to
that same env by switching from its general Triton zeroing kernel
(``_zero_kv_blocks_kernel``) to the native ``_rocm_C::zero_kv_blocks`` op:

    RuntimeError: ROCm shuffled KV cache requires the native
    _rocm_C::zero_kv_blocks op

That op lives in vLLM's own ``csrc/rocm/kv_cache_zero.cu`` but is not compiled
into the precompiled ``_rocm_C`` extension shipped with the M3-AMD build we run,
so shuffled-layout startup dies during warm-up.

Rather than patch ``KVBlockZeroer`` (which vLLM imports too early during plugin
registration for a reliable class-level monkeypatch), we register the missing op
into the ``_rocm_C`` namespace ourselves and back it with the Triton kernel that
already sits next to it in ``vllm.v1.worker.utils``. That kernel clears exactly
the bytes the native op would -- for each (block, segment) it zeroes
``page_size`` int32 elements at ``seg_addr + block_id * block_stride`` -- so
vLLM's native branch runs unchanged and correct, just Triton-launched.

Drop this patch once the M3-AMD build ships a ``_rocm_C`` with the native op.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("atom")

_applied = False
# Cache (blk_size, max_chunks) per segment-metadata tensor so the hot path does
# not force a device->host sync on every step. Keyed by the page-size tensor's
# storage pointer, which is stable for the lifetime of a KVBlockZeroer.
_tiling_cache: dict[tuple[int, int], tuple[int, int]] = {}


def _zero_kv_blocks_impl(seg_addrs, seg_block_strides, seg_page_sizes, block_ids):
    from vllm.v1.worker.utils import _zero_kv_blocks_kernel

    n_blocks = block_ids.numel()
    n_segs = seg_addrs.numel()
    if n_blocks == 0 or n_segs == 0:
        return

    key = (seg_page_sizes.data_ptr(), n_segs)
    tiling = _tiling_cache.get(key)
    if tiling is None:
        max_page = int(seg_page_sizes.max().item())
        if max_page <= 0:
            return
        blk_size = min(1 << (max_page - 1).bit_length(), 1024)
        max_chunks = (max_page + blk_size - 1) // blk_size
        tiling = (blk_size, max_chunks)
        _tiling_cache[key] = tiling
    blk_size, max_chunks = tiling

    grid = (n_blocks, n_segs, max_chunks)
    _zero_kv_blocks_kernel[grid](
        seg_addrs,
        seg_block_strides,
        seg_page_sizes,
        block_ids,
        BLOCK_SIZE=blk_size,
    )


def apply_vllm_kv_block_zeroer_patch() -> None:
    global _applied
    if _applied:
        return

    import torch

    if getattr(getattr(torch.ops, "_rocm_C", None), "zero_kv_blocks", None) is not None:
        # A build with the native op needs no help.
        _applied = True
        return

    lib = torch.library.Library("_rocm_C", "FRAGMENT")
    lib.define(
        "zero_kv_blocks(Tensor seg_addrs, Tensor seg_block_strides, "
        "Tensor seg_page_sizes, Tensor block_ids) -> ()"
    )
    lib.impl("zero_kv_blocks", _zero_kv_blocks_impl, "CUDA")
    # Zeroing runs during runner input-prep, never inside a captured/compiled
    # region, but register a Meta no-op so any fake-tensor pass stays happy.
    lib.impl("zero_kv_blocks", lambda *a: None, "Meta")

    # Keep a reference so the Library (and its registrations) outlive this call.
    global _KV_ZERO_LIB
    _KV_ZERO_LIB = lib
    _applied = True
    logger.info(
        "ATOM plugin: registered a Triton-backed _rocm_C::zero_kv_blocks op "
        "(native op absent in this build) for the shuffled-KV M3 gluon path."
    )
