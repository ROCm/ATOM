# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""HiSparse swap-in kernels — thin shim over the aiter ``module_hisparse_swap`` op.

The fused decode hot path (miss-detect + LRU evict + swap + translate) and the
new-token backup live in aiter (``aiter/csrc/py_itfs_cu/hisparse_swap_kernels.cu``),
JIT-compiled and cached on first call. This module re-exports the four ops so the
coordinator imports from one place, and keeps a ``load_inline`` fallback for the
two simple functions (device-pointer translation + plain gather) when the aiter
build is unavailable.

On this platform ``XNACK`` is disabled (``gfx950 ... xnack-``), so a GPU kernel
cannot dereference a raw host VA. Cold-pool pointers must be translated with
``host_get_device_pointer`` (``hipHostGetDevicePointer``) and the mapped pointer
passed to the kernels; the coordinator caches the result once per cold pool.
"""

import torch

_FALLBACK_MODULE = None  # None = not attempted, False = unavailable, else module


def _aiter():
    """Return the aiter hisparse ops module, or None if aiter is unavailable."""
    try:
        import aiter

        return aiter
    except ImportError:
        return None


# Fallback: the Phase-0 load_inline gather (device-ptr translation + plain gather).
# Only used if aiter cannot be imported. Written in CUDA idiom; torch's
# cpp_extension runs hipify on ROCm (cuda* -> hip*).
_HIP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int WARP_SIZE = 64;

__device__ __forceinline__ void transfer_item_warp(
    int lane_id, const void* __restrict__ src_addr,
    void* __restrict__ dst_addr, int64_t item_size_bytes) {
  const auto* src = static_cast<const char*>(src_addr);
  auto* dst = static_cast<char*>(dst_addr);
  const int64_t word_count = item_size_bytes / (int64_t)sizeof(uint64_t);
  const auto* src_words = reinterpret_cast<const uint64_t*>(src);
  auto* dst_words = reinterpret_cast<uint64_t*>(dst);
  for (int64_t i = lane_id; i < word_count; i += WARP_SIZE) {
    dst_words[i] = src_words[i];
  }
  const int64_t tail = word_count * (int64_t)sizeof(uint64_t);
  for (int64_t i = tail + lane_id; i < item_size_bytes; i += WARP_SIZE) {
    dst[i] = src[i];
  }
}

__global__ void hisparse_gather_kernel(
    const char* __restrict__ host_cache, char* __restrict__ device_buffer,
    const int32_t* __restrict__ src_locs, const int32_t* __restrict__ dst_locs,
    int num_misses, int64_t item_size_bytes) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;
  for (int m = warp_id; m < num_misses; m += num_warps) {
    const int64_t src = (int64_t)src_locs[m] * item_size_bytes;
    const int64_t dst = (int64_t)dst_locs[m] * item_size_bytes;
    transfer_item_warp(lane_id, host_cache + src, device_buffer + dst,
                       item_size_bytes);
  }
}

int64_t host_get_device_pointer(at::Tensor pinned_host_tensor) {
  TORCH_CHECK(pinned_host_tensor.is_pinned(),
              "host_get_device_pointer: tensor must be pinned host memory");
  void* host_ptr = pinned_host_tensor.data_ptr();
  void* dev_ptr = nullptr;
  cudaError_t e = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
  TORCH_CHECK(e == cudaSuccess,
              "cudaHostGetDevicePointer failed: ", cudaGetErrorString(e));
  return reinterpret_cast<int64_t>(dev_ptr);
}

void hisparse_swap_in(
    int64_t host_cache_dev_ptr, at::Tensor device_buffer,
    at::Tensor src_locs, at::Tensor dst_locs, int64_t item_size_bytes) {
  const int num_misses = (int)src_locs.numel();
  if (num_misses == 0) return;
  const char* host_cache = reinterpret_cast<const char*>(host_cache_dev_ptr);
  char* dev_buf = reinterpret_cast<char*>(device_buffer.data_ptr());
  const int block = 256;
  const int grid = (num_misses * WARP_SIZE + block - 1) / block;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  hisparse_gather_kernel<<<dim3(grid), dim3(block), 0, stream>>>(
      host_cache, dev_buf, src_locs.data_ptr<int32_t>(),
      dst_locs.data_ptr<int32_t>(), num_misses, item_size_bytes);
}
"""

_CPP_GLUE = r"""
#include <torch/extension.h>
#include <cstdint>
int64_t host_get_device_pointer(at::Tensor pinned_host_tensor);
void hisparse_swap_in(int64_t host_cache_dev_ptr, at::Tensor device_buffer,
                      at::Tensor src_locs, at::Tensor dst_locs,
                      int64_t item_size_bytes);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("host_get_device_pointer", &host_get_device_pointer);
  m.def("hisparse_swap_in", &hisparse_swap_in);
}
"""


def _fallback():
    """Lazily JIT-compile the load_inline fallback module (or None)."""
    global _FALLBACK_MODULE
    if _FALLBACK_MODULE is not None:
        return _FALLBACK_MODULE or None
    from torch.utils.cpp_extension import load_inline

    _FALLBACK_MODULE = load_inline(
        name="atom_hisparse_swap_fallback",
        cpp_sources=[_CPP_GLUE],
        cuda_sources=[_HIP_SOURCE],
        with_cuda=True,
        verbose=False,
    )
    return _FALLBACK_MODULE


def host_get_device_pointer(pinned_host_tensor: torch.Tensor) -> int:
    """Translate a pinned host tensor to a device-mapped pointer (int VA).

    Required before feeding a cold-pool pointer to the swap kernels on this
    xnack- platform. Cache the result — the mapping is stable for the allocation.
    """
    a = _aiter()
    if a is not None:
        return a.hisparse_host_get_device_pointer(pinned_host_tensor)
    return _fallback().host_get_device_pointer(pinned_host_tensor)


def hisparse_swap_in(
    host_cache_dev_ptr: int,
    device_buffer: torch.Tensor,
    src_locs: torch.Tensor,
    dst_locs: torch.Tensor,
    item_size_bytes: int,
) -> None:
    """Gather scattered tokens from the pinned host cold pool into the hot buffer.

    One wavefront64 per miss token, word-wise copy, on the current HIP stream.
    """
    a = _aiter()
    if a is not None:
        a.hisparse_swap_in(
            host_cache_dev_ptr,
            device_buffer,
            src_locs.to(torch.int32),
            dst_locs.to(torch.int32),
            item_size_bytes,
        )
        return
    _fallback().hisparse_swap_in(
        host_cache_dev_ptr,
        device_buffer,
        src_locs.to(torch.int32),
        dst_locs.to(torch.int32),
        item_size_bytes,
    )


def hisparse_swap_and_translate(
    cold_pool_dev_ptr: int,
    hot_buffer: torch.Tensor,
    topk_logical: torch.Tensor,
    indptr: torch.Tensor,
    req_slots: torch.Tensor,
    slot_token: torch.Tensor,
    last_used: torch.Tensor,
    token_to_slot: torch.Tensor,
    recency: torch.Tensor,
    out_translated: torch.Tensor,
    item_size_bytes: int,
    hot_slots: int,
    cold_depth: int,
    topk: int,
) -> None:
    """Fused per-layer decode hot path (aiter). See the aiter op for semantics."""
    a = _aiter()
    if a is None:
        raise RuntimeError(
            "hisparse_swap_and_translate requires the aiter module_hisparse_swap "
            "op; aiter could not be imported."
        )
    a.hisparse_swap_and_translate(
        cold_pool_dev_ptr,
        hot_buffer,
        topk_logical,
        indptr,
        req_slots,
        slot_token,
        last_used,
        token_to_slot,
        recency,
        out_translated,
        item_size_bytes,
        hot_slots,
        cold_depth,
        topk,
    )


def hisparse_backup_new_token(
    cold_pool_dev_ptr: int,
    hot_buffer: torch.Tensor,
    layer_kv: torch.Tensor,
    src_slots: torch.Tensor,
    req_slots: torch.Tensor,
    logical_pos: torch.Tensor,
    slot_token: torch.Tensor,
    last_used: torch.Tensor,
    token_to_slot: torch.Tensor,
    recency: torch.Tensor,
    item_size_bytes: int,
    hot_slots: int,
    cold_depth: int,
) -> None:
    """Batched new-token backup for one layer (aiter). See the aiter op."""
    a = _aiter()
    if a is None:
        raise RuntimeError(
            "hisparse_backup_new_token requires the aiter module_hisparse_swap "
            "op; aiter could not be imported."
        )
    a.hisparse_backup_new_token(
        cold_pool_dev_ptr,
        hot_buffer,
        layer_kv,
        src_slots,
        req_slots,
        logical_pos,
        slot_token,
        last_used,
        token_to_slot,
        recency,
        item_size_bytes,
        hot_slots,
        cold_depth,
    )
