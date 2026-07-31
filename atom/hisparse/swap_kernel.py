# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""JIT-compiled HIP gather kernel for HiSparse swap-in.

A single GPU kernel gathers scattered top-k tokens directly from pinned host
memory (cold pool) into the GPU hot buffer over PCIe/XGMI — no staging buffer,
no cudaMemcpy. One wavefront64 copies one token's ``item_size_bytes`` word-wise,
mirroring SGLang's ``transfer_item_warp`` (ROCm branch) in ``hisparse.cuh``.

On this platform ``XNACK`` is disabled (``gfx950 ... xnack-``), so a GPU kernel
cannot dereference a raw host VA — it faults. The cold-pool pinned tensor must
first be translated with ``hipHostGetDevicePointer`` and the returned
device-mapped pointer passed to the kernel. ``host_get_device_pointer`` exposes
that translation; the coordinator caches the result once per cold-pool tensor.
"""

import torch

_SWAP_MODULE = None  # None = not attempted, False = unavailable, else module


# Written in CUDA idiom; torch's cpp_extension runs hipify on ROCm, translating
# cuda* -> hip* (including cudaHostGetDevicePointer -> hipHostGetDevicePointer).
_HIP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int WARP_SIZE = 64;  // wavefront64 on gfx950

// Word-wise warp copy, mirrors transfer_item_warp (ROCm branch) in hisparse.cuh
__device__ __forceinline__ void transfer_item_warp(
    int lane_id, const void* __restrict__ src_addr,
    void* __restrict__ dst_addr, int64_t item_size_bytes) {
  const auto* src = static_cast<const char*>(src_addr);
  auto* dst = static_cast<char*>(dst_addr);
  const int64_t word_count = item_size_bytes / (int64_t)sizeof(uint64_t);
  const auto* src_words = reinterpret_cast<const uint64_t*>(src);
  auto* dst_words = reinterpret_cast<uint64_t*>(dst);
  for (int64_t i = lane_id; i < word_count; i += WARP_SIZE) {
    dst_words[i] = src_words[i];  // reads pinned HOST memory over the bus
  }
  const int64_t tail = word_count * (int64_t)sizeof(uint64_t);
  for (int64_t i = tail + lane_id; i < item_size_bytes; i += WARP_SIZE) {
    dst[i] = src[i];
  }
}

// One warp per miss token. host_cache is a device-mapped host pointer.
__global__ void hisparse_gather_kernel(
    const char* __restrict__ host_cache,   // device-mapped pinned host memory
    char* __restrict__ device_buffer,      // device HBM
    const int32_t* __restrict__ src_locs,  // scattered host token slots
    const int32_t* __restrict__ dst_locs,  // scattered device buffer slots
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

// Translate a pinned host allocation to a device-mapped pointer (int64 VA).
// xnack- environment: the kernel faults on a raw host VA, so callers must feed
// the mapped pointer this returns instead of tensor.data_ptr().
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
    int64_t host_cache_dev_ptr,  // from host_get_device_pointer
    at::Tensor device_buffer,    // CUDA, contiguous, row = item_size_bytes
    at::Tensor src_locs,         // CUDA int32 [num_misses]
    at::Tensor dst_locs,         // CUDA int32 [num_misses]
    int64_t item_size_bytes) {
  const int num_misses = (int)src_locs.numel();
  if (num_misses == 0) return;
  TORCH_CHECK(device_buffer.is_cuda(), "device_buffer must be CUDA");
  TORCH_CHECK(src_locs.is_cuda() && dst_locs.is_cuda(),
              "src_locs/dst_locs must be CUDA");
  TORCH_CHECK(src_locs.scalar_type() == at::kInt &&
                  dst_locs.scalar_type() == at::kInt,
              "src_locs/dst_locs must be int32");
  TORCH_CHECK(dst_locs.numel() == num_misses, "src/dst length mismatch");

  const char* host_cache = reinterpret_cast<const char*>(host_cache_dev_ptr);
  char* dev_buf = reinterpret_cast<char*>(device_buffer.data_ptr());

  const int block = 256;  // 4 warps/block
  const int grid = (num_misses * WARP_SIZE + block - 1) / block;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  hisparse_gather_kernel<<<dim3(grid), dim3(block), 0, stream>>>(
      host_cache, dev_buf, src_locs.data_ptr<int32_t>(),
      dst_locs.data_ptr<int32_t>(), num_misses, item_size_bytes);
}
"""


# Explicit pybind glue (main.cpp, host c++). We bind the functions ourselves
# instead of via load_inline's `functions=` (which routes through
# torch::wrap_pybind_function and injects a cross-arg device guard that rejects
# our int-pointer + device-tensor signature). Forward-declare the definitions
# that live in the hipcc-compiled cuda_sources.
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


def _get_swap_module():
    """Lazily JIT-compile the HIP swap kernel. Returns the module or None."""
    global _SWAP_MODULE
    if _SWAP_MODULE is not None:
        return _SWAP_MODULE or None

    from torch.utils.cpp_extension import load_inline

    _SWAP_MODULE = load_inline(
        name="atom_hisparse_swap",
        cpp_sources=[_CPP_GLUE],
        cuda_sources=[_HIP_SOURCE],
        with_cuda=True,
        verbose=False,
    )
    return _SWAP_MODULE


def host_get_device_pointer(pinned_host_tensor: torch.Tensor) -> int:
    """Translate a pinned host tensor to a device-mapped pointer (int VA).

    Required before feeding a cold-pool pointer to :func:`hisparse_swap_in` on
    this xnack- platform. Cache the result — the mapping is stable for the
    lifetime of the allocation.
    """
    return _get_swap_module().host_get_device_pointer(pinned_host_tensor)


def hisparse_swap_in(
    host_cache_dev_ptr: int,
    device_buffer: torch.Tensor,
    src_locs: torch.Tensor,
    dst_locs: torch.Tensor,
    item_size_bytes: int,
) -> None:
    """Gather scattered tokens from pinned host cold pool into the GPU hot buffer.

    One wavefront64 per miss token, word-wise copy. Launches on the current HIP
    stream (CUDAGraph-capturable: fixed device_buffer address, no allocation).

    Args:
        host_cache_dev_ptr: device-mapped pointer for one layer's cold pool,
            from :func:`host_get_device_pointer`.
        device_buffer: one layer's GPU hot buffer, contiguous, each row spanning
            ``item_size_bytes``.
        src_locs: int32 CUDA tensor of cold-pool token slots to read.
        dst_locs: int32 CUDA tensor of hot-buffer slots to write (paired with
            ``src_locs``).
        item_size_bytes: bytes per token per layer (kv_dim * dtype.itemsize).
    """
    _get_swap_module().hisparse_swap_in(
        host_cache_dev_ptr, device_buffer, src_locs, dst_locs, item_size_bytes
    )
