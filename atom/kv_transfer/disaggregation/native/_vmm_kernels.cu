// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Device copy kernels for the fabric (scale-out / IFOE) path. A shader-driven
// copy resolves the fabric peer where hipMemcpy / SDMA livelock (and can wedge
// the GPU) on gfx1250. Compiled by hipcc (this is a .cu), unlike _vmm_ext.cpp
// which the host compiler builds.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#define KCK(expr)                                                              \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess)                                                      \
      throw std::runtime_error(std::string(#expr) + ": " +                     \
                               hipGetErrorString(_e));                         \
  } while (0)

__global__ void _copy16(uint4 *d, const uint4 *s, size_t n4) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += (size_t)gridDim.x * blockDim.x)
    d[i] = s[i];
}

__global__ void _copy1(char *d, const char *s, size_t off, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    d[off + i] = s[off + i];
}

// D2D copy via a kernel (safe on a fabric mapping). 16-byte bulk + byte tail.
void vmm_copy_kernel(int64_t dst_ptr, int64_t src_ptr, int64_t nbytes) {
  size_t nb = static_cast<size_t>(nbytes), n4 = nb / 16, tail = nb - n4 * 16;
  const int TH = 256;
  auto grid = [&](size_t n) {
    long g = (long)((n + TH - 1) / TH);
    return (int)(g < 1 ? 1 : (g > 2048 ? 2048 : g));
  };
  if (n4)
    _copy16<<<grid(n4), TH>>>(reinterpret_cast<uint4 *>(dst_ptr),
                              reinterpret_cast<const uint4 *>(src_ptr), n4);
  if (tail)
    _copy1<<<grid(tail), TH>>>(reinterpret_cast<char *>(dst_ptr),
                               reinterpret_cast<const char *>(src_ptr),
                               n4 * 16, tail);
  KCK(hipGetLastError());
  KCK(hipDeviceSynchronize());
}
