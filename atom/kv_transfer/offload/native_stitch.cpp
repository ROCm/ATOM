// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <vector>

void stitch_chunk_buffers(
    torch::Tensor dst,
    std::vector<torch::Tensor> chunk_buffers,
    std::vector<int64_t> chunk_block_counts,
    std::vector<int64_t> seg_block_bytes) {
  TORCH_CHECK(dst.device().is_cpu(), "dst must be a CPU tensor");
  TORCH_CHECK(dst.dtype() == torch::kUInt8, "dst must be uint8");
  TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
  TORCH_CHECK(
      chunk_buffers.size() == chunk_block_counts.size(),
      "chunk_buffers and chunk_block_counts size mismatch");

  const int64_t nchunks = static_cast<int64_t>(chunk_buffers.size());
  const int64_t nsegs = static_cast<int64_t>(seg_block_bytes.size());
  int64_t total_blocks = 0;
  for (int64_t nblocks : chunk_block_counts) {
    TORCH_CHECK(nblocks >= 0, "chunk block count must be non-negative");
    total_blocks += nblocks;
  }

  std::vector<int64_t> dst_bases(nsegs);
  int64_t acc = 0;
  for (int64_t seg = 0; seg < nsegs; ++seg) {
    const int64_t nb = seg_block_bytes[seg];
    TORCH_CHECK(nb >= 0, "segment byte count must be non-negative");
    dst_bases[seg] = acc;
    acc += nb * total_blocks;
  }
  TORCH_CHECK(dst.numel() >= acc, "dst is smaller than stitched output");

  std::vector<const uint8_t*> src_ptrs(nchunks);
  std::vector<int64_t> src_offsets(nchunks * nsegs);
  for (int64_t c = 0; c < nchunks; ++c) {
    const auto& src = chunk_buffers[c];
    TORCH_CHECK(src.device().is_cpu(), "chunk buffer must be a CPU tensor");
    TORCH_CHECK(src.dtype() == torch::kUInt8, "chunk buffer must be uint8");
    TORCH_CHECK(src.is_contiguous(), "chunk buffer must be contiguous");
    src_ptrs[c] = src.data_ptr<uint8_t>();

    int64_t src_acc = 0;
    const int64_t nblocks = chunk_block_counts[c];
    for (int64_t seg = 0; seg < nsegs; ++seg) {
      src_offsets[c * nsegs + seg] = src_acc;
      src_acc += seg_block_bytes[seg] * nblocks;
    }
    TORCH_CHECK(src.numel() >= src_acc, "chunk buffer is smaller than expected");
  }

  auto* dst_ptr = dst.data_ptr<uint8_t>();
  at::parallel_for(0, nsegs, 1, [&](int64_t begin, int64_t end) {
    for (int64_t seg = begin; seg < end; ++seg) {
      const int64_t nb = seg_block_bytes[seg];
      int64_t logical_block_start = 0;
      for (int64_t c = 0; c < nchunks; ++c) {
        const int64_t nblocks = chunk_block_counts[c];
        const int64_t nbytes = nblocks * nb;
        if (nbytes > 0) {
          std::memcpy(
              dst_ptr + dst_bases[seg] + logical_block_start * nb,
              src_ptrs[c] + src_offsets[c * nsegs + seg],
              static_cast<size_t>(nbytes));
        }
        logical_block_start += nblocks;
      }
    }
  });
}

void split_request_buffer(
    torch::Tensor src,
    std::vector<torch::Tensor> chunk_buffers,
    std::vector<int64_t> chunk_block_counts,
    std::vector<int64_t> seg_block_bytes) {
  TORCH_CHECK(src.device().is_cpu(), "src must be a CPU tensor");
  TORCH_CHECK(src.dtype() == torch::kUInt8, "src must be uint8");
  TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
  TORCH_CHECK(
      chunk_buffers.size() == chunk_block_counts.size(),
      "chunk_buffers and chunk_block_counts size mismatch");

  const int64_t nchunks = static_cast<int64_t>(chunk_buffers.size());
  const int64_t nsegs = static_cast<int64_t>(seg_block_bytes.size());
  int64_t total_blocks = 0;
  for (int64_t nblocks : chunk_block_counts) {
    TORCH_CHECK(nblocks >= 0, "chunk block count must be non-negative");
    total_blocks += nblocks;
  }

  std::vector<int64_t> src_bases(nsegs);
  int64_t acc = 0;
  for (int64_t seg = 0; seg < nsegs; ++seg) {
    const int64_t nb = seg_block_bytes[seg];
    TORCH_CHECK(nb >= 0, "segment byte count must be non-negative");
    src_bases[seg] = acc;
    acc += nb * total_blocks;
  }
  TORCH_CHECK(src.numel() >= acc, "src is smaller than split input");

  std::vector<uint8_t*> dst_ptrs(nchunks);
  std::vector<int64_t> dst_offsets(nchunks * nsegs);
  for (int64_t c = 0; c < nchunks; ++c) {
    auto& dst = chunk_buffers[c];
    TORCH_CHECK(dst.device().is_cpu(), "chunk buffer must be a CPU tensor");
    TORCH_CHECK(dst.dtype() == torch::kUInt8, "chunk buffer must be uint8");
    TORCH_CHECK(dst.is_contiguous(), "chunk buffer must be contiguous");
    dst_ptrs[c] = dst.data_ptr<uint8_t>();

    int64_t dst_acc = 0;
    const int64_t nblocks = chunk_block_counts[c];
    for (int64_t seg = 0; seg < nsegs; ++seg) {
      dst_offsets[c * nsegs + seg] = dst_acc;
      dst_acc += seg_block_bytes[seg] * nblocks;
    }
    TORCH_CHECK(dst.numel() >= dst_acc, "chunk buffer is smaller than expected");
  }

  const auto* src_ptr = src.data_ptr<uint8_t>();
  at::parallel_for(0, nsegs, 1, [&](int64_t begin, int64_t end) {
    for (int64_t seg = begin; seg < end; ++seg) {
      const int64_t nb = seg_block_bytes[seg];
      int64_t logical_block_start = 0;
      for (int64_t c = 0; c < nchunks; ++c) {
        const int64_t nblocks = chunk_block_counts[c];
        const int64_t nbytes = nblocks * nb;
        if (nbytes > 0) {
          std::memcpy(
              dst_ptrs[c] + dst_offsets[c * nsegs + seg],
              src_ptr + src_bases[seg] + logical_block_start * nb,
              static_cast<size_t>(nbytes));
        }
        logical_block_start += nblocks;
      }
    }
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stitch_chunk_buffers", &stitch_chunk_buffers);
  m.def("split_request_buffer", &split_request_buffer);
}
