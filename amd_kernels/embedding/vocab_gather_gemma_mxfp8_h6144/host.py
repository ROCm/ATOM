# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Stateless fused lookup using an existing, published VocabGather mapping."""
import functools
import struct

import torch

from amd_kernels.embedding.vocab_gather_h6144.ipc import _run_compiled
from .kernel import compile_vocab_gather_gemma_mxfp8
from .kernel_fourwave import compile_vocab_gather_gemma_mxfp8_fourwave


@functools.cache
def get_kernel(tiled, id_bits, fourwave):
    compiler = (compile_vocab_gather_gemma_mxfp8_fourwave if fourwave
                else compile_vocab_gather_gemma_mxfp8)
    return compiler(tiled, id_bits)


def vocab_gather_gemma_mxfp8(ids, lookup, weight, eps=1e-6, code_tile_rows=16):
    """Return codes, shuffled scales and the raw BF16 residual.

    ``lookup`` is a live ``VocabGather``. Its immutable peer weight mappings
    remain alive through every eager call and graph replay using this function.
    Code rows use row-major layout at tile 0, or the standard 16x16 layout
    padded to ``code_tile_rows``. Padding rows remain unwritten.
    """
    assert ids.ndim == 1 and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)
    assert ids.device == lookup.weight.device == weight.device
    assert weight.shape == (6144,) and weight.dtype == torch.bfloat16 and weight.is_contiguous()
    assert code_tile_rows >= 0 and code_tile_rows % 16 == 0
    m = ids.numel()
    assert 0 <= m <= 2048
    tile = max(code_tile_rows, 1)
    rows = (m + tile - 1) // tile * tile
    q = torch.empty((rows, 6144), device=ids.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(((rows + 31) // 32 * 32, 192), device=ids.device, dtype=torch.uint8)
    residual = torch.empty((m, 6144), device=ids.device, dtype=torch.bfloat16)
    if m:
        _run_compiled(
            get_kernel(code_tile_rows > 0, 32 if ids.dtype == torch.int32 else 64, m > 512),
            ids.data_ptr(), lookup.peers.data_ptr(), lookup.shard_vocab, weight.data_ptr(),
            q.data_ptr(), scales.data_ptr(), residual.data_ptr(), q.numel(), scales.numel(), m,
            struct.unpack('<i', struct.pack('<f', eps))[0], torch.cuda.current_stream(),
        )
    return q, scales, residual


def warmup_vocab_gather_gemma_mxfp8(lookup, weight):
    """Compile both physical-wave variants, ID widths and code layouts."""
    for m in (16, 513):
        for dtype in (torch.int32, torch.int64):
            ids = torch.zeros(m, device=weight.device, dtype=dtype)
            for tile in (0, 16):
                vocab_gather_gemma_mxfp8(ids, lookup, weight, code_tile_rows=tile)
