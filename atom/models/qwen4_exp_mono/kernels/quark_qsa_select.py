# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reuse AITER stable QSA selection in LDS and expand indices before publication.

The Qwen metadata ABI uses int64 logical positions and int32 request IDs and
sequence lengths. Keep the position load wide even for short decode contexts.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.topk.radix_topk_one_block import (
    build_radix_topk_one_block_body,
)
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.memory import raw_tensor
from atom.mono.device.ops import kernel_symbol, launcher, rsrc
from atom.mono.runtime.abi import KernelAbi

_STREAM = fx.Stream(None)

ABI = KernelAbi(
    (
        "logits",
        "contexts",
        "positions",
        "requests",
        "out",
        "logit_stride",
        "num_requests",
    )
)


@cache
def build(rows):
    assert rows in (1, 2, 4)
    threads = 512
    shared, body = build_radix_topk_one_block_body(
        512, threads, False, True, True, wave_size=64, arch="gfx950"
    )

    @fx.struct
    class Smem:
        select: shared
        indices: fx.Array[fx.Int32, 512, 16]

    name = kernel_symbol("qwen_quark_qsa_select_expand", b=rows)

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kernel(
        logits: Int64,
        contexts: Int64,
        positions: Int64,
        requests: Int64,
        out: Int64,
        logit_stride: Int64,
        num_requests: Int64,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        req = fx.Int32(bo.buffer_load(rsrc(requests), row, vec_width=1, dtype=T.i32))
        pos = fx.Int64(bo.buffer_load(rsrc(positions), row, vec_width=1, dtype=T.i64))
        context = fx.Int32(
            bo.buffer_load(
                rsrc(contexts, nbytes=num_requests * 4),
                fx.max(req, 0),
                vec_width=1,
                dtype=T.i32,
                mask=(req >= 0) & (req < num_requests),
            )
        )
        valid = (req >= 0) & (req < num_requests) & (pos >= 0) & (pos < context)
        length = fx.Int32(valid.select(pos + 1, fx.Int64(0)))
        full = length // 4
        sm = fx.SharedAllocator().allocate(Smem)
        indices = sm.indices.peek().view(fx.make_layout(512, 1))
        input_row = raw_tensor(logits + fx.Int64(row * logit_stride * 4), fx.Float32)
        body(sm.select, fx.get_iter(input_row), full, indices, input_row, fx.Int32(0))
        gpu.barrier()
        history_end = fx.min(full, 512) * 4
        for step in range_constexpr((2051 + threads - 1) // threads):
            col = tid + threads * step
            group = indices[fx.min(col // 4, 511)]
            history = ((group >= 0) & (group < full)).select(
                group * 4 + col % 4, fx.Int32(-1)
            )
            tail = full * 4 + col - history_end
            tail = ((tail >= full * 4) & (tail < length)).select(tail, fx.Int32(-1))
            value = (col < history_end).select(history, tail)
            if col < 2051:
                bo.buffer_store(value, rsrc(out), row * 2051 + col)

    @launcher(name)
    def launch(
        logits: Int64,
        contexts: Int64,
        positions: Int64,
        requests: Int64,
        out: Int64,
        logit_stride: Int64,
        num_requests: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES
        kernel(
            logits, contexts, positions, requests, out, logit_stride, num_requests
        ).launch(grid=(rows,), block=(threads,), stream=stream)

    return launch


def select(module, query, metadata, kernel):
    """Ordinary Quark decode: unchanged scorer and stable 512-group budget."""
    import torch

    from atom.model_ops.qwen4_exp.ops.qsa import qsa_paged_mqa_logits

    ratio = module.indexer.compress_ratio
    assert ratio == 4 and module.indexer.token_topk == 2048
    assert 2048 <= metadata.max_seq_len <= 16384
    assert metadata.logical_positions.dtype == torch.int64
    assert metadata.token_to_req.dtype == metadata.seq_lens.dtype == torch.int32
    reachable = (metadata.max_seq_len + ratio - 1) // ratio
    bucket = 1 << (max(reachable, 1) - 1).bit_length()
    columns = min(
        metadata.block_tables.shape[1] * module.compressed_key_cache.shape[1], bucket
    )
    logits, _ = qsa_paged_mqa_logits(
        query,
        module.compressed_key_cache,
        metadata.block_tables,
        metadata.token_to_req,
        metadata.logical_positions,
        metadata.seq_lens,
        ratio,
        max_columns=columns,
        allow_flydsl=False,
        compressed_k_scale=module.compressed_key_scale,
    )
    selected = torch.empty(
        (query.shape[0], 2051), device=query.device, dtype=torch.int32
    )
    kernel(
        logits.data_ptr(),
        metadata.seq_lens.data_ptr(),
        metadata.logical_positions.data_ptr(),
        metadata.token_to_req.data_ptr(),
        selected.data_ptr(),
        logits.stride(0),
        metadata.seq_lens.numel(),
        stream=torch.cuda.current_stream(),
    )
    return selected
