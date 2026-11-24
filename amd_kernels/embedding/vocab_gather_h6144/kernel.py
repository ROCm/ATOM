# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Read a vocabulary row directly from its owning rank's immutable bf16 weight."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr.typing import Vector as Vec
from flydsl.expr import range_constexpr

H = 6144
THREADS = 256
PIECES = 3  # three contiguous 4096-byte pieces of each row


def compile_vocab_gather(id_bits):
    name=f"vocab_gather_ranks4_h6144_i{id_bits}"
    @flyc.kernel(name=name,known_block_size=[THREADS,1,1])
    def kernel(ids: fx.Tensor, peers: fx.Tensor, out: fx.Tensor,
               rows: fx.Int32, shard_vocab: fx.Int32):
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        row, piece = bx // PIECES, bx % PIECES
        token = fx.Int64(ids[row])
        valid = (token >= 0) & (token < fx.Int64(shard_vocab * 4))
        safe = valid.select(token,fx.Int64(0))
        owner = (safe // fx.Int64(shard_vocab)).to(fx.Int32)
        local = safe % fx.Int64(shard_vocab)
        base = fx.Int64(peers[owner]) + local * (H * 2)
        src = buffer_ops.create_buffer_resource_from_addr(base,num_records_bytes=H*2)
        # Static weights need no system-coherent polling; the ordinary GPU
        # cache also serves repeated token rows in real decode batches.
        v = Vec(buffer_ops.buffer_load(src,piece*1024+tx*4,vec_width=4,dtype=fx.Int32,mask=valid))
        # AITER sums one nonzero owner with positive zeros. Its fp32 FTZ
        # turns bf16 subnormals and either signed zero into positive zero.
        clean = []
        for i in range_constexpr(4):
            word = fx.Int32(v[i])
            low = ((word & 0x7F80) == 0).select(fx.Int32(0),word & 0xFFFF)
            high = ((word & 0x7F800000) == 0).select(fx.Int32(0),word & fx.Int32(-65536))
            clean.append(low | high)
        v = Vec.from_elements(clean,fx.Int32)
        dst = buffer_ops.create_buffer_resource(out,num_records_bytes=rows*H*2)
        buffer_ops.buffer_store(v,dst,row*(H//2)+piece*1024+tx*4)

    @flyc.jit
    def launch_vocab_gather_h6144(ids: fx.Tensor,peers: fx.Tensor,out: fx.Tensor,
                                 rows: fx.Int32,shard_vocab: fx.Int32,stream: fx.Stream):
        kernel(ids,peers,out,rows,shard_vocab).launch(
            grid=(fx.Int64(rows*PIECES),1,1),block=(THREADS,1,1),stream=stream)
    launch_vocab_gather_h6144.kernel_name=name
    return launch_vocab_gather_h6144
