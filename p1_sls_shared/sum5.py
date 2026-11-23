# SPDX-License-Identifier: Apache-2.0
"""p1/sp-local-shared: the TP4 pull reduce-scatter's final sum with the owner's shared expert as a fifth term.

``compile_sum4`` of amd_kernels.collectives.pull_collectives_ranks4 (fd233ef) with one more bf16 source: out =
bf16((((p0 + p1) + p2) + p3) + shared) in fp32, p_r the four staged bf16 slot-sum blocks in rank order and ``shared``
this rank's own rows of the full shared expert. The shared output is added once, unrounded into any partial (the
partials then carry the routed slots only; the stage kernel gets a zero shared input)."""
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops
from amd_kernels.collectives.pull_collectives_ranks4.kernel import NT, STEP, SUM_WGS, UNROLL, _as_f32, _pk_bf16
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from .common.launch import _run_compiled

WORLD = 4


@functools.cache
def compile_sum5():
    @flyc.kernel(name="p1_sls_pull_ranks4_sum4_plus_shared", known_block_size=[NT, 1, 1])
    def kernel(arg_out: fx.Int64, arg_s0: fx.Int64, arg_s1: fx.Int64, arg_s2: fx.Int64, arg_s3: fx.Int64,
               arg_sh: fx.Int64, i64_block: fx.Int64):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        srcs = [buffer_ops.create_buffer_resource_from_addr(a, num_records_bytes=i64_block)
                for a in (arg_s0, arg_s1, arg_s2, arg_s3, arg_sh)]
        out = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=i64_block)
        n_dw = fx.Int32(i64_block // 4)
        per_block = (n_dw + (SUM_WGS - 1)) // SUM_WGS
        per_block = (per_block + (STEP - 1)) // STEP * STEP
        start = bid * per_block
        end = start + per_block
        off = start + tx * 4
        while off < end:
            words = []
            for u in range_constexpr(UNROLL):
                o = off + u * NT * 4
                words.append([Vec(buffer_ops.buffer_load(srcs[r], o, vec_width=4, dtype=fx.Int32, mask=o < n_dw))
                              for r in range_constexpr(WORLD + 1)])
            for u in range_constexpr(UNROLL):
                o = off + u * NT * 4
                packed = []
                for i in range_constexpr(4):
                    lo = fx.Float32(0.0)
                    hi = fx.Float32(0.0)
                    for r in range_constexpr(WORLD + 1):
                        w = fx.Int32(words[u][r][i])
                        lo = lo + _as_f32(w << 16)
                        hi = hi + _as_f32(w & fx.Int32(-65536))
                    packed.append(_pk_bf16(lo, hi))
                buffer_ops.buffer_store(Vec.from_elements(packed, fx.Int32), out, o, mask=o < n_dw)
            off = off + STEP

    @flyc.jit
    def launch(arg_out: fx.Int64, arg_s0: fx.Int64, arg_s1: fx.Int64, arg_s2: fx.Int64, arg_s3: fx.Int64,
               arg_sh: fx.Int64, i64_block: fx.Int64, stream: fx.Stream):
        kernel(arg_out, arg_s0, arg_s1, arg_s2, arg_s3, arg_sh, i64_block).launch(
            grid=(SUM_WGS, 1, 1), block=(NT, 1, 1), stream=stream)

    launch.kernel_name = "p1_sls_pull_ranks4_sum4_plus_shared"
    return launch


_ZEROS = {}


def _zeros(rows, device):
    z = _ZEROS.get(device)
    if z is None or z.shape[0] < rows:
        z = _ZEROS[device] = torch.zeros((rows, 6144), dtype=torch.bfloat16, device=device)
    return z[:rows]


def slots4_reduce_scatter_owner_shared(rs, slots, shared_own, out=None):
    """``SlotsReduceScatterComm.slots4_reduce_scatter`` (pull_collectives_ranks4) with the shared expert of this
    rank's own rows ``shared_own`` [rows / 4, H] added in the final fp32 sum: the stage + pull launch with a zero
    shared input (its partials = bf16 slot sums), then ``compile_sum5``."""
    rows = slots.shape[0]
    staging = rs._stage_slots(slots, _zeros(rows, slots.device))
    own = rows // WORLD
    assert shared_own.shape == (own, 6144) and shared_own.dtype == torch.bfloat16 and shared_own.is_contiguous()
    if out is None:
        out = torch.empty((own, 6144), dtype=torch.bfloat16, device=slots.device)
    block = rows * 6144 * 2 // WORLD
    srcs = [staging.data_ptr() + r * block for r in range(WORLD)]
    _run_compiled(compile_sum5(), out.data_ptr(), *srcs, shared_own.data_ptr(), block,
                  torch.cuda.current_stream(slots.device))
    return out
