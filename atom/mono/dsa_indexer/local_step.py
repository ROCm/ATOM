"""DSA-only reset: local scratch for short steps, original fence otherwise.

Shared whole-model runners retain their own reset and peer fence.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import Int32, Int64, T

from atom.mono.device.ops import (
    CM_DEV,
    CM_SYS,
    ballot,
    launcher,
    load_ptr64,
    rsrc,
)
from atom.mono.dsa_indexer.build import local_step_symbol
from atom.mono.dsa_indexer.kernel import scratch_layout
from atom.mono.dsa_indexer.selection_plan import SHORT_VERIFY_CTX, short_flag_count
from atom.mono.plan.build_key import source_digest
from atom.mono.plan.execution import BLOCKS, THREADS
from atom.mono.runtime.step_begin import (
    ARRIVED,
    EPOCH,
    _fence,
    _ld64,
    _wait_arrived,
    _zero,
)
from atom.utils.forward_context import get_forward_context

_STREAM = fx.Stream(None)


@cache
def build(rows, qlen, world, compact, debug=False):
    assert rows <= (256 if qlen == 6 else 64) and rows % qlen == 0 and qlen in (1, 6, 8)
    prefix, flag_stride, suffix, total = compact
    name = local_step_symbol(rows, qlen, world, compact, debug)
    keyed = (name, source_digest("mono"))

    @flyc.kernel(
        name=name,
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(
        scratch: Int64,
        size: Int64,
        peers: Int64,
        peer_size: Int64,
        rank: Int32,
        state: Int64,
        ctx: Int64,
        bids: Int64,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tid % 64
        row = fx.min(lane, fx.Int32(rows // qlen - 1)) * qlen
        request = fx.Int32(bo.buffer_load(rsrc(bids), row, vec_width=1, dtype=T.i32))
        length = fx.Int32(
            bo.buffer_load(
                rsrc(ctx), fx.max(request, fx.Int32(0)), vec_width=1, dtype=T.i32
            )
        )
        large = (lane < rows // qlen) & (request >= 0) & (length > SHORT_VERIFY_CTX)
        local = ballot(large) == 0
        first, stride = bid * THREADS + tid, BLOCKS * THREADS
        if ~local:
            epoch = _ld64(rsrc(state), EPOCH) + 1
            _zero(load_ptr64(peers, rank), peer_size, first, stride, CM_SYS)
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if (bid != 0) & (tid == 0):
                bo.buffer_store(
                    fx.Int32(epoch & 0xFFFFFFFF),
                    rsrc(state),
                    ARRIVED * 2 + bid,
                    cache_modifier=CM_DEV,
                )
            if (bid == 0) & (tid < 64):
                _wait_arrived(state, epoch)
                _fence(peers, peer_size, rank, state, epoch, world, debug)
        if local:
            _zero(scratch, fx.Int64(prefix), first, stride)
            _zero(scratch + fx.Int64(suffix), fx.Int64(total - suffix), first, stride)
            # Only score units in the runtime short-context window are polled.
            short_flags = short_flag_count(flag_stride)
            for i in range(first, rows * short_flags, stride):
                flag_row, chunk = i // short_flags, i % short_flags
                at = prefix + flag_row * flag_stride + chunk * 8
                bo.buffer_store(
                    fx.Vector.filled(2, 0, fx.Int32),
                    rsrc(scratch),
                    at // 4,
                    cache_modifier=CM_DEV,
                )
        else:
            _zero(scratch, size, first, stride)

    @launcher(name)
    def launch(
        scratch: Int64,
        size: Int64,
        peers: Int64,
        peer_size: Int64,
        rank: Int32,
        state: Int64,
        ctx: Int64,
        bids: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kernel(scratch, size, peers, peer_size, rank, state, ctx, bids).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch


@cache
def _compact_layout(key):
    layout = scratch_layout(key)
    start, size = layout["sflag"]
    return (
        start,
        size // key.tokens,
        start + size,
        max(o + n for o, n in layout.values()),
    )


def begin(runner, rows, qlen):
    key = runner.ix.key(rows, qlen)
    if not key.short_owned_verify:
        runner.mailboxes.begin_step()
        return
    md = get_forward_context().attn_metadata
    _run_compiled(
        build(rows, qlen, key.world, _compact_layout(key), runner.mailboxes._debug),
        runner.scratch.data_ptr(),
        runner.scratch.nbytes,
        runner.peers.addresses.data_ptr(),
        runner.peers.bytes.nbytes,
        runner.rank,
        runner.mailboxes._state.data_ptr(),
        md.context_lens.data_ptr(),
        md.batch_id_per_q_token.data_ptr(),
        torch.cuda.current_stream(),
    )
