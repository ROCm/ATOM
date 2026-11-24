# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The staged bf16 stream's buffer layout and batched two-stage handshake, shared by the streamed TP4 all-reduce + add +
Gemma RMSNorm kernels (the coded staged stream, the registered stream and batch pulls, the MXFP8 streams).

Grid: 4 groups of workgroups of R consecutive rows of a chunk (chunk c = rows [c q, (c + 1) q), q = ceil(rows / 4)).
Group 0 (dispatched first) reduces chunk ``me``, groups 1-3 pull chunks me + 1 .. me + 3. A puller signals each
sub-batch of ``sub_rows(R)`` rows; the reducer publishes the reduced batch (``publish_stream``) and the pullers wait
for it (``wait_reduced_stream``)."""

import flydsl.compiler as flyc
import flydsl.expr as fx

from ..add_rmsnorm.add_rmsnorm import NW
from ..add_rmsnorm.common import H
from .common import LINE, WORLD, _flag_store, _spin, sel4

NT = 64 * NW  # 768 threads
ROWB = H * 2
MAX_ROWS = 8192  # rows < MAX_ROWS (the buffer layout the kernels address; the communicator takes 1-4095)
RED = MAX_ROWS * LINE  # flag word RED + b * LINE: row b's sub-batch is published
RED_OFF = 2 * MAX_ROWS * LINE * 4  # [MAX_ROWS / 4, H] bf16 reduced rows (chunk me)
STG_OFF = RED_OFF + (MAX_ROWS // WORLD) * ROWB  # [MAX_ROWS, H] bf16 staged partials
STREAM_BUF_BYTES = STG_OFF + MAX_ROWS * ROWB


def sub_rows(rpw):
    """Rows per staging handshake: 2 up to R16, 4 at R32 (measured on the slots4 stream)."""
    return 2 if rpw <= 16 else 4


@flyc.jit
def publish_stream(b0, b1, b2, b3, me, b, e, reducer):
    lane = fx.thread_idx.x
    if reducer & (lane < WORLD - 1):
        p = (me + 1 + lane) % WORLD
        _flag_store(sel4(p, b0, b1, b2, b3) + fx.Int64((RED + b * LINE) * 4), e)


@flyc.jit
def wait_reduced_stream(mine, b, e):
    if fx.thread_idx.x == 0:
        _spin(mine + fx.Int64((RED + b * LINE) * 4), e)
    fx.gpu.barrier()
