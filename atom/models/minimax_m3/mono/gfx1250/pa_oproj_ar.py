# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host wrapper of the sparse PA + o_proj + k5 one-shot AR + residual + RMSNorm launch
(``kernels/pa_oproj_ar_gfx1250.py``)."""

from __future__ import annotations

import torch

from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, _run_compiled, current_stream_arg, ptr_arg

_SYNCED = set()


def first_launch_sync(launch, args) -> None:
    """Before a kernel's first launch in this process: compile it without launching, then barrier with the
    other ranks. Its in-launch all-reduce waits on the peers' pushes, so a rank that JIT-compiles seconds
    later would otherwise run the peers' polls into their bound (seen: k5 err 32 on one rank at a first
    call), and a timed-out round leaves the all-reduce rounds of every later call misaligned."""
    if id(launch) in _SYNCED:
        return
    _preload_compiled(launch, *args)
    # no barrier inside a graph capture (not capturable); the compile is done here, and a captured first launch
    # only runs at the graph's first replay, which every rank reaches together
    if torch.distributed.is_initialized() and not torch.cuda.is_current_stream_capturing():
        torch.distributed.barrier()
    _SYNCED.add(id(launch))


def flydsl_pa_oproj_ar(
    pa_args: tuple,
    traits,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    res: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    inbox: list[int],
    ctr: int,
    slot: int,
    rank: int,
    ctl: torch.Tensor,
    *,
    m: int,
    lda: int,
    maxit: int = 20000000,
    max_rows: int = 256,
    out: torch.Tensor | None = None,
    res_out: torch.Tensor | None = None,
    timeline: torch.Tensor | None = None,
    stream=None,
    dev_out: int = 1,
    poll_sleep: int = 16,
    gen_sync: int = 1,
    cluster_poll: int = 0,
    w_delay: int = 0,
    a_l2: int = 0,
    cpush: int = 0,
    w_cpol: int = 1,
    recv_late: int = 0,
    w_late: int = 0,
    exit_cl: int = 0,
    role_mode: int = 0,
    rsh_tab: torch.Tensor | None = None,
    r1: int = 0,
):
    """``pa_args``: the production PA launch's arguments without its stream, in the order
    ``pa_decode`` passes them (the PA writes the MXFP8 attention output that o_proj reads, ``lda``
    its row stride). ``traits``: that launch's ``Gfx1250Traits``. ``w`` / ``w_scale``: the o_proj
    weight in the decode GEMM's layout; ``res`` [m, 6144] bf16, ``norm_w`` [6144] (Gemma, the kernel
    adds 1). ``inbox`` / ``ctr`` / ``slot`` / ``rank``: the k5 context's. ``ctl``: int32 [16], zeroed
    once (the launch resets it). Returns (out, res_out)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from .kernels.pa_oproj_ar import compile_pa_oproj_ar

    n = 6144
    assert 0 < m <= 8 and len(inbox) == 4 and ctl.dtype == torch.int32 and ctl.numel() >= 16
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=res.device)
    if res_out is None:
        res_out = torch.empty((m, n), dtype=torch.bfloat16, device=res.device)
    if stream is None:
        stream = current_stream_arg()
    tl = 0 if timeline is None else 1
    if timeline is not None:
        from .kernels.pa_oproj_ar import GRID, TLW

        assert timeline.dtype == torch.int32 and timeline.numel() >= GRID * TLW
    launch = compile_pa_oproj_ar(traits, max_rows=max_rows, tl=tl, dev_out=dev_out, poll_sleep=poll_sleep,
                                 gen_sync=gen_sync, cluster_poll=cluster_poll, w_delay=w_delay,
                                 a_l2=a_l2, cpush=cpush, w_cpol=w_cpol, recv_late=recv_late,
                                 w_late=w_late, exit_cl=exit_cl, role_mode=role_mode,
                                 r1=int(bool(r1) or rsh_tab is not None), r1_router=int(rsh_tab is not None))
    args = (
        *pa_args,
        ptr_arg(w), ptr_arg(w_scale), ptr_arg(res), ptr_arg(norm_w), ptr_arg(out), ptr_arg(res_out),
        flyc.from_c_void_p(fx.Int32, int(ctr)),
        ptr_arg(ctl, fx.Int32),
        ptr_arg(timeline if timeline is not None else ctl, fx.Int32),
        ptr_arg(rsh_tab if rsh_tab is not None else ctl.view(torch.int64), fx.Int64),
        int(inbox[0]), int(inbox[1]), int(inbox[2]), int(inbox[3]),
        int(m), int(lda), int(slot), int(rank), int(maxit), float(eps),
        stream,
    )  # fmt: skip
    first_launch_sync(launch, args)
    _run_compiled(launch, *args)
    return out, res_out
