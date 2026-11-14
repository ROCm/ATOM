# SPDX-License-Identifier: MIT
"""TP all-reduce + Gemma RMSNorm + MXFP8 writes the separate path's bytes.

`fused_allreduce_gemma_rms_norm_mxfp8` replaces AR + RMSNorm followed by
`quantize_fp8`, so on every rank its E4M3 values, E8M0 scales, residual and
(with `emit_bf16`) bf16 rows must equal the separate path's bit for bit, eager
and replayed from a CUDA graph (the custom all-reduce's registered-buffer
path). Row counts straddle the 1-stage limit (37 / 38 rows of 6144 at TP4)
past which the separate path runs, and a prefill-sized input.
On an aiter without the fused epilogue every case takes the separate path and
still passes.
"""

import multiprocessing as mp
import time

import pytest
import torch

pytest.importorskip("aiter", reason="the fused all-reduce lives in aiter")

H = 6144
EPS = 1e-6
ROWS = (1, 2, 16, 32, 37, 38, 42, 1355)


def _matches(rank, rows):
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
    )
    from aiter.dist.parallel_state import graph_capture

    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.layernorm import _fused_ar_gemma_rms_norm_mxfp8

    g = torch.Generator(device="cuda").manual_seed(1000 * rows + rank)
    x = (torch.randn(rows, H, device="cuda", generator=g) * 3).to(torch.bfloat16)
    res = torch.randn(rows, H, device="cuda", generator=g).to(torch.bfloat16)
    w = (
        torch.randn(H, device="cuda", generator=torch.Generator("cuda").manual_seed(7))
        * 0.1
    ).to(torch.bfloat16)
    if rows > 3:
        x[3, :64] = 0
        res[3, :64] = 0  # all-zero groups after the norm

    normed, rr = tensor_model_parallel_fused_allreduce_rmsnorm(
        x, res, w, EPS, gemma_norm=True
    )
    rq, rs = quantize_fp8(normed)

    def same(out, emit):
        q, s, r, n = out
        ok = (
            torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
            and torch.equal(s.view(torch.uint8), rs.view(torch.uint8))
            and torch.equal(r, rr)
        )
        return ok and (torch.equal(n, normed) if emit else n.numel() == 0)

    ok = all(
        same(_fused_ar_gemma_rms_norm_mxfp8(x, res, w, EPS, e), e)
        for e in (False, True)
    )
    if rows <= 64:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            out = _fused_ar_gemma_rms_norm_mxfp8(x, res, w, EPS, True)
        for t in out:
            t.zero_()
        graph.replay()
        torch.cuda.synchronize()
        ok = ok and same(out, True)
    return ok


def _rank(rank, world, init_method, results):
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(world, 1)
    try:
        results[rank] = tuple(_matches(rank, rows) for rows in ROWS)
    finally:
        torch.cuda.synchronize()
        get_tp_group().barrier()
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("world", [4])
def test_fused_ar_rmsnorm_mxfp8_matches_separate_path(world):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        results = manager.dict()
        procs = [
            ctx.Process(target=_rank, args=(r, world, init_method, results))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        deadline = time.monotonic() + 600
        for p in procs:
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        hung = [p for p in procs if p.is_alive()]
        for p in hung:
            p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
        assert not hung and all(p.exitcode == 0 for p in procs)
        want = (True,) * len(ROWS)
        assert dict(results) == {r: want for r in range(world)}
