# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.5 MTP drafter all-reduce fusions on 4 gfx1250 GPUs (TP4), on the ATOM
call sites the drafter runs, against the unfused sequences they replace.

- ATOM_MTP_FUSE_ALLREDUCE_NORM: o_proj + all-reduce + post-attention norm in
  aiter's GEMM + all-reduce + RMSNorm, and the MoE all-reduce + final norm in the
  fused all-reduce + RMSNorm (the target's kernels).
- ATOM_FUSED_MOE_TAIL_AR_NORM: `routed + sigmoid(gate) * shared` read by the
  all-reduce + RMSNorm kernel itself (bitwise the sigmoid_mul + fused kernel).
- ATOM_MTP_SELECT_ROWS_BEFORE_O_PROJ: step 0 keeps its rows before o_proj; the
  output gate gathers them and the GEMM + all-reduce kernel gathers the residual
  (bitwise the full-row path followed by the two index_selects).

Run as a script with --bench for the graph-replay timings (slowest rank, us a
sequence, the sequence captured 64 times a graph with rotating o_proj weights).
"""

import argparse
import multiprocessing as mp
import os
import sys

import pytest
import torch

WORLD_SIZE = 4
HIDDEN, K, EPS = 2048, 1024, 1e-6
QKVG = 2 * K + 512  # gate | q | k | v of a rank (4 q heads, 1 kv head of 256)


def _skip_reason():
    if torch.cuda.device_count() < WORLD_SIZE:
        return "needs 4 GPUs"
    try:
        from aiter.jit.utils.chip_info import get_gfx
    except ImportError:
        return "needs aiter"
    return None if get_gfx() == "gfx1250" else "needs gfx1250"


def _rand(gen, *shape, scale=1.0):
    return scale * torch.randn(
        shape, generator=gen, dtype=torch.bfloat16, device="cuda"
    )


class _Inputs:
    """One rank's inputs: TP-partial ones from a per-rank seed, replicated ones
    (residual, router logits, norm weight) from a shared seed."""

    def __init__(self, rank, rows, weights=1):
        mine = torch.Generator(device="cuda").manual_seed(100 * rows + rank)
        same = torch.Generator(device="cuda").manual_seed(100 * rows + 7)
        self.attn = _rand(mine, rows, K)
        self.qkvg = _rand(mine, rows, QKVG, scale=2.0)
        self.gate = self.qkvg[:, :K]
        self.w_o = [_rand(mine, HIDDEN, K, scale=K**-0.5) for _ in range(weights)]
        self.routed = _rand(mine, rows, HIDDEN, scale=0.5)
        self.shared = _rand(mine, rows, HIDDEN, scale=0.5)
        self.residual = _rand(same, rows, HIDDEN)
        self.logits = _rand(same, rows, 257, scale=3.0)
        self.shared_gate = self.logits[:, 256:]
        self.norm_w = _rand(same, HIDDEN, scale=0.1)


def _sequences(rows_sel=None):
    """The step's sequences by name: each maps (inputs, weight index) to its
    outputs."""
    from aiter.dist.communication_op import (
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_fused_allreduce_rmsnorm,
    )
    from aiter.tuned_gemm import tgemm

    from atom.model_ops.layernorm import (
        GemmaRMSNorm,
        fused_allreduce_moe_tail_gemma_rmsnorm,
        gluon_gemm_allreduce_gemma_rmsnorm,
    )
    from atom.model_ops.sigmoid_gate import (
        output_gate_mul,
        output_gate_rows,
        shared_expert_gate_add,
    )

    def plain_norm(t, x, residual):
        norm = GemmaRMSNorm(HIDDEN, eps=EPS)
        norm.weight.data = t.norm_w
        return norm.forward_cuda(x, residual)

    def o_proj_old(t, i):
        # kernels 8-11: gate, o_proj GEMM, all-reduce, post-attention norm
        g = output_gate_mul(t.attn, t.gate, True)
        y = tensor_model_parallel_all_reduce(tgemm.mm(g, t.w_o[i], None))
        return plain_norm(t, y, t.residual)

    def o_proj_fused(t, i):
        g = output_gate_mul(t.attn, t.gate, True)
        return gluon_gemm_allreduce_gemma_rmsnorm(
            g, t.w_o[i], t.residual, t.norm_w, EPS
        )

    def pick(outs):
        return tuple(torch.index_select(o, 0, rows_sel) for o in outs)

    def step0_old(t, i):  # then the two index_selects
        return pick(o_proj_old(t, i))

    def step0_fused(t, i):
        return pick(o_proj_fused(t, i))

    def step0_rows_first(t, i):
        g = output_gate_rows(t.attn, t.gate, rows_sel)
        return gluon_gemm_allreduce_gemma_rmsnorm(
            g, t.w_o[i], t.residual, t.norm_w, EPS, rows_sel
        )

    def moe_old(t, i):
        # kernels 21-23: gated shared-expert add, all-reduce, final norm
        x = shared_expert_gate_add(t.shared, t.shared_gate, t.routed, True)
        return plain_norm(t, tensor_model_parallel_all_reduce(x), t.residual)

    def moe_fused(t, i):
        x = shared_expert_gate_add(t.shared, t.shared_gate, t.routed, True)
        return tensor_model_parallel_fused_allreduce_rmsnorm(
            x, t.residual, t.norm_w, EPS, gemma_norm=True
        )

    def moe_gated(t, i):
        return fused_allreduce_moe_tail_gemma_rmsnorm(
            t.routed, t.shared, t.shared_gate, t.residual, t.norm_w, EPS
        )

    if rows_sel is not None:
        return {
            "o_proj+AR+norm+2 index_select": step0_old,
            "gemm_ar+2 index_select": step0_fused,
            "gate_rows+gemm_ar(residual rows)": step0_rows_first,
        }
    return {
        "o_proj+AR+norm": o_proj_old,
        "gemm_ar": o_proj_fused,
        "moe: gate_add+AR+norm": moe_old,
        "moe: gate_add+AR_norm": moe_fused,
        "moe: gated AR_norm": moe_gated,
    }


def _diff(a, b):
    """How far a is from b: the share of differing values, the largest
    difference, and the largest one in bf16 ulps of b where |b| >= 1/16."""
    a, b = a.float(), b.float()
    d = (a - b).abs()
    big = b.abs() >= 1 / 16
    ulps = (d[big] / (b[big].abs() * 2.0**-7)).max() if bool(big.any()) else 0.0
    return (
        f"{float((d > 0).float().mean()) * 100:.1f}% differ, max {float(d.max()):.4f}"
        f" ({float(ulps):.1f} ulp)"
    )


def _check(rank):
    failures, notes = [], []
    for rows in (16, 64, 65, 80):
        t = _Inputs(rank, rows)
        seq = _sequences()
        old = seq["o_proj+AR+norm"](t, 0)
        new = seq["gemm_ar"](t, 0)
        # Not bitwise: the unfused path rounds the GEMM's and the all-reduce's
        # sums to bf16, the fused kernel adds the 4 bf16 partials and the
        # residual in fp32 once (as the target's layers do).
        notes.append(
            f"rows {rows}: gemm_ar vs unfused: out {_diff(new[0], old[0])}; "
            f"residual {_diff(new[1], old[1])}"
        )
        torch.testing.assert_close(new[1], old[1], atol=0.05, rtol=0.02)
        torch.testing.assert_close(new[0], old[0], atol=0.05, rtol=0.02)
        ref = seq["moe: gate_add+AR_norm"](t, 0)
        for name in ("moe: gate_add+AR+norm", "moe: gated AR_norm"):
            out = seq[name](t, 0)
            if name == "moe: gated AR_norm":
                if not all(torch.equal(a, b) for a, b in zip(out, ref)):
                    failures.append(f"rank {rank} rows {rows}: {name} not bitwise")
            else:
                notes.append(
                    f"rows {rows}: fused AR_norm vs AR+norm: out "
                    f"{_diff(ref[0], out[0])}; residual {_diff(ref[1], out[1])}"
                )
    for total in (65, 80, 320):
        sel = total // 5
        # The last of each sequence's 5 verify rows.
        rows_sel = torch.arange(sel, device="cuda") * 5 + 4
        t = _Inputs(rank, total)
        seq = _sequences(rows_sel)
        ref = seq["gemm_ar+2 index_select"](t, 0)
        out = seq["gate_rows+gemm_ar(residual rows)"](t, 0)
        if not all(torch.equal(a, b) for a, b in zip(out, ref)):
            failures.append(f"rank {rank} rows {total}: rows-first not bitwise")
        # Under a graph too.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = seq["gate_rows+gemm_ar(residual rows)"](t, 0)
        graph.replay()
        torch.cuda.synchronize()
        if not all(torch.equal(a, b) for a, b in zip(outs, ref)):
            failures.append(f"rank {rank} rows {total}: rows-first graph not bitwise")
    return failures, notes


def _bench(rank, iters=64, replays=20):
    import torch.distributed as dist
    from aiter.dist.parallel_state import get_tp_group

    group = get_tp_group().cpu_group
    weights = 32  # 32 x 4 MiB of o_proj weight a rank: past the caches
    cases = [(r, None) for r in (16, 64, 80, 320)]
    cases += [(r, r // 5) for r in (65, 80, 320)]
    results = []
    for rows, sel in cases:
        t = _Inputs(rank, rows, weights)
        rows_sel = None
        if sel is not None:
            rows_sel = torch.arange(sel, device="cuda") * 5 + 4
        for name, fn in _sequences(rows_sel).items():
            if sel is not None and name.startswith("moe"):
                continue
            for i in range(3):
                fn(t, i)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(iters):
                    fn(t, i % weights)
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graph.replay()
            stop.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(stop) * 1e3 / (replays * iters)
            per_rank = [None] * WORLD_SIZE
            dist.all_gather_object(per_rank, us, group=group)
            label = f"{rows}" if sel is None else f"{rows} -> {sel}"
            results.append((label, name, max(per_rank)))
            del graph
    return results


def _worker(rank, init_method, mode):
    os.environ["AITER_GLUON_FUSED_AR"] = "1"  # read when the communicator is made
    torch.cuda.set_device(rank)
    import aiter

    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    try:
        from aiter.dist.parallel_state import get_tp_group

        from atom.model_ops.layernorm import gluon_gemm_ar_init

        gluon_gemm_ar_init()
        state = get_tp_group().device_communicator.gluon_ar
        assert state is not None, "no Gluon all-reduce + RMSNorm inboxes"
        out = _check(rank) if mode == "check" else _bench(rank)
        torch.cuda.synchronize()
        assert state[1][-1].item() == 0, "a Gluon poll timed out"
        return out
    finally:
        aiter.destroy_dist_env()


def _run(mode):
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    ctx = mp.get_context("spawn")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with ctx.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_worker, (rank, init_method, mode))
            for rank in range(WORLD_SIZE)
        ]
        return [r.get(timeout=900) for r in results]


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
def test_mtp_tp4_fusions():
    results = _run("check")
    failures = [f for fails, _ in results for f in fails]
    for note in results[0][1]:
        print(note)
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if _skip_reason() is not None:
        sys.exit(f"skipped: {_skip_reason()}")
    if args.bench:
        rows = _run("bench")[0]
        print("| rows | sequence | us (graph replay, slowest rank) |")
        print("|---|---|---:|")
        for label, name, us in rows:
            print(f"| {label} | {name} | {us:.2f} |")
    else:
        fails, notes = zip(*_run("check"))
        print("\n".join(notes[0]))
        print("FAILURES:\n" + "\n".join(f for fs in fails for f in fs) or "ok")
