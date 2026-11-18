# SPDX-License-Identifier: MIT
"""TP MegaMoE on gfx950 (4 GPUs), against the production MoE path.

The test relaunches itself under ``torchrun --nproc_per_node=4`` and checks, on
synthetic MXFP4 experts of MiniMax-M3.1's TP4 shape (hidden 6144, 3072 / 4
intermediate, 128 routed + 1 fused shared expert, top-4 + 1, SwiGLU limit 7):

* ``TPMegaMoE.forward`` (the in-kernel all-reduce, the shared expert and the pad
  rows of a step that is not a multiple of tp) against a bf16-activation torch
  reference and against ``aiter.fused_moe`` + all-reduce, which is what the
  production path computes, at the row counts decode and prefill steps use;
* identical outputs on every rank, no watchdog bit;
* CUDA-graph replay on new inputs equal to the eager run.

Needs >= 4 gfx950 GPUs and an AITER with ``aiter.ops.flydsl.mega_moe_tp``'s
small-M ``ar`` engine (``MegaMoeTPConfig.bigm_min_tokens`` / ``act_dtype``).
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROWS = (1, 3, 4, 13, 16, 64, 100, 128, 256, 512)
GRAPH_ROWS = (3, 16, 256)
HIDDEN, INTER_GLOBAL, ROUTED, TOP_K, LIMIT = 6144, 3072, 128, 4, 7.0


def _skip_reason():
    try:
        import torch

        if torch.cuda.device_count() < 4:
            return "needs 4 GPUs"
        if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
            return "needs gfx950"
        import dataclasses

        from aiter.ops.flydsl.mega_moe_tp import MegaMoeTPConfig

        fields = {f.name for f in dataclasses.fields(MegaMoeTPConfig)}
        if not {"bigm_min_tokens", "act_dtype", "fit_max_local"} <= fields:
            return "AITER MegaMoeTP without the small-M ar engine options"
    except Exception as exc:  # noqa: BLE001 - any import problem is a skip
        return f"no MegaMoeTP here: {exc!r}"
    return None


def test_tp_mega_moe_matches_the_production_path_on_gfx950():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        os.path.abspath(__file__),
    ]
    env = {**os.environ, "HSA_ENABLE_IPC_MODE_LEGACY": "1"}
    done = subprocess.run(
        cmd, check=False, env=env, capture_output=True, text=True, timeout=3600
    )
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]


# -- worker (one process per GPU) ----------------------------------------------


def _quant_experts(experts, rows, cols, magnitude, seed, device):
    import aiter
    import torch
    from aiter import dtypes

    quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    q = torch.empty((experts, rows, cols // 2), dtype=torch.uint8, device=device)
    scale = None
    gen = torch.Generator(device=device)
    step = 16
    for start in range(0, experts, step):
        n = min(step, experts - start)
        gen.manual_seed(seed + start)
        w = torch.randn(
            (n, rows, cols), dtype=torch.bfloat16, device=device, generator=gen
        )
        qq, ss = quant(w.mul_(magnitude), quant_dtype=dtypes.fp4x2)
        q[start : start + n] = qq.view(n, rows, cols // 2).view(torch.uint8)
        if scale is None:
            scale = torch.empty(
                (experts * rows, ss.shape[-1]), dtype=ss.dtype, device=device
            )
        scale[start * rows : (start + n) * rows] = ss.view(n * rows, -1)
    return q.view(dtypes.fp4x2), scale


def _dequant(q, scale, rows, cols):
    import torch
    from aiter.utility import fp4_utils

    f = fp4_utils.mxfp4_to_f32(q.view(torch.uint8).reshape(-1, cols // 2))
    s = fp4_utils.e8m0_to_f32(scale.view(torch.uint8).reshape(-1, cols // 32))
    return (f.view(-1, cols // 32, 32) * s.view(-1, cols // 32, 1)).view(-1, rows, cols)


def _truth(w1, w1s, w2, w2s, inter, x, weights, ids, experts):
    """bf16-activation partial sum of this rank's slice, ATOM's SwiGLU-OAI."""
    import torch

    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in range(experts):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        a = _dequant(
            w1[e : e + 1], w1s[e * 2 * inter : (e + 1) * 2 * inter], 2 * inter, HIDDEN
        )[0]
        b = _dequant(w2[e : e + 1], w2s[e * HIDDEN : (e + 1) * HIDDEN], HIDDEN, inter)[
            0
        ]
        h = x[tok].float() @ a.t()
        g, u = h[:, :inter].clamp(max=LIMIT), h[:, inter:].clamp(-LIMIT, LIMIT)
        act = g * torch.sigmoid(1.702 * g) * (u + 1.0)
        out.index_add_(0, tok, (act @ b.t()) * weights[tok, slot, None])
    return out


def _route(rows, gen, device, experts):
    import torch

    score = torch.randn((rows, ROUTED), device=device, generator=gen)
    w, ids = torch.softmax(score, dim=-1).topk(TOP_K, dim=-1)
    # The fused shared expert is the last column, the way ATOM's topk widens it.
    ids = torch.cat([ids, torch.full((rows, 1), experts - 1, device=device)], 1)
    w = torch.cat([w, w.new_full((rows, 1), 0.3)], 1)
    w = w / w.sum(dim=-1, keepdim=True)
    return w.float().contiguous(), ids.to(torch.int32).contiguous()


def _worker():
    import torch
    import torch.distributed as dist
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig
    from aiter.ops.flydsl.moe_common import GateMode
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import fp4_utils

    from atom.model_ops.fused_moe.sp_mega_moe import (
        TP_CONFIG,
        TPMegaMoE,
        _weights,
        tp_local_token_tile,
    )

    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    cpu_group = dist.new_group(backend="gloo")
    experts, inter = ROUTED + 1, INTER_GLOBAL // tp

    def rel_l2(a, b):
        d = (a.float() - b.float()).pow(2).sum()
        n = b.float().pow(2).sum()
        t = torch.stack([d, n])
        dist.all_reduce(t)
        return float((t[0] / t[1]).sqrt())

    base = 1000 * rank
    w1, w1s = _quant_experts(experts, 2 * inter, HIDDEN, HIDDEN**-0.25, base, device)
    w2, w2s = _quant_experts(experts, HIDDEN, inter, inter**-0.25, base + 7, device)
    layer = SimpleNamespace(
        hidden_size=HIDDEN,
        top_k=TOP_K,
        intermediate_size_per_partition=inter,
        local_num_experts=experts,
        activation=ActivationType.Swiglu,
        swiglu_limit=LIMIT,
        w13_weight=shuffle_weight(w1, layout=(16, 16)),
        w13_weight_scale=fp4_utils.e8m0_shuffle(w1s),
        w2_weight=shuffle_weight(w2, layout=(16, 16)),
        w2_weight_scale=fp4_utils.e8m0_shuffle(w2s),
    )
    runtime = MegaMoeTP(
        MegaMoeTPConfig(
            rank=rank,
            world_size=tp,
            model_dim=HIDDEN,
            inter_dim=inter,
            experts=experts,
            topk=TOP_K + 1,
            max_local_tokens=tp_local_token_tile(max(ROWS), tp),
            activation=ActivationType.Swiglu,
            swiglu_limit=LIMIT,
            fit_max_local=True,
            **TP_CONFIG,
        ),
        **_weights(layer),
        group=cpu_group,
        device=device,
    )
    runtime.prepare(range(1, runtime.max_local_tokens + 1))
    backend = TPMegaMoE(runtime, layer, 1, cpu_group)
    assert backend.runs(max(ROWS))
    routing = {}
    backend._route = lambda layer_, hidden, logits: routing["wi"]
    failures = []

    def production(x, weights, ids):
        out = fused_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            weights,
            ids,
            expert_mask=None,
            activation=ActivationType.Swiglu,
            quant_type=QuantType.per_1x32,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            doweight_stage1=False,
            gate_mode=GateMode.SEPARATED.value,
            swiglu_limit=LIMIT,
        )
        dist.all_reduce(out)
        return out

    for rows in ROWS:
        gen = torch.Generator(device=device).manual_seed(77 + rows)
        x = torch.randn(
            (rows, HIDDEN), dtype=torch.bfloat16, device=device, generator=gen
        )
        weights, ids = _route(rows, gen, device, experts)
        for t in (x, weights, ids):
            dist.broadcast(t, src=0)
        routing["wi"] = (weights, ids)
        backend.runtime.clear_errors()
        out = backend.forward(layer, x, None)
        torch.cuda.synchronize()
        truth = _truth(w1, w1s, w2, w2s, inter, x, weights, ids, experts)
        dist.all_reduce(truth)
        err_mega = rel_l2(out, truth)
        err_prod = rel_l2(production(x, weights, ids), truth)
        # a16w4 below 256 global tokens (as fused_moe's decode): near-exact; above,
        # both run FP4 activations and carry the same quantization error.
        bound = 0.02 if rows < 256 else 1.15 * err_prod + 0.02
        gathered = [torch.empty_like(out) for _ in range(tp)]
        dist.all_gather(gathered, out)
        same = all(torch.equal(gathered[0], g) for g in gathered)
        flags = torch.tensor([backend.runtime.poll_errors()], device=device)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        ok = err_mega <= bound and same and not int(flags[0]) and out.shape == x.shape
        if rank == 0:
            print(
                f"rows={rows:4d} mega-vs-truth={err_mega:.4f} "
                f"production-vs-truth={err_prod:.4f} bound={bound:.4f} "
                f"ranks-equal={same} watchdog={int(flags[0])} {'ok' if ok else 'FAIL'}",
                flush=True,
            )
        if not ok:
            failures.append(rows)

    # Graph replay on new inputs == eager (the layer's launch args are baked at capture).
    for rows in GRAPH_ROWS:
        gen = torch.Generator(device=device).manual_seed(5 + rows)
        cases = []
        for _ in range(3):
            x = torch.randn(
                (rows, HIDDEN), dtype=torch.bfloat16, device=device, generator=gen
            )
            case = (x, *_route(rows, gen, device, experts))
            for t in case:
                dist.broadcast(t, src=0)
            cases.append(case)
        static = [t.clone() for t in cases[0]]
        routing["wi"] = (static[1], static[2])
        eager = []
        for x, w, i in cases:
            for dst, src in zip(static, (x, w, i)):
                dst.copy_(src)
            eager.append(backend.forward(layer, static[0], None).clone())
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                backend.forward(layer, static[0], None)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = backend.forward(layer, static[0], None)
        equal = True
        for k in (0, 1, 2, 1, 0, 2):
            for dst, src in zip(static, cases[k]):
                dst.copy_(src)
            graph.replay()
            equal = equal and torch.equal(captured, eager[k])
        torch.cuda.synchronize()
        flags = torch.tensor(
            [backend.runtime.poll_errors(), int(not equal)], device=device
        )
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                f"graph rows={rows}: replay==eager {not int(flags[1])} watchdog={int(flags[0])}",
                flush=True,
            )
        if int(flags[0]) or int(flags[1]):
            failures.append(f"graph{rows}")

    fail = torch.tensor([len(failures)], device=device)
    dist.all_reduce(fail, op=dist.ReduceOp.MAX)
    backend.close()
    dist.destroy_process_group()
    sys.exit(1 if int(fail[0]) else 0)


if __name__ == "__main__":
    _worker()
