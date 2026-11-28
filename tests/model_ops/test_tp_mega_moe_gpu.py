# SPDX-License-Identifier: MIT
"""TP MegaMoE on gfx950 (4 GPUs), against the standard MoE path.

The test relaunches itself under ``torchrun --nproc_per_node=4`` and checks, on
synthetic weights of MiniMax-M3.1's TP4 shape (hidden 6144, 3072 / 4
intermediate, 128 MXFP4 routed experts top-4, routed scale 2, one MXFP8 shared
expert, SwiGLU-OAI limit 7):

* ``TPMegaMoE._routes``: sigmoid top-4 + bias + renorm, scaled by the routed
  scale, the shared expert appended as AITER's last expert at weight 1;
* ``TPMegaMoE.forward`` (routed + MXFP8 shared expert + in-kernel all-reduce,
  the pad rows of a step that is not a multiple of tp) against an fp32 torch
  reference and against the standard path (``aiter.fused_moe`` * routed scale
  + the shared expert, all-reduced), at decode and prefill row counts of the
  small-M and LB tiers;
* identical outputs on every rank, no watchdog bit;
* CUDA-graph replay on new inputs equal to the eager run.

Needs >= 4 gfx950 GPUs and an AITER whose MegaMoeTP takes the MXFP8 shared
expert (``w1_shared``, ROCm/aiter#8270).
"""

import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROWS = (1, 3, 4, 13, 16, 64, 100, 256, 257, 512, 1024, 4096)
GRAPH_ROWS = (3, 16, 256, 512)
HIDDEN, INTER_GLOBAL, ROUTED, TOP_K, LIMIT, SCALE = 6144, 3072, 128, 4, 7.0, 2.0


def _skip_reason():
    try:
        import torch

        if torch.cuda.device_count() < 4:
            return "needs 4 GPUs"
        if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
            return "needs gfx950"
        from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP

        if "w1_shared" not in inspect.signature(MegaMoeTP.__init__).parameters:
            return "AITER MegaMoeTP without the MXFP8 shared expert (w1_shared)"
    except Exception as exc:  # noqa: BLE001 - any import problem is a skip
        return f"no MegaMoeTP here: {exc!r}"
    return None


def test_tp_mega_moe_matches_the_standard_path_on_gfx950():
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


def _quant_mxfp8(rows, cols, magnitude, seed, device):
    """A row-major MXFP8 weight as the checkpoint holds it: e4m3 [rows, cols]
    and an e8m0 scale per 32 [rows, cols / 32]."""
    import torch

    gen = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn((rows, cols), device=device, generator=gen) * magnitude
    blk = w.view(rows, cols // 32, 32)
    e = torch.ceil(torch.log2(blk.abs().amax(-1).clamp_min(1e-30) / 448.0))
    e = e.clamp(-127, 127)
    q = (blk / torch.exp2(e)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(rows, cols), (e + 127).to(torch.uint8).view(torch.float8_e8m0fnu)


def _dequant8(q, scale):
    import torch

    s = torch.exp2(scale.view(torch.uint8).float() - 127.0)
    return q.float() * s.repeat_interleave(32, dim=1)


def _dequant4(q, scale, rows, cols):
    import torch
    from aiter.utility import fp4_utils

    f = fp4_utils.mxfp4_to_f32(q.view(torch.uint8).reshape(-1, cols // 2))
    s = fp4_utils.e8m0_to_f32(scale.view(torch.uint8).reshape(-1, cols // 32))
    return (f.view(-1, cols // 32, 32) * s.view(-1, cols // 32, 1)).view(-1, rows, cols)


def _fp4_round(x):
    """x through MXFP4 (per 32, as the kernels quantize activations)."""
    import aiter
    import torch
    from aiter import dtypes

    rows, cols = x.shape
    q, scale = aiter.get_torch_quant(aiter.QuantType.per_1x32)(
        x.to(torch.bfloat16).contiguous(), quant_dtype=dtypes.fp4x2
    )
    return _dequant4(q, scale, rows, cols)[0]


def _swiglu_oai(h, inter):
    import torch

    g, u = h[:, :inter].clamp(max=LIMIT), h[:, inter:].clamp(-LIMIT, LIMIT)
    return g * torch.sigmoid(1.702 * g) * (u + 1.0)


def _truth(w, inter, x, weights, ids, a4=False):
    """fp32 partial sum of this rank's slice: the routed experts at the routed
    weights (scaled) + the shared expert at weight 1. ``a4``: the activations
    of both GEMMs through MXFP4, as the kernels' a4w4 tiers run them (the
    shared expert's too: its weights stay MXFP8)."""
    import torch

    rnd = _fp4_round if a4 else (lambda t: t.float())
    x = rnd(x)
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in range(ROUTED):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        a = _dequant4(
            w.w1[e : e + 1],
            w.w1s[e * 2 * inter : (e + 1) * 2 * inter],
            2 * inter,
            HIDDEN,
        )[0]
        b = _dequant4(
            w.w2[e : e + 1], w.w2s[e * HIDDEN : (e + 1) * HIDDEN], HIDDEN, inter
        )[0]
        act = rnd(_swiglu_oai(x[tok] @ a.t(), inter))
        out.index_add_(0, tok, (act @ b.t()) * weights[tok, slot, None])
    return out + _shared(w, inter, x, rnd)


def _shared(w, inter, x, rnd=lambda t: t.float()):
    act = rnd(_swiglu_oai(rnd(x) @ _dequant8(w.sh1, w.sh1s).t(), inter))
    return act @ _dequant8(w.sh2, w.sh2s).t()


def _worker():
    import torch
    import torch.distributed as dist
    from aiter import ActivationType, QuantType, topk_gating
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
        tp_shared_expert_weights,
    )

    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    cpu_group = dist.new_group(backend="gloo")
    inter = INTER_GLOBAL // tp

    def rel_l2(a, b):
        d = (a.float() - b.float()).pow(2).sum()
        n = b.float().pow(2).sum()
        t = torch.stack([d, n])
        dist.all_reduce(t)
        return float((t[0] / t[1]).sqrt())

    base = 1000 * rank
    w = SimpleNamespace()
    w.w1, w.w1s = _quant_experts(ROUTED, 2 * inter, HIDDEN, HIDDEN**-0.25, base, device)
    w.w2, w.w2s = _quant_experts(ROUTED, HIDDEN, inter, inter**-0.25, base + 7, device)
    w.sh1, w.sh1s = _quant_mxfp8(2 * inter, HIDDEN, HIDDEN**-0.25, base + 31, device)
    w.sh2, w.sh2s = _quant_mxfp8(HIDDEN, inter, inter**-0.25, base + 37, device)
    bias = torch.randn(ROUTED, device=device) * 0.1
    dist.broadcast(bias, src=0)
    layer = SimpleNamespace(
        hidden_size=HIDDEN,
        top_k=TOP_K,
        intermediate_size_per_partition=inter,
        local_num_experts=ROUTED,
        activation=ActivationType.Swiglu,
        swiglu_limit=LIMIT,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=SCALE,
        scoring_func="sigmoid",
        w13_weight=shuffle_weight(w.w1, layout=(16, 16)),
        w13_weight_scale=fp4_utils.e8m0_shuffle(w.w1s),
        w2_weight=shuffle_weight(w.w2, layout=(16, 16)),
        w2_weight_scale=fp4_utils.e8m0_shuffle(w.w2s),
        tp_mega_shared=tp_shared_expert_weights(w.sh1, w.sh1s, w.sh2, w.sh2s),
    )
    # As initialize_sp_mega_moe builds it, without the big-M path (<= 4096 rows).
    runtime = MegaMoeTP(
        MegaMoeTPConfig(
            rank=rank,
            world_size=tp,
            model_dim=HIDDEN,
            inter_dim=inter,
            experts=ROUTED + 1,
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
    assert backend.runs(max(ROWS)) and not backend.bigm_rows
    failures = []

    def logits_for(rows, gen):
        logits = torch.randn(
            (rows, ROUTED), dtype=torch.bfloat16, device=device, generator=gen
        )
        dist.broadcast(logits, src=0)
        return logits

    def standard(x, logits):
        """The standard path: routing unscaled, fused_moe * routed scale + the
        shared expert (here its fp32 reference), all-reduced."""
        rows = x.shape[0]
        weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=device)
        ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
        topk_gating(weights, ids, logits, bias, True, 1.0, score_func="sigmoid")
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
        out = out.float() * SCALE + _shared(w, inter, x)
        dist.all_reduce(out)
        return out, weights, ids

    for rows in ROWS:
        gen = torch.Generator(device=device).manual_seed(77 + rows)
        x = torch.randn(
            (rows, HIDDEN), dtype=torch.bfloat16, device=device, generator=gen
        )
        dist.broadcast(x, src=0)
        logits = logits_for(rows, gen)
        backend.runtime.clear_errors()
        out = backend.forward(layer, x, logits)
        torch.cuda.synchronize()
        ref, weights, ids = standard(x, logits)
        # The routing the kernel took: the standard one, scaled, + the shared.
        mw, mi = backend._shared_routes[rows]
        routes_ok = (
            torch.equal(mi[:, :TOP_K], ids)
            and bool((mi[:, TOP_K] == ROUTED).all())
            and torch.allclose(mw[:, :TOP_K], weights * SCALE)
            and bool((mw[:, TOP_K] == 1.0).all())
        )
        # bf16 activations below 256 global tokens (as fused_moe's decode):
        # against the fp32 truth; from 256 the kernel quantizes every
        # activation to MXFP4, the shared expert's included (the standard path
        # quantizes the routed experts' only): against that truth.
        a4 = rows >= 256
        truth = _truth(w, inter, x, weights * SCALE, ids, a4=a4)
        dist.all_reduce(truth)
        err_mega = rel_l2(out, truth)
        err_std = rel_l2(ref, truth)
        bound = 0.10 if a4 else 0.02
        gathered = [torch.empty_like(out) for _ in range(tp)]
        dist.all_gather(gathered, out)
        same = all(torch.equal(gathered[0], g) for g in gathered)
        flags = torch.tensor([backend.runtime.poll_errors()], device=device)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        ok = (
            err_mega <= bound
            and routes_ok
            and same
            and not int(flags[0])
            and out.shape == x.shape
        )
        if rank == 0:
            print(
                f"rows={rows:4d} truth={'a4' if a4 else 'fp32'} "
                f"mega-vs-truth={err_mega:.4f} standard-vs-truth={err_std:.4f} "
                f"bound={bound:.4f} "
                f"routes={routes_ok} ranks-equal={same} watchdog={int(flags[0])} "
                f"{'ok' if ok else 'FAIL'}",
                flush=True,
            )
        if not ok:
            failures.append(rows)

    # Graph replay on new inputs == eager (the launch args are baked at capture).
    for rows in GRAPH_ROWS:
        gen = torch.Generator(device=device).manual_seed(5 + rows)
        cases = []
        for _ in range(3):
            x = torch.randn(
                (rows, HIDDEN), dtype=torch.bfloat16, device=device, generator=gen
            )
            dist.broadcast(x, src=0)
            cases.append((x, logits_for(rows, gen)))
        static = [t.clone() for t in cases[0]]
        eager = []
        for case in cases:
            for dst, src in zip(static, case):
                dst.copy_(src)
            eager.append(backend.forward(layer, *static).clone())
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                backend.forward(layer, *static)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = backend.forward(layer, *static)
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
                f"graph rows={rows}: replay==eager {not int(flags[1])} "
                f"watchdog={int(flags[0])}",
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
