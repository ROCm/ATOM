"""The attention side of a Kimi-K3 KDA decode layer on one gfx1250 GPU, in the
serving order and under one CUDA graph, the layers' weights rotated so every
launch starts as cold as in serving: attention AttnRes, K1, MLP AttnRes, the
replicated front (layer_bench_tail's setup; o_proj and the rest of K2 are not
launched). Prints the graph's time a layer and each kernel's mean a layer
under the profiler.

    ROWS=40,48,56,64 QLEN=8 python tools/kimi_k3_mono_attn_bench.py"""

import collections
import json
import os
import statistics
import tempfile
from types import SimpleNamespace

import torch
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

import atom.models.kimi_k3.mono.runner_gfx1250 as rm
from atom.models.kimi_k3.mono.config import EXPERTS, HIDDEN, LATENT
from atom.models.kimi_k3.mono.k1_local_runner_gfx1250 import K1LocalRuntime
from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    K1LocalBuildSpec,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as front
from atom.mono.plan.architecture import GFX1250
from atom.quantization.quark.utils import quantize_weight_to_fp8_128x128_blockscale

COPIES = int(os.environ.get("COPIES", "4"))
LAYERS = int(os.environ.get("LAYERS", "8"))
REPS = int(os.environ.get("REPS", "7"))
WIDTHS = [int(r) for r in os.environ.get("ROWS", "40,48,56,64").split(",")]
QLEN = int(os.environ.get("QLEN", "8"))
FUSED, PROJ, SH, NB = 6284, 1536, 768, 4
torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
gen = torch.Generator(device=dev).manual_seed(4)
rm.envs.ATOM_KIMI_MONO_ATTN_RES = 1
rm.envs.ATOM_KIMI_MONO_ATTN_RES_SLICED = int(os.environ.get("SLICED", "4"))


def bf16(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)


def fp8_preshuffled(rows, cols, pad=False):
    w = (torch.randn(rows, cols, generator=gen, device=dev) * 0.02).to(torch.bfloat16)
    q, s = quantize_weight_to_fp8_128x128_blockscale(
        w, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0
    )
    if pad:
        q = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, -rows % 128)).view(
            dtypes.fp8
        )
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()


def site():
    return SimpleNamespace(
        score_weight=(torch.randn(HIDDEN, generator=gen, device=dev) * 0.05).float(),
        eps=1e-6,
        out_eps=1e-5,
        enabled=True,
        out_norm=SimpleNamespace(weight=bf16(HIDDEN) + 1.0),
    )


layers = []
for _ in range(COPIES):
    w_in, w_in_s = fp8_preshuffled(FUSED, HIDDEN, pad=True)
    sgu, sgu_s = fp8_preshuffled(2 * SH, HIDDEN)
    layers.append(
        SimpleNamespace(
            w_in=w_in,
            w_in_scale=w_in_s,
            w_gate=bf16(EXPERTS, HIDDEN, scale=0.02),
            w_ld=bf16(LATENT, HIDDEN, scale=0.02),
            w_sgu=sgu,
            w_sgu_scale=sgu_s,
            attn_site=site(),
            mlp_site=site(),
            in_nw=bf16(HIDDEN),
            w_fb=bf16(1536, 128, scale=0.05),
            conv_w=(torch.randn((4608, 4), generator=gen, device=dev) * 0.3).float(),
            a_log=(torch.randn(12, generator=gen, device=dev) * 0.5).float(),
            dt_bias=(torch.randn(1536, generator=gen, device=dev) * 0.5).float(),
            on_w=torch.randn(128, generator=gen, device=dev).float(),
        )
    )


def build_width(rows):
    req = rows // QLEN
    spec = K1LocalBuildSpec(
        qlen=QLEN,
        delta=False,
        state_len=QLEN + 2,
        eps=1e-6,
        out_eps=1e-6,
        onorm_eps=1e-5,
        lower_bound=-5.0,
        write_idx=-1,
        quant_core=False,
        external_seam=True,
        fp8_projection=True,
    )
    k1 = (
        Gfx1250KimiBuildPlan(
            threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=1
        )
        .make_k1_local_runner(spec)
        .prepare(dev)
    )
    fr = front.build_replicated_front(
        Gfx1250KimiBuildPlan(
            threads=256,
            blocks=256,
            lds_bytes=GFX1250.max_workgroup_lds_bytes,
            tokens=rows,
            nblocks=NB,
        ),
        fp8_shared=True,
    )
    blk1 = bf16(rows, 1, HIDDEN)
    conv = bf16(req, 4608, QLEN + 2, scale=0.5)
    rs = (
        torch.randn((req * QLEN + req, 12, 128, 128), generator=gen, device=dev) * 0.05
    ).float()
    st = (torch.arange(req * QLEN, device=dev, dtype=torch.int32) + req).view(req, QLEN)
    st[:, 0] = torch.arange(req, device=dev, dtype=torch.int32)
    num_acc = torch.full((req,), QLEN, device=dev, dtype=torch.int32)
    core = torch.empty((rows, PROJ), dtype=torch.bfloat16, device=dev)
    blocks = bf16(rows, NB + 1, HIDDEN)
    prefix, add, attn_out = bf16(rows, HIDDEN), bf16(rows, HIDDEN), bf16(rows, HIDDEN)
    logits = torch.empty(rows, EXPERTS, device=dev)
    latent = torch.empty(rows, LATENT // 8, dtype=torch.int32, device=dev)
    scales = torch.empty(rows, LATENT // 32, dtype=torch.uint8, device=dev)
    shared = torch.empty(rows, SH, dtype=torch.bfloat16, device=dev)

    def layer(i, stream):
        w = layers[i % COPIES]
        normed, pref = rm._mono_attn_res(w.attn_site, prefix, blocks[:, :NB], add, None)
        k1(
            K1LocalRuntime(
                prefix=normed, delta=None, blocks=blk1, blk_sm=blk1.stride(0),
                blk_sr=blk1.stride(1), ares_nw=w.in_nw, ares_qk=w.in_nw,
                in_nw=w.in_nw, w_in=w.w_in, w_in_scale=w.w_in_scale, w_fb=w.w_fb,
                conv_w=w.conv_w, conv_st=conv, cs_seq=conv.stride(0),
                cs_dim=conv.stride(1), cs_tok=conv.stride(2), a_log=w.a_log,
                dt_bias=w.dt_bias, on_w=w.on_w, rstate=rs, rs_seq=rs.stride(0),
                st_idx=st, st_stride=st.stride(0), num_acc=num_acc, core_out=core,
                core_q=None, core_scale=None,
            ),
            stream=stream,
        )  # fmt: skip
        x, _ = rm._mono_attn_res(w.mlp_site, pref, blocks[:, :NB], attn_out, None)
        fr.launcher(
            x.data_ptr(), w.w_gate.data_ptr(), w.w_ld.data_ptr(), w.w_sgu.data_ptr(),
            w.w_sgu_scale.data_ptr(), logits.data_ptr(), latent.data_ptr(),
            scales.data_ptr(), shared.data_ptr(), stream=stream,
        )  # fmt: skip

    layer.keep = (k1, fr)
    return layer


KINDS = (
    ("attn_res", "attn_res"),
    ("k3_k1_gfx1250", "K1"),
    ("replicated_front", "front"),
)


def measure(layer):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(COPIES):
            layer(i, s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(LAYERS):
            layer(i, torch.cuda.current_stream())
    g.replay()
    torch.cuda.synchronize()
    totals = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        totals.append(a.elapsed_time(b) * 1000 / LAYERS)
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
    path = tempfile.mktemp(suffix=".json")
    prof.export_chrome_trace(path)
    times = collections.defaultdict(list)
    for e in json.load(open(path))["traceEvents"]:
        if e.get("ph") == "X" and e.get("cat") == "kernel":
            for key, label in KINDS:
                if key in e["name"]:
                    times[label].append(e["dur"])
                    break
    per_layer = {k: round(sum(v) / (3 * LAYERS), 2) for k, v in times.items()}
    return g, round(statistics.median(totals), 1), per_layer


keep = []
for rows in WIDTHS:
    layer = build_width(rows)
    g, layer_us, kernels = measure(layer)
    keep.append((layer, g))
    print(
        json.dumps(
            {
                "tag": os.environ.get("TAG", ""),
                "rows": rows,
                "qlen": QLEN,
                "layer_us": layer_us,
                "kernels_us_per_layer": kernels,
            }
        ),
        flush=True,
    )
print("done", flush=True)
