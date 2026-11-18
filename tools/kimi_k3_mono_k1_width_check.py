"""K1 at a wide step against the same requests run as narrower steps, same
inputs (layer_bench_tail's K1 setup, external seam, FP8 in-projection): the
projection mailbox, core_out, and the updated recurrent and conv state, every
element compared bit for bit, each run on fresh clones of the state. A token's
projection and its request's recurrence do not depend on the other rows, so a
wide build must equal its parts run alone. Also: the projection against a torch
fp32 reference of prefix @ dequant(w_in).T, and a repeat (determinism).

Run on one gfx1250 GPU:
    WIDTHS=50:5,60:5,64:8 PART=25 python tools/kimi_k3_mono_k1_width_check.py
``PART``: the narrow steps' rows at most (whole requests)."""

import json
import os

import torch
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.k1_local_runner_gfx1250 import K1LocalRuntime
from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    K1LocalBuildSpec,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.quantization.quark.utils import quantize_weight_to_fp8_128x128_blockscale

WIDTHS = [
    tuple(int(v) for v in w.split(":"))
    for w in os.environ.get("WIDTHS", "50:5,55:5,60:5").split(",")
]
PART = int(os.environ.get("PART", "48"))
FUSED, PROJ = 6284, 1536
torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
gen = torch.Generator(device=dev).manual_seed(4)


def bf16(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)


w = (torch.randn(FUSED, HIDDEN, generator=gen, device=dev) * 0.02).to(torch.bfloat16)
q, s = quantize_weight_to_fp8_128x128_blockscale(
    w, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0
)
w_ref = (
    q.float().view(FUSED, HIDDEN // 128, 128)
    * s.float().repeat_interleave(128, 0)[:FUSED].view(FUSED, HIDDEN // 128, 1)
).view(FUSED, HIDDEN)
qp = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, -FUSED % 128)).view(
    dtypes.fp8
)
w_in, w_in_scale = shuffle_weight(qp, layout=(16, 16)).contiguous(), s.contiguous()
runners = {}


def runner(rows, qlen):
    if (rows, qlen) not in runners:
        plan = Gfx1250KimiBuildPlan(
            threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=1
        )
        spec = K1LocalBuildSpec(
            qlen=qlen,
            delta=False,
            state_len=qlen + 2,
            eps=1e-6,
            out_eps=1e-6,
            onorm_eps=1e-5,
            lower_bound=-5.0,
            write_idx=-1,
            quant_core=False,
            external_seam=True,
            fp8_projection=True,
        )
        runners[rows, qlen] = plan.make_k1_local_runner(spec).prepare(dev)
    return runners[rows, qlen]


def diff(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return {
        "max_abs": float(f"{d.max().item():.3g}"),
        "max_rel": float(f"{(d / b.abs().clamp_min(1e-2)).max().item():.3g}"),
        "mismatch": int((a != b).sum().item()),
        "n": a.numel(),
    }


def check(rows, qlen):
    req = rows // qlen
    prefix = bf16(rows, HIDDEN)
    blk1 = bf16(rows, 1, HIDDEN)
    conv0 = bf16(req, 4608, qlen + 2, scale=0.5)
    rs0 = (
        torch.randn((req * qlen + req, 12, 128, 128), generator=gen, device=dev) * 0.05
    ).float()
    st = (torch.arange(req * qlen, device=dev, dtype=torch.int32) + req).view(req, qlen)
    st[:, 0] = torch.arange(req, device=dev, dtype=torch.int32)
    in_nw = bf16(HIDDEN)
    w_fb = bf16(1536, 128, scale=0.05)
    conv_w = (torch.randn((4608, 4), generator=gen, device=dev) * 0.3).float()
    a_log = (torch.randn(12, generator=gen, device=dev) * 0.5).float()
    dt_bias = (torch.randn(1536, generator=gen, device=dev) * 0.5).float()
    on_w = torch.randn(128, generator=gen, device=dev).float()
    num_acc = torch.randint(1, qlen + 1, (req,), generator=gen, device=dev).to(
        torch.int32
    )

    def run(parts):
        """Requests [r0, r1) a launch each, on one fresh copy of the state."""
        conv, rs = conv0.clone(), rs0.clone()
        core = torch.zeros((rows, PROJ), dtype=torch.bfloat16, device=dev)
        proj = []
        for r0, r1 in parts:
            t0, t1 = r0 * qlen, r1 * qlen
            k1 = runner(t1 - t0, qlen)
            k1._scratch.zero_()
            sub_st = st[r0:r1]
            sub_blk = blk1[t0:t1]
            k1(
                K1LocalRuntime(
                    prefix=prefix[t0:t1],
                    delta=None,
                    blocks=sub_blk,
                    blk_sm=sub_blk.stride(0),
                    blk_sr=sub_blk.stride(1),
                    ares_nw=in_nw,
                    ares_qk=in_nw,
                    in_nw=in_nw,
                    w_in=w_in,
                    w_in_scale=w_in_scale,
                    w_fb=w_fb,
                    conv_w=conv_w,
                    conv_st=conv,
                    cs_seq=conv.stride(0),
                    cs_dim=conv.stride(1),
                    cs_tok=conv.stride(2),
                    a_log=a_log,
                    dt_bias=dt_bias,
                    on_w=on_w,
                    rstate=rs,
                    rs_seq=rs.stride(0),
                    st_idx=sub_st,
                    st_stride=sub_st.stride(0),
                    num_acc=num_acc[r0:r1],
                    core_out=core[t0:t1],
                    core_q=None,
                    core_scale=None,
                ),
                stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()
            off, size = k1.layout["proj"]
            proj.append(
                k1._scratch[off : off + size]
                .view(torch.int32)
                .view(t1 - t0, -1, 2)
                .clone()
            )
        return {
            "proj": torch.cat(proj),
            "core_out": core,
            "rstate": rs,
            "conv_st": conv,
        }

    per = max(1, PART // qlen)
    parts = [(r, min(r + per, req)) for r in range(0, req, per)]
    wide = run([(0, req)])
    again = run([(0, req)])
    split = run(parts)
    report = {
        "rows": rows,
        "qlen": qlen,
        "parts": [(a * qlen, b * qlen) for a, b in parts],
        "name": runner(rows, qlen).entrypoint.name,
    }
    for name in wide:
        report[name] = diff(wide[name], split[name])
    ref = prefix.float() @ w_ref.T
    cols = min(wide["proj"].shape[1], FUSED)
    for word in (0, 1):
        v = wide["proj"][..., word].view(torch.float32)[:, :cols]
        report[f"ref_word{word}"] = diff(v, ref[:, :cols])
    report["repeat_mismatch"] = {
        n: int((again[n] != wide[n]).sum().item()) for n in again
    }
    print(json.dumps(report), flush=True)


for rows, qlen in WIDTHS:
    check(rows, qlen)
print("done", flush=True)
