"""K1 with its whole heads split over SPLIT CTAs (ATOM_KIMI_MONO_K1_HEAD_SPLIT,
default auto) against one CTA a head, same inputs (layer_bench_tail's K1 setup,
external seam, qlen 5): the projection mailbox, core_out, and the updated
recurrent and conv state, each run on fresh clones of the state, every element
compared bit for bit, and a repeat of the split kernel (determinism).

Run on one gfx1250 GPU: ROWS=20,40 python tools/kimi_k3_mono_k1_head_split_check.py"""

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

SPLIT = os.environ.get("SPLIT", "auto")
ORDER = os.environ.get("ORDER", "base,split").split(",")
WIDTHS = [int(r) for r in os.environ.get("ROWS", "10,20,40").split(",")]
FUSED, PROJ, QLEN = 6284, 1536, 5
torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
gen = torch.Generator(device=dev).manual_seed(4)


def bf16(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)


w = (torch.randn(FUSED, HIDDEN, generator=gen, device=dev) * 0.02).to(torch.bfloat16)
q, s = quantize_weight_to_fp8_128x128_blockscale(
    w, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0
)
qp = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, -FUSED % 128)).view(
    dtypes.fp8
)
w_in, w_in_scale = shuffle_weight(qp, layout=(16, 16)).contiguous(), s.contiguous()


def diff(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return {
        "max_abs": float(f"{d.max().item():.3g}"),
        "max_rel": float(f"{(d / b.abs().clamp_min(1e-2)).max().item():.3g}"),
        "mismatch": int((a != b).sum().item()),
        "n": a.numel(),
    }


def check(rows):
    req = rows // QLEN
    plan = Gfx1250KimiBuildPlan(
        threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=1
    )
    base_spec = K1LocalBuildSpec(
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
    runners = {}
    for tag, value in (("base", ""), ("split", SPLIT)):
        os.environ["ATOM_KIMI_MONO_K1_HEAD_SPLIT"] = value
        runners[tag] = plan.make_k1_local_runner(base_spec).prepare(dev)
    os.environ["ATOM_KIMI_MONO_K1_HEAD_SPLIT"] = ""
    prefix = bf16(rows, HIDDEN)
    blk1 = bf16(rows, 1, HIDDEN)
    conv0 = bf16(req, 4608, QLEN + 2, scale=0.5)
    rs0 = (
        torch.randn((req * QLEN + req, 12, 128, 128), generator=gen, device=dev) * 0.05
    ).float()
    st = (torch.arange(req * QLEN, device=dev, dtype=torch.int32) + req).view(req, QLEN)
    st[:, 0] = torch.arange(req, device=dev, dtype=torch.int32)
    in_nw = bf16(HIDDEN)
    w_fb, conv_w = (
        bf16(1536, 128, scale=0.05),
        (torch.randn((4608, 4), generator=gen, device=dev) * 0.3).float(),
    )
    a_log = (torch.randn(12, generator=gen, device=dev) * 0.5).float()
    dt_bias = (torch.randn(1536, generator=gen, device=dev) * 0.5).float()
    on_w = torch.randn(128, generator=gen, device=dev).float()
    num_acc = torch.full((req,), QLEN, device=dev, dtype=torch.int32)

    def run(k1):
        conv, rs = conv0.clone(), rs0.clone()
        core = torch.zeros((rows, PROJ), dtype=torch.bfloat16, device=dev)
        k1._scratch.zero_()
        k1(
            K1LocalRuntime(
                prefix=prefix,
                delta=None,
                blocks=blk1,
                blk_sm=blk1.stride(0),
                blk_sr=blk1.stride(1),
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
                st_idx=st,
                st_stride=st.stride(0),
                num_acc=num_acc,
                core_out=core,
                core_q=None,
                core_scale=None,
            ),
            stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        off, size = k1.layout["proj"]
        pairs = (
            k1._scratch[off : off + size].view(torch.int32).view(rows, -1, 2).clone()
        )
        return {"proj": pairs, "core_out": core, "rstate": rs, "conv_st": conv}

    out = {}
    for tag in ORDER:
        print("run", rows, tag, runners[tag].entrypoint.name, flush=True)
        out[tag] = run(runners[tag])
    again = run(runners["split"])
    report = {"rows": rows, "names": {t: k.entrypoint.name for t, k in runners.items()}}
    # which word of a pair is the f32 value: the one that varies across rows
    for word in (0, 1):
        report[f"proj_word{word}"] = diff(
            out["split"]["proj"][..., word].view(torch.float32),
            out["base"]["proj"][..., word].view(torch.float32),
        )
    for name in ("core_out", "rstate", "conv_st"):
        report[name] = diff(out["split"][name], out["base"][name])
    report["split_repeat_mismatch"] = {
        n: int((again[n] != out["split"][n]).sum().item()) for n in again
    }
    print(json.dumps(report), flush=True)
    return runners, out


keep = [check(rows) for rows in WIDTHS]
print("done", flush=True)
