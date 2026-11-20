"""The replicated front and the AttnRes mixes at a wide step, on one gfx1250
GPU, layer_bench_tail's weights and widths:

* front: the router logits, MXFP4 latent and its scales, the shared experts'
  hidden at ``rows`` against the same rows run as steps of at most ``PART``,
  bit for bit (a row's outputs read only its own x);
* AttnRes, both gfx1250 mixes (``ATOM_KIMI_MONO_ATTN_RES_SLICED`` on and off):
  against the production Triton ``_attn_res_fused_kernel`` (allclose at
  bf16's tolerance, max diffs reported) and against its parts, bit for bit.

    ROWS=50,55,60,64 PART=25 python tools/kimi_k3_mono_attn_width_check.py"""

import json
import os
from types import SimpleNamespace

import torch
import triton
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

import atom.models.kimi_k3.mono.runner_gfx1250 as rm
from atom.model_ops.kimi_k3.attention_residual import (
    _ATTN_RES_BF16_CONFIG,
    _attn_res_fused_kernel,
)
from atom.models.kimi_k3.mono.config import EXPERTS, HIDDEN, LATENT
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as front
from atom.mono.plan.architecture import GFX1250
from atom.quantization.quark.utils import quantize_weight_to_fp8_128x128_blockscale

WIDTHS = [int(r) for r in os.environ.get("ROWS", "50,55,60,64").split(",")]
PART = int(os.environ.get("PART", "25"))
SH, NB = 768, 4
torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
gen = torch.Generator(device=dev).manual_seed(4)


def bf16(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)


def fp8_preshuffled(rows, cols):
    w = (torch.randn(rows, cols, generator=gen, device=dev) * 0.02).to(torch.bfloat16)
    q, s = quantize_weight_to_fp8_128x128_blockscale(
        w, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0
    )
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()


def diff(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return {
        "max_abs": float(f"{d.max().item():.3g}"),
        "mismatch": int((a != b).sum().item()),
        "n": a.numel(),
    }


w_gate, w_ld = bf16(EXPERTS, HIDDEN, scale=0.02), bf16(LATENT, HIDDEN, scale=0.02)
w_sgu, w_sgu_s = fp8_preshuffled(2 * SH, HIDDEN)
fronts = {}


def run_front(x):
    rows = x.size(0)
    if rows not in fronts:
        plan = Gfx1250KimiBuildPlan(
            threads=256,
            blocks=256,
            lds_bytes=GFX1250.max_workgroup_lds_bytes,
            tokens=rows,
            nblocks=NB,
        )
        fronts[rows] = front.build_replicated_front(plan, fp8_shared=True)
    out = {
        "router_logits": torch.empty(rows, EXPERTS, device=dev),
        "latent_fp4": torch.empty(rows, LATENT // 8, dtype=torch.int32, device=dev),
        "latent_scales": torch.empty(rows, LATENT // 32, dtype=torch.uint8, device=dev),
        "shared_hidden": torch.empty(rows, SH, dtype=torch.bfloat16, device=dev),
    }
    fronts[rows].launcher(
        x.data_ptr(),
        w_gate.data_ptr(),
        w_ld.data_ptr(),
        w_sgu.data_ptr(),
        w_sgu_s.data_ptr(),
        out["router_logits"].data_ptr(),
        out["latent_fp4"].data_ptr(),
        out["latent_scales"].data_ptr(),
        out["shared_hidden"].data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    return out


site = SimpleNamespace(
    score_weight=(torch.randn(HIDDEN, generator=gen, device=dev) * 0.05).float(),
    eps=1e-6,
    out_eps=1e-5,
    enabled=True,
    out_norm=SimpleNamespace(weight=bf16(HIDDEN) + 1.0),
)


def mono_attn_res(blocks, prefix, add, sliced):
    rm.envs.ATOM_KIMI_MONO_ATTN_RES = 1
    rm.envs.ATOM_KIMI_MONO_ATTN_RES_SLICED = 4 if sliced else 0
    out = rm._mono_attn_res(site, prefix, blocks[:, :NB], add, None)
    torch.cuda.synchronize()
    return out


def triton_attn_res(blocks, prefix, add):
    rows = prefix.size(0)
    out, pref = torch.empty_like(prefix), torch.empty_like(prefix)
    warps, stages, tile = _ATTN_RES_BF16_CONFIG
    view = blocks[:, :NB]
    _attn_res_fused_kernel[(rows,)](
        view, prefix, site.score_weight, out, out, add, add, pref,
        site.out_norm.weight, NB, NB + 1, HIDDEN, 1e-6, 1e-5, 1.0, 1.0,
        view.stride(0), view.stride(1), prefix.stride(0), out.stride(0),
        add.stride(0), add.stride(0), pref.stride(0), BL=tile,
        BD=triton.next_power_of_2(HIDDEN), num_stages=stages, num_warps=warps,
        DO_ADD=True, DO_ADD2=False, WRITE_PREF=True, OUT_NORM=True, QUANT=False,
    )  # fmt: skip
    torch.cuda.synchronize()
    return out, pref


for rows in WIDTHS:
    parts = [(r, min(r + PART, rows)) for r in range(0, rows, PART)]
    x = bf16(rows, HIDDEN)
    wide = run_front(x)
    split = [run_front(x[a:b].contiguous()) for a, b in parts]
    report = {"rows": rows, "parts": parts, "front": fronts[rows].name}
    for name in wide:
        report[f"front_{name}"] = diff(wide[name], torch.cat([s[name] for s in split]))
    report["front_repeat_mismatch"] = sum(
        int((run_front(x)[n] != wide[n]).sum().item()) for n in wide
    )
    blocks, prefix, add = bf16(rows, NB + 1, HIDDEN), bf16(rows, HIDDEN), bf16(rows, HIDDEN)
    ref = triton_attn_res(blocks, prefix, add)
    for sliced in (True, False):
        tag = "sliced" if sliced else "mix"
        got = mono_attn_res(blocks, prefix, add, sliced)
        pieces = [
            mono_attn_res(
                blocks[a:b].contiguous(),
                prefix[a:b].contiguous(),
                add[a:b].contiguous(),
                sliced,
            )
            for a, b in parts
        ]
        for i, name in enumerate(("out", "prefix")):
            report[f"{tag}_{name}_vs_triton"] = diff(got[i], ref[i])
            report[f"{tag}_{name}_allclose"] = bool(
                torch.allclose(got[i].float(), ref[i].float(), atol=2e-2, rtol=2e-2)
            )
            report[f"{tag}_{name}_vs_parts"] = diff(
                got[i], torch.cat([p[i] for p in pieces])
            )
    print(json.dumps(report), flush=True)
print("done", flush=True)
