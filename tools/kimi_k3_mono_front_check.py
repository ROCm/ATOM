# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The replicated gfx1250 K2 front (``k2_front_tdm_gfx1250``) built with
``ATOM_KIMI_MONO_FRONT_PIPE`` sites against the default build, bit for bit,
on one GPU: per width, both launched on the same inputs (BF16 tokens, the
BF16 router and latent weights, the shared gate/up as ATOM's online
per_1x128 FP8, preshuffled), ITERS fresh inputs each, and every output
(router logits, latent FP4 and its E8M0 scales, shared hidden) compared as
bytes; each launch's resources and ``runner_gfx1250._resident_hazards``.

    python tools/kimi_k3_mono_front_check.py
    env: ROWS (1, the qlen-5 widths 5..40 and the qlen-8 ones 8..64),
    SITES (late,early: the variant's sites, "+"-separated variants), ITERS
    (2), INF (1: one more input with infinities in some rows, so latent
    groups take the E8M0 code 254 and row 0's outputs are NaN)
"""

import json
import os

import torch
from aiter import QuantType, dtypes
from aiter.ops.shuffle import shuffle_weight

from atom.models.kimi_k3.mono.config import EXPERTS, HIDDEN, LATENT
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as kf
from atom.models.kimi_k3.mono.runner_gfx1250 import _resident_hazards
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.compile import compile_kernels, launcher_resources
from atom.quantization.quark.utils import quant_weight_online

DEFAULT_ROWS = sorted({1, *range(5, 41, 5), *range(8, 65, 8)})
ROWS = [
    int(r) for r in os.environ.get("ROWS", ",".join(map(str, DEFAULT_ROWS))).split(",")
]
VARIANTS = [
    frozenset(s for s in v.split(",") if s)
    for v in os.environ.get("SITES", "late,early").split("+")
]
ITERS = int(os.environ.get("ITERS", "2"))
INF = os.environ.get("INF", "1") == "1"
dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(11)


def bf16(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)


w_router = bf16(EXPERTS, HIDDEN, scale=0.02)
w_latent = bf16(LATENT, HIDDEN, scale=0.02)
q, s = quant_weight_online(
    bf16(kf.SHARED_ROWS, HIDDEN, scale=0.02),
    QuantType.per_1x128,
    dtypes.fp8,
    scale_dtype=dtypes.fp8_e8m0,
)
w_shared, s_shared = shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()


def outputs(rows):
    return {
        "router_logits": torch.full((rows, EXPERTS), -7.0, device=dev),
        "latent_fp4": torch.full(
            (rows, LATENT // 8), -1, dtype=torch.int32, device=dev
        ),
        "latent_scales": torch.full(
            (rows, LATENT // 32), 7, dtype=torch.uint8, device=dev
        ),
        "shared_hidden": torch.full(
            (rows, kf.SHARED_COLUMNS), -3.0, dtype=torch.bfloat16, device=dev
        ),
    }


def launch(kernel, x, out):
    args = {
        "x": x.data_ptr(),
        "router_weight": w_router.data_ptr(),
        "latent_weight": w_latent.data_ptr(),
        "shared_weight": w_shared.data_ptr(),
        "shared_scale": s_shared.data_ptr(),
        **{name: t.data_ptr() for name, t in out.items()},
    }
    kernel.launcher(*kernel.abi.pack(args), stream=torch.cuda.current_stream())


def resources(kernel):
    compile_kernels([(kernel.launcher, kernel.abi)])
    launcher = getattr(kernel.launcher, "launch", kernel.launcher)
    res = launcher_resources(launcher)
    return {
        "vgpr": max(r["vgpr"] for r in res),
        "lds": max(r["lds"] for r in res),
        "hazards": list(_resident_hazards(launcher)),
    }


failed = []
for rows in ROWS:
    plan = Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=GFX1250.max_workgroup_lds_bytes,
        tokens=rows,
        nblocks=4,
    )
    base = kf.build_replicated_front(plan, fp8_shared=True, pipe_sites=frozenset())
    for sites in VARIANTS:
        variant = kf.build_replicated_front(plan, fp8_shared=True, pipe_sites=sites)
        mismatched = {}
        for it in range(ITERS + INF):
            x = bf16(rows, HIDDEN)
            if it == ITERS:
                # one infinite column: every sum infinite, its latent groups
                # code 254; row 0 a NaN-producing mix (code 0)
                x[:, 5] = float("inf")
                x[0, ::97] = float("-inf")
            want, got = outputs(rows), outputs(rows)
            launch(base, x, want)
            launch(variant, x, got)
            torch.cuda.synchronize()
            for name in want:
                a = want[name].view(torch.uint8)
                b = got[name].view(torch.uint8)
                if not torch.equal(a, b):
                    mismatched[name] = int((a != b).sum())
        rec = {
            "rows": rows,
            "sites": sorted(sites),
            "base": base.name,
            "variant": variant.name,
            "bitwise": not mismatched,
            "mismatched_bytes": mismatched,
            "resources": resources(variant),
        }
        if mismatched or rec["resources"]["hazards"]:
            failed.append((rows, sorted(sites)))
        print(json.dumps(rec), flush=True)
print(json.dumps({"failed": failed}), flush=True)
