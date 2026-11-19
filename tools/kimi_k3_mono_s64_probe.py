# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every gfx1250 Kimi-K3 mono decode kernel of a width, built as
``K3Gfx1250MonoDecodeRunner._build_width`` builds it (GUGU experts, replicated
finalize, FP8 in-projection and shared experts, the TP's in-kernel all-reduces)
and compiled without a launch: one JSON line a kernel with its resources and a
hash of its ``.text``, or the refusal its builder raised.

    ARCH=gfx1250 WIDTHS=40:5,60:5,64:1 python tools/kimi_k3_mono_s64_probe.py

env: WIDTHS (``rows:qlen`` list), KINDS (k1, k1q, oproj, front, finish, moe,
gather, allreduce, finalize, tail, legacy; default all but legacy), TPS (8,4).
AITER_FLYDSL_USE_HERD and the ATOM_KIMI_MONO_* switches as in serving.
"""

import hashlib
import json
import os
import struct
import traceback

from aiter.ops.triton.utils._triton import arch_info

# no device to probe: AITER's Triton modules ask for the arch at import
arch_info._CACHED_ARCH = arch_info._CACHED_ARCH or os.environ.get("ARCH", "gfx1250")

from atom.models.kimi_k3.mono import runner_gfx1250 as rm  # noqa: E402
from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    K1LocalBuildSpec,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_finalize_tdm_gfx1250 as fin
from atom.models.kimi_k3.mono.kernels.stages import (
    k2_finish_persistent_gfx1250 as lite,
)
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as front
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages import k2_tail_gfx1250 as tail
from atom.models.kimi_k3.mono.kernels.stages import moe_tdm_gfx1250 as moe
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.compile import _code_object, compile_kernels, kernel_resources
from atom.utils import envs

WIDTHS = [
    tuple(int(v) for v in w.split(":"))
    for w in os.environ.get("WIDTHS", "40:5,45:5,50:5,55:5,60:5,64:1").split(",")
]
KINDS = os.environ.get(
    "KINDS", "k1,k1q,oproj,front,finish,moe,gather,allreduce,finalize,tail"
).split(",")
TPS = [int(t) for t in os.environ.get("TPS", "8,4").split(",")]
NB = 4
PROJ = 1536


def text(elf: bytes) -> bytes:
    """The ELF64's ``.text`` section."""
    shoff, = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3A)

    def section(i):
        name, _, _, _, offset, size = struct.unpack_from(
            "<IIQQQQ", elf, shoff + i * shentsize
        )
        return name, offset, size

    _, stroff, _ = section(shstrndx)
    for i in range(shnum):
        name, offset, size = section(i)
        if elf[stroff + name : elf.index(b"\0", stroff + name)] == b".text":
            return elf[offset : offset + size]
    raise RuntimeError("no .text")


def plan(rows, nblocks):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=GFX1250.max_workgroup_lds_bytes,
        tokens=rows,
        nblocks=nblocks,
    )


def k1(rows, qlen, quant):
    spec = K1LocalBuildSpec(
        qlen=qlen,
        delta=False,
        state_len=qlen + 2,
        eps=1e-6,
        out_eps=1e-6,
        onorm_eps=1e-5,
        lower_bound=-5.0,
        write_idx=-1,
        quant_core=quant,
        external_seam=True,
        fp8_projection=True,
    )
    ep = plan(rows, 1).build_k1_local(spec).kernel
    return [(ep.launcher, ep.abi)]


def kernels(kind, rows, qlen):
    herd = rm._FLYDSL_USE_HERD and rm._FLYDSL_HERD_MIN_M <= rows <= rm._FLYDSL_HERD_MAX_M
    fused = rows <= envs.ATOM_KIMI_MONO_FUSED_MOE_ROWS
    if kind == "k1":
        return k1(rows, qlen, False)
    if kind == "k1q":
        if rows > rm.K1_PTPC_FUSION_MAX_ROWS:
            return []
        return k1(rows, qlen, True)
    if kind == "oproj":
        out = []
        for tp in TPS:
            k = oar.build_oproj_allreduce(plan(rows, 1), tp, PROJ)
            out.append((k.launcher, k.abi))
        return out
    if kind == "front":
        k = front.build_replicated_front(plan(rows, NB), fp8_shared=True)
        return [(k.launcher, k.abi)]
    if kind == "moe":
        if not fused:
            return []
        k = moe.build_moe(rows)
        from atom.mono.runtime.abi import KernelAbi

        abi = KernelAbi(
            ("ids", "latent", "scales", "w13", "w13s", "w2", "w2s", "inter",
             "grouped", "flags", "state")
        )  # fmt: skip
        return [(k.launcher, abi)]
    if kind == "finish":
        parallel = herd and fused and envs.ATOM_KIMI_MONO_PARALLEL_HERD
        out = []
        for ph in sorted({parallel, False} if herd else {False}):
            k = lite.build_packed_finish(
                plan(rows, NB),
                gu_interleave=True,
                include_routed=False,
                replicated_front=True,
                herd=herd,
                fp8_shared_down=True,
                precompute_herd=herd,
                parallel_herd=ph,
            )
            out.append((k.launcher, k.abi))
        return out
    if kind == "gather":
        out = []
        for tp in TPS:
            k = ar.build_gather_allreduce(tp, rows)
            out.append((k.launcher, k.abi))
        return out
    if kind == "allreduce":
        out = []
        for tp in TPS:
            k = ar.build_allreduce(tp, rows)
            out.append((k.launcher, k.abi))
        return out
    if kind == "finalize":
        k = fin.build_replicated_finalize(plan(rows, NB))
        return [(k.launcher, k.abi)]
    if kind == "tail":
        if rows > tail.tail_max_rows():
            return []
        out = []
        for tp in TPS:
            k = tail.build_k2_tail(tp, rows, 1e-5, identity_routes=True)
            out.append((k.launcher, k.abi))
        return out
    if kind == "legacy":
        from atom.models.kimi_k3.mono.kernels.stages import (
            k2_front_role_lds_gfx1250 as role_front,
        )
        from atom.models.kimi_k3.mono.kernels.stages import (
            k2_local_kernels_gfx1250 as local,
        )

        p = plan(rows, NB)
        built = [
            role_front.build_role_lds_front(p, include_attention=False, reset=False),
            lite.build_packed_finish(p, gu_interleave=False),
            local.build_finalize_local(p, input_normalized=True),
        ]
        return [(k.launcher, k.abi) for k in built]
    raise ValueError(kind)


for rows, qlen in WIDTHS:
    for kind in KINDS:
        try:
            pairs = kernels(kind, rows, qlen)
            compile_kernels(pairs)
        except Exception as error:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "rows": rows,
                        "qlen": qlen,
                        "kind": kind,
                        "refused": f"{type(error).__name__}: {error}"[:300],
                        "where": traceback.format_exc().splitlines()[-3][:200],
                    }
                ),
                flush=True,
            )
            continue
        for launcher, _ in pairs:
            for artifact in launcher._mem_cache.values():
                elf = _code_object(artifact._ir_text)
                body = text(elf)
                print(
                    json.dumps(
                        {
                            "rows": rows,
                            "qlen": qlen,
                            "kind": kind,
                            **kernel_resources(elf),
                            "kib": round(len(body) / 1024, 1),
                            "sha": hashlib.sha256(body).hexdigest()[:16],
                        }
                    ),
                    flush=True,
                )
