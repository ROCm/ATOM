# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Byte-exact comparisons against the two production calls, on gfx1250.

Run with pytest on the ROCm host. CPU CI reports these as pending/skipped;
the gate and head-projection tests in the sibling file run on CPU.
"""

import math
from dataclasses import dataclass

import pytest
import torch


def _tdm_gpu():
    if not torch.cuda.is_available():
        return False
    from aiter.ops.triton.utils._triton import arch_info

    return arch_info.is_tdm_avail()


pytestmark = pytest.mark.skipif(not _tdm_gpu(), reason="requires a TDM GPU")


@dataclass(frozen=True)
class Case:
    m: int
    h: int = 16  # DeepSeek-R1 TP8: 128 / 8 query heads
    l: int = 512
    p: int = 128
    r: int = 64
    bs: int = 64
    scale: float | None = 0.37
    neox: bool = True
    amplitude: float = 1.0
    padded_heads: int = 0


def exercise_case(case, *, replay=False):
    from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
    from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16

    from atom.model_ops.mla_fused.fused_kup_qpe import kup_qpe_fused

    c = case
    torch.manual_seed(3498)
    device = "cuda"
    dtype = torch.float8_e4m3fn
    # Real q projection views and fused KV projection views are strided.
    q = (
        torch.randn(c.m, c.h, c.p + c.r, device=device, dtype=torch.bfloat16)
        * c.amplitude
    )
    qn, qp = q.split([c.p, c.r], dim=-1)
    wk = torch.randn(c.h, c.l, c.p, device=device, dtype=torch.bfloat16)
    kv = (
        torch.randn(c.m, 1, c.l + c.r, device=device, dtype=torch.bfloat16)
        * c.amplitude
    )
    kn, kp = kv.split([c.l, c.r], dim=-1)
    positions = (torch.arange(c.m, device=device, dtype=torch.int64) * 7) % 4096
    phase = torch.arange(4096, device=device)[:, None] / (
        1 + torch.arange(c.r // 2, device=device)[None, :]
    )
    cos, sin = phase.cos().to(torch.bfloat16), phase.sin().to(torch.bfloat16)
    scale = None if c.scale is None else torch.tensor([c.scale], device=device)
    blocks = math.ceil((c.m + 64) / c.bs) + 2
    # Dispersed unique slots cover both 16-token groups and page transitions.
    slots = (torch.arange(c.m, device=device, dtype=torch.int64) * 17) % (blocks * c.bs)
    # Choose a coprime multiplier to keep slots unique for every block size.
    if math.gcd(17, blocks * c.bs) != 1:
        slots = torch.arange(c.m, device=device, dtype=torch.int64)
    slots[::7] = -1
    guard = 4096

    def guarded(shape):
        n = math.prod(shape)
        raw = torch.full((n + 2 * guard,), 0x55, dtype=torch.uint8, device=device)
        return raw, raw[guard:-guard].view(dtype).view(shape)

    ref_q_raw, ref_q_full = guarded((c.m, c.h + c.padded_heads, c.l + c.r))
    new_q_raw, new_q_full = guarded((c.m, c.h + c.padded_heads, c.l + c.r))
    ref_q, new_q = ref_q_full[:, : c.h], new_q_full[:, : c.h]
    ref_k_raw, ref_k = guarded((blocks, 1, c.bs, c.l + c.r))
    new_k_raw, new_k = guarded((blocks, 1, c.bs, c.l + c.r))

    def baseline():
        # Exactly MLAAttention._q_proj_and_k_up_proj's BF16 prestored call,
        # followed by its full production RoPE/cache call; no estimated share.
        batched_gemm_bf16(qn.transpose(0, 1), wk, YQ=ref_q[..., : c.l].transpose(0, 1))
        fused_qk_rope_cat_and_cache_mla(
            ref_q[..., : c.l],
            qp,
            kn,
            kp,
            ref_k,
            slots,
            positions,
            cos,
            sin,
            scale,
            c.neox,
            num_decode_toks_for_zeros=0,
            apply_scale=True,
            q_out=ref_q,
            shuffled_kv_cache=True,
            q_nope_prestored=True,
        )

    def candidate():
        kup_qpe_fused(
            qn.transpose(0, 1),
            wk,
            qp,
            kn,
            kp,
            new_k,
            slots,
            positions,
            cos,
            sin,
            scale,
            c.neox,
            new_q,
        )

    def compare():
        torch.cuda.synchronize()
        assert torch.equal(ref_q_raw, new_q_raw), f"q_out/padding byte mismatch: {c}"
        assert torch.equal(ref_k_raw, new_k_raw), f"KV/sentinel byte mismatch: {c}"
        for raw in (ref_q_raw, new_q_raw, ref_k_raw, new_k_raw):
            assert bool((raw[:guard] == 0x55).all())
            assert bool((raw[-guard:] == 0x55).all())

    baseline()
    candidate()
    compare()
    if replay:
        # Capture with warmed kernels, then change values and sentinel masks
        # in place. This tests descriptor lifetime and padded graph replay.
        graphs = []
        for op in (baseline, candidate):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op()
            graphs.append(graph)
        for i in range(3):
            q.add_(0.125)
            kv.sub_(0.0625)
            positions.copy_((positions + 13) % 4096)
            slots.copy_((torch.arange(c.m, device=device) + 19 * i) % (blocks * c.bs))
            slots[i::5] = -1
            for raw in (ref_q_raw, new_q_raw, ref_k_raw, new_k_raw):
                raw.fill_(0x55)
            for graph in graphs:
                graph.replay()
            compare()
    return baseline, candidate


@pytest.mark.parametrize("m", [64, 65, 128, 512, 1024, 1280, 1512, 1536, 2049])
@pytest.mark.parametrize("scale", [None, 1.0, 0.37])
@pytest.mark.parametrize("neox", [False, True])
def test_production_bytes(m, scale, neox):
    exercise_case(Case(m=m, scale=scale, neox=neox))


@pytest.mark.parametrize(
    "l,p,r",
    [
        (l, p, r)
        for l in (256, 512, 1024)
        for p in (32, 64, 128)
        for r in (32, 64, 128, 256)
        if (l, p, r) != (256, 128, 256)
    ],
)
def test_supported_shapes_compile_and_match(l, p, r):
    exercise_case(Case(m=65, l=l, p=p, r=r))


@pytest.mark.parametrize("m", [64, 65, 128, 512, 1512])
def test_graph_replay_bytes(m):
    exercise_case(Case(m=m), replay=True)


@pytest.mark.parametrize(
    "case",
    [
        Case(m=65, amplitude=100),
        Case(m=1512, amplitude=100),
        Case(m=65, padded_heads=2),
        Case(m=128, bs=16),
        Case(m=128, bs=128),
        Case(m=65, h=1),
        Case(m=65, h=128),
    ],
)
def test_saturation_strides_and_page_sizes(case):
    exercise_case(case)
