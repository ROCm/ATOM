# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_PREFILL_AR_QR_MIN_TOKENS on real devices, under torch.compile.

The token-count choice between the fused all-reduce + Gemma RMSNorm and
quick-reduce + a separate norm is made inside a custom op so that one compiled
graph serves every size. This runs GemmaRMSNorm(fused_allreduce=True) at TP2
with INT4 quick-reduce, compiles it once at one token count and runs it at the
other (16 and 4096, threshold 2048) without a recompile, and checks the path
taken and the result against an fp32 all-reduce + Gemma norm reference.

Needs two ROCm devices; skipped otherwise.
"""

import os
import tempfile

import pytest
import torch
import torch.multiprocessing as mp

WORLD = 2
HIDDEN = 4096
MIN_TOKENS = 2048
SMALL, LARGE = 16, 4096
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD,
    reason=f"needs {WORLD} ROCm devices",
)


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _reference(x, residual, weight):
    """fp32 all-reduce, residual add, Gemma RMSNorm: x * (1 + w)."""
    import torch.distributed as dist

    xs = x.float()
    dist.all_reduce(xs)
    res = xs + residual.float()
    normed = res * torch.rsqrt(res.pow(2).mean(-1, keepdim=True) + EPS)
    return normed * (1.0 + weight.float()), res


def _inputs(rank, m, device):
    g = torch.Generator(device="cpu").manual_seed(1000 * m + rank)
    x = torch.randn(m, HIDDEN, generator=g).to(device, torch.bfloat16)
    g.manual_seed(m)  # residual: the same on every rank
    residual = torch.randn(m, HIDDEN, generator=g).to(device, torch.bfloat16)
    return x, residual


def _check(norm, per_token, rank, trace_m, run_m, split_calls):
    import torch._dynamo

    torch._dynamo.reset()
    compiled = torch.compile(norm, fullgraph=True)
    for i, m in enumerate((trace_m, run_m)):
        x, residual = _inputs(rank, m, f"cuda:{rank}")
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(residual, 0)
        before = split_calls[0]
        # The second size must reuse the graph traced at the first.
        with torch._dynamo.config.patch(error_on_recompile=i > 0):
            outs = compiled(x, residual)
        torch.cuda.synchronize()
        took_split = split_calls[0] - before
        assert took_split == (1 if m >= MIN_TOKENS else 0), (m, took_split)

        out_ref, res_ref = _reference(x, residual, norm.weight)
        if per_token:
            out_fp8, scale, out_bf16, res_out = outs
            assert out_fp8.shape == (m, HIDDEN) and scale.shape == (m, 1)
            assert out_bf16.shape == (m, HIDDEN)
            out = out_bf16
            deq = out_fp8.float() * scale.float()
            assert _cos(deq, out_ref) > 0.98, (m, _cos(deq, out_ref))  # FP8
        else:
            out, res_out = outs
        # INT4 quick-reduce is lossy; a norm with x * w instead of x * (1 + w)
        # would land near 0 here because w is centred on 0.
        floor = 0.98 if took_split else 0.999
        assert _cos(res_out, res_ref) > floor, (m, _cos(res_out, res_ref))
        assert _cos(out, out_ref) > floor, (m, _cos(out, out_ref))


def _worker(rank, init):
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
    os.environ["ATOM_PREFILL_AR_QR_MIN_TOKENS"] = str(MIN_TOKENS)
    os.environ["ATOM_FUSED_NORM_PER_TOKEN_QUANT"] = "1"
    torch.cuda.set_device(rank)

    from types import SimpleNamespace

    from aiter import QuantType, destroy_dist_env, init_dist_env

    init_dist_env(
        WORLD,
        rankID=rank,
        backend="nccl",
        distributed_init_method=init,
        local_rank=rank,
    )
    try:
        from atom.model_ops import layernorm as ln

        assert ln._tp_quick_reduce() is not None, "quick-reduce is off"

        split_calls = [0]
        split = ln._quick_reduce_split_norm

        def counting_split(*args, **kwargs):
            split_calls[0] += 1
            return split(*args, **kwargs)

        ln._quick_reduce_split_norm = counting_split

        g = torch.Generator(device="cpu").manual_seed(7)
        weight = (0.1 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16)
        for per_token in (False, True):
            qc = SimpleNamespace(quant_type=QuantType.per_Token) if per_token else None
            norm = ln.GemmaRMSNorm(
                HIDDEN, eps=EPS, quant_config=qc, write_bf16=True, fused_allreduce=True
            ).to(f"cuda:{rank}", torch.bfloat16)
            norm.weight.data.copy_(weight)
            assert norm.ar_split_min_tokens == MIN_TOKENS
            assert norm.per_token_quant is per_token
            for trace_m, run_m in ((SMALL, LARGE), (LARGE, SMALL)):
                _check(norm, per_token, rank, trace_m, run_m, split_calls)
    finally:
        destroy_dist_env()


def test_routed_norm_under_torch_compile_at_16_and_4096_tokens():
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_worker, args=(f"file://{d}/store",), nprocs=WORLD, join=True)
