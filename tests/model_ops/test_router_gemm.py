# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The padded bf16 MoE router GEMM (``atom/model_ops/router_gemm.py``).

The table checks run anywhere. The GPU checks hold the dispatch to what the
model needs from it: every decode M and the prefill fallback agree with an fp32
reference, the padding rows read back as zeros, a CUDA graph replays the tuned
kernel on fresh inputs, and the routing kernels read the padded-stride logits
the same as contiguous ones.
"""

import re

import pytest
import torch

from atom.model_ops.router_gemm import (
    _FLYDSL_ROUTER_KERNELS,
    ROUTER_ROW_ALIGN,
    padded_router_rows,
    select_router_kernel,
)

# aiter's tuned-CSV FlyDSL hgemm name, bf16 in and out, no bias, no K tail.
_KERNEL_NAME = re.compile(
    r"flydsl_hgemm_abf16_wbf16_bf16_t\d+x\d+x\d+x\d+_ks\d+_w\d+x\d+x\d+"
    r"_bias0_ktail0_gm\d+_p(ft|ht)_gfx\d+"
)


@pytest.mark.parametrize(
    ("rows", "padded"), [(1, 16), (16, 16), (257, 272), (512, 512), (513, 528)]
)
def test_router_rows_pad_to_the_alignment(rows, padded):
    assert padded_router_rows(rows) == padded
    assert padded % ROUTER_ROW_ALIGN == 0


def test_router_tables_are_well_formed():
    assert _FLYDSL_ROUTER_KERNELS, "no router kernel table"
    for (gfx, cu_num, padded_rows, k), table in _FLYDSL_ROUTER_KERNELS.items():
        assert padded_rows % ROUTER_ROW_ALIGN == 0
        assert k % 64 == 0 and cu_num > 0
        bounds = [largest_m for largest_m, _ in table]
        assert bounds == sorted(set(bounds)), f"M buckets not ascending: {bounds}"
        for _, name in table:
            assert _KERNEL_NAME.fullmatch(name), name
            assert name.endswith(f"_{gfx}"), name


def test_select_router_kernel_buckets_m():
    table = ((4, "a"), (16, "b"), (64, "c"))
    assert [select_router_kernel(table, m) for m in (1, 4, 5, 16, 17, 64)] == [
        "a",
        "a",
        "b",
        "b",
        "c",
        "c",
    ]
    assert select_router_kernel(table, 65) is None


# ── GPU ────────────────────────────────────────────────────────────────────

gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the router GEMM runs AITER kernels"
)

N_ROUTED, K = 512, 4096
N = N_ROUTED + 1  # Qwen3.5-397B: 512 routed experts + the shared_expert_gate row


def _padded_weight(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    padded = torch.nn.functional.pad(w, (0, 0, 0, padded_router_rows(N) - N))
    return w, padded


def _check_against_fp32(out, x, w, out_dtype=torch.bfloat16):
    ref = x.float() @ w.float().t()
    if out_dtype == torch.float32:
        # fp32 split-K atomics: reassociation only.
        torch.testing.assert_close(out[:, :N], ref, atol=1e-4, rtol=1e-4)
    else:
        # bf16 split-K atomics round every partial: a few bf16 ulps, not one
        # (up to 0.056 absolute on |logit| <= 7 with the checkpoint's weights).
        # A lost or doubled K partition is off by O(1) and still fails.
        torch.testing.assert_close(out[:, :N].float(), ref, atol=6e-2, rtol=3e-2)


def _require_table():
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    from atom.model_ops.router_gemm import router_kernel_table

    table = router_kernel_table(get_gfx(), get_cu_num(), padded_router_rows(N), K)
    if not table:
        pytest.skip(f"no router kernel table for {get_gfx()} / {get_cu_num()} CUs")
    return table


@gpu
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "m", [1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 128, 200, 256, 1024]
)
def test_router_gemm_matches_fp32(m, out_dtype):
    from atom.model_ops.linear import padded_router_gemm

    _require_table()
    w, padded = _padded_weight()
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    out = padded_router_gemm(x, padded, out_dtype)
    assert out.shape == (m, padded.shape[0]) and out.dtype == out_dtype
    _check_against_fp32(out, x, w, out_dtype)
    assert torch.all(out[:, N:] == 0), "zero padding rows must produce zeros"


@gpu
def test_every_table_kernel_resolves_and_runs():
    from atom.model_ops.linear import _router_flydsl_params, padded_router_gemm

    table = _require_table()
    w, padded = _padded_weight()
    for largest_m, name in table:
        assert _router_flydsl_params(name) is not None, name
        x = torch.randn(largest_m, K, device="cuda", dtype=torch.bfloat16)
        _check_against_fp32(padded_router_gemm(x, padded, torch.bfloat16), x, w)


@gpu
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m", [1, 4, 32, 64, 128, 256])
def test_router_gemm_replays_in_a_cuda_graph(m, out_dtype):
    from atom.model_ops.linear import padded_router_gemm

    _require_table()
    w, padded = _padded_weight()
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):  # compile and allocate outside the capture
            padded_router_gemm(x, padded, out_dtype)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Two calls in one graph share the split-K semaphores, as 60 layers do.
        out = padded_router_gemm(x, padded, out_dtype)
        out2 = padded_router_gemm(x, padded, out_dtype)
    for _ in range(3):  # split-K tickets must hand over across replays
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        _check_against_fp32(out, x, w, out_dtype)
        _check_against_fp32(out2, x, w, out_dtype)


@gpu
def test_router_linear_pads_after_load_and_slices_the_output(monkeypatch):
    from types import SimpleNamespace

    from atom.model_ops import linear as linear_mod

    _require_table()
    monkeypatch.setattr(
        linear_mod,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    w, _ = _padded_weight()
    x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
    outs = {}
    for enabled in (False, True):
        monkeypatch.setenv("ATOM_ROUTER_FLYDSL_GEMM", "1" if enabled else "0")
        with torch.device("cuda"):
            layer = linear_mod.MergedReplicatedRouterLinear(
                K, [N_ROUTED, 1], bias=False, quant_config=None, prefix="mlp.gate"
            )
        layer.weight_loader(layer.weight, w[:N_ROUTED], 0)
        layer.weight_loader(layer.weight, w[N_ROUTED:], 1)
        layer.process_weights_after_loading()
        assert layer.uses_router_gemm is enabled
        assert layer.weight.shape[0] == (padded_router_rows(N) if enabled else N)
        y = layer(x)
        assert y.shape == (4, N) and y.dtype == torch.bfloat16
        _check_against_fp32(y, x, w)
        y32 = layer(x, otype=torch.float32)
        assert y32.shape == (4, N) and y32.dtype == torch.float32
        _check_against_fp32(y32, x, w, torch.float32)
        outs[enabled] = y
    # The routed / shared-gate split the MoE block takes keeps both halves.
    y = outs[True]
    assert y[:, :N_ROUTED].shape == (4, N_ROUTED) and y[:, N_ROUTED:].shape == (4, 1)


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 4, 40])
def test_topk_reads_padded_stride_logits_like_contiguous(tokens, dtype):
    """The router output is a view with row stride 528, not 513 (or 512)."""
    from atom.model_ops.topK import rocm_aiter_topk_softmax

    padded = torch.randn(tokens, padded_router_rows(N), device="cuda").to(dtype)
    routed_view = padded[:, :N_ROUTED]
    assert routed_view.stride(0) == padded_router_rows(N)
    w_view, ids_view = rocm_aiter_topk_softmax(routed_view, 10, True)
    w_ref, ids_ref = rocm_aiter_topk_softmax(routed_view.contiguous(), 10, True)
    torch.testing.assert_close(ids_view, ids_ref, atol=0, rtol=0)
    torch.testing.assert_close(w_view, w_ref, atol=0, rtol=0)


@gpu
@pytest.mark.parametrize("otype", [torch.bfloat16, torch.float32])
def test_router_linear_traces_under_torch_compile(monkeypatch, otype):
    """Qwen3.5's model is @support_torch_compile: the M-bucketed dispatch has
    to stay inside the op, so one dynamic-M graph serves every batch."""
    from types import SimpleNamespace

    import torch.nn.functional as F

    from atom.model_ops import linear as linear_mod

    _require_table()
    monkeypatch.setattr(
        linear_mod,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    monkeypatch.setenv("ATOM_ROUTER_FLYDSL_GEMM", "1")
    w, _ = _padded_weight()
    with torch.device("cuda"):
        layer = linear_mod.MergedReplicatedRouterLinear(
            K, [N_ROUTED, 1], bias=False, quant_config=None, prefix="mlp.gate"
        )
    layer.weight_loader(layer.weight, w[:N_ROUTED], 0)
    layer.weight_loader(layer.weight, w[N_ROUTED:], 1)
    layer.process_weights_after_loading()
    assert layer.uses_router_gemm

    def moe_gate(x, shared):
        # The Qwen3NextSparseMoeBlock pattern: routed slice + gated shared.
        logits = layer(x, otype=otype)
        return (
            logits[:, :N_ROUTED],
            F.sigmoid(logits[:, N_ROUTED:]).to(shared.dtype) * shared,
        )

    compiled = torch.compile(moe_gate, dynamic=True, fullgraph=True)
    torch._dynamo.reset()
    for m in (4, 32, 100, 300):
        x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
        shared = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        routed, gated = compiled(x, shared)
        ref = x.float() @ w.float().t()
        assert routed.dtype == otype and gated.dtype == torch.bfloat16
        _check_against_fp32(
            torch.cat([routed, ref[:, N_ROUTED:].to(otype)], 1), x, w, otype
        )
        torch.testing.assert_close(
            gated.float(),
            (torch.sigmoid(ref[:, N_ROUTED:]) * shared.float()),
            atol=2e-2,
            rtol=2e-2,
        )


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_shared_topk_reads_padded_stride_logits(dtype):
    """The FP8 path hands all 513 columns (sigmoid-scored shared expert last)."""
    import atom.model_ops.topK as topK_mod

    topK_mod.init_aiter_topK_meta_data.cache_clear()
    topK_mod.init_aiter_topK_meta_data(
        n_routed_experts=N_ROUTED,
        n_shared_experts=1,
        top_k=10,
        tp_rank=0,
        tp_size=1,
        max_num_tokens=256,
        is_EP=False,
    )
    padded = torch.randn(40, padded_router_rows(N), device="cuda").to(dtype)
    view = padded[:, :N]

    def route(logits):
        w, ids = topK_mod.rocm_aiter_topk_softmax(
            logits,
            10,
            True,
            num_fused_shared_experts=1,
            num_routing_experts=N_ROUTED,
            fused_shared_experts_scoring_func="sigmoid",
        )
        return w.clone(), ids.clone()

    w_view, ids_view = route(view)
    w_ref, ids_ref = route(view.contiguous())
    torch.testing.assert_close(ids_view, ids_ref, atol=0, rtol=0)
    torch.testing.assert_close(w_view, w_ref, atol=0, rtol=0)
    torch.testing.assert_close(
        w_view[:, 10], torch.sigmoid(view[:, N_ROUTED].float()), atol=1e-6, rtol=1e-6
    )
