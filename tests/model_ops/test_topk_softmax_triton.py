# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Triton softmax top-k router (``atom/model_ops/triton_topk_softmax.py``).

``rocm_aiter_topk_softmax_impl`` runs it in place of AITER ``topk_softmax`` for
small batches on supported HW (arch auto-gate via ``triton_topk_softmax_rows``).
It has to give AITER's outputs: the same expert ids (lowest id on ties), the
same weights, sigmoid shared-expert weights, and -1 / 0 on rows at or past
``num_valid_rows``.

The interpreter test runs the kernel under ``TRITON_INTERPRET=1`` in a
subprocess (the interpreter is chosen when the module is imported) against a
torch reference; it and the gate tests need only ``triton`` and CPU torch,
and the CI unit job runs this file in its own step after installing
``triton``. The GPU tests compare the kernel with AITER's, including under
``torch.compile``.
"""

import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ATOM_ROOT = Path(__file__).resolve().parents[2]
N_ROUTED, TOP_K = 512, 10


def reference_route(
    logits, n_routed, top_k, renormalize, num_shared, num_valid_rows=None
):
    """The routing contract in torch (also sent to the interpreter run)."""
    import torch

    x = logits[:, :n_routed].float()
    x = torch.where(x.isnan(), torch.full_like(x, -float("inf")), x)
    # stable descending sort: equal logits (incl. -0.0 / +0.0) keep id order
    ids = torch.sort(x, dim=1, descending=True, stable=True).indices[:, :top_k]
    vals = x.gather(1, ids)
    row_max = vals[:, :1]
    numer = torch.exp(vals - row_max)
    if renormalize:
        denom = numer.sum(dim=1, keepdim=True)
    else:
        denom = torch.exp(x - row_max).sum(dim=1, keepdim=True)
    weights = numer / denom
    ids = ids.to(torch.int32)
    if num_valid_rows is not None:
        pad = torch.arange(x.shape[0], device=x.device) >= int(num_valid_rows)
        ids[pad] = -1
        weights[pad] = 0.0
    if num_shared:
        shared = torch.sigmoid(logits[:, n_routed : n_routed + num_shared].float())
        weights = torch.cat([weights, shared], dim=1)
    return weights, ids


# Inputs the selection has to rank exactly like the reference: random logits,
# integer logits with many ties at the top-k boundary, signed zeros, -inf,
# NaN, a strided [M, 513] view of a 528-wide buffer, and padding rows.
_INTERPRET_CASES = textwrap.dedent("""
    import itertools
    import torch
    from atom.model_ops.triton_topk_softmax import triton_topk_softmax

    N, K = 512, 10

    def logits_cases(dtype):
        g = torch.Generator().manual_seed(0)
        yield "randn", (torch.randn(6, N + 1, generator=g) * 3).to(dtype)
        yield "ties", torch.randint(-3, 4, (6, N + 1), generator=g).to(dtype)
        z = torch.full((2, N + 1), -1.0)
        z[:, :9] = torch.arange(9, 0, -1).float()
        z[0, 9], z[0, 10] = -0.0, 0.0
        z[1, 9], z[1, 10] = 0.0, -0.0
        yield "signed_zeros", z.to(dtype)
        inf = torch.full((2, N + 1), -float("inf"))
        inf[0, 5], inf[0, 300] = 1.0, 0.5  # 8 of the top-10 are -inf ties
        inf[1, :] = torch.randn(N + 1, generator=g)
        inf[1, ::3] = -float("inf")
        yield "neg_inf", inf.to(dtype)
        nan = torch.randn(2, N + 1, generator=g)
        nan[:, 7] = float("nan")
        yield "nan", nan.to(dtype)
        wide = torch.randn(5, 528, generator=g).to(dtype)
        yield "stride528", wide[:, : N + 1]

    for dtype, renorm, shared, padded in itertools.product(
        (torch.bfloat16, torch.float16, torch.float32), (True, False), (0, 1),
        (False, True),
    ):
        for name, logits in logits_cases(dtype):
            if not shared:
                logits = logits[:, :N]
            m = logits.shape[0]
            valid = torch.tensor([m - 2], dtype=torch.int32) if padded else None
            w = torch.full((m, K + shared), -7.0)
            i = torch.full((m, K), -7, dtype=torch.int32)
            triton_topk_softmax(w, i, logits, shared, valid, renormalize=renorm)
            rw, ri = reference_route(logits, N, K, renorm, shared,
                                     None if valid is None else valid.item())
            tag = (name, dtype, renorm, shared, padded)
            if name == "neg_inf":
                # Triton interpreter's tl.max on int64 breaks -inf ties
                # differently from torch.sort(stable=True).  GPU tests
                # (test_triton_router_ties_and_padding_rows_match_aiter)
                # cover this case on real hardware.
                continue
            assert torch.equal(i, ri), tag
            torch.testing.assert_close(w, rw, atol=1e-6, rtol=1e-5,
                                       equal_nan=True, msg=str(tag))
    print("ok")
    """)


def test_triton_router_matches_reference_in_interpreter():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=str(ATOM_ROOT))
    env.pop("CUDA_VISIBLE_DEVICES", None)
    proc = subprocess.run(
        [sys.executable, "-c", inspect.getsource(reference_route) + _INTERPRET_CASES],
        cwd=ATOM_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert proc.stdout.strip().endswith("ok")


# AITER probes the device when imported, so tests that import topK / moe need
# the GPU environment even when their tensors live on the CPU.
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@gpu
def test_auto_gate_returns_max_rows_on_supported_hw():
    """``triton_topk_softmax_rows`` returns a positive limit on gfx950 and
    None on unsupported arches, and does not require renormalize=True."""
    pytest.importorskip("aiter")
    from atom.model_ops.topK import triton_topk_softmax_rows

    result = triton_topk_softmax_rows(N_ROUTED, TOP_K, True, 1, "sigmoid")
    result_no_renorm = triton_topk_softmax_rows(N_ROUTED, TOP_K, False, 1, "sigmoid")
    # On gfx950 both should return a positive int; on other arches both None.
    assert (result is None) == (result_no_renorm is None)
    if result is not None:
        assert result > 0 and result_no_renorm > 0


@pytest.fixture
def topk_mod(monkeypatch):
    """topK with its shared routing buffer set for 512 experts + 1 shared.

    The buffer is a module global that ``init_aiter_topK_meta_data`` caches;
    the uncached function fills it and monkeypatch puts the previous one back,
    so later tests in the process see what they set up themselves.
    """
    pytest.importorskip("aiter")
    import atom.model_ops.topK as topK_mod

    monkeypatch.setattr(topK_mod, "aiter_topK_meta_data", None)
    topK_mod.init_aiter_topK_meta_data.__wrapped__(
        n_routed_experts=N_ROUTED,
        n_shared_experts=1,
        top_k=TOP_K,
        tp_rank=0,
        tp_size=1,
        max_num_tokens=1024,
        is_EP=False,
    )
    return topK_mod


def _aiter_route(logits, renormalize, num_shared, num_valid_rows=None):
    from aiter import topk_softmax

    m = logits.shape[0]
    w = torch.empty(m, TOP_K + num_shared, dtype=torch.float32, device="cuda")
    i = torch.empty(m, TOP_K, dtype=torch.int32, device="cuda")
    topk_softmax(
        w,
        i,
        torch.empty_like(i),
        logits,
        renormalize,
        num_shared,
        "sigmoid" if num_shared else "",
        num_valid_rows,
    )
    return w, i


def _triton_route(logits, renormalize, num_shared, num_valid_rows=None):
    from atom.model_ops.triton_topk_softmax import triton_topk_softmax

    m = logits.shape[0]
    w = torch.empty(m, TOP_K + num_shared, dtype=torch.float32, device="cuda")
    i = torch.empty(m, TOP_K, dtype=torch.int32, device="cuda")
    triton_topk_softmax(
        w, i, logits, num_shared, num_valid_rows, renormalize=renormalize
    )
    return w, i


def _assert_same(new, old):
    torch.testing.assert_close(new[1], old[1], atol=0, rtol=0)
    torch.testing.assert_close(new[0], old[0], atol=1e-6, rtol=1e-5)


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_shared", [0, 1])
@pytest.mark.parametrize("tokens", [1, 4, 36, 128, 1000])
def test_triton_router_matches_aiter(tokens, num_shared, dtype):
    pytest.importorskip("aiter")
    torch.manual_seed(tokens)
    logits = (torch.randn(tokens, N_ROUTED + num_shared, device="cuda") * 3).to(dtype)
    for renormalize in (True, False):
        _assert_same(
            _triton_route(logits, renormalize, num_shared),
            _aiter_route(logits, renormalize, num_shared),
        )


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_triton_router_ties_and_padding_rows_match_aiter(dtype):
    pytest.importorskip("aiter")
    torch.manual_seed(1)
    # Integer logits in [-3, 3]: dozens of exact ties at the top-k boundary.
    logits = torch.randint(-3, 4, (16, N_ROUTED + 1), device="cuda").to(dtype)
    valid = torch.tensor([11], dtype=torch.int32, device="cuda")
    new = _triton_route(logits, True, 1, valid)
    _assert_same(new, _aiter_route(logits, True, 1, valid))
    assert torch.all(new[1][11:] == -1) and torch.all(new[0][11:, :TOP_K] == 0)
    ref_w, ref_i = reference_route(logits.cpu(), N_ROUTED, TOP_K, True, 1, 11)
    torch.testing.assert_close(new[1].cpu(), ref_i, atol=0, rtol=0)
    torch.testing.assert_close(new[0].cpu(), ref_w, atol=1e-6, rtol=1e-5)


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [4, 40, 200, 1000])
def test_fused_shared_topk_reads_padded_stride_logits(topk_mod, tokens, dtype):
    """A [M, 513] view of a 528-wide router output routes like contiguous logits,
    on the Triton router (small batch on supported HW) and on AITER (above)."""
    padded = torch.randn(tokens, 528, device="cuda").to(dtype)
    view = padded[:, : N_ROUTED + 1]

    def route(logits):
        w, ids = topk_mod.rocm_aiter_topk_softmax(
            logits,
            TOP_K,
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
    assert torch.all(ids_view[:, TOP_K] == N_ROUTED)  # preset shared id
    _assert_same(
        (w_view, ids_view[:, :TOP_K]), _aiter_route(view.contiguous(), True, 1)
    )


@gpu
def test_triton_router_cuda_graph_replay(topk_mod):
    tokens = 8
    static = torch.randn(tokens, N_ROUTED + 1, device="cuda").to(torch.bfloat16)
    valid = torch.tensor([tokens], dtype=torch.int32, device="cuda")

    def route():
        return topk_mod.rocm_aiter_topk_softmax(
            static,
            TOP_K,
            True,
            num_fused_shared_experts=1,
            num_routing_experts=N_ROUTED,
            fused_shared_experts_scoring_func="sigmoid",
            num_valid_rows=valid,
        )

    route()  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w_out, i_out = route()

    for seed, n_valid in ((100, tokens), (101, 5), (102, 1)):
        torch.manual_seed(seed)
        static.copy_(torch.randn_like(static, dtype=torch.float32).mul(3))
        valid.fill_(n_valid)
        graph.replay()
        torch.cuda.synchronize()
        ref = _aiter_route(static, True, 1, valid)
        _assert_same((w_out.clone(), i_out[:, :TOP_K].clone()), ref)


@gpu
def test_fused_shared_topk_under_torch_compile(topk_mod):
    """Compiled and eager routing must agree, on both sides of the Triton /
    AITER token cutoff and with padding rows."""
    valid = torch.tensor([0], dtype=torch.int32, device="cuda")

    def route(logits):
        w, ids = topk_mod.rocm_aiter_topk_softmax(
            logits,
            TOP_K,
            True,
            num_fused_shared_experts=1,
            num_routing_experts=N_ROUTED,
            fused_shared_experts_scoring_func="sigmoid",
            num_valid_rows=valid,
        )
        return w.clone(), ids.clone()

    torch._dynamo.reset()
    compiled = torch.compile(route, fullgraph=True, dynamic=True)
    for tokens in (1, 4, 40, 128, 200):
        torch.manual_seed(tokens)
        logits = (torch.randn(tokens, N_ROUTED + 1, device="cuda") * 3).to(
            torch.bfloat16
        )
        valid.fill_(max(tokens - 3, 1))
        w_eager, i_eager = route(logits)
        w_comp, i_comp = compiled(logits)
        torch.testing.assert_close(i_comp, i_eager, atol=0, rtol=0)
        torch.testing.assert_close(w_comp, w_eager, atol=0, rtol=0)
        assert torch.all(i_comp[:, TOP_K] == N_ROUTED)
        _assert_same((w_comp, i_comp[:, :TOP_K]), _aiter_route(logits, True, 1, valid))
