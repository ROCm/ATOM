# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""HERD in a partly filled CUDA graph bucket: the Top-(K+1) candidates
`Mxfp4MoEMethod.select_experts` routes and the pass row count fused_moe gets
as `num_local_tokens` keep the pad rows out of HERD's batch popularity, so on
every replay the real rows keep the experts and weights an eager run over them
alone gives. fused_moe(herd_topk) over those routes is aiter's to check
(op_tests/test_flydsl_herd.py::test_fused_moe_herd_skips_pad_rows)."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("runs gfx1250 HERD kernels; needs a real GPU", allow_module_level=True)

aiter = pytest.importorskip("aiter")

from aiter.jit.utils.chip_info import get_gfx_runtime

if get_gfx_runtime() != "gfx1250":
    pytest.skip("fused_moe(herd_topk) runs on gfx1250 only", allow_module_level=True)

# herd_route_psum is the route fused_moe(herd_topk) runs: aiter v0.1.29.
herd_topk = pytest.importorskip("aiter.ops.flydsl.herd_topk")

from atom.model_ops import moe
from atom.utils import forward_context as fc

# scoring, grouped top-k over one group, experts, top_k, routed scale
PROFILES = {
    "deepseek_v4": ("sqrtsoftplus", False, 256, 6, 1.5),
    "kimi_k3": ("sigmoid", True, 896, 16, 2.5),
}
BUCKET, LIVE, TILE_M = 32, 17, 16


def _layer(scoring_func, grouped, experts, top_k, scale):
    return SimpleNamespace(
        scoring_func=scoring_func,
        use_grouped_topk=grouped,
        num_expert_group=1 if grouped else None,
        topk_group=1 if grouped else None,
        top_k=top_k,
        routed_scaling_factor=scale,
        renormalize=True,
        custom_routing_function=None,
        apply_router_weight_on_input=False,
        num_fused_shared_experts=0,
        e_score_correction_bias=torch.zeros(
            experts, dtype=torch.float32, device="cuda"
        ),
    )


def _method():
    method = object.__new__(moe.Mxfp4MoEMethod)
    method.use_flydsl_herd = True
    method.use_triton = False
    method.fused_experts = None
    return method


def _logits(experts, top_k):
    """The review's case: every real row's candidates are 0..K, the pad rows'
    0..K-2, K and K+1. Counted, the pad rows would make K-1 the real rows'
    least popular candidate in place of K."""
    logits = torch.full((BUCKET, experts), -8.0)
    scores = 4.0 - 0.25 * torch.arange(top_k + 1, dtype=torch.float32)
    logits[:LIVE, : top_k + 1] = scores
    logits[LIVE:, list(range(top_k - 1)) + [top_k, top_k + 1]] = scores
    return logits.to(device="cuda", dtype=torch.bfloat16)


def _pass(monkeypatch, rows, count):
    context = fc.Context(
        positions=torch.zeros(rows, dtype=torch.int64, device="cuda"),
        running_tokens=rows,
        scheduled_tokens_gpu=count,
    )
    monkeypatch.setattr(fc, "_forward_context", fc.ForwardContext(context=context))


def _herd(method, layer, logits):
    """The routes a HERD step's fused_moe runs: the method's candidates,
    dropped under the count FusedMoE._run_experts passes for these rows."""
    weights, ids = method.select_experts(layer, None, logits)
    assert ids.shape[1] == layer.top_k + 1
    kept_weights, kept_ids, masked_m, _, _ = herd_topk.herd_route_psum(
        weights,
        ids,
        logits.shape[1],
        TILE_M,
        layer.routed_scaling_factor,
        num_valid_rows=fc.pass_scheduled_tokens_gpu(logits.shape[0]),
    )
    return kept_weights, kept_ids, masked_m


def _eager_real_rows(monkeypatch, method, layer, logits, n):
    count = torch.tensor([n], dtype=torch.int32, device="cuda")
    _pass(monkeypatch, n, count)
    return _herd(method, layer, logits[:n].contiguous())


def _bits(w):
    return w.view(torch.int32)


@pytest.mark.parametrize("profile", list(PROFILES))
def test_uncounted_pad_rows_move_the_real_rows(monkeypatch, profile):
    """The case is live: given no count, the pad rows' votes change the real
    rows' kept experts."""
    monkeypatch.setattr(moe, "_FLYDSL_HERD_MIN_M", 1)
    layer, method = _layer(*PROFILES[profile]), _method()
    logits = _logits(*PROFILES[profile][2:4])

    _pass(monkeypatch, BUCKET, None)
    _, uncounted_ids, _ = _herd(method, layer, logits)
    _, ref_ids, _ = _eager_real_rows(monkeypatch, method, layer, logits, LIVE)

    assert not torch.equal(uncounted_ids[:LIVE], ref_ids)


@pytest.mark.parametrize("profile", list(PROFILES))
def test_replayed_bucket_routes_the_real_rows_as_alone(monkeypatch, profile):
    """A bs=32 graph replayed at 17, 1, 31 and 32 real rows: the real rows'
    kept experts, weights and per-expert counts equal an eager run over them
    alone, bit for bit, and the pad rows route nowhere."""
    # The eager references run below aiter's HERD window.
    monkeypatch.setattr(moe, "_FLYDSL_HERD_MIN_M", 1)
    layer, method = _layer(*PROFILES[profile]), _method()
    logits = _logits(*PROFILES[profile][2:4])
    count = torch.tensor([LIVE], dtype=torch.int32, device="cuda")

    _pass(monkeypatch, BUCKET, count)
    _herd(method, layer, logits)  # compiles outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_weights, graph_ids, graph_m = _herd(method, layer, logits)

    for n in (LIVE, 1, BUCKET - 1, BUCKET, LIVE):
        count.fill_(n)
        graph.replay()
        torch.cuda.synchronize()
        ref_weights, ref_ids, ref_m = _eager_real_rows(
            monkeypatch, method, layer, logits, n
        )
        assert torch.equal(graph_ids[:n], ref_ids), f"n={n}"
        assert torch.equal(_bits(graph_weights[:n]), _bits(ref_weights)), f"n={n}"
        assert torch.equal(graph_m, ref_m), f"n={n}"
        assert bool((graph_ids[n:] == -1).all()), f"n={n}"
        assert bool((_bits(graph_weights[n:]) == 0).all()), f"n={n}"
