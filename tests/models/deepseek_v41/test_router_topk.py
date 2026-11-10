# SPDX-License-Identifier: MIT
"""V4.1 decode-width routing top-k matches aiter's topk_gating bit for bit."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from aiter.jit.utils.chip_info import get_gfx_runtime

if get_gfx_runtime() != "gfx1250":
    pytest.skip("the kernel mirrors gfx1250's topk_gating", allow_module_level=True)

from aiter import topk_gating

from atom.model_ops.deepseek_v41.router_topk import (
    EXPERTS,
    MAX_ROWS,
    route_topk,
    route_topk_supported,
)

SCALE = 1.5


def _reference(logits, bias, topk, valid):
    rows = logits.shape[0]
    ids = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    weights = torch.empty(rows, topk, dtype=torch.float32, device="cuda")
    topk_gating(
        weights,
        ids,
        logits,
        bias,
        True,
        SCALE,
        score_func="sqrtsoftplus",
        num_valid_rows=valid,
    )
    return weights, ids


def _logits(rows, seed, ties):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, EXPERTS, generator=g, device="cuda") * 4
    if ties:
        # few distinct values: many exact ties within and across waves
        logits = (logits * 2).round() / 2
        logits[0, ::7] = float("nan")
    return logits.to(torch.bfloat16)


@pytest.mark.parametrize("rows", [1, 7, 96, 576, MAX_ROWS])
@pytest.mark.parametrize("topk", [6, 8])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_route_topk_is_topk_gating_bitwise(rows, topk, ties, with_bias):
    logits = _logits(rows, 17 * rows + topk, ties)
    bias = (
        torch.linspace(-1, 1, EXPERTS, device="cuda", dtype=torch.float32)
        if with_bias
        else None
    )
    valid = torch.tensor([max(1, rows - 3)], dtype=torch.int32, device="cuda")
    assert route_topk_supported(logits, bias, topk, True)
    weights, ids = route_topk(logits, bias, topk, SCALE, valid)
    ref_weights, ref_ids = _reference(logits, bias, topk, valid)
    live = slice(0, int(valid))
    assert torch.equal(ids[live], ref_ids[live])
    assert torch.equal(
        weights[live].view(torch.int32), ref_weights[live].view(torch.int32)
    )
    assert (ids[int(valid) :] == -1).all()
    assert (weights[int(valid) :] == 0).all()


def test_route_topk_replays_with_a_changing_row_count():
    rows, topk = 96, 6
    logits = _logits(rows, 3, False)
    bias = torch.linspace(-1, 1, EXPERTS, device="cuda", dtype=torch.float32)
    valid = torch.tensor([rows], dtype=torch.int32, device="cuda")
    out = {}
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        route_topk(logits, bias, topk, SCALE, valid)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out["w"], out["i"] = route_topk(logits, bias, topk, SCALE, valid)
    for count in (rows, 40, 1):
        valid.fill_(count)
        graph.replay()
        ref_weights, ref_ids = _reference(logits, bias, topk, valid)
        assert torch.equal(out["i"][:count], ref_ids[:count])
        assert torch.equal(
            out["w"][:count].view(torch.int32), ref_weights[:count].view(torch.int32)
        )
        assert (out["i"][count:] == -1).all()
