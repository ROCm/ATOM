# SPDX-License-Identifier: MIT
"""Shared image router: sentinel/hash and explicit-mask callers agree."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from atom.model_ops.topK import mm_topk


def oracle(logits, bias, bias_vl, mask, ids, table, renormalize):
    scores = torch.nn.functional.softplus(logits.double()).sqrt()
    choices = scores + torch.where(mask[:, None], bias_vl, bias)
    picked = choices.argsort(dim=-1, descending=True, stable=True)[:, :6]
    if table is not None:
        hashed = table[ids.clamp(0, table.shape[0] - 1)].long()
        picked = torch.where(mask[:, None], picked, hashed)
    weights = scores.gather(-1, picked)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
    return picked, weights * 1.5


@pytest.mark.parametrize("rows", [0, 1, 17, 257])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("hashed", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_mm_topk_strided_inputs_and_shared_output_columns(
    rows, explicit, hashed, renormalize
):
    torch.manual_seed(2149)
    logits = torch.randn(rows, 768, device="cuda")[:, ::2]
    mask = (torch.arange(rows * 2, device="cuda").reshape(rows, 2) % 3 == 0)[:, 0]
    ids = torch.randint(0, 64, (rows, 2), device="cuda")[:, 0]
    if not explicit:
        ids[mask] = 66
    bias = torch.linspace(-2, 2, 384, device="cuda")
    table = (
        torch.arange(64 * 6, dtype=torch.int32, device="cuda").reshape(64, 6)
        if hashed
        else None
    )
    out_ids = torch.full((rows, 8), -7, dtype=torch.int32, device="cuda")
    out_weights = torch.full((rows, 8), -7.0, device="cuda")
    passed_ids = ids if hashed or not explicit else None
    mm_topk(
        passed_ids,
        logits,
        bias,
        -bias,
        table,
        64,
        renormalize,
        1.5,
        out_ids[:, :6],
        out_weights[:, :6],
        image_mask=mask if explicit else None,
    )
    expected_ids, expected_weights = oracle(
        logits, bias, -bias, mask, ids, table, renormalize
    )
    order, expected_order = out_ids[:, :6].argsort(-1), expected_ids.argsort(-1)
    torch.testing.assert_close(
        out_ids[:, :6].gather(-1, order).long(),
        expected_ids.gather(-1, expected_order),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        out_weights[:, :6].gather(-1, order).double(),
        expected_weights.gather(-1, expected_order),
        rtol=2e-6,
        atol=2e-7,
    )
    assert torch.all(out_ids[:, 6:] == -7) and torch.all(out_weights[:, 6:] == -7)


def test_explicit_mask_graph_replay_observes_new_mask_and_logits():
    rows = 129
    logits = torch.zeros(rows, 384, device="cuda")
    bias = torch.zeros(384, device="cuda")
    alt = torch.linspace(-2, 2, 384, device="cuda")
    mask = torch.zeros(rows, dtype=torch.bool, device="cuda")
    ids = torch.empty(rows, 6, dtype=torch.int32, device="cuda")
    weights = torch.empty(rows, 6, device="cuda")

    def run():
        mm_topk(
            None, logits, bias, alt, None, 0, True, 1.5, ids, weights, image_mask=mask
        )

    run()
    from aiter.jit.utils.chip_info import get_gfx

    stride = 64 if get_gfx().startswith("gfx9") else 1
    torch.testing.assert_close(
        ids,
        (torch.arange(6, dtype=torch.int32, device="cuda") * stride).expand(rows, 6),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for mixed in [True, False]:
        mask[::2] = mixed
        logits.normal_()
        graph.replay()
        expected_ids, expected_weights = oracle(
            logits, bias, alt, mask, None, None, True
        )
        torch.testing.assert_close(ids.long(), expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(
            weights.double(), expected_weights, rtol=2e-6, atol=2e-7
        )


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_v4_equal_score_experts_match_native_router(renormalize, dtype):
    from aiter.jit.utils.chip_info import get_gfx

    from atom.model_ops.moe import FusedMoE

    if not get_gfx().startswith("gfx9"):
        pytest.skip("V4's 384-expert wave64 tie order")
    torch.manual_seed(2149)
    rows = 2048
    # Discrete scores exercise both cross-lane ties and the non-stable order
    # within each lane's six-expert sorting network.
    logits = torch.randint(-4, 5, (rows, 384), device="cuda").to(dtype)
    bias = torch.randint(-2, 3, (384,), device="cuda").float()
    alt = torch.randint(-2, 3, (384,), device="cuda").float()
    logits[0].zero_()
    # Real-input boundary: expert 193 (lane 1) must beat 150 (lane 22).
    logits[1].fill_(-20)
    logits[1, [45, 140, 220, 29, 205]] = 20
    logits[1, [150, 193]] = -2.171875
    bias.zero_()
    alt[[45, 140, 220, 29, 205]] = 30
    alt[150] = alt[193] = 21.24795150756836
    mask = torch.arange(rows, device="cuda") % 2 == 1
    ids = torch.empty((rows, 6), dtype=torch.int32, device="cuda")
    weights = torch.empty((rows, 6), device="cuda")
    routes = [
        FusedMoE.select_experts(
            hidden_states=logits,
            router_logits=logits,
            top_k=6,
            use_grouped_topk=False,
            renormalize=renormalize,
            scoring_func="sqrtsoftplus",
            e_score_correction_bias=b,
            routed_scaling_factor=1.5,
        )
        for b in (bias, alt)
    ]
    expected_weights, expected_ids = [
        torch.where(mask[:, None], image, text) for text, image in zip(*routes)
    ]
    mm_topk(
        None,
        logits,
        bias,
        alt,
        None,
        0,
        renormalize,
        1.5,
        ids,
        weights,
        image_mask=mask,
    )
    assert 193 in ids[1].tolist() and 150 not in ids[1].tolist()
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
