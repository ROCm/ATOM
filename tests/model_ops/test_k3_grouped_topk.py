import pytest
import torch


def _is_gfx1250() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return arch.split(":", 1)[0] == "gfx1250"


pytestmark = pytest.mark.skipif(not _is_gfx1250(), reason="requires gfx1250")

E = 896
K = 16


def _reference(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    groups: int = 1,
    topk_groups: int = 1,
    renorm: bool = True,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(logits.float())
    choice = scores + bias.float().unsqueeze(0)
    if groups > 1:
        grouped = choice.view(logits.shape[0], groups, E // groups)
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        selected = group_scores.topk(topk_groups, dim=-1).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, selected, True)
        expert_mask = (
            group_mask.unsqueeze(-1)
            .expand(logits.shape[0], groups, E // groups)
            .reshape(logits.shape[0], E)
        )
        choice = choice.masked_fill(~expert_mask, float("-inf"))
    ids = choice.topk(K, dim=-1, sorted=True).indices
    weights = scores.gather(1, ids)
    if renorm:
        weights /= weights.sum(dim=-1, keepdim=True)
    return weights * scale, ids.int()


def _run_original(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    groups: int = 1,
    topk_groups: int = 1,
    renorm: bool = True,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.topk import biased_grouped_topk_hip

    weights = torch.empty((logits.shape[0], K), dtype=torch.float32, device="cuda")
    ids = torch.empty((logits.shape[0], K), dtype=torch.int32, device="cuda")
    biased_grouped_topk_hip(
        logits,
        bias,
        weights,
        ids,
        groups,
        topk_groups,
        renorm,
        scale,
    )
    return weights, ids


def _run_candidate(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    renorm: bool = True,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from atom.model_ops.triton_k3_grouped_topk import k3_biased_grouped_topk

    weights = torch.empty((logits.shape[0], K), dtype=torch.float32, device="cuda")
    ids = torch.empty((logits.shape[0], K), dtype=torch.int32, device="cuda")
    k3_biased_grouped_topk(logits, bias, weights, ids, renorm, scale)
    return weights, ids


@pytest.mark.parametrize("m", [1, 2, 4, 8, 32, 128])
@pytest.mark.parametrize("renorm,scale", [(True, 1.0), (False, 2.5)])
def test_k3_candidate_matches_fp32_reference(m, renorm, scale):
    torch.manual_seed(17 + m)
    logits = torch.randn((m, E), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(E, dtype=torch.bfloat16, device="cuda") * 0.1

    weights, ids = _run_candidate(logits, bias, renorm=renorm, scale=scale)
    ref_weights, ref_ids = _reference(
        logits,
        bias,
        renorm=renorm,
        scale=scale,
    )

    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("case", ["all_tie", "plateau", "extreme", "nan"])
def test_k3_candidate_preserves_original_edge_semantics(case):
    torch.manual_seed(2026)
    bias = torch.zeros(E, dtype=torch.bfloat16, device="cuda")
    if case == "all_tie":
        logits = torch.zeros((8, E), dtype=torch.bfloat16, device="cuda")
    elif case == "plateau":
        logits = (
            (torch.arange(E, device="cuda") % 7).sub_(3).repeat(8, 1).to(torch.bfloat16)
        )
    elif case == "extreme":
        logits = torch.empty((8, E), dtype=torch.bfloat16, device="cuda")
        logits[:, 0::4] = 100
        logits[:, 1::4] = -100
        logits[:, 2::4] = 0
        logits[:, 3::4] = torch.linspace(-20, 20, E // 4, device="cuda")
    else:
        logits = torch.randn((8, E), dtype=torch.bfloat16, device="cuda")
        logits[:, ::113] = float("nan")

    original = _run_original(logits, bias)
    candidate = _run_candidate(logits, bias)
    assert torch.equal(candidate[1], original[1])
    torch.testing.assert_close(
        candidate[0],
        original[0],
        atol=2e-5,
        rtol=2e-5,
        equal_nan=True,
    )


def test_non_k3_groups_keep_original_path():
    from atom.model_ops.topK import rocm_aiter_biased_grouped_topk_impl

    torch.manual_seed(11)
    logits = torch.randn((32, E), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(E, dtype=torch.bfloat16, device="cuda") * 0.1
    weights, ids = rocm_aiter_biased_grouped_topk_impl(
        logits,
        bias,
        num_expert_group=8,
        topk_group=4,
        need_renorm=True,
        topk=K,
        routed_scaling_factor=1.0,
    )
    ref_weights, ref_ids = _reference(
        logits,
        bias,
        groups=8,
        topk_groups=4,
    )
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights, atol=2e-5, rtol=2e-5)


def test_production_dispatch_uses_k3_candidate(monkeypatch):
    import aiter

    from atom.model_ops.topK import rocm_aiter_biased_grouped_topk_impl

    def fail_fallback(*args, **kwargs):
        raise AssertionError("K3 production predicate unexpectedly used AITER fallback")

    monkeypatch.setattr(aiter, "biased_grouped_topk", fail_fallback)
    torch.manual_seed(9)
    logits = torch.randn((8, E), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(E, dtype=torch.bfloat16, device="cuda") * 0.1
    weights, ids = rocm_aiter_biased_grouped_topk_impl(
        logits,
        bias,
        num_expert_group=1,
        topk_group=1,
        need_renorm=True,
        topk=K,
        routed_scaling_factor=1.0,
    )
    ref_weights, ref_ids = _reference(logits, bias)
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights, atol=2e-5, rtol=2e-5)
