# SPDX-License-Identifier: MIT
"""Exercise zero active rows through routing and AITER sorting on graph replay.

This isolates the idle DP rank's local expert path. Multi-rank communication
and model accuracy still require a separate DP workload.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("requires a real ROCm GPU", allow_module_level=True)

pytest.importorskip("aiter")

from aiter.fused_moe import moe_sorting

from atom.model_ops import moe, topK
from atom.model_ops.fused_moe.expert_layout import SharedExpertMode
from atom.utils import forward_context as fc


@pytest.mark.parametrize(
    "rows", [1, 33, 384], ids=["decode", "padded-decode", "prefill"]
)
@pytest.mark.parametrize("router", ["aiter", "triton"])
@pytest.mark.parametrize("shared", [0, 1])
def test_zero_expert_rows_sort_and_replay(monkeypatch, rows, router, shared):
    experts, top_k, model_dim = 32, 4, 128
    logits = torch.randn(rows, experts + shared, device="cuda", dtype=torch.bfloat16)
    count = torch.tensor([rows], dtype=torch.int32, device="cuda")
    context = fc.Context(
        positions=torch.zeros(rows, dtype=torch.int64, device="cuda"),
        running_tokens=rows,
        scheduled_tokens=rows,
        scheduled_tokens_gpu=count,
    )
    monkeypatch.setattr(fc, "_forward_context", fc.ForwardContext(context=context))
    monkeypatch.setattr(
        topK,
        "triton_topk_softmax_rows",
        lambda *args: rows if router == "triton" else None,
    )
    # The shared ID is prefilled once, as it is for LEGACY_AITER. Reassignment
    # must not turn a replay's invalid routed rows back into shared work.
    meta_w = torch.ones(rows, top_k + shared, device="cuda")
    meta_ids = torch.zeros(rows, top_k + shared, dtype=torch.int32, device="cuda")
    meta_ids[:, top_k:] = experts
    monkeypatch.setattr(topK, "aiter_topK_meta_data", (meta_w, meta_ids))
    layer = SimpleNamespace(
        expert_layout=SimpleNamespace(mode=SharedExpertMode.LEGACY_AITER), top_k=top_k
    )

    def route_and_sort():
        weights, ids = moe.FusedMoE.select_experts(
            hidden_states=None,
            router_logits=logits,
            top_k=top_k,
            use_grouped_topk=False,
            renormalize=True,
            num_routing_experts=experts,
            num_fused_shared_experts=shared,
            fused_shared_experts_scoring_func="sigmoid" if shared else None,
        )
        # A gather owns its output rather than the prefilled metadata table.
        ids = ids.clone()
        if shared:
            topK.reassign_gathered_shared_ids(layer, ids)
        sorted_rows = moe_sorting(
            ids,
            weights,
            experts + shared,
            model_dim,
            torch.bfloat16,
            num_local_tokens=count,
        )
        return ids, sorted_rows[3], sorted_rows[4]

    # Compile and initialize sorting outside capture on a non-default stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            route_and_sort()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_ids, graph_valid, graph_output = route_and_sort()
    assert graph_output.shape == (rows, model_dim)

    real_ids = None
    real_valid = None
    for active in (rows, 0, rows):
        count.fill_(active)
        # Sorting clears only the active rows. A dummy replay clears none and
        # leaves its rows stale, like padding rows; nothing reads them.
        graph_output.fill_(7)
        graph.replay()
        torch.cuda.synchronize()
        assert (graph_output[:active] == 0).all()
        if active == 0:
            assert (graph_ids == -1).all()
            assert graph_valid[0].item() == 0
        elif real_ids is None:
            assert (graph_ids >= 0).all()
            assert graph_valid[0].item() > 0
            real_ids, real_valid = graph_ids.clone(), graph_valid[0].item()
        else:
            assert torch.equal(graph_ids, real_ids)
            assert graph_valid[0].item() == real_valid
