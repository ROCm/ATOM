"""Softmax top-k with a fused shared expert returns weights and ids for this batch only.

The fused-shared path writes into the preallocated (max_num_tokens, top_k + 1)
buffers; both outputs must be sliced to the batch. Returning the whole weights
buffer next to sliced ids makes AITER's fused MXFP4 decode sort+quant kernel
reject the call (``topk_weights.numel() == topk_ids.numel()``).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="topK metadata is CUDA"
)


@pytest.mark.parametrize("tokens", [1, 5, 128])
def test_fused_shared_topk_softmax_is_sliced_to_batch(tokens):
    import atom.model_ops.topK as topK_mod

    n_routed, top_k = 512, 10
    topK_mod.init_aiter_topK_meta_data.cache_clear()
    topK_mod.init_aiter_topK_meta_data(
        n_routed_experts=n_routed,
        n_shared_experts=1,
        top_k=top_k,
        tp_rank=0,
        tp_size=1,
        max_num_tokens=256,
        is_EP=False,
    )
    logits = torch.randn(tokens, n_routed, device="cuda")
    weights, ids = topK_mod.rocm_aiter_topk_softmax(
        logits, top_k, True, num_fused_shared_experts=1, num_routing_experts=n_routed
    )
    assert weights.shape == ids.shape == (tokens, top_k + 1)
    assert weights.is_contiguous() and ids.is_contiguous()
    assert torch.all(ids[:, top_k] == n_routed)
    assert torch.all(weights[:, top_k] == 1.0)
    routed = ids[:, :top_k].long()
    assert torch.equal(
        routed.sort(dim=1).values,
        logits.topk(top_k, dim=1).indices.sort(dim=1).values,
    )
    torch.testing.assert_close(
        weights[:, :top_k].sum(dim=1), torch.ones(tokens, device="cuda")
    )
