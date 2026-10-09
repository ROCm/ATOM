from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.fused_moe.config import moe_kernel_token_capacity


@pytest.mark.parametrize(
    "mbt, enable_dpa, dp_size, use_all2all, dp_logical_ratio, expected",
    [
        # DPA + TP MoE: all-gather, topK must cover dp * mbt (GLM-5.2 CI: 16384*4).
        (16384, True, 4, False, 1, 65536),
        # DPA + EP all2all: topK stays on local tokens, keep per-rank mbt.
        (16384, True, 4, True, 1, 16384),
        # No DPA: no gather, capacity stays mbt even if dp_size > 1.
        (16384, False, 4, False, 1, 16384),
        # Simulated DP: gathered width also multiplies dp_logical_ratio.
        (1024, True, 2, False, 2, 4096),
    ],
)
def test_moe_kernel_token_capacity(
    mbt, enable_dpa, dp_size, use_all2all, dp_logical_ratio, expected
):
    cfg = SimpleNamespace(max_num_batched_tokens=mbt, enable_dp_attention=enable_dpa)
    assert (
        moe_kernel_token_capacity(
            cfg,
            dp_size=dp_size,
            use_all2all=use_all2all,
            dp_logical_ratio=dp_logical_ratio,
        )
        == expected
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="topK metadata is CUDA")
def test_dpa_capacity_fits_gathered_topk_metadata():
    import atom.model_ops.topK as topK_mod

    cfg = SimpleNamespace(max_num_batched_tokens=16384, enable_dp_attention=True)
    cap = moe_kernel_token_capacity(cfg, dp_size=4, use_all2all=False)
    topK_mod.init_aiter_topK_meta_data.cache_clear()
    topK_mod.init_aiter_topK_meta_data(
        n_routed_experts=160,
        n_shared_experts=1,
        top_k=8,
        tp_rank=0,
        tp_size=1,
        max_num_tokens=cap,
        is_EP=False,
    )
    weights, ids = topK_mod.aiter_topK_meta_data
    assert weights.shape[0] >= 65536
    assert ids.shape[0] >= 65536
