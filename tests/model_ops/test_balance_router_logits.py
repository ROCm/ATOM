import pytest
import torch

pytest.importorskip(
    "aiter", reason="init_balance_router_logits lives in atom.model_ops.moe"
)

from atom.model_ops.fused_moe.expert_layout import MoEExpertLayout
from atom.model_ops.moe import _FAKE_EPLB_LOGIT, init_balance_router_logits


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="balance router logits are CUDA-only"
)
@pytest.mark.parametrize(
    "layout_kwargs",
    [
        pytest.param(
            {
                "num_routed": 256,
                "num_fused_shared_experts": 1,
                "num_configured_redundant": 0,
                "ep_size": 8,
                "use_all2all": True,
                "eplb_enabled": False,
            },
            id="local_replica_shared",
        ),
        pytest.param(
            {
                "num_routed": 256,
                "num_fused_shared_experts": 1,
                "num_configured_redundant": 32,
                "ep_size": 8,
                "use_all2all": True,
                "eplb_enabled": True,
            },
            id="eplb_routed_shared",
        ),
    ],
)
def test_balance_router_logits_matches_routed_gate_width(layout_kwargs):
    layout = MoEExpertLayout.make(**layout_kwargs)
    assert layout.num_physical > layout.num_routed

    top_k = 8
    ep_size = layout_kwargs["ep_size"]
    max_tokens = 16
    logits = init_balance_router_logits(
        layout.num_routed,
        top_k,
        ep_size,
        max_num_tokens=max_tokens,
    )

    assert logits.shape == (max_tokens, layout.num_routed)

    selected = (logits == _FAKE_EPLB_LOGIT).nonzero(as_tuple=False)[:, 1]
    assert selected.numel() == max_tokens * top_k
    assert selected.min().item() >= 0
    assert selected.max().item() < layout.num_routed
