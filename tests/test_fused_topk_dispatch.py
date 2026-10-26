"""The fused route must use the current pass's device row count."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("count", [None, 0, 3, 4])
def test_current_pass_count_reaches_aiter(monkeypatch, count):
    import aiter.ops.triton.moe.fused_topk_dispatch as op

    import atom.model_ops.fused_moe.mori_v2_prepare_finalize as mori
    import atom.utils.forward_context as fc
    from atom import config
    from atom.model_ops.fused_moe.fused_topk_dispatch import try_fused_topk
    from atom.utils.tbo import ubatching

    class Backend:
        pass

    backend = Backend()
    backend.prepare_finalize = SimpleNamespace(
        mega=SimpleNamespace(
            _config=SimpleNamespace(dispatch_backend="mori"),
            _total_recv=torch.tensor(9, device="cpu", dtype=torch.int32),
        )
    )
    method = SimpleNamespace(fused_experts=backend, use_triton_ep=False)
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(pcp_merge=False),
        top_k=8,
        renormalize=True,
        scoring_func="softmax",
        use_grouped_topk=False,
        custom_routing_function=None,
        e_score_correction_bias=None,
        shared_expert_scoring_func=None,
        num_fused_shared_experts=0,
        expert_layout=SimpleNamespace(
            num_routed=256, uses_dispatch_remap=False, shared_is_routed=False
        ),
    )
    valid = (
        None if count is None else torch.tensor(count, device="cpu", dtype=torch.int32)
    )
    context = SimpleNamespace(running_tokens=4, scheduled_tokens_gpu=valid)
    monkeypatch.setenv("ATOM_FUSED_TOPK_DISPATCH", "1")
    monkeypatch.setattr(mori, "MoriV2ModularKernel", Backend)
    monkeypatch.setattr(
        fc, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    monkeypatch.setattr(
        config, "get_current_atom_config", lambda: SimpleNamespace(eplb_enable=False)
    )
    monkeypatch.setattr(ubatching, "tbo_active", lambda: False)
    calls = []

    def launch(logits, weights, ids, cu, requests, pad, recv, **kwargs):
        assert cu is None and requests is None
        assert kwargs["num_valid_rows"] is valid
        assert kwargs["no_padding"] == (valid is None)
        calls.append(ids)

    monkeypatch.setattr(op, "fused_topk_dispatch", launch)
    logits = torch.zeros((4, 256), device="cpu", dtype=torch.bfloat16)
    _, ids = try_fused_topk(method, layer, logits)
    assert len(calls) == 1 and calls[0] is ids
    assert backend._fused_topk_result[0] is ids
    # An unsupported layout must keep the normal route and launch nothing.
    layer.expert_layout.uses_dispatch_remap = True
    assert try_fused_topk(method, layer, logits) is None
    assert len(calls) == 1
