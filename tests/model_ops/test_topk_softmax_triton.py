"""Triton small-M router matches AITER topk_softmax on the Qwen3.5 routing shape.

Qwen3-Next / Qwen3.5 route with softmax top-10 over 512 experts plus one fused
shared expert whose weight is sigmoid(gate logit 512). At or below
ATOM_TRITON_TOPK_SOFTMAX_MAX_TOKENS the Triton router replaces AITER's kernel;
it must give the same ids and weights, return views sliced to the batch, and
replay correctly from a CUDA graph.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton router is GPU-only"
)

N_ROUTED, TOP_K = 512, 10


@pytest.fixture
def topk_mod():
    import atom.model_ops.topK as topK_mod

    topK_mod.init_aiter_topK_meta_data.cache_clear()
    topK_mod.init_aiter_topK_meta_data(
        n_routed_experts=N_ROUTED,
        n_shared_experts=1,
        top_k=TOP_K,
        tp_rank=0,
        tp_size=1,
        max_num_tokens=256,
        is_EP=False,
    )
    yield topK_mod
    topK_mod.init_aiter_topK_meta_data.cache_clear()


def _route(topk_mod, monkeypatch, logits, max_tokens):
    monkeypatch.setattr(topk_mod, "_triton_topk_softmax_max_tokens", lambda: max_tokens)
    return topk_mod.rocm_aiter_topk_softmax(
        logits,
        TOP_K,
        True,
        num_fused_shared_experts=1,
        num_routing_experts=N_ROUTED,
        fused_shared_experts_scoring_func="sigmoid",
    )


def _by_id(weights, ids):
    order = ids[:, :TOP_K].argsort(dim=1)
    return weights[:, :TOP_K].gather(1, order), ids[:, :TOP_K].gather(1, order)


def _assert_same_routing(new, old, logits):
    (w_new, i_new), (w_old, i_old) = new, old
    w_new_s, i_new_s = _by_id(w_new, i_new)
    w_old_s, i_old_s = _by_id(w_old, i_old)
    assert torch.equal(i_new_s, i_old_s)
    torch.testing.assert_close(w_new_s, w_old_s, atol=1e-6, rtol=1e-5)
    assert torch.all(i_new[:, TOP_K] == N_ROUTED)
    torch.testing.assert_close(
        w_new[:, TOP_K], torch.sigmoid(logits[:, N_ROUTED].float()), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 4, 36, 128])
def test_triton_router_matches_aiter(topk_mod, monkeypatch, tokens, dtype):
    torch.manual_seed(tokens)
    logits = (torch.randn(tokens, N_ROUTED + 1, device="cuda") * 3).to(dtype)
    assert topk_mod._use_triton_topk_softmax(logits, 1, "sigmoid")
    w, i = _route(topk_mod, monkeypatch, logits, 128)
    assert w.shape == i.shape == (tokens, TOP_K + 1)
    assert w.is_contiguous() and i.is_contiguous()
    new = (w.clone(), i.clone())
    old = tuple(t.clone() for t in _route(topk_mod, monkeypatch, logits, 0))
    _assert_same_routing(new, old, logits)
    torch.testing.assert_close(
        new[0][:, :TOP_K].sum(dim=1), torch.ones(tokens, device="cuda")
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_triton_router_ties_pick_lowest_expert_id(topk_mod, monkeypatch, dtype):
    # Integer logits in [-3, 3]: dozens of exact ties at the top-k boundary.
    torch.manual_seed(1)
    logits = torch.randint(-3, 4, (16, N_ROUTED + 1), device="cuda").to(dtype)
    new = tuple(t.clone() for t in _route(topk_mod, monkeypatch, logits, 128))
    old = tuple(t.clone() for t in _route(topk_mod, monkeypatch, logits, 0))
    assert torch.equal(new[1], old[1])  # same ids in the same order
    _assert_same_routing(new, old, logits)


def test_gate_keeps_aiter_outside_small_m_fused_sigmoid(topk_mod, monkeypatch):
    monkeypatch.setattr(topk_mod, "_triton_topk_softmax_max_tokens", lambda: 128)
    small = torch.empty(128, N_ROUTED + 1, device="cuda")
    large = torch.empty(129, N_ROUTED + 1, device="cuda")
    assert topk_mod._use_triton_topk_softmax(small, 1, "sigmoid")
    assert not topk_mod._use_triton_topk_softmax(large, 1, "sigmoid")
    assert not topk_mod._use_triton_topk_softmax(small, 1, "")
    assert not topk_mod._use_triton_topk_softmax(small, 0, "sigmoid")
    assert not topk_mod._use_triton_topk_softmax(small[:, :N_ROUTED].t(), 1, "sigmoid")


@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("num_shared", [0, 1])
def test_triton_router_kernel_vs_torch(renormalize, num_shared):
    from atom.model_ops.triton_topk_softmax import triton_topk_softmax

    torch.manual_seed(0)
    tokens = 9
    logits = torch.randn(tokens, N_ROUTED + num_shared, device="cuda")
    w = torch.full((tokens, TOP_K + 1), -1.0, device="cuda")
    i = torch.full((tokens, TOP_K + 1), -1, dtype=torch.int32, device="cuda")
    triton_topk_softmax(w, i, logits, TOP_K, renormalize, num_shared)

    probs = logits[:, :N_ROUTED].softmax(dim=1)
    ref_w, ref_i = probs.topk(TOP_K, dim=1)
    if renormalize:
        ref_w = ref_w / ref_w.sum(dim=1, keepdim=True)
    assert torch.equal(i[:, :TOP_K].long(), ref_i)
    torch.testing.assert_close(w[:, :TOP_K], ref_w, atol=1e-6, rtol=1e-5)
    assert torch.all(i[:, TOP_K] == -1)  # shared ids are preset, never written
    if num_shared:
        torch.testing.assert_close(w[:, TOP_K], torch.sigmoid(logits[:, N_ROUTED]))
    else:
        assert torch.all(w[:, TOP_K] == -1.0)


def test_triton_router_cuda_graph_replay(topk_mod, monkeypatch):
    tokens = 8
    static = torch.randn(tokens, N_ROUTED + 1, device="cuda").to(torch.bfloat16)
    _route(topk_mod, monkeypatch, static, 128)  # compile outside capture
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w_out, i_out = _route(topk_mod, monkeypatch, static, 128)
    assert w_out.shape == i_out.shape == (tokens, TOP_K + 1)

    for seed in range(3):
        torch.manual_seed(100 + seed)
        fresh = (torch.randn(tokens, N_ROUTED + 1, device="cuda") * 3).to(
            torch.bfloat16
        )
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        new = (w_out.clone(), i_out.clone())
        old = tuple(t.clone() for t in _route(topk_mod, monkeypatch, fresh, 0))
        _assert_same_routing(new, old, fresh)
