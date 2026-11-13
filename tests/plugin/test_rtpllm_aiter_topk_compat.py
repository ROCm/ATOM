"""AITER top-k's optional valid-row argument differs across installed wheels."""

import pytest
import torch

if not torch.cuda.is_available():
    # Native unit CI collects tests/ without AITER/GPU; avoid importing the
    # GPU-only top-k module before pytest can report this as a skip.
    pytest.skip("GPU and AITER required", allow_module_level=True)

pytest.importorskip("aiter")

from atom.plugin.rtpllm.utils.topk_compat import (
    rtp_topk_softmax as rocm_aiter_topk_softmax_impl,
)


def test_topk_signature_without_text_signature_uses_legacy_abi(monkeypatch):
    from atom.plugin.rtpllm.utils import topk_compat

    def unavailable_signature(_function):
        raise ValueError("no signature found")

    monkeypatch.setattr(topk_compat.inspect, "signature", unavailable_signature)
    assert topk_compat._accepts_valid_rows(lambda: None) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_old_aiter_topk_masks_graph_padding_on_device(monkeypatch):
    import aiter

    def old_topk(weights, ids, token_ids, logits, renorm, shared=0, scoring=""):
        weights.fill_(0.5)
        ids.fill_(3)
        token_ids.fill_(3)

    monkeypatch.setattr(aiter, "topk_softmax", old_topk)
    logits = torch.ones((4, 8), device="cuda")
    valid_rows = torch.tensor([2], device="cuda", dtype=torch.int32)
    weights, ids = rocm_aiter_topk_softmax_impl(
        logits, 2, True, num_valid_rows=valid_rows
    )
    assert torch.all(weights[:2] == 0.5)
    assert torch.all(ids[:2] == 3)
    assert torch.all(weights[2:] == 0)
    assert torch.all(ids[2:] == -1)

    # The count is a device value, not a Python/capture-time constant.
    valid_rows.fill_(1)
    weights, ids = rocm_aiter_topk_softmax_impl(
        logits, 2, True, num_valid_rows=valid_rows
    )
    assert torch.all(weights[1:] == 0)
    assert torch.all(ids[1:] == -1)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_weights, graph_ids = rocm_aiter_topk_softmax_impl(
            logits, 2, True, num_valid_rows=valid_rows
        )
    valid_rows.fill_(3)
    graph.replay()
    assert torch.all(graph_ids[:3] == 3)
    assert torch.all(graph_ids[3:] == -1)
    assert torch.all(graph_weights[3:] == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_new_aiter_topk_receives_valid_row_argument(monkeypatch):
    import aiter

    seen = []

    def new_topk(
        weights,
        ids,
        token_ids,
        logits,
        renorm,
        shared=0,
        scoring="",
        num_valid_rows=None,
    ):
        seen.append(num_valid_rows)
        weights.fill_(0.25)
        ids.fill_(4)

    monkeypatch.setattr(aiter, "topk_softmax", new_topk)
    logits = torch.ones((2, 8), device="cuda")
    valid_rows = torch.tensor([1], device="cuda", dtype=torch.int32)
    weights, ids = rocm_aiter_topk_softmax_impl(
        logits, 2, True, num_valid_rows=valid_rows
    )
    assert seen == [valid_rows]
    assert torch.all(weights == 0.25)
    assert torch.all(ids == 4)
