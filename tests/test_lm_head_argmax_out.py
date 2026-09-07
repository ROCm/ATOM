import pytest
import torch

embed_head = pytest.importorskip(
    "atom.model_ops.embed_head",
    reason="embed_head requires the Triton/AITER runtime",
    exc_type=ImportError,
)


def _tp1_head(logits: torch.Tensor, monkeypatch) -> embed_head.ParallelLMHead:
    head = embed_head.ParallelLMHead.__new__(embed_head.ParallelLMHead)
    torch.nn.Module.__init__(head)
    head.tp_size = 1
    head.weight = torch.nn.Parameter(torch.empty(1), requires_grad=False)
    head.bias = None
    monkeypatch.setattr(embed_head.tgemm, "mm", lambda *_args, **_kwargs: logits)
    return head


def test_argmax_can_write_directly_to_caller_storage(monkeypatch):
    logits = torch.tensor([[1.0, 4.0, 2.0], [9.0, 3.0, 5.0]])
    head = _tp1_head(logits, monkeypatch)
    hidden = torch.empty(2, 4)
    out = torch.empty(2, dtype=torch.long)

    token_ids = head.compute_argmax_token(hidden, out=out)

    assert token_ids.data_ptr() == out.data_ptr()
    assert token_ids.tolist() == [1, 0]


@pytest.mark.parametrize(
    "out",
    [
        torch.empty(3, dtype=torch.long),
        torch.empty(2, dtype=torch.int32),
    ],
)
def test_argmax_rejects_invalid_output_storage(monkeypatch, out):
    head = _tp1_head(torch.empty(2, 3), monkeypatch)
    with pytest.raises(AssertionError, match="argmax out"):
        head.compute_argmax_token(torch.empty(2, 4), out=out)
