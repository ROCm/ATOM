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
    out = torch.empty(2, dtype=torch.int32)

    token_ids = head.compute_argmax_token(hidden, out=out)

    assert token_ids.data_ptr() == out.data_ptr()
    assert token_ids.tolist() == [1, 0]


def test_argmax_refuses_to_allocate_for_the_caller(monkeypatch):
    """No ``out``, no answer: the storage belongs to whoever knows its lifetime.

    A returned tensor would let a caller holding a CUDA-graph buffer read a
    fresh one and leave the buffer -- whose address the capture baked -- stale.
    """
    head = _tp1_head(torch.empty(2, 3), monkeypatch)
    with pytest.raises(TypeError, match="out"):
        head.compute_argmax_token(torch.empty(2, 4))


def test_empty_token_ids_is_the_engines_token_dtype():
    """int32, like every `Sampler` path and the target's own id buffer.

    This is what a caller without storage hands the head, and what the draft
    loop stages into a graph buffer -- `DraftGraph._stage_one` asserts the two
    dtypes agree, so an int64 here is a startup crash, not a silent widening.
    Sized off the row axis, so a V4 draft's ``[N, hc, dim]`` gives ``[N]`` too.
    """
    assert embed_head.empty_token_ids(torch.empty(5, 4)).dtype == torch.int32
    assert embed_head.empty_token_ids(torch.empty(5, 2, 4)).shape == (5,)


@pytest.mark.parametrize(
    "out",
    [
        torch.empty(3, dtype=torch.int32),
        torch.empty(2, dtype=torch.long),
    ],
)
def test_argmax_rejects_invalid_output_storage(monkeypatch, out):
    head = _tp1_head(torch.empty(2, 3), monkeypatch)
    with pytest.raises(AssertionError, match="argmax out"):
        head.compute_argmax_token(torch.empty(2, 4), out=out)
