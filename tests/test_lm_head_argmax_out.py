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
    monkeypatch.setattr(embed_head, "topk_select", _torch_topk_select)
    return head


def _torch_topk_select(input, topk, *, tie=None, output_idx=None, **_kwargs):
    """`aiter.topk_select` over torch, for a runner with no GPU.

    Stubbed for the same reason `tgemm.mm` above is: the selection is aiter's
    and has its own tests on a machine that can run it. What is ATOM's, and what
    these tests are for, is that the caller's buffer is the one handed down and
    the one handed back -- a contract a stub can hold and a GPU is not needed to
    check. `tie="low"` is asserted rather than honoured, since `torch.argmax`
    already breaks ties that way.
    """
    assert tie == "low", f"the head must ask for the lowest-index tie, not {tie!r}"
    idx = torch.topk(input, topk, dim=-1, sorted=True).indices.to(torch.int32)
    if output_idx is None:
        return None, idx
    output_idx.copy_(idx)
    return None, output_idx


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
