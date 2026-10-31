import pytest
import torch

from atom.model_engine.structured_output_mask import StructuredOutputMask


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab", [1, 31, 32, 33, 127, 129, 131073])
def test_packed_mask_preserves_exact_logits_and_unrequested_rows(device, dtype, vocab):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    torch.manual_seed(19)
    # Strided vocabulary and batch axes, and reordered speculative target/bonus
    # rows. High bits of signed int32 words must work as well as low bits.
    storage = torch.randn(14, vocab * 2, dtype=dtype, device=device)
    logits = storage[::2, ::2]
    expected = storage.clone()
    rows = [5, 0, 3]
    buffer = StructuredOutputMask(vocab)
    masks = buffer.prepare(len(rows), logits.device)
    masks.random_(-(2**31), 2**31 - 1)
    masks[0].zero_()
    masks[1].fill_(-1)
    tokens = torch.arange(vocab)
    for i, row in enumerate(rows):
        allowed = ((masks[i, tokens // 32] >> (tokens % 32)) & 1).bool()
        expected[::2, ::2][row].masked_fill_(~allowed.to(device), -float("inf"))
    buffer.apply(logits, rows)
    assert torch.equal(storage, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_buffer_reuse_growth_and_changing_batches(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    buffer = StructuredOutputMask(65)
    pointers = {}
    for size in [1, 8, 2, 0, 7, 16, 1]:
        logits = torch.ones(max(size, 1), 65, device=device)
        masks = buffer.prepare(size, logits.device)
        masks.fill_(-1 if size % 2 else 0)
        pointer = buffer.host_masks.data_ptr()
        if buffer.capacity in pointers:
            assert pointer == pointers[buffer.capacity]
        pointers[buffer.capacity] = pointer
        buffer.apply(logits, list(range(size)))
        assert torch.all(logits == (1 if size % 2 or size == 0 else -float("inf")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_async_reuse_across_streams_keeps_each_batch_mask():
    buffer = StructuredOutputMask(4097)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for i in range(40):
        with torch.cuda.stream(streams[i % 2]):
            logits = torch.ones(8, 4097, device="cuda")
            masks = buffer.prepare(8, logits.device)
            masks.fill_(-1 if i % 2 else 0)
            buffer.apply(logits, list(range(8)))
            outputs.append(logits)
    for stream in streams:
        stream.synchronize()
    for i, logits in enumerate(outputs):
        assert torch.all(logits == (1 if i % 2 else -float("inf")))
