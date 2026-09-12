"""Numeric parity: ATOM's PLE n-gram hashing vs the vLLM reference."""

from types import SimpleNamespace

import pytest
import torch

from tests.qwen3_8_flash_next.parity_harness import init_single_rank, load_ngram_reference

NGRAM_SIZE = 3
HEADS_PER_NGRAM = 2
NGRAM_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
HEAD_DIM = 8
EMBED_DIM = NGRAM_HEADS * HEAD_DIM
VOCAB = 512
EOS = 7


def _config():
    return SimpleNamespace(
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS_PER_NGRAM,
        vocab_size=VOCAB,
        split_ngram_parts=4,
        seed=1234,
        ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=EOS,
        hidden_size=32,
    )


@pytest.fixture(scope="module")
def pair():
    init_single_rank()
    reference = load_ngram_reference()
    from atom.model_ops.qwen3_8_flash_next.ngram import Qwen3_8FlashNextNGramEmbedding

    config = _config()
    ref = reference["Qwen3_8FlashNextNGramEmbedding"](
        config, EMBED_DIM, 0, 64, 8, "ple_embedding"
    )
    mine = Qwen3_8FlashNextNGramEmbedding(config, EMBED_DIM, 0, 64, 8, "ple_embedding").cuda()
    return ref, mine


def test_hash_tables_match(pair):
    """Per-head prime vocab sizes, offsets and multipliers are derived, not loaded."""
    ref, mine = pair
    for name in (
        "layer_multipliers",
        "ngram_heads_vocab_sizes",
        "ngram_heads_offsets",
    ):
        torch.testing.assert_close(
            getattr(mine, name).cpu(), getattr(ref, name), rtol=0, atol=0
        )
    assert mine.head_dim == ref.head_dim
    assert mine.table_rows == ref.ngram_embedding.org_vocab_size


@pytest.mark.parametrize(
    "lengths",
    [
        [6],  # single request
        [4, 3, 5],  # ragged batch
        [1, 1],  # decode-shaped
    ],
)
def test_ngram_ids_match_reference(pair, lengths):
    ref, mine = pair
    torch.manual_seed(len(lengths))
    num_tokens = sum(lengths)
    input_ids = torch.randint(0, VOCAB, (num_tokens,), dtype=torch.long)
    # Force EOS inside a request so the document-boundary masking is exercised.
    input_ids[1] = EOS
    query_start_loc = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    ngram_context = torch.randint(
        0, VOCAB, (len(lengths), NGRAM_SIZE - 1), dtype=torch.long
    )

    ref(input_ids, query_start_loc, ngram_context)
    expected = ref.ngram_embedding.last_ids
    got = mine.compute_ngram_ids(
        input_ids.cuda(), query_start_loc.cuda(), ngram_context.cuda()
    )

    assert expected.shape == (num_tokens, NGRAM_HEADS)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
