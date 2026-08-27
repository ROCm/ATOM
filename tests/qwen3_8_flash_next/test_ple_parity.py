"""Numeric parity: ATOM's PLE layer vs the vLLM reference (full forward)."""

from types import SimpleNamespace

import pytest
import torch

from tests.qwen3_8_flash_next.parity_harness import init_single_rank, load_ple_reference

H, HC = 32, 4
NGRAM_SIZE, HEADS_PER_NGRAM = 3, 2
NGRAM_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
PLE_EMBED_DIM = NGRAM_HEADS * 8
VOCAB, EOS, CONV_K = 512, 7, 4


def _config():
    return SimpleNamespace(
        hidden_size=H,
        hc_count=HC,
        ple_conv_kernel_size=CONV_K,
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS_PER_NGRAM,
        ple_embed_dim=PLE_EMBED_DIM,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        split_ngram_parts=4,
        seed=1234,
        ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=EOS,
    )


@pytest.fixture(scope="module")
def pair():
    init_single_rank()
    reference = load_ple_reference()
    from atom.model_ops.qwen3_8_flash_next.ple import Qwen3_8FlashNextPLELayer

    config = _config()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float32),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=8),
        num_speculative_tokens=0,
    )
    ref = reference["Qwen3_8FlashNextPLELayer"](
        config, vllm_config, layer_idx=1, ple_dense_layer_id=0, prefix="ple"
    ).float()
    reference["_forward_contexts"]["ple"] = ref

    mine = (
        Qwen3_8FlashNextPLELayer(config, 64, 8, ple_dense_layer_id=0, prefix="ple")
        .cuda()
        .float()
    )

    torch.manual_seed(0)
    with torch.no_grad():
        table = torch.randn_like(ref.ple_embedding.ngram_embedding.weight) * 0.05
        ref.ple_embedding.ngram_embedding.weight.copy_(table)
        mine.ple_embedding.ngram_embedding.weight.copy_(table.cuda())
        for name in ("key_proj", "value_proj"):
            w = torch.randn_like(getattr(ref, name).weight) * 0.05
            getattr(ref, name).weight.copy_(w)
            getattr(mine, name).weight.copy_(w.cuda())
        for name in ("norm_key", "norm_query", "norm_conv"):
            w = torch.randn(HC * H) * 0.1
            getattr(ref, name).weight.copy_(w)
            getattr(mine, name).weight.copy_(w.cuda())
        cw = torch.randn_like(ref.conv1d.weight) * 0.1
        ref.conv1d.weight.copy_(cw)
        mine.conv1d.weight.copy_(cw.cuda())
    return ref, mine


@pytest.mark.parametrize("lengths", [[6], [4, 3, 5]])
def test_ple_forward_matches_reference(pair, lengths):
    """Whole-sequence path: every request present in full, no carried state."""
    ref, mine = pair
    torch.manual_seed(len(lengths))
    num_tokens = sum(lengths)
    hidden = torch.randn(num_tokens, HC * H)
    input_ids = torch.randint(0, VOCAB, (num_tokens,), dtype=torch.long)
    input_ids[1] = EOS
    query_start_loc = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    ngram_context = torch.randint(
        0, VOCAB, (len(lengths), NGRAM_SIZE - 1), dtype=torch.long
    )

    with torch.no_grad():
        expected = ref(hidden, input_ids, query_start_loc, ngram_context)
        got = mine(
            hidden.cuda(),
            input_ids.cuda(),
            query_start_loc.cuda(),
            ngram_context.cuda(),
        )

    assert expected.shape == (num_tokens, HC * H)
    torch.testing.assert_close(got.cpu(), expected, rtol=2e-6, atol=2e-6)


def test_state_shape_matches_config(pair):
    """conv_state_len = (kernel - 1) * dilation, dilation being ngram_size."""
    _, mine = pair
    assert mine.conv_state_len == (CONV_K - 1) * NGRAM_SIZE
    assert mine.state_shape == (mine.conv_state_len, HC * H)
