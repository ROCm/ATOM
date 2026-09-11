import os
import sys
import types

import pytest
import torch

MODEL = "/data/DeepSeek-V4.1-Flash"
REF = os.path.join(MODEL, "inference")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(REF), reason="DeepSeek-V4.1-Flash checkpoint not present"
)
from transformers import AutoTokenizer

from atom.models.deepseek_v41_engram import (
    EngramLayout,
    NgramHashState,
    cached_compressed_token_map,
)


def test_engram_hash_parity_with_reference():
    sys.path.insert(0, REF)
    import engram as ref

    cfg = dict(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[384006168, 384016682],
        engram_max_ngram_size=4,
        engram_vocab_size=16000000,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_pad_id=2,
        engram_compressed_vocab_size=99092,
        max_batch_size=2,
        max_seq_len=4096,
    )
    args = types.SimpleNamespace(**cfg)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print("tokenizer len", len(tok))

    ref_layout = ref.EngramLayout.from_args(args)
    mine_layout = EngramLayout.from_args(args)
    assert ref_layout.primes == mine_layout.primes, "prime layout mismatch"
    print(
        "primes match; per-layer sums:",
        [sum(p for ng in layer for p in ng) for layer in mine_layout.primes],
    )
    print("config num_embeddings:", cfg["engram_num_embeddings"])

    ref_state = ref.NgramHashState(args, ref_layout, tok)
    lookup, size = cached_compressed_token_map(tok, MODEL)
    print("compressed vocab size", size)
    assert size == cfg["engram_compressed_vocab_size"], size
    mine = NgramHashState(mine_layout, lookup, size, cfg["engram_pad_id"])
    assert torch.equal(ref_state.multipliers, mine.multipliers), "multiplier mismatch"
    assert torch.equal(ref_state.offsets, mine.offsets), "offset mismatch"
    assert torch.equal(ref_state.token_map, mine.token_map), "token map mismatch"

    text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."
    ids = torch.tensor(tok(text)["input_ids"], dtype=torch.int64)
    L = ids.numel()
    ref_out = ref_state(ids.unsqueeze(0), 0)[0]

    flat = ids
    positions = torch.arange(L)
    cu = torch.tensor([0, L])
    slot = torch.full((1, 3), -1, dtype=torch.int64)
    hist = mine.build_history(flat, positions, cu, slot)
    mine_out = mine(flat, positions, hist)
    assert torch.equal(ref_out, mine_out), (ref_out[:4], mine_out[:4])
    print("PREFILL parity OK", tuple(mine_out.shape))

    # split prefill into two chunks + token-by-token decode, exercising carried history
    slot2 = torch.full((1, 3), -1, dtype=torch.int64)
    outs = []
    cuts = [0, 7, 13, *range(14, L + 1)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        chunk, pos = flat[a:b], positions[a:b]
        h = mine.build_history(chunk, pos, torch.tensor([0, b - a]), slot2)
        outs.append(mine(chunk, pos, h))
    chunked = torch.cat(outs, 0)
    assert torch.equal(ref_out, chunked), "chunked mismatch"
    print("CHUNKED+DECODE parity OK")

    # two sequences batched flat
    ids2 = torch.tensor(
        tok("The quick brown fox jumps over the lazy dog.")["input_ids"],
        dtype=torch.int64,
    )
    ref2 = ref_state(ids2.unsqueeze(0), 0)[0]
    both = torch.cat([ids, ids2])
    pos_both = torch.cat([torch.arange(L), torch.arange(ids2.numel())])
    cu_both = torch.tensor([0, L, L + ids2.numel()])
    slot3 = torch.full((2, 3), -1, dtype=torch.int64)
    hb = mine.build_history(both, pos_both, cu_both, slot3)
    ob = mine(both, pos_both, hb)
    assert torch.equal(ob[:L], ref_out) and torch.equal(
        ob[L:], ref2
    ), "batched mismatch"
    print("BATCHED parity OK")
    print("ALL ENGRAM HASH PARITY TESTS PASSED")
