# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mono's step prologue (``StepPrologue``): which tables it takes, and on
a GPU its ``h0``, ``emb`` and n-gram state bit for bit against ATOM's ops
(the embedding and its HC copy, ``advance_ngram_state``,
``compute_ngram_ids``, the n-gram lookup)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("aiter")

from atom.models.qwen4_exp.mono import prologue as P

VOCAB, H, HC, NGRAM, HEADS, EOS = 4096, 2560, 4, 3, 8, 4000
HEAD_DIM = H // ((NGRAM - 1) * HEADS)
BF16 = torch.bfloat16


def _lookup(rows, cols, dtype=BF16, tp_size=1):
    return SimpleNamespace(weight=torch.zeros(rows, cols, dtype=dtype), tp_size=tp_size)


def _ple(table):
    return SimpleNamespace(ngram_embedding=table, head_dim=HEAD_DIM)


@pytest.mark.parametrize(
    "embed, table, width, reason",
    [
        (_lookup(8, H, tp_size=2), _lookup(8, HEAD_DIM), HC * H, "sharded"),
        (_lookup(8, H), _lookup(8, HEAD_DIM, tp_size=2), HC * H, "sharded"),
        (_lookup(8, H, torch.float32), _lookup(8, HEAD_DIM), HC * H, "not a bf16"),
        (_lookup(8, H), _lookup(8, HEAD_DIM + 1), HC * H, "one head wide"),
        (_lookup(8, H), _lookup(8, HEAD_DIM), HC * H + 8, "whole embeddings"),
    ],
)
def test_tables_it_does_not_take_are_refused(embed, table, width, reason):
    assert reason in P.refusal(embed, _ple(table), width)


def test_the_models_tables_are_taken():
    assert P.refusal(_lookup(8, H), _ple(_lookup(8, HEAD_DIM)), HC * H) is None


@pytest.fixture(scope="module")
def model():
    from atom.model_ops.qwen4_exp.ple_layer import (
        _SPLITMIX_GAMMA,
        nth_prime_after,
        splitmix64,
    )

    dev = "cuda"
    g = torch.Generator().manual_seed(0)
    embed = torch.randn(VOCAB, H, generator=g).to(BF16).to(dev)
    half = max(1, ((1 << 63) - 1) // VOCAB // 2)
    mult = [
        2 * (splitmix64(1234 + _SPLITMIX_GAMMA * (i + 1)) % half) + 1
        for i in range(NGRAM)
    ]
    sizes, offsets = [], [0]
    for k in range((NGRAM - 1) * HEADS):
        sizes.append(nth_prime_after(97, k + 1))
        offsets.append(offsets[-1] + sizes[-1])
    table = torch.randn(offsets[-1], HEAD_DIM, generator=g).to(BF16).to(dev)

    def long(v):
        return torch.tensor(v, dtype=torch.long, device=dev)

    ple = SimpleNamespace(
        ngram_embedding=SimpleNamespace(weight=table),
        layer_multipliers=long(mult),
        ngram_heads_vocab_sizes=long(sizes),
        ngram_heads_offsets=long(offsets[:-1]),
        ngram_size=NGRAM,
        heads_per_ngram=HEADS,
        head_dim=HEAD_DIM,
        eos_token_id=EOS,
    )
    return embed, ple, P.StepPrologue(SimpleNamespace(weight=embed), ple)


def _tokens(n, g):
    t = torch.randint(0, VOCAB, (n,), generator=g)
    t[torch.rand(n, generator=g) < 0.15] = EOS
    return t


CASES = {
    # (span lengths, speculative, padded tail tokens, padded requests, forked, cold)
    "verify c1 k3": ([4], True, 0, (), False, ()),
    "verify c1 k3 cold": ([4], True, 0, (), False, (0,)),
    "groups 4x4": ([4] * 4, True, 0, (), False, ()),
    "groups 4x4 one padded": ([4] * 4, True, 0, (3,), False, ()),
    "groups 2 live, tail 8": ([4, 4, 0, 0], True, 8, (), False, ()),
    "decode 16": ([1] * 16, False, 0, (5, 9), False, (2,)),
    "decode 8 forked": ([1] * 8, False, 0, (), True, (1,)),
    "decode 1": ([1], False, 0, (), False, ()),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("ids_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("case", list(CASES))
def test_the_prologue_is_the_ops_bit_for_bit(model, case, ids_dtype):
    from atom.model_ops.qwen4_exp.ops.ple import advance_ngram_state, compute_ngram_ids

    embed, ple, prologue = model
    lens, spec, tail, padded, forked, cold = CASES[case]
    dev = embed.device
    g = torch.Generator().manual_seed(len(case))
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n)
    n_tok, n_req = starts[-1] + tail, len(lens)
    k = 3 if spec else 0
    slots = 3 * n_req + 2
    state = _tokens(slots * (NGRAM - 1 + k), g).view(slots, -1).to(dev)
    s_out = torch.randperm(slots, generator=g)[:n_req].to(torch.int32)
    s_in = s_out.clone()
    if forked:  # a fork's source is read only: no request's destination
        s_in = torch.tensor(
            [s for s in range(slots) if s not in s_out.tolist()][:n_req],
            dtype=torch.int32,
        )
    for r in padded:
        s_in[r] = s_out[r] = -1
    has = torch.ones(n_req, dtype=torch.bool)
    has[list(cold)] = False
    accepted = None
    if spec:
        accepted = torch.randint(1, k + 2, (n_req,), generator=g, dtype=torch.int32)
    md = SimpleNamespace(
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device=dev),
        ngram_state=state,
        state_indices_in=s_in.to(dev),
        state_indices_out=s_out.to(dev),
        has_initial_state=has.to(dev),
        num_accepted_tokens=None if accepted is None else accepted.to(dev),
    )
    ids = _tokens(n_tok, g).to(dev, ids_dtype)

    ref_state = state.clone()
    context = advance_ngram_state(
        ids,
        md.query_start_loc,
        ref_state,
        md.state_indices_in,
        md.state_indices_out,
        md.has_initial_state,
        EOS,
        history_width=NGRAM - 1,
        num_accepted_tokens=md.num_accepted_tokens,
    )
    ngrams = compute_ngram_ids(
        ids,
        md.query_start_loc,
        context,
        ple.layer_multipliers,
        ple.ngram_heads_vocab_sizes,
        ple.ngram_heads_offsets,
        NGRAM,
        HEADS,
        EOS,
    )
    table = ple.ngram_embedding.weight
    emb_ref = F.embedding(ngrams.reshape(-1), table).reshape(n_tok, H)
    h0_ref = F.embedding(ids, embed).unsqueeze(1).expand(-1, HC, -1)

    h0 = torch.zeros(n_tok, HC * H, dtype=BF16, device=dev)
    emb = torch.zeros(n_tok, H, dtype=BF16, device=dev)
    prologue(ids, md, h0, emb)
    assert torch.equal(h0, h0_ref.reshape(n_tok, HC * H))
    assert torch.equal(emb, emb_ref)
    assert torch.equal(state, ref_state)
