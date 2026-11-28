# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MTP_DRAFT_VOCAB on the GPU: draft logits over a subset of the vocabulary.

* the subset head's logits are the full head's at the subset's columns (bf16 exact,
  FP8 close);
* a draft drawn over the subset is the draw of the full-vocabulary sampler on logits
  that are -inf off the subset, the same uniforms and presence penalty given: token
  and record (columns as vocabulary ids, probabilities, top-p threshold);
* with rejection sampling the first emitted token still follows the official
  sampler's own draws (lossless), and a draft is accepted at sum(min(p, q)) with q
  zero outside the subset;
* the draft presence penalty reaches the subset's columns (a view of the first
  range, or each step's rows gathered) and keeps drafts as column ids;
* the draw, its maps and the presence record replay in a CUDA graph, and the fused
  record store writes vocabulary ids as `store_step` does."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from types import SimpleNamespace

from atom.model_ops.draft_vocab import DraftVocab, parse_vocab_ranges
from atom.model_ops.presence_penalty import DraftPresencePenalty, PresencePenaltyState
from atom.model_ops.sampler import fused_topk_topp_sample
from atom.model_ops.spec_rejection import _EPS, SpecRejectionSampling, _kb

DEV = "cuda"
V, H, K, S = 32768, 256, 20, 3
N, REPS = 2048, 4
F32 = {"dtype": torch.float32, "device": DEV}
I32 = {"dtype": torch.int32, "device": DEV}
VIEW = "0:16256,32640:32768"  # 16384 ids, a 128-id tail: the presence mask is a view
STAGED = "0:8192,16384:24576"  # a tail half the subset: rows gathered a step


@pytest.fixture(autouse=True)
def _bf16_default():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    yield
    torch.set_default_dtype(old)


def _vocab(spec=VIEW, fp8=False, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = (torch.randn(V, H, generator=g, **F32) * 0.05).to(torch.bfloat16)
    return w, DraftVocab(w, parse_vocab_ranges(spec, V), fp8=fp8)


def test_mapping():
    _, dv = _vocab()
    assert dv.size == 16384
    sub = torch.tensor([0, 16255, 16256, 16383, 16384 + 5], **I32)
    assert dv.to_vocab(sub, out=sub).tolist() == [0, 16255, 32640, 32767, -1]
    rec = torch.randint(0, dv.size, (7, 3 * 32 + 2), **I32)
    raw = rec.clone()
    dv.map_cols_(rec[:, :K])
    assert torch.equal(rec[:, :K], dv.ids[raw[:, :K].long()])
    assert torch.equal(rec[:, K:], raw[:, K:])


@pytest.mark.parametrize("fp8", [False, True])
def test_subset_logits(fp8):
    w, dv = _vocab(fp8=fp8)
    x = torch.randn(8, H, **F32).to(torch.bfloat16)
    full = (x.float() @ w.float().t())[:, dv.ids.long()]
    sub = dv.mm(x).float()
    assert sub.shape == (8, dv.size)
    rel = ((sub - full).norm() / full.norm()).item()
    assert rel < (0.06 if fp8 else 1e-2), rel


def _presence(dv, rows):
    state = PresencePenaltyState(V, max_num_seqs=rows, max_extra=S, device=DEV)
    dp = DraftPresencePenalty(state, max_rows=rows, max_drafts=S, device=DEV)
    dp.set_vocab(dv)
    return state, dp


@pytest.mark.parametrize("spec", [VIEW, STAGED])
def test_subset_draw_is_the_masked_full_draw(spec):
    """The subset draw, its token and its record are the full-vocabulary sampler's
    on logits -inf off the subset (same uniforms; the staged presence mask is exact,
    the view one is exact on the first range, so the marks stay there)."""
    from aiter.ops.triton.fusions import topk_topp_sample as tts

    rows = 64
    _, dv = _vocab(spec)
    state, dp = _presence(dv, rows)
    g = torch.Generator(device=DEV).manual_seed(3)
    a0, b0, _ = dv.ranges[0]
    marks = torch.randint(a0, b0, (state.num_slots, 200), device=DEV, generator=g)
    state.mask.scatter_(1, marks, 1)
    dp.slots[:rows] = torch.randperm(state.num_slots, device=DEV, generator=g)[
        :rows
    ].int()
    dp.penalties[:rows] = 1.5
    dp.stage_vocab(rows)
    rs = SpecRejectionSampling(
        max_rows=rows, mtp_k=S, max_num_seqs=rows, device=DEV, draft_k=K
    )
    rs.stage_step(list(range(rows)), torch.ones(rows, **F32), 0.95)
    sub = (3.0 * torch.randn(rows, dv.size, generator=g, **F32)).to(torch.bfloat16)
    sub[
        torch.arange(rows, device=DEV),
        torch.randint(0, dv.size, (rows,), device=DEV, generator=g),
    ] = 9.0
    u = torch.rand(rows, generator=g, **F32)
    rs.u_draft[:rows].copy_(u)
    out = torch.empty(rows, **I32)
    # the draw itself, with the uniforms fixed (sample_draft draws its own)
    tts.topk_topp_sample(
        sub,
        rs.row_temp,
        K,
        rs.row_topp,
        uniform=u,
        eps=_EPS,
        out=rs.col_scratch[:rows],
        debug=rs.dbg_scratch[:rows],
        presence=rs._presence_args(dp),
        gluon_select=tts.topk_topp_sample_gluon_select_supported(sub, K),
    )
    dv.to_vocab(rs.col_scratch[:rows], out=rs.tok_scratch[:rows])
    dv.map_cols_(rs.dbg_scratch[:rows, :K])
    out.copy_(rs.tok_scratch[:rows])
    full = torch.full((rows, V), float("-inf"), dtype=torch.bfloat16, device=DEV)
    full[:, dv.ids.long()] = sub
    tok_full = torch.empty(rows, **I32)
    kb = _kb(K)
    dbg_full = torch.empty(rows, 3 * kb + 2, **I32)
    tts.topk_topp_sample(
        full,
        rs.row_temp,
        K,
        rs.row_topp,
        uniform=u,
        eps=_EPS,
        out=tok_full,
        debug=dbg_full,
        presence={
            "input_ids": dp.ctx,
            "row_seq": dp.row_seq,
            "row_pos": dp.pos,
            "seq_start": dp.seq_start,
            "slots": dp.slots,
            "penalties": dp.penalties,
            "mask": state.mask,
            "max_extra": state.max_extra,
        },
        gluon_select=tts.topk_topp_sample_gluon_select_supported(full, K),
    )
    assert torch.equal(out, tok_full)
    assert torch.equal(rs.dbg_scratch[:rows, :K], dbg_full[:, :K])
    for lo, hi in ((kb, kb + K), (3 * kb, 3 * kb + 1)):
        d = rs.dbg_scratch[:rows, lo:hi].view(torch.float32) - dbg_full[:, lo:hi].view(
            torch.float32
        )
        assert float(d.abs().max()) < 1e-6
    assert torch.equal(dv.ids[rs.col_scratch[:rows].long()], out)


def _dist(rec, k):
    kb = _kb(k)
    cols = rec[:, :k].long()
    p = rec[:, kb : kb + k].view(torch.float32)
    thr = rec[:, 3 * kb].view(torch.float32)[:, None]
    gt = (p[:, None, :] * (p[:, None, :] > p[:, :, None])).sum(-1)
    w = torch.where(gt < thr, p, 0.0)
    w = w / w.sum(-1, keepdim=True)
    return torch.zeros(rec.shape[0], V, **F32).scatter_add_(1, cols, w)


def _freq(tokens):
    return torch.bincount(tokens.long(), minlength=V).float() / tokens.numel()


def _row(gen, hot_cols):
    x = torch.randn(V, generator=gen, **F32)
    x[hot_cols] += 15.0 * torch.linspace(1.0, 0.55, hot_cols.numel(), **F32)
    return x


def test_rs_over_the_subset_is_lossless():
    gen = torch.Generator(device=DEV).manual_seed(7)
    _, dv = _vocab()
    # hot columns on both sides of the subset, so q misses some of p's mass
    hot = torch.tensor([3, 17, 900, 16000, 20000, 25000, 32700, 32750], device=DEV)
    pl = [
        _row(gen, hot[torch.randperm(8, generator=gen, device=DEV)])
        for _ in range(S + 1)
    ]
    ql = [(pl[i] + 0.6 * torch.randn(V, generator=gen, **F32)) for i in range(S)]
    ql_sub = [q[dv.ids.long()].to(torch.bfloat16) for q in ql]
    rs = SpecRejectionSampling(
        max_rows=N, mtp_k=S, max_num_seqs=(REPS + 2) * N, device=DEV, draft_k=K
    )
    rs.warm(V, draft_vocab=dv)
    b = torch.arange(N, device=DEV)
    meta = SimpleNamespace(
        cu_num_draft_tokens=((b + 1) * S).to(torch.int32),
        target_logits_indices=(
            b[:, None] * (S + 1) + torch.arange(S, device=DEV)
        ).reshape(-1),
        bonus_logits_indices=b * (S + 1) + S,
        num_spec_steps=S,
    )
    logits = torch.stack([pl[i % (S + 1)] for i in range(N * (S + 1))]).to(
        torch.bfloat16
    )
    temps = torch.ones(N, **F32)
    kb = _kb(K)
    dbg = torch.empty(N * (S + 1), 3 * kb + 2, **I32)
    official = []
    for _ in range(REPS):
        tok = fused_topk_topp_sample(
            logits,
            temps,
            K,
            0.95,
            verify_rows=meta.target_logits_indices,
            bonus_rows=meta.bonus_logits_indices,
            cu_num_draft_tokens=meta.cu_num_draft_tokens,
            debug=dbg,
        )
        official.append(tok[: N * S].view(N, S)[:, 0].clone())
    p0 = _dist(dbg[:1], K)[0]
    allowed = torch.zeros(V, dtype=torch.bool, device=DEV)
    allowed[dv.ids.long()] = True
    first, accepted, q0 = [], [], None
    for rep in range(REPS):
        rs.stage_step(list(range(rep * N, (rep + 1) * N)), temps, 0.95)
        ids = torch.zeros(N, S + 1, dtype=torch.int64, device=DEV)
        out = torch.empty(N, **I32)
        for i in range(S):
            rs.sample_draft(ql_sub[i].expand(N, dv.size).contiguous(), out, vocab=dv)
            assert bool(allowed[out.long()].all()), "a draft outside the subset"
            assert torch.equal(dv.ids[rs.draft_columns(N).long()], out)
            ids[:, 1 + i] = out
            rs.store_step(i, N)
            if i == 0:
                q0 = _dist(rs.dbg_scratch[:1], K)[0]
                assert float(q0[~allowed].sum()) == 0.0
        meta.input_ids = ids.reshape(-1)
        out_v, _ = rs.verify(logits, meta, temps, K, 0.95)
        t0 = out_v.view(N, S)[:, 0]
        first.append(t0.clone())
        accepted.append(t0 == ids[:, 1].int())
    tv = 0.5 * (_freq(torch.cat(first)) - _freq(torch.cat(official))).abs().sum().item()
    assert tv < 0.05, tv
    acc = torch.cat(accepted).float().mean().item()
    assert abs(acc - torch.minimum(p0, q0).sum().item()) < 0.02


@pytest.mark.parametrize(
    "spec,marked,hits",
    [
        # the view: column 10 is id 10; column 16266 (id 32650) reads id 16266's bit
        (VIEW, [10, 32650, 20000], [10]),
        # gathered: exact on every range
        (STAGED, [10, 20000, 30000], [10, 8192 + (20000 - 16384)]),
    ],
)
def test_presence_over_the_subset(spec, marked, hits):
    _, dv = _vocab(spec)
    state = PresencePenaltyState(V, max_num_seqs=8, max_extra=S, device=DEV)
    dp = DraftPresencePenalty(state, max_rows=4, max_drafts=S, device=DEV)
    dp.set_vocab(dv)
    assert (dp.mask_vocab is None) == (spec == VIEW)
    state.mask[5, marked] = 1
    dp.slots[:1] = 5
    dp.penalties[:1] = 1.5
    dp.stage_vocab(1)
    logits = torch.zeros(1, dv.size, dtype=torch.bfloat16, device=DEV)
    dp.apply(logits)
    assert (logits[0] != 0).nonzero().view(-1).tolist() == hits
    # a draft (column 77) counts for the next pass
    dp.record(torch.tensor([77], **I32))
    logits.zero_()
    dp.apply(logits)
    assert float(logits[0, 77]) == -1.5


def test_draw_maps_and_record_replay_in_a_graph():
    """The recorded draft pass: draw over the subset, the two maps, the presence
    record of the columns, replayed on new logits."""
    rows = 40
    _, dv = _vocab()
    _, dp = _presence(dv, rows)
    dp.slots[:rows] = torch.arange(rows, **I32)
    dp.penalties[:rows] = 1.5
    rs = SpecRejectionSampling(
        max_rows=rows, mtp_k=S, max_num_seqs=rows, device=DEV, draft_k=K
    )
    rs.stage_step(list(range(rows)), torch.ones(rows, **F32), 0.95)
    rs.warm(V, draft_presence=dp, draft_vocab=dv)
    logits = torch.zeros(rows, dv.size, dtype=torch.bfloat16, device=DEV)
    out = torch.zeros(rows, **I32)

    def draft_pass():
        rs.sample_draft(logits, out, dp, vocab=dv)
        dp.record(rs.draft_columns(rows))

    draft_pass()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        draft_pass()
    g = torch.Generator(device=DEV).manual_seed(11)
    for rep in range(3):
        dp.pos.zero_()
        dp.ctx.fill_(-1)
        hot = torch.randint(0, dv.size, (rows,), device=DEV, generator=g)
        logits.copy_(
            (2.0 * torch.randn(rows, dv.size, generator=g, **F32)).to(torch.bfloat16)
        )
        logits[torch.arange(rows, device=DEV), hot] = 30.0
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, dv.ids[hot]), rep  # the hot column, as a vocabulary id
        assert torch.equal(rs.tok_scratch[:rows], out)
        assert torch.equal(rs.dbg_scratch[:rows, 0], out)  # the record's first column
        cols = rs.dbg_scratch[:rows, :K].long()
        allowed = torch.zeros(V, dtype=torch.bool, device=DEV)
        allowed[dv.ids.long()] = True
        assert bool(allowed[cols].all())
        assert torch.equal(dp.ctx.view(rows, -1)[:, 1], hot.int())  # columns recorded
        assert dp.pos[:rows].tolist() == [1] * rows


def test_fused_record_store_keeps_vocabulary_ids():
    from atom.spec_decode.eagle_fused_glue import store_last_draft_pass

    rows = 16
    _, dv = _vocab()
    rs = SpecRejectionSampling(
        max_rows=rows, mtp_k=S, max_num_seqs=rows, device=DEV, draft_k=K
    )
    rs.stage_step(list(range(rows)), torch.ones(rows, **F32), 0.95)
    logits = torch.randn(rows, dv.size, **F32).to(torch.bfloat16)
    out = torch.zeros(rows, **I32)
    rs.sample_draft(logits, out, vocab=dv)
    rs.store_step(S - 1, rows)
    want = rs.records.clone(), rs.record_tok.clone()
    rs.records.zero_()
    rs.record_tok.fill_(-1)
    draft = torch.zeros(rows, S, dtype=torch.int64, device=DEV)
    store_last_draft_pass(out, draft, S - 1, rows, records=rs.record_args(S - 1, rows))
    assert torch.equal(rs.records, want[0]) and torch.equal(rs.record_tok, want[1])
    slots = rs.step_slots[:rows]
    assert torch.equal(rs.record_tok[slots, S - 1], out)
    assert torch.equal(draft[:, S - 1], out.long())
