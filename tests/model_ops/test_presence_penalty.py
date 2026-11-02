# SPDX-License-Identifier: MIT
"""presence_penalty against a plain reference: MTP4-shaped verify rows (anchor + 4
drafts a request, with repeats among the drafts and drafts already generated), the
full vocab and one TP shard's slice, emitted-token marking with -1 holes, slot reuse
across requests, plain decode rows, and a prefill step that only clears."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

V, TP = 248320, 4


def _batch(req_ids, q, pens, prefill=0):
    return SimpleNamespace(
        req_ids=req_ids,
        num_scheduled_tokens=np.asarray(q, np.int32),
        presence_penalties=np.asarray(pens, np.float32),
        is_dummy_run=False,
        total_tokens_num_prefill=prefill,
    )


def test_presence_penalty_matches_reference():
    from atom.model_ops.presence_penalty import PresencePenaltyState

    dev = "cuda"
    g = torch.Generator().manual_seed(0)
    state = PresencePenaltyState(V, max_num_seqs=8, max_extra=8, device=dev)
    history = {}  # request -> its generated tokens (the reference)

    def step(req_ids, pens, q=5, shard=None):
        """One verify step: [anchor, d0..d3] a request, penalize the logits, then
        emit a random accepted prefix."""
        n = len(req_ids)
        ids = []
        for r in req_ids:
            hist = sorted(history[r]) if history[r] else [7]
            anchor = hist[torch.randint(len(hist), (1,), generator=g).item()]
            drafts = torch.randint(0, 300, (q - 1,), generator=g).tolist()
            if q > 2:
                drafts[1] = drafts[0]  # a repeated draft
            if q > 3 and history[r]:
                drafts[2] = hist[0]  # a draft already generated
            ids += [anchor] + drafts
            history[r].add(anchor)  # a decode query starts at the last emitted token
        input_ids = torch.tensor(ids, dtype=torch.int32, device=dev)
        st = state.begin_step(_batch(req_ids, [q] * n, pens), input_ids)
        lo, hi = (0, V) if shard is None else (shard * V // TP, (shard + 1) * V // TP)
        logits = torch.randn(n * q, hi - lo, generator=g).to(dev, torch.bfloat16)
        ref = logits.float().clone()
        for i, r in enumerate(req_ids):
            for j in range(q):
                present = set(history[r]) | set(ids[i * q + 1 : i * q + 1 + j])
                cols = [t - lo for t in present if lo <= t < hi]
                if cols:
                    ref[i * q + j, cols] -= pens[i]
        st.apply(logits, lo)
        torch.testing.assert_close(
            logits.float(), ref.to(torch.bfloat16).float(), atol=1e-2, rtol=0
        )
        out = torch.full((n, q), -1, dtype=torch.int32)
        for i, r in enumerate(req_ids):
            a = torch.randint(1, q + 1, (1,), generator=g).item()
            toks = torch.randint(0, 300, (a,), generator=g).tolist()
            out[i, :a] = torch.tensor(toks)
            history[r].update(toks)
        st.mark(out.to(dev))

    reqs = ["a", "b", "c"]
    for r in reqs:
        history[r] = set()
    # a prefill step clears the slots and penalizes nothing
    assert (
        state.begin_step(_batch(reqs, [100, 50, 7], [1.5, 1.5, 0.0], prefill=157), None)
        is None
    )
    for _ in range(3):
        step(reqs, [1.5, 1.5, 0.0])
    step(reqs, [1.5, 1.5, 0.0], shard=2)
    # 20 new requests churn the 8 slots: the running ones keep theirs
    for k in range(20):
        rid = f"n{k}"
        history[rid] = set()
        step(["a", "b", rid], [1.5, 1.5, 1.0])
    step(["a", "b"], [1.5, 1.5], shard=3)
    step(["a", "n19"], [1.5, 1.0], q=1)  # plain decode: one query token a request
    mask_a = state.mask[state._slot_of["a"]].nonzero().flatten().tolist()
    assert sorted(mask_a) == sorted(history["a"])
