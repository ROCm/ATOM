# SPDX-License-Identifier: MIT
"""DCP sparse PREFILL index filter: a top-k -> this rank's compacted slot list.

Covers ``triton_filter_and_convert_dcp_index_prefill`` in
``atom/model_ops/dcp_ops.py``. Decode had a twin until the fused merge
(``aiter.flydsl_dcp_topk_merge``) took over, emitting this rank's owned slots
directly with nothing left to filter; its tests live in test_dcp_topk.py.

Why the checks go past "it does not crash": ``cp_lse_ag_out_rs`` rebuilds a
global softmax out of per-rank partial attentions, and that is only valid when
the ranks' candidate sets form a DISJOINT PARTITION of one global top-k. A
filter that silently drops or double-claims a token produces no fault and no
NaN -- just a quietly wrong answer. So values, order, per-region lengths and the
partition property are all asserted.

The expected slots come from the WRITE side (``writer_slot`` below), not from a
second copy of the filter's own arithmetic, so a wrong slot formula cannot agree
with itself and pass.

The compacted layout must also be hole-free: an earlier "fixed length + -1
sentinel" layout is what broke aiter's lse path (the ASM decode kernel does not
honour the -1 mask the way the Triton one does), so a -1 anywhere inside a
written region is a failure.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

try:
    from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
    from atom.model_ops.dcp_ops import triton_filter_and_convert_dcp_index_prefill
except ImportError as _e:  # triton/aiter absent on a CPU-only runner
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("Triton kernels need a GPU", allow_module_level=True)

DEV = "cuda"


def writer_slot(block_table_row, pos, rank, world, page, interleave=1):
    """Where the WRITE side physically put this token; -1 if another rank owns it.

    This is the independent authority the expected values are built from. A
    hand-written copy of the filter's own slot formula would check compaction,
    ownership and ordering, but could never catch the formula itself being
    wrong -- and a wrong slot is the one bug class that makes attention read
    another sequence's KV, silently. So the reference calls the function that
    builds decode's slot_mapping: the KV really does live wherever that put it,
    and the two sides now break loudly if they ever drift apart.
    """
    stub = SimpleNamespace(
        dcp_world_size=world,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
        model_runner=SimpleNamespace(block_size=page),
    )
    return int(
        AiterMLAMetadataBuilder._dcp_round_robin_slot(stub, block_table_row, pos)
    )


# ────────────────────────────────────────────────────────────── prefill side ──

PRE_W = 8  # overridden per-test below
PRE_PAGE = 16
PRE_TOPK = (
    256  # multiple of BLOCK_N=128; production runs index_topk=2048 (see the prod case)
)


def _build_prefill_case(seq_lens):
    """One prefill batch: `seq_lens` fresh sequences, candidates = past tokens."""
    bs = len(seq_lens)
    cu_k = np.zeros(bs + 1, dtype=np.int32)
    np.cumsum(seq_lens, out=cu_k[1:])
    num_tokens = int(cu_k[bs])

    token_to_seq = np.repeat(np.arange(bs, dtype=np.int32), seq_lens)
    # position of each query token within its sequence
    local_off = np.concatenate([np.arange(s, dtype=np.int32) for s in seq_lens])
    counts = np.minimum(local_off + 1, PRE_TOPK).astype(np.int32)

    kv_indptr = np.zeros(num_tokens + 1, dtype=np.int32)
    np.cumsum(counts, out=kv_indptr[1:])

    rng = np.random.default_rng(0)

    # topk_indices holds FLAT KV indices here (not within-sequence positions).
    # Token t of seq b selects `counts[t]` positions of its own causal window;
    # a strided-then-wrapped pick keeps the selection non-contiguous.
    topk = np.full((num_tokens, PRE_TOPK), -1, dtype=np.int32)
    for t in range(num_tokens):
        b = token_to_seq[t]
        n = counts[t]
        p = local_off[t]
        sel = (np.arange(n, dtype=np.int64) * 7919) % (p + 1)
        # np.unique also SORTS. The real indexer emits its top-k in score order,
        # and compaction has to preserve whatever order it gets (that is what
        # makes the fp accumulation order reproducible), so feeding a sorted
        # selection would make "kept in order" and "sorted on the way out"
        # indistinguishable. Shuffle it back.
        sel = np.unique(sel)
        rng.shuffle(sel)
        topk[t, : len(sel)] = cu_k[b] + sel.astype(np.int32)
        kv_indptr[t + 1] = kv_indptr[t] + len(sel)

    max_blocks = int(
        max((s + PRE_PAGE * PRE_W - 1) // (PRE_PAGE * PRE_W) for s in seq_lens)
    )
    block_table = rng.permutation(bs * max_blocks).reshape(bs, max_blocks)
    return {
        "bs": bs,
        "num_tokens": num_tokens,
        "cu_k": cu_k,
        "token_to_seq": token_to_seq,
        "kv_indptr": kv_indptr,
        "topk": topk,
        "block_table": block_table.astype(np.int32),
    }


def _run_prefill(case, interleave=1):
    n = case["num_tokens"]
    g = {
        k: torch.from_numpy(v).to(DEV)
        for k, v in case.items()
        if isinstance(v, np.ndarray)
    }
    out_buf = torch.full((n * PRE_TOPK,), -7, dtype=torch.int32, device=DEV)
    indptr = torch.zeros(n + 1, dtype=torch.int32, device=DEV)
    counts_scratch = torch.zeros(n, dtype=torch.int32, device=DEV)

    per_rank = []
    for r in range(PRE_W):
        out_buf.fill_(-7)
        counts_scratch.fill_(-1)
        triton_filter_and_convert_dcp_index_prefill(
            g["kv_indptr"],
            g["token_to_seq"],
            g["topk"],
            g["cu_k"],
            g["block_table"],
            r,
            PRE_W,
            PRE_PAGE,
            out_kv_indptr=indptr,
            owned_counts=counts_scratch,
            NUM_TOPK_TOKENS=PRE_TOPK,
            out=out_buf,
            cp_kv_cache_interleave_size=interleave,
        )
        torch.cuda.synchronize()
        per_rank.append(
            (
                indptr.cpu().numpy().copy(),
                out_buf.cpu().numpy().copy(),
                counts_scratch.cpu().numpy().copy(),
            )
        )
    return per_rank


@pytest.mark.parametrize("world", [8, 4, 2])  # production ships dcp8; 4/2 untested
@pytest.mark.parametrize("interleave", [1, 4])  # 4 divides PRE_PAGE=16
@pytest.mark.parametrize(
    "seq_lens", [[400], [300, 240], [17, 5, 1]], ids=["single", "two-seq", "tiny"]
)
def test_prefill_filter(seq_lens, interleave, world):
    global PRE_W
    PRE_W = world
    case = _build_prefill_case(np.asarray(seq_lens, dtype=np.int32))
    per_rank = _run_prefill(case, interleave)

    cu_k = case["cu_k"]
    tts = case["token_to_seq"]
    bt = case["block_table"]
    topk = case["topk"]
    kvp = case["kv_indptr"]

    # (seq, within-seq position) -> (owning rank, slot), asked of the write side.
    # Memoised: query tokens reselect the same positions constantly, so this is a
    # few hundred lookups instead of a few hundred thousand.
    owner_of = {}

    def _writer(b, p):
        key = (int(b), int(p))
        if key not in owner_of:
            claims = [
                (r, writer_slot(bt[b], int(p), r, PRE_W, PRE_PAGE, interleave))
                for r in range(PRE_W)
            ]
            claims = [(r, s) for r, s in claims if s >= 0]
            assert (
                len(claims) == 1
            ), f"seq {b} position {p}: the writer gives it {len(claims)} owners"
            owner_of[key] = claims[0]
        return owner_of[key]

    n_empty = 0
    for t in range(case["num_tokens"]):
        b = tts[t]
        want = topk[t, : kvp[t + 1] - kvp[t]]
        want = want[want >= 0]
        n_claimed = 0
        for r in range(PRE_W):
            ind, buf, cnts = per_rank[r]
            slots = buf[ind[t] : ind[t + 1]]
            exp_slots = [
                slot
                for p in (want - cu_k[b])
                for owner, slot in [_writer(b, p)]
                if owner == r
            ]

            # Contract: a row this rank owns nothing of gets one valid dummy
            # slot so persistent MLA metadata never sees a zero-length row.
            # owned_counts remains 0, allowing the caller to replace its
            # attention result with O=0/LSE=-inf before the DCP merge.
            # Counted off the kernel's own indptr, never off the reference:
            # summing the reference's per-rank splits would reproduce `want` by
            # construction and assert nothing.
            n_claimed += int(cnts[t])

            if not exp_slots:
                n_empty += 1
                assert list(slots) == [0] and cnts[t] == 0, (
                    f"token={t} rank={r}: unowned row must hold one dummy, got "
                    f"slots={list(slots)} count={cnts[t]}"
                )
                continue

            assert cnts[t] == len(
                exp_slots
            ), f"token={t} rank={r}: owned_count {cnts[t]} != {len(exp_slots)}"
            assert list(slots) == exp_slots, (
                f"token={t} rank={r}: slots {list(slots)[:8]}... "
                f"!= {exp_slots[:8]}..."
            )

        # Partition: every candidate is claimed by exactly one rank -- neither
        # dropped nor double-claimed, which is what cp_lse_ag_out_rs needs.
        assert n_claimed == len(
            want
        ), f"token={t}: ranks kept {n_claimed} of {len(want)} candidates"

    # Every batch starts with tokens whose causal window is shorter than W, so
    # the zero-owned path must have been exercised -- guard against the empty-row
    # assertions passing vacuously.
    assert n_empty > 0, "expected some rank to own nothing for the early tokens"
