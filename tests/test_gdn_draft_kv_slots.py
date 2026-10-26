# SPDX-License-Identifier: MIT
"""Where an MTP draft mid-step writes its KV on a GDN hybrid (Qwen3.5 / Next).

`EagleProposer.propose` takes the draft's write slots from what
`prepare_mtp_decode` returns, and falls back to `kv_indices[kv_indptr[1:] - 1]`
when it returns none. That fallback is a token slot only at page size 1 (MLA).
The GDN builder used to return none, so at `block_size` 16 the draft wrote its
K/V at token slot = PAGE id, i.e. into block `id // 16` -- another request's KV
or the units of a PAGE state checkpoint. A Qwen3.5 checkpoint image overwritten
that way restores as garbage, and the resumed request answers in one token
(multi-turn GSM8K, ATOM_GDN_STATE_MIDSTEP=1).
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_ops.attentions import gdn_attn
from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder

BLOCK = 16
COLS = 8
# Three real rows and one pad row. Page ids are far apart from anything a
# token slot of these rows could be, so `id` read as a slot is never right.
TABLES = [
    [700, 701, 702, 0, 0, 0, 0, 0],
    [300, 905, 0, 0, 0, 0, 0, 0],
    [41, 42, 43, 44, 45, 0, 0, 0],
    [0] * COLS,
]
# Context after the proposer's bump: the newest token sits at ctx - 1.
CTX = [35, 17, 80, 0]


def expected(rows=3):
    out = []
    for table, ctx in zip(TABLES[:rows], CTX[:rows]):
        pos = ctx - 1
        out.append(table[pos // BLOCK] * BLOCK + pos % BLOCK)
    return out + [-1] * (len(TABLES) - rows)


def _cpu_kv_indices(block_tables, kv_indices, kv_indptr, block_ratio, max_seqlen_k):
    del block_ratio, max_seqlen_k
    for r in range(block_tables.shape[0]):
        a, b = int(kv_indptr[r]), int(kv_indptr[r + 1])
        kv_indices[a:b] = block_tables[r, : b - a]


def builder(monkeypatch):
    """The GDN builder with just what `prepare_mtp_decode` reads."""
    monkeypatch.setattr(gdn_attn, "kv_indices_generate_triton", _cpu_kv_indices)
    # Not a dummy run. `raising=False`: the builder did not read the forward
    # context before the fix, and the test has to load against that tree too.
    monkeypatch.setattr(
        gdn_attn,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(is_dummy_run=False)),
        raising=False,
    )
    tables = torch.tensor(TABLES, dtype=torch.int32)
    ctx = torch.tensor(CTX, dtype=torch.int32)
    blocks = [(c + BLOCK - 1) // BLOCK for c in CTX]
    indptr = torch.tensor([0] + list(torch.tensor(blocks).cumsum(0)), dtype=torch.int32)
    var = {
        "block_tables": SimpleNamespace(gpu=tables),
        "context_lens": SimpleNamespace(gpu=ctx),
        "kv_indptr": SimpleNamespace(gpu=indptr),
        "kv_indices": SimpleNamespace(gpu=torch.zeros(64, dtype=torch.int32)),
        "slot_mapping": SimpleNamespace(
            gpu=torch.full((16,), 12345, dtype=torch.int64)
        ),
    }
    b = object.__new__(GDNAttentionMetadataBuilder)
    b.model_runner = SimpleNamespace(forward_vars=var, block_size=BLOCK)
    b.block_ratio = 1
    b.block_size = BLOCK
    b.refresh_flydsl_plan = lambda context_lens: None
    return b, var


def proposer_slots(workinfos, var, running_bs):
    """The slots `EagleProposer.propose` hands the draft's KV write."""
    if "slot_mapping" in workinfos:
        return workinfos["slot_mapping"][:running_bs]
    indptr = var["kv_indptr"].gpu
    return var["kv_indices"].gpu[(indptr[1 : running_bs + 1] - 1).long()].long()


def test_the_draft_writes_where_the_target_would(monkeypatch):
    b, var = builder(monkeypatch)
    running_bs = len(TABLES)
    workinfos = b.prepare_mtp_decode(
        3, 1, max(CTX), torch.zeros(running_bs, dtype=torch.int32)
    )
    slots = proposer_slots(workinfos, var, running_bs)
    assert slots[:3].tolist() == expected()[:3]


def test_no_slot_lands_in_a_block_the_row_does_not_own(monkeypatch):
    b, var = builder(monkeypatch)
    running_bs = len(TABLES)
    workinfos = b.prepare_mtp_decode(
        3, 1, max(CTX), torch.zeros(running_bs, dtype=torch.int32)
    )
    slots = proposer_slots(workinfos, var, running_bs)
    for row in range(3):
        assert (
            int(slots[row]) // BLOCK in TABLES[row][: (CTX[row] + BLOCK - 1) // BLOCK]
        )


def test_pad_rows_write_nowhere():
    out = torch.full((len(TABLES),), 7, dtype=torch.int64)
    gdn_attn.draft_kv_write_slots(
        torch.tensor(CTX, dtype=torch.int32),
        torch.tensor(TABLES, dtype=torch.int32),
        BLOCK,
        out,
        3,
    )
    assert out.tolist() == expected()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel path needs a GPU")
def test_the_kernel_matches_the_reference():
    tables = torch.tensor(TABLES, dtype=torch.int32)
    ctx = torch.tensor(CTX, dtype=torch.int32)
    ref = torch.empty(len(TABLES), dtype=torch.int64)
    gdn_attn.draft_kv_write_slots(ctx, tables, BLOCK, ref, 3)
    got = torch.full((len(TABLES),), 7, dtype=torch.int64, device="cuda")
    gdn_attn.draft_kv_write_slots(ctx.cuda(), tables.cuda(), BLOCK, got, 3)
    assert got.cpu().tolist() == ref.tolist() == expected()
