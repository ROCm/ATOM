# SPDX-License-Identifier: MIT

"""`kv_indices_{swa,csa,hca}` carry no information in their own length.

`_attach_v4_paged_decode_meta` publishes these whole rather than sliced to
`indptr_np[T]`. The slice was never wrong, but its length is data-dependent --
it is the cumsum of per-token KV spans -- and under AF_PIECEWISE the attention
core is captured into its own cudagraph, which bakes whatever shape it was
handed. A length that varies with the batch's sequence lengths, baked once at
capture, only stays correct because the backing buffer is sized for the worst
case. Publishing the whole buffer removes the coincidence.

That rests on one property of both consumers: neither derives a bound from the
tensor's own length. The writer's grid comes from `topk_local.shape` and it
writes `[indptr[t], indptr[t] + valid_k[t])`; the reader walks
`kv_indices[indptr[t] : indptr[t+1]]`. So both are exercised here against an
exactly-sized destination and an oversized one, and required to agree.

The production decode reader is aiter's `mla_decode_fwd_v4_nm`, which these
tests cannot reach -- `paged_decode.py` already documents the same contract for
it ("in production `kv_indices.shape[0]` is a padded bucket whose value is
unrelated to the true per-token kv_len"), and `sparse_attn_v4_paged_decode`
passes the tensor straight through with no length-derived argument. What is
locked here is the in-repo half.
"""

from __future__ import annotations

import torch

from atom.model_ops.v4_kernels.csa_translate_pack import csa_translate_pack_reference
from atom.model_ops.v4_kernels.paged_decode import (
    sparse_attn_v4_paged_decode_reference,
)

ENVELOPE_ROWS = 8
CSA_BLOCK_CAPACITY = 64
WINDOW_SIZE = 128
SLACK = 97  # deliberately not a round number, and not a multiple of index_topk


def _decode_batch(bs: int, tokens_per_seq: int, index_topk: int):
    """A ragged decode batch: per-token spans differ, and a CG pad tail follows.

    `positions` drives `skip = min(pos + 1, WINDOW_SIZE)` inline, so varying
    them across tokens is what makes `valid_k` -- and therefore the exact
    destination length -- data-dependent in the first place.
    """
    g = torch.Generator().manual_seed(bs * 1000 + tokens_per_seq)
    t_real = bs * tokens_per_seq
    t_pad = t_real + 3  # CG padding: batch_id -1, contributes nothing

    batch_id = torch.full((t_pad,), -1, dtype=torch.int32)
    batch_id[:t_real] = torch.repeat_interleave(
        torch.arange(bs, dtype=torch.int32), tokens_per_seq
    )
    positions = torch.zeros(t_pad, dtype=torch.int32)
    positions[:t_real] = torch.randint(
        WINDOW_SIZE, WINDOW_SIZE * 6, (t_real,), generator=g, dtype=torch.int32
    )

    # Slice length per token = skip + valid_k, with valid_k ragged across tokens.
    skip = torch.minimum(
        positions[:t_real].to(torch.int64) + 1, torch.tensor(WINDOW_SIZE)
    )
    valid_k = torch.randint(1, index_topk + 1, (t_real,), generator=g)
    spans = torch.zeros(t_pad, dtype=torch.int64)
    spans[:t_real] = skip + valid_k

    indptr = torch.zeros(t_pad + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(spans, 0).to(torch.int32)

    topk_local = torch.randint(
        0, CSA_BLOCK_CAPACITY * 4, (t_pad, index_topk), generator=g, dtype=torch.int32
    )
    block_tables = torch.randint(
        1, 5000, (max(bs, 1), 16), generator=g, dtype=torch.int32
    )
    return topk_local, block_tables, positions, indptr, batch_id, int(indptr[t_pad])


def _run_writer(dest_len: int, batch) -> torch.Tensor:
    topk_local, block_tables, positions, indptr, batch_id, _ = batch
    dest = torch.full((dest_len,), -7, dtype=torch.int32)
    csa_translate_pack_reference(
        topk_local,
        block_tables,
        positions,
        indptr,
        batch_id,
        None,
        dest,
        envelope_rows=ENVELOPE_ROWS,
        csa_block_capacity=CSA_BLOCK_CAPACITY,
        window_size=WINDOW_SIZE,
    )
    return dest


def test_writer_ignores_the_destination_length():
    batch = _decode_batch(bs=5, tokens_per_seq=6, index_topk=32)
    exact = batch[-1]

    tight = _run_writer(exact, batch)
    loose = _run_writer(exact + SLACK, batch)

    torch.testing.assert_close(loose[:exact], tight)
    assert torch.all(loose[exact:] == -7), (
        "an oversized destination must leave its tail untouched -- a writer that "
        "sized anything off `kv_indices.numel()` would have scribbled into it"
    )


def test_writer_is_length_invariant_across_shapes():
    """The same, over batch shapes whose exact lengths differ widely."""
    for bs, tokens_per_seq, index_topk in [
        (1, 1, 16),
        (3, 4, 32),
        (8, 6, 64),
        (17, 2, 128),
    ]:
        batch = _decode_batch(bs, tokens_per_seq, index_topk)
        exact = batch[-1]
        tight = _run_writer(exact, batch)
        loose = _run_writer(exact + SLACK, batch)
        torch.testing.assert_close(
            loose[:exact],
            tight,
            msg=f"bs={bs} tokens_per_seq={tokens_per_seq} topk={index_topk}",
        )


def test_reader_ignores_the_indices_length():
    """Attention output is identical over an exact vs an oversized `kv_indices`.

    The oversized tail is filled with in-range but WRONG slot ids, so a reader
    that walked past `indptr[T]` would change its answer rather than fault.
    """
    torch.manual_seed(0)
    t, heads, dim, pages = 6, 4, 32, 512

    spans = torch.tensor([3, 1, 7, 0, 4, 2])
    indptr = torch.zeros(t + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(spans, 0).to(torch.int32)
    exact = int(indptr[t])

    q = torch.randn(t, heads, dim)
    unified_kv = torch.randn(pages, dim)
    attn_sink = torch.randn(heads)
    tight = torch.randint(0, pages, (exact,), dtype=torch.int32)
    loose = torch.cat([tight, torch.randint(0, pages, (SLACK,), dtype=torch.int32)])

    out_tight = sparse_attn_v4_paged_decode_reference(
        q, unified_kv, tight, indptr, attn_sink, dim**-0.5
    )
    out_loose = sparse_attn_v4_paged_decode_reference(
        q, unified_kv, loose, indptr, attn_sink, dim**-0.5
    )
    torch.testing.assert_close(out_tight, out_loose)
