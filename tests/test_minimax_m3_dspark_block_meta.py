# SPDX-License-Identifier: MIT
"""The MiniMax-M3.1 DSpark block metadata kernel (`dspark._block_meta_kernel`)
against the torch path it replaces (the proposer's slots / context lengths,
`AllocationCap.apply`, `_page16_table`, `_window_rows`): positions, slots and
lengths equal, and every table entry a row reads equal."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

T, ENTRY, WINDOW = 7, 128, 4096


def _reference(block_tables, anchors, alloc_pages, scheduled_bs, pad_ctx):
    from atom.model_ops.minimax_m3.sparse_attn import PAGES_PER_SPARSE_BLOCK
    from atom.models.dspark_draft import AllocationCap
    from atom.models.minimax_m3.dspark import _page16_table, _window_rows

    dev = block_tables.device
    running_bs, W = block_tables.shape
    pos = anchors.view(-1, 1) + torch.arange(1, T + 1, device=dev).view(1, T)
    page_idx = torch.div(pos, ENTRY, rounding_mode="floor")
    in_table = page_idx < W
    page_idx = page_idx.clamp(max=W - 1)
    slots = torch.gather(block_tables, 1, page_idx).long() * ENTRY + pos % ENTRY
    slots.masked_fill_(~in_table, -1)
    ctx = anchors + 1 + T
    slots[scheduled_bs:] = -1
    ctx[scheduled_bs:] = pad_ctx
    cap = AllocationCap(running_bs, dev)
    alloc = cap.apply(alloc_pages, scheduled_bs, pos, slots, ENTRY)
    ctx_eff = torch.minimum(ctx.int(), alloc)
    P = PAGES_PER_SPARSE_BLOCK * W
    flat = torch.zeros((running_bs + 1) * P, dtype=torch.int32, device=dev)
    bt16 = flat[: running_bs * P].view(running_bs, P)
    sub = torch.arange(8, dtype=torch.int32, device=dev)
    _page16_table(block_tables, bt16, sub)
    if running_bs > scheduled_bs:
        bt16[scheduled_bs:, :8].copy_(sub)
    rows = running_bs * T
    row_req = torch.arange(running_bs, device=dev).repeat_interleave(T)
    tables = torch.zeros(rows, P, dtype=torch.int32, device=dev)
    lens = torch.zeros(rows, dtype=torch.int32, device=dev)
    start16 = torch.zeros(rows, dtype=torch.int64, device=dev)
    _window_rows(
        flat,
        ctx_eff,
        pos.view(-1),
        row_req,
        WINDOW,
        scheduled_bs * T,
        start16,
        tables,
        lens,
    )
    return pos, slots, ctx.int(), tables, lens


@pytest.mark.parametrize("stage", [True, False])
@pytest.mark.parametrize(
    "running_bs,scheduled_bs", [(1, 1), (8, 5), (16, 16), (64, 61)]
)
def test_block_meta_matches_torch_path(running_bs, scheduled_bs, stage):
    import triton

    from atom.models.minimax_m3.dspark import _block_meta_kernel

    torch.manual_seed(running_bs)
    dev = torch.device("cuda")
    W = 96  # 12K-token table
    block_tables = torch.randint(
        1, 50000, (running_bs, W), dtype=torch.int32, device=dev
    )
    # anchors: short contexts, contexts inside the window, past it, and at the table's end
    choices = [0, 5, 127, 3000, 4100, 9000, W * ENTRY - 3, W * ENTRY - T - 1]
    anchors = torch.tensor(
        [choices[i % len(choices)] for i in range(running_bs)],
        dtype=torch.int64,
        device=dev,
    )
    anchors[scheduled_bs:] = 0
    # allocation: through the anchor's page, sometimes one more page
    alloc_pages = np.array(
        [int(a) // ENTRY + 1 + (i % 2) for i, a in enumerate(anchors.tolist())],
        dtype=np.int32,
    )[:scheduled_bs]
    pad_ctx = 1 + T
    ref = _reference(block_tables, anchors, alloc_pages, scheduled_bs, pad_ctx)

    pw = triton.cdiv(WINDOW + T + 16, 16) + 1
    pw = triton.cdiv(pw, 8) * 8
    pos = torch.empty(running_bs, T, dtype=torch.int64, device=dev)
    slots = torch.empty(running_bs, T, dtype=torch.int64, device=dev)
    ctx = torch.empty(running_bs, dtype=torch.int32, device=dev)
    rows_bt = torch.full((running_bs * T, pw), -7, dtype=torch.int32, device=dev)
    rows_len = torch.empty(running_bs * T, dtype=torch.int32, device=dev)
    # the published form: real rows' page counts, stale garbage past them
    alloc = torch.full((running_bs,), -5, dtype=torch.int32, device=dev)
    alloc[:scheduled_bs] = torch.from_numpy(alloc_pages).to(dev)
    staged_anchor = torch.full_like(anchors, -1) if stage else anchors.clone()
    ids_out = torch.full((running_bs,), -1, dtype=torch.int32, device=dev)
    # STAGE: anchors come from positions[rows]; next ids are clamped to the vocab
    positions = torch.zeros(10 * running_bs * 8, device=dev, dtype=torch.int64)
    rows = torch.arange(scheduled_bs, device=dev, dtype=torch.int64) * 8 + 7
    positions[rows] = anchors[:scheduled_bs]
    next_ids = torch.randint(-3, 210000, (scheduled_bs,), device=dev, dtype=torch.int32)
    if not stage:  # the default: every staging pointer aliases the anchors
        next_ids = positions = rows = ids_out = staged_anchor
    _block_meta_kernel[(running_bs,)](
        staged_anchor, block_tables, block_tables.stride(0), alloc,
        next_ids, positions, rows, ids_out, 200063,
        pos, slots, ctx, rows_bt, rows_len,
        scheduled_bs, W, T=T, TP=8, ENTRY=ENTRY, WINDOW=WINDOW, PW=pw,
        PWP=triton.next_power_of_2(pw), PAD_CTX=pad_ctx, STAGE=stage, num_warps=4,
    )  # fmt: skip
    torch.testing.assert_close(staged_anchor, anchors)
    if stage:
        want_ids = torch.zeros(running_bs, dtype=torch.int32, device=dev)
        want_ids[:scheduled_bs] = next_ids.clamp(0, 200063)
        torch.testing.assert_close(ids_out, want_ids)
    r_pos, r_slots, r_ctx, r_tables, r_lens = ref
    torch.testing.assert_close(pos, r_pos)
    torch.testing.assert_close(slots, r_slots)
    torch.testing.assert_close(ctx, r_ctx)
    torch.testing.assert_close(rows_len, r_lens)
    for r in range(running_bs * T):
        n = (int(r_lens[r]) + 15) // 16
        assert n <= pw, (r, n, pw)
        torch.testing.assert_close(rows_bt[r, :n], r_tables[r, :n], msg=f"row {r}")


@pytest.mark.parametrize("bs", [1, 3, 64])
@pytest.mark.parametrize("shard", [0, 2])
def test_block_embedding(bs, shard):
    """`MiniMaxM3DSpark.embed_block` (one rank, tp 1 so no all-reduce) equals
    the masked lookup of ``[anchor, MASK, ...]`` on that rank's vocab shard."""
    from types import SimpleNamespace

    from atom.models.minimax_m3.dspark import MiniMaxM3DSpark

    torch.manual_seed(bs + shard)
    block, V, D, tp = 7, 4096, 6144, 4
    n = V // tp
    v0 = shard * n
    weight = torch.randn(n, D, device="cuda").to(torch.bfloat16)
    noise = 2 * n + 5  # in shard 2 only
    anchors = torch.randint(0, V, (bs,), device="cuda", dtype=torch.int32)
    anchors[0] = v0 + 3
    model = SimpleNamespace(
        embed_tokens=SimpleNamespace(weight=weight, vocab_start_idx=v0, tp_size=1),
        noise_token_id=noise,
    )
    out = MiniMaxM3DSpark.embed_block(model, anchors, block)
    ids = torch.full((bs, block), noise, device="cuda", dtype=torch.int32)
    ids[:, 0] = anchors
    ids = ids.view(-1)
    hit = (ids >= v0) & (ids < v0 + n)
    ref = torch.where(hit[:, None], weight[(ids - v0).clamp(0, n - 1).long()], 0)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("running_bs,scheduled_bs", [(1, 1), (8, 5)])
def test_build_block_metadata_published_equals_host_upload(running_bs, scheduled_bs):
    """`build_block_metadata` with the allocation published on the device (stale
    past the scheduled rows) equals the host-upload fallback."""
    from types import SimpleNamespace

    from atom.models.minimax_m3.dspark import MiniMaxM3DSpark

    torch.manual_seed(running_bs)
    dev, W = torch.device("cuda"), 96
    model = SimpleNamespace(sliding_window=WINDOW, max_num_seqs=64, vocab_size=1000)
    block_tables = torch.randint(
        1, 5000, (running_bs, W), dtype=torch.int32, device=dev
    )
    anchors = torch.randint(0, W * ENTRY - T - 1, (running_bs,), device=dev)
    anchors[scheduled_bs:] = 0
    alloc_host = np.array(
        [int(a) // ENTRY + 1 + (i % 2) for i, a in enumerate(anchors.tolist())],
        dtype=np.int32,
    )[:scheduled_bs]
    alloc_dev = torch.full((running_bs,), 9, dtype=torch.int32, device=dev)
    alloc_dev[:scheduled_bs] = torch.from_numpy(alloc_host).to(dev)

    def run(alloc):
        out = (
            torch.empty(running_bs, T, dtype=torch.int64, device=dev),
            torch.empty(running_bs, T, dtype=torch.int64, device=dev),
            torch.empty(running_bs, dtype=torch.int32, device=dev),
        )
        meta = SimpleNamespace()
        MiniMaxM3DSpark.build_block_metadata(
            model, meta, block_tables, anchors, *out, alloc,
            scheduled_bs, ENTRY, 1 + T,
        )  # fmt: skip
        return *out, meta.dspark_bt16_rows.clone(), meta.dspark_ctx_rows.clone()

    for got, want in zip(run(alloc_dev), run(alloc_host)):
        assert torch.equal(got, want)
