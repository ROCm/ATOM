# SPDX-License-Identifier: MIT
"""The planar FP4 prefill paths (`packed_prefill_rows`): the distinct-row
remap on CPU; on a GPU, the decoded shadow against the tiled path's staged
rows (bit for bit) and the three modes' outputs against each other."""

import pytest

torch = pytest.importorskip("torch")

from atom.model_ops.attentions.deepseek_v41.packed_prefill_rows import (  # noqa: E402
    BatchPages,
    _page_rows,
    _remap_rows_torch,
    distinct_rows,
)

RPP, RING = 8, 133


def _paged_csr(tokens, cap, pages_total, slots_total, requests, gen, device="cpu"):
    """A CSR over `requests` requests, each owning random pages (rows page * RPP + j) and a ring
    (rows slot * RING + k), with the `BatchPages` that describe them."""
    perm = torch.randperm(pages_total, generator=gen)
    columns = [int(c) for c in torch.randint(1, 9, (requests,), generator=gen)]
    width = max(columns) + 2
    tables = torch.zeros(requests, width, dtype=torch.int32)
    used = 0
    for b, n in enumerate(columns):
        tables[b, :n] = perm[used : used + n]
        used += n
    slots = [int(x) for x in torch.randperm(slots_total, generator=gen)[:requests]]
    counts = torch.randint(1, cap + 1, (tokens,), generator=gen)
    indptr = torch.zeros(tokens + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(counts, 0)
    n = tokens * cap
    req = torch.randint(0, requests, (n,), generator=gen)
    win = torch.rand(n, generator=gen) < 0.3
    col = (torch.rand(n, generator=gen) * torch.tensor(columns)[req]).long()
    main = tables[req, col].long() * RPP + torch.randint(0, RPP, (n,), generator=gen)
    ring = torch.tensor(slots)[req] * RING + torch.randint(0, RING, (n,), generator=gen)
    indices = torch.where(win, ring * 2 + 1, main * 2)
    indices[int(indptr[-1]) :] = -1
    pages = BatchPages(tables.to(device), tuple(columns), tuple(slots), RPP, RING)
    return indices.to(device), indptr.to(device), pages


def test_page_rows_remap_every_named_row():
    gen = torch.Generator().manual_seed(3)
    indices, indptr, pages = _paged_csr(41, 32, 64, 8, 3, gen)
    tagged, local = _page_rows(indices, indptr, 64 * RPP, 8 * RING, pages)
    n = int(indptr[-1])
    assert tagged.numel() == sum(pages.columns) * RPP + 3 * RING
    assert torch.equal(tagged[local[:n].long()], indices[:n])


def test_page_rows_of_a_window_only_layer():
    gen = torch.Generator().manual_seed(4)
    indices, indptr, pages = _paged_csr(9, 8, 16, 4, 2, gen)
    keep = (indices & 1) == 1
    indices = torch.where(keep, indices, torch.full_like(indices, 1))   # window rows only
    pages = BatchPages(pages.tables, pages.columns, pages.slots, 0, RING)
    tagged, local = _page_rows(indices, indptr, 0, 4 * RING, pages)
    n = int(indptr[-1])
    assert tagged.numel() == 2 * RING and int(local.min()) >= 0
    named = indices[:n] != 1
    assert torch.equal(tagged[local[:n].long()][named], indices[:n][named])


def _csr(tokens, cap, main_rows, window_rows, gen, device="cpu"):
    counts = torch.randint(1, cap + 1, (tokens,), generator=gen)
    indptr = torch.zeros(tokens + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(counts, 0)
    n = tokens * cap
    win = torch.rand(n, generator=gen) < 0.3
    rows = torch.where(
        win,
        torch.randint(0, window_rows, (n,), generator=gen),
        torch.randint(0, main_rows, (n,), generator=gen),
    )
    indices = rows * 2 + win.long()
    indices[int(indptr[-1]) :] = -1   # capacity past the CSR: never read
    return indices.to(device), indptr.to(device)


def test_distinct_rows_remaps_every_named_row():
    gen = torch.Generator().manual_seed(0)
    indices, indptr = _csr(37, 24, 300, 50, gen)
    tagged, local = distinct_rows(indices, indptr, 300, 50)
    n = int(indptr[-1])
    assert torch.equal(tagged[local[:n].long()], indices[:n])
    assert tagged.unique().numel() == tagged.numel()
    assert local.dtype == torch.int32 and int(local.max()) < tagged.numel()


def test_distinct_rows_without_main_rows():
    # a window-only layer: the padding's key 0 is window row 0, whose tag must stay the window's
    indices = torch.tensor([1, 3, 1, -1, -1], dtype=torch.int64)
    tagged, local = distinct_rows(indices, torch.tensor([0, 2, 3], dtype=torch.int32), 0, 4)
    assert tagged.tolist() == [1, 3] and local.tolist() == [0, 1, 0, 0, 0]


def test_distinct_rows_with_an_empty_csr():
    indices = torch.full((8,), -1, dtype=torch.int64)
    tagged, local = distinct_rows(indices, torch.zeros(3, dtype=torch.int32), 4, 4)
    assert tagged.tolist() == [0] and local.tolist() == [0] * 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("tokens", [7, 300])
def test_the_prefill_modes_agree(tokens, monkeypatch):
    from atom.model_ops.attentions.deepseek_v41.packed_attention import packed_prefill
    from atom.model_ops.attentions.deepseek_v41.packed_rows import gather_prefix_rows

    dev, gen = "cuda", torch.Generator().manual_seed(1)
    M, W, H, D, cap = 4096, 1064, 16, 512, 64
    main = torch.randint(0, 256, (M, 256), dtype=torch.uint8, generator=gen).to(dev)
    main_scale = torch.randint(1, 0x50, (M, 32), dtype=torch.uint8, generator=gen).to(dev)
    window = torch.randint(0, 0x70, (W, 512), dtype=torch.uint8, generator=gen).to(dev)
    window_scale = torch.randint(110, 140, (W, 16), dtype=torch.uint8, generator=gen).to(dev)
    pool = ((main, main_scale), (window, window_scale))
    indices, indptr = _csr(tokens, cap, M, W, gen, dev)
    indices[int(indptr[-1]) :] = 0
    q = (torch.randn(tokens, H, D, generator=gen) * 0.5).to(dev, torch.bfloat16)
    kv = (torch.randn(tokens, D, generator=gen) * 0.5).to(dev, torch.bfloat16)
    ext = torch.cat([torch.arange(max(0, t - 3), t + 1) for t in range(tokens)]).to(dev, torch.int32)
    ext_ptr = torch.zeros(tokens + 1, dtype=torch.int32)
    ext_ptr[1:] = torch.cumsum(torch.tensor([min(t, 3) + 1 for t in range(tokens)]), 0)
    ext_ptr = ext_ptr.to(dev)
    sink = torch.randn(H, generator=gen).to(dev)
    scale = D**-0.5

    # the shadow holds the tiled path's staged values, bit for bit
    tagged, local = distinct_rows(indices, indptr, M, W)
    shadow = torch.empty(tagged.numel(), D, dtype=torch.bfloat16, device=dev)
    gather_prefix_rows(pool, tagged, torch.tensor([0, tagged.numel()], dtype=torch.int32, device=dev), shadow, 0, 1)
    staged = torch.empty(int(indptr[-1]), D, dtype=torch.bfloat16, device=dev)
    gather_prefix_rows(pool, indices, indptr, staged, 0, tokens)
    n = int(indptr[-1])
    assert torch.equal(shadow[local[:n].long()].view(torch.int16), staged.view(torch.int16))

    outs = {}
    for mode in ("tile32", "tile256", "rows"):
        monkeypatch.setenv("ATOM_DSV41_PACKED_PREFILL", mode)
        outs[mode] = packed_prefill(q, pool, indices, indptr, kv, ext, ext_ptr, sink, scale).float()
    monkeypatch.setenv("ATOM_DSV41_PACKED_PREFILL", "rows")   # (again: the plane's cached remap)
    outs["rows_cached"] = packed_prefill(q, pool, indices, indptr, kv, ext, ext_ptr, sink, scale).float()
    # the batch's pages: a CSR over 3 requests' pages and rings of the same pool (M = 512 pages x 8, W = 8 x 133)
    pidx, pptr, pages = _paged_csr(tokens, cap, M // RPP, W // RING, 3, gen, dev)
    pidx[int(pptr[-1]) :] = 0
    tagged, local = _page_rows(pidx, pptr, M, W, pages)   # (the Triton pass on the GPU)
    inv_page = torch.zeros(M // RPP, dtype=torch.int32, device=dev)
    inv_slot = torch.zeros(W // RING, dtype=torch.int32, device=dev)
    nm = sum(pages.columns) * RPP
    pg = torch.cat([pages.tables[b, :c] for b, c in enumerate(pages.columns)]).long()
    inv_page[pg] = torch.arange(pg.numel(), dtype=torch.int32, device=dev)
    inv_slot[torch.tensor(pages.slots, device=dev)] = torch.arange(3, dtype=torch.int32, device=dev)
    assert torch.equal(local, _remap_rows_torch(pidx, pptr, inv_page, inv_slot, nm, RPP, RING))
    pn = int(pptr[-1])
    assert torch.equal(tagged[local[:pn].long()], pidx[:pn])
    monkeypatch.setenv("ATOM_DSV41_PACKED_PREFILL", "tile32")
    outs["pages_ref"] = packed_prefill(q, pool, pidx, pptr, kv, ext, ext_ptr, sink, scale).float()
    monkeypatch.setenv("ATOM_DSV41_PACKED_PREFILL", "auto")
    outs["pages"] = packed_prefill(q, pool, pidx, pptr, kv, ext, ext_ptr, sink, scale, batch_pages=pages).float()
    err = (outs["pages"] - outs["pages_ref"]).abs().max().item()
    print(f"pages vs tile32: max abs {err:.3e}")
    assert err == 0 or ((outs["pages"] - outs["pages_ref"]).norm() / outs["pages_ref"].norm()).item() < 1e-2
    ref = outs["tile32"]
    for mode in ("tile256", "rows", "rows_cached"):
        err = (outs[mode] - ref).abs().max().item()
        rel = ((outs[mode] - ref).norm() / ref.norm()).item()
        print(f"{mode} vs tile32: max abs {err:.3e}, rel l2 {rel:.3e}")
        assert rel < 1e-2, (mode, err, rel)
