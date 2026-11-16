# SPDX-License-Identifier: MIT
"""The planar FP4 prefill paths (`packed_prefill_rows`): the distinct-row
remap on CPU; on a GPU, the decoded shadow against the tiled path's staged
rows (bit for bit) and the three modes' outputs against each other."""

import pytest

torch = pytest.importorskip("torch")

from atom.model_ops.attentions.deepseek_v41.packed_prefill_rows import (  # noqa: E402
    distinct_rows,
)


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
    ref = outs["tile32"]
    for mode in ("tile256", "rows", "rows_cached"):
        err = (outs[mode] - ref).abs().max().item()
        rel = ((outs[mode] - ref).norm() / ref.norm()).item()
        print(f"{mode} vs tile32: max abs {err:.3e}, rel l2 {rel:.3e}")
        assert rel < 1e-2, (mode, err, rel)
