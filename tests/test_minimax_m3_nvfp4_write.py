# SPDX-License-Identifier: MIT
"""NVFP4 page writer (the writer oracle) against a torch reference, byte for byte.

The reference below quantizes with the documented math (fp32 scale = e4m3(amax/6/gs),
nearest e2m1 grid value, ties to the even code) and places bytes by the page offsets
of ``nvfp4_pages``, one (token, head) row at a time, as plain torch indexing --
independent of the kernel's comparisons, atomics and index arithmetic. Compared:
whole pages, so a stray byte anywhere fails, on pages pre-filled with garbage (a
V nibble must clear its old value) and with slots scattered over pages, skipped
(-1), and sharing 8-token V octets within one launch.

Needs a GPU (Triton), like the other kernel tests; ``TRITON_INTERPRET=1`` runs the
same kernels on CPU tensors (arithmetic-level check only: not the GPU's e4m3 convert).
"""

import os

import pytest
import torch

from atom.model_ops.minimax_m3.nvfp4_pages import (
    HEAD_SIDE_BYTES,
    INDEX_PAGE_BYTES,
    head_regions,
    kv_page_bytes,
)

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEV = "cpu" if INTERPRET else "cuda"
pytestmark = pytest.mark.skipif(
    not (INTERPRET or torch.cuda.is_available()), reason="needs a GPU"
)

GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def ref_quantize(x, gs):
    """x fp32 [128] -> (codes int64 [128], scale bytes uint8 [8])."""
    g = x.view(8, 16)
    one_sixth = torch.tensor(1 / 6, dtype=torch.float32)
    gs_t = torch.tensor(gs, dtype=torch.float32)
    sf = (g.abs().amax(-1) * one_sixth * (1.0 / gs_t)).clamp(max=448)
    sf8 = sf.to(torch.float8_e4m3fn)
    sv = sf8.float()
    quant = torch.where(sv != 0, 1.0 / (sv * gs_t), torch.zeros(()))
    y = g * quant[:, None]
    dist = (y.abs()[..., None] - GRID.to(x.device)).abs()
    is_min = dist == dist.amin(-1, keepdim=True)
    ids = torch.arange(8, device=x.device)
    even = torch.where(is_min & (ids % 2 == 0), ids, 8).amin(-1)
    mag = torch.where(even < 8, even, dist.argmin(-1))
    codes = mag | (torch.signbit(y).long() << 3)
    return codes.flatten(), sf8.view(torch.uint8)


def k_scale_off(t, g):
    r = t % 32
    n = 4 * (r // 8) + r % 4
    return ((((t // 32) * 4 + g // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + g % 2


def ref_insert_kv(pages, k, v, slots, ks, vs):
    flat = pages.view(-1)  # contiguous pages in these tests
    stride = pages.stride(0)
    heads = k.shape[1]
    for side, (x, gs) in enumerate(((k, ks), (v, vs))):
        for t in range(x.shape[0]):
            if slots[t] < 0:
                continue
            blk, pos = int(slots[t]) // 128, int(slots[t]) % 128
            for h in range(heads):
                codes, sb = ref_quantize(x[t, h].float(), gs)
                db, sc = head_regions(side, h, heads)
                page = blk * stride
                for g in range(8):
                    if side == 0:
                        flat[page + sc + k_scale_off(pos, g)] = sb[g]
                    else:
                        flat[page + sc + g * 128 + pos] = sb[g]
                for d in range(128):
                    c = int(codes[d])
                    if side == 0:
                        i = page + db + pos * 64 + d // 2
                        sh = (d % 2) * 4
                    else:
                        i = page + db + ((pos // 8) * 128 + d) * 4 + (pos % 8) // 2
                        sh = (pos % 2) * 4
                    flat[i] = (int(flat[i]) & ~(0xF << sh) & 0xFF) | (c << sh)


def ref_insert_index(pages, keys, slots):
    flat = pages.view(-1)
    for t in range(keys.shape[0]):
        if slots[t] < 0:
            continue
        blk, pos = int(slots[t]) // 128, int(slots[t]) % 128
        codes, sb = ref_quantize(keys[t].float(), 1.0)
        base = blk * pages.stride(0)
        for d in range(0, 128, 2):
            flat[base + pos * 64 + d // 2] = int(codes[d]) | (int(codes[d + 1]) << 4)
        flat[base + 128 * 64 + pos * 8 : base + 128 * 64 + pos * 8 + 8] = sb


def make_rows(tokens, heads, seed):
    """Rows covering the cases: random scales, zero rows / groups, saturation, subnormal
    magnitudes, exact e2m1 midpoints (amax 6 so the scale is exactly 1) and signed zeros.
    """
    gen = torch.Generator().manual_seed(seed)
    exp = torch.randint(-12, 12, (tokens, heads, 8, 1), generator=gen)
    x = torch.randn(tokens, heads, 8, 16, generator=gen) * torch.exp2(exp.float())
    x[0, :, 0] = 0.0  # a zero group
    x[1] = 0.0  # a zero row
    # clamp: scale 448 and y up to ~1e2 > 6 saturates to code 7. (Not 1e30: there y ~ 2^90
    # and the reference's `|y| - grid` distance absorbs the grid in fp32, picks code 0,
    # where the hardware convert, and so this kernel, saturates.)
    x[2, :, 1] = 1e4 * torch.randn(heads, 16, generator=gen)
    x[3, :, 2] = 1e-35 * torch.randn(heads, 16, generator=gen)  # scale underflows to 0
    mids = torch.tensor(
        [
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            6.0,
            0.0,
            -0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
        ]
    )
    x[4] = mids  # amax 6 -> scale 1.0 -> y = x: every tie, and signed zero
    x[5] = -mids
    x[6] = mids * 448  # amax 2688 -> scale 448
    return x.flatten(-2).to(torch.bfloat16)


def scatter_slots(npages, tokens, seed, page=None, first=0):
    """Distinct slots over ``npages`` pages (a slot written twice has no defined
    result); with ``page``, the first 8 tokens fill one V octet of it from ``first``."""
    perm = torch.randperm(npages * 128, generator=torch.Generator().manual_seed(seed))
    if page is None:
        return perm[:tokens]
    octet = page * 128 + first + torch.arange(8)
    rest = perm[~torch.isin(perm, octet)][: tokens - 8]
    return torch.cat([octet, rest])


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("gs", [1.0, 0.37])
def test_kv_pages_equal_the_reference_byte_for_byte(heads, gs):
    from nvfp4_write_oracle import nvfp4_insert_kv

    dev = DEV
    tokens, npages = 40, 5
    k = make_rows(tokens, heads, 1)
    v = make_rows(tokens, heads, 2).flip(0)
    # scattered slots: 8 tokens fill one octet of page 3 (shared V words), others are
    # spread over pages, two are skipped
    slots = scatter_slots(npages, tokens, 3, page=3, first=16)
    slots[10] = slots[11] = -1
    pb = kv_page_bytes(heads)
    base = torch.randint(
        0,
        256,
        (npages, pb),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(4),
    )
    got = base.clone().to(dev)
    want = base.clone()
    ks = torch.tensor([gs], dtype=torch.float32, device=dev)
    vs = torch.tensor([gs * 0.5], dtype=torch.float32, device=dev)
    nvfp4_insert_kv(k.to(dev), v.to(dev), got, slots.to(dev), ks, vs)
    ref_insert_kv(want, k, v, slots, gs, gs * 0.5)
    assert torch.equal(got.cpu(), want)


def test_index_pages_equal_the_reference_byte_for_byte():
    from nvfp4_write_oracle import nvfp4_insert_index

    dev = DEV
    tokens, npages = 40, 4
    keys = make_rows(tokens, 1, 5)[:, 0]
    slots = scatter_slots(npages, tokens, 6)
    slots[3] = -1
    base = torch.randint(
        0,
        256,
        (npages, INDEX_PAGE_BYTES),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(7),
    )
    got, want = base.clone().to(dev), base.clone()
    nvfp4_insert_index(keys.to(dev), got, slots.to(dev))
    ref_insert_index(want, keys, slots)
    assert torch.equal(got.cpu(), want)


def test_a_page_stride_wider_than_a_page_leaves_the_gap_alone():
    from nvfp4_write_oracle import nvfp4_insert_kv

    dev, heads, npages = DEV, 1, 3
    pb = kv_page_bytes(heads)
    wide = torch.full((npages, pb + 256), 0xA5, dtype=torch.uint8, device=dev)
    pages = wide[:, :pb]
    k = make_rows(8, heads, 8).to(dev)
    one = torch.ones(1, device=dev)
    nvfp4_insert_kv(k, k, pages, torch.arange(8, device=dev) + 128, one, one)
    assert (wide[:, pb:] == 0xA5).all() and (wide[0] == 0xA5).all()
    assert not (wide[1, :pb] == 0xA5).all()
    assert HEAD_SIDE_BYTES * 2 == pb
