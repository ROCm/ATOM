#!/usr/bin/env python3
"""Offline P2 gate for M1 paged decode-indices: kernel vs reference, no model."""

import torch
import pytest

try:
    import atom.model_ops.v4_kernels  # noqa: F401  (triggers the heavy import chain)
except Exception as _e:  # pre-existing atom.config circular import under bare pytest
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)
from atom.model_ops.v4_kernels.paged_decode_indices import (
    write_v4_paged_decode_indices,
    write_v4_paged_decode_indices_reference,
)

dev = "cuda"
torch.manual_seed(0)
win = 8
block_size = 8
bs = 3
max_blocks = 8

# One decode token per seq (T = bs); positions vary so n=min(pos+1,win) and
# windows span multiple blocks (exercises per-window-position block lookup).
positions = torch.tensor([5, 20, 13], dtype=torch.int32, device=dev)
batch_id_per_token = torch.tensor([0, 1, 2], dtype=torch.int32, device=dev)
T = bs
block_tables = torch.randint(1, 50, (bs, max_blocks), dtype=torch.int32, device=dev)

# Build ragged indptrs: SWA-only buffer (n per token), and CSA/HCA with a
# compress head section (arbitrary nonzero head sizes) + SWA tail.
n_per = torch.minimum(positions + 1, torch.full_like(positions, win)).tolist()
csa_head = [3, 0, 5]
hca_head = [1, 2, 0]


def indptr(heads):
    v = [0]
    for t in range(T):
        v.append(v[-1] + heads[t] + n_per[t])
    return torch.tensor(v, dtype=torch.int32, device=dev)


swa_indptr = indptr([0, 0, 0])
csa_indptr = indptr(csa_head)
hca_indptr = indptr(hca_head)


def run(fn):
    swa = torch.full((int(swa_indptr[-1]),), -7, dtype=torch.int32, device=dev)
    csa = torch.full((int(csa_indptr[-1]),), -7, dtype=torch.int32, device=dev)
    hca = torch.full((int(hca_indptr[-1]),), -7, dtype=torch.int32, device=dev)
    fn(
        block_tables=block_tables,
        batch_id_per_token=batch_id_per_token,
        positions=positions,
        swa_indptr=swa_indptr,
        csa_indptr=csa_indptr,
        hca_indptr=hca_indptr,
        swa_indices=swa,
        csa_indices=csa,
        hca_indices=hca,
        T=T,
        win=win,
        block_size=block_size,
    )
    return swa, csa, hca


rs, rc, rh = run(write_v4_paged_decode_indices_reference)
ks, kc, kh = run(write_v4_paged_decode_indices)
torch.cuda.synchronize()

# SWA buffer fully written by kernel → must match everywhere.
assert torch.equal(ks, rs), f"swa mismatch\nref={rs}\nker={ks}"
# CSA/HCA: only the SWA-tail of each slice is written here (heads left at -7).
assert torch.equal(kc, rc), f"csa mismatch\nref={rc}\nker={kc}"
assert torch.equal(kh, rh), f"hca mismatch\nref={rh}\nker={kh}"
print("PASS: decode-indices kernel == reference (swa/csa-tail/hca-tail)")

# Spot check: seq1 pos=20, n=win=8 → window [13..20]; first window pos 13 →
# blk 13//8=1, phys=block_tables[1,1]; paged=phys*8 + 13%8.
phys = int(block_tables[1, 13 // block_size])
exp = phys * block_size + 13 % block_size
got = int(rs[int(swa_indptr[1])])  # seq1 SWA slice start (swa head=0)
assert got == exp, f"spot check {got} != {exp}"
print(f"PASS: seq1 window-start pos=13 -> phys={phys} paged={exp}")


# ---------------------------------------------------------------------------
# HCA compress paged offsets (decode path) at k2_hca > 1 — regression for the
# HCA paged-gather bug. `_attach_v4_paged_decode_meta` builds the HCA compress
# section with `hca_compress_paged_offsets`; with V4 block_size=256, ratio=128
# each physical block packs k2_hca=2 HCA entries, so entry e -> block
# block_tables[bid, e//k2] slot e%k2 -> swa_pages + phys*k2 + slot. The pre-fix
# math used swa_pages + block_tables[bid, e] (k2==1), reading wrong blocks.
import numpy as np  # noqa: E402
from atom.model_ops.v4_kernels import hca_compress_paged_offsets  # noqa: E402

k2 = 2
swa_pages_h = 10_000
bt_np = np.array([[5, 9, 13, 17], [2, 6, 10, 14]], dtype=np.int32)  # [bs, blocks]
entry_idx = np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int64)  # seq0:4 entries, seq1:3
bid_per_entry = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
got_h = hca_compress_paged_offsets(entry_idx, bid_per_entry, bt_np, swa_pages_h, k2)
exp_h = np.array(
    [swa_pages_h + int(bt_np[b][e // k2]) * k2 + e % k2
     for e, b in zip(entry_idx.tolist(), bid_per_entry.tolist())],
    dtype=np.int32,
)
assert np.array_equal(got_h, exp_h), (
    f"k2={k2} decode HCA compress offset wrong (the HCA paged-gather bug)\n"
    f"got={got_h.tolist()}\nexp={exp_h.tolist()}"
)
# k2==1 must reduce to the legacy swa_pages + block_tables[bid, e]
got_1 = hca_compress_paged_offsets(entry_idx, bid_per_entry, bt_np, swa_pages_h, 1)
exp_1 = np.array(
    [swa_pages_h + int(bt_np[b][e]) for e, b in zip(entry_idx.tolist(), bid_per_entry.tolist())],
    dtype=np.int32,
)
assert np.array_equal(got_1, exp_1), "k2==1 must equal legacy swa_pages + bt"
print(f"PASS: k2_hca={k2} decode HCA compress offsets = {got_h.tolist()} (block-packed)")
print("ALL OK")
