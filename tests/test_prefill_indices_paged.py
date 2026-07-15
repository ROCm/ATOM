#!/usr/bin/env python3
"""Offline P2 gate for M1 paged prefill-indices: kernel vs reference, no model.
Exercises prefix_swa_count>0 (the cross-request prefix-hit boundary path)."""

import numpy as np
import torch
import pytest

try:
    import atom.model_ops.v4_kernels  # noqa: F401  (triggers the heavy import chain)
except Exception as _e:  # pre-existing atom.config circular import under bare pytest
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)
from atom.model_ops.v4_kernels.paged_prefill_indices import (
    write_v4_paged_prefill_indices,
    write_v4_paged_prefill_indices_reference,
)

dev = "cuda"
torch.manual_seed(0)
win = 8
block_size = 8
hca_ratio = 8
max_blocks = 8

# seq0: fresh chunk (chunk_start=0) pos 0..4 -> prefix_swa_count=0 (extend only)
# seq1: prefix-cache hit, chunk_start=16, recompute pos 16..19 -> prefix_swa_count>0
bid_per_token = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=dev)
positions = torch.tensor([0, 1, 2, 3, 4, 16, 17, 18, 19], dtype=torch.int32, device=dev)
chunk_start = torch.tensor([0, 16], dtype=torch.int32, device=dev)
cu_q = torch.tensor([0, 5], dtype=torch.int32, device=dev)  # per-seq kv start
n_hca_per_seq = torch.tensor([0, 2], dtype=torch.int32, device=dev)
state_slot = torch.tensor([0, 1], dtype=torch.int32, device=dev)  # unused now
block_tables = torch.randint(1, 40, (2, max_blocks), dtype=torch.int32, device=dev)
swa_block_tables = torch.randint(1, 40, (2, max_blocks), dtype=torch.int32, device=dev)
T = positions.shape[0]

# Per-token counts (mirror _build_paged_prefill_meta formulas).
pos = positions.cpu().numpy().astype(np.int64)
bid = bid_per_token.cpu().numpy()
cs_pt = chunk_start.cpu().numpy()[bid]
tpic = pos - cs_pt
swa_low = np.maximum(pos - win + 1, 0)
extend_count = np.minimum(tpic + 1, win)
prefix_swa_count = np.maximum(cs_pt - swa_low, 0)
n_hca = np.minimum((pos + 1) // hca_ratio, n_hca_per_seq.cpu().numpy()[bid])
csa_head = np.array(
    [2, 1, 0, 3, 2, 1, 0, 2, 1]
)  # arbitrary CSA topk head sizes (not written here)


def indptr(counts):
    v = np.zeros(T + 1, np.int64)
    v[1:] = np.cumsum(counts)
    return torch.tensor(v, dtype=torch.int32, device=dev)


ext_indptr = indptr(extend_count)
swa_indptr = indptr(prefix_swa_count)
csa_indptr = indptr(prefix_swa_count + csa_head)
hca_indptr = indptr(prefix_swa_count + n_hca)


def run(fn):
    ext = torch.full((int(ext_indptr[-1]),), -9, dtype=torch.int32, device=dev)
    swa = torch.full((int(swa_indptr[-1]),), -9, dtype=torch.int32, device=dev)
    csa = torch.full((int(csa_indptr[-1]),), -9, dtype=torch.int32, device=dev)
    hca = torch.full((int(hca_indptr[-1]),), -9, dtype=torch.int32, device=dev)
    fn(
        positions=positions,
        bid_per_token=bid_per_token,
        chunk_start_per_seq=chunk_start,
        cu_seqlens_q_per_seq=cu_q,
        state_slot_per_seq=state_slot,
        n_committed_hca_per_seq=n_hca_per_seq,
        block_tables=block_tables,
        swa_block_tables=swa_block_tables,
        extend_indptr=ext_indptr,
        prefix_swa_indptr=swa_indptr,
        prefix_csa_indptr=csa_indptr,
        prefix_hca_indptr=hca_indptr,
        extend_indices=ext,
        prefix_swa_indices=swa,
        prefix_csa_indices=csa,
        prefix_hca_indices=hca,
        T=T,
        win=win,
        block_size=block_size,
        swa_pages=10_000,
        hca_ratio=hca_ratio,
    )
    return ext, swa, csa, hca


re_, rs, rc, rh = run(write_v4_paged_prefill_indices_reference)
ke, ks, kc, kh = run(write_v4_paged_prefill_indices)
torch.cuda.synchronize()

assert torch.equal(ke, re_), f"extend mismatch\nref={re_}\nker={ke}"
assert torch.equal(ks, rs), f"swa mismatch\nref={rs}\nker={ks}"
assert torch.equal(kc, rc), f"csa mismatch\nref={rc}\nker={kc}"
assert torch.equal(kh, rh), f"hca mismatch\nref={rh}\nker={kh}"
print(
    f"PASS: prefill-indices kernel == reference (prefix_swa_count={prefix_swa_count.tolist()})"
)

# Spot: seq1 first token pos=16, global_pos starts swa_low=9 -> blk 9//8=1,
# phys=block_tables[1,1]; paged=phys*8 + 9%8.
phys = int(swa_block_tables[1, 9 // block_size])
exp = phys * block_size + 9 % block_size
got = int(rs[int(swa_indptr[5])])  # token t=5 (seq1 first) swa slice start
assert got == exp, f"spot {got} != {exp}"
print(f"PASS: seq1 prefix-swa start global_pos=9 -> phys={phys} paged={exp}")


# ---------------------------------------------------------------------------
# k2_hca > 1 coverage (regression for the HCA paged-gather bug). With V4
# block_size=256, ratio=128 the compressor packs k2_hca = block_size // ratio = 2
# HCA compress entries per physical block (cache view [num_blocks, k2_hca, D]), so
# entry e -> physical block block_tables[bid, e // k2] at slot e % k2 -> unified
# row swa_pages + phys * k2 + slot. The pre-fix gather emitted swa_pages +
# block_tables[bid, e] (assumed k2==1), silently reading wrong physical blocks for
# e >= (blocks_per_seq). block_size=16, ratio=8 -> k2=2 mirrors 256 // 128.
bs2, hr2 = 16, 8
k2 = bs2 // hr2
sp2 = 10_000
# one seq, token at pos=31 -> n_hca = min(32 // 8, 4) = 4 entries spanning 2 blocks
pos2 = torch.tensor([31], dtype=torch.int32, device=dev)
bidt2 = torch.tensor([0], dtype=torch.int32, device=dev)
cstart2 = torch.tensor([0], dtype=torch.int32, device=dev)  # fresh -> no SWA prefix
cuq2 = torch.tensor([0], dtype=torch.int32, device=dev)
nhca2 = torch.tensor([4], dtype=torch.int32, device=dev)
sslot2 = torch.tensor([0], dtype=torch.int32, device=dev)
bt2 = torch.tensor([[5, 9, 13, 17, 21, 25, 29, 33]], dtype=torch.int32, device=dev)
sbt2 = torch.tensor([[2, 3, 4, 6, 7, 8, 10, 11]], dtype=torch.int32, device=dev)
T2 = 1
p2 = pos2.cpu().numpy().astype(np.int64)
psc2 = np.maximum(0 - np.maximum(p2 - win + 1, 0), 0)  # 0 (chunk_start=0)
nh2 = np.minimum((p2 + 1) // hr2, nhca2.cpu().numpy())
ec2 = np.minimum(p2 + 1, win)
assert int(psc2[0]) == 0 and int(nh2[0]) == 4


def _idp2(counts):
    v = np.zeros(T2 + 1, np.int64)
    v[1:] = np.cumsum(counts)
    return torch.tensor(v, dtype=torch.int32, device=dev)


ext_i2, swa_i2 = _idp2(ec2), _idp2(psc2)
csa_i2, hca_i2 = _idp2(psc2), _idp2(psc2 + nh2)


def _run2(fn):
    e = torch.full((max(int(ext_i2[-1]), 1),), -9, dtype=torch.int32, device=dev)
    s = torch.full((max(int(swa_i2[-1]), 1),), -9, dtype=torch.int32, device=dev)
    c = torch.full((max(int(csa_i2[-1]), 1),), -9, dtype=torch.int32, device=dev)
    h = torch.full((int(hca_i2[-1]),), -9, dtype=torch.int32, device=dev)
    fn(
        positions=pos2, bid_per_token=bidt2, chunk_start_per_seq=cstart2,
        cu_seqlens_q_per_seq=cuq2, state_slot_per_seq=sslot2,
        n_committed_hca_per_seq=nhca2, block_tables=bt2, swa_block_tables=sbt2,
        extend_indptr=ext_i2, prefix_swa_indptr=swa_i2, prefix_csa_indptr=csa_i2,
        prefix_hca_indptr=hca_i2, extend_indices=e, prefix_swa_indices=s,
        prefix_csa_indices=c, prefix_hca_indices=h, T=T2, win=win,
        block_size=bs2, swa_pages=sp2, hca_ratio=hr2, k2_hca=k2,
    )
    return h


h_ref2 = _run2(write_v4_paged_prefill_indices_reference)
h_ker2 = _run2(write_v4_paged_prefill_indices)
torch.cuda.synchronize()
assert torch.equal(h_ker2, h_ref2), f"k2=2 HCA kernel!=ref\nref={h_ref2}\nker={h_ker2}"
btn = bt2.cpu().numpy()[0]
# independent oracle = where the compressor writes entry e (phys*k2 + slot)
oracle2 = torch.tensor(
    [sp2 + int(btn[e // k2]) * k2 + e % k2 for e in range(4)],
    dtype=torch.int32, device=dev,
)
assert torch.equal(h_ker2, oracle2), (
    f"k2={k2} HCA compress offset wrong (the HCA paged-gather bug)\n"
    f"got={h_ker2.tolist()}\nexp={oracle2.tolist()}"
)
print(f"PASS: k2_hca={k2} HCA compress offsets = {h_ker2.tolist()} (block-packed)")
print("ALL OK")
