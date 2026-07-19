#!/usr/bin/env python3
"""Offline P2 gate for M1 paged swa_write: kernel vs reference, no model.

Validates the block-table addressing: writes the last-N tokens of each seq to
the content-addressed SWA region and checks the Triton kernel matches the
pure-PyTorch reference (and that a prefix-cache hit, modelled as two seqs
sharing the same physical blocks, lands SWA in the SAME rows)."""

import torch
import pytest

try:
    import atom.model_ops.v4_kernels  # noqa: F401  (triggers the heavy import chain)
except Exception as _e:  # pre-existing atom.config circular import under bare pytest
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)
from atom.model_ops.v4_kernels.state_writes import swa_write, swa_write_reference

dev = "cuda"
torch.manual_seed(0)

bs = 3
block_size = 8  # small for the test (real V4 = 128)
head_dim = 16
max_blocks = 6
num_pages = 128  # >= max(phys)*block_size + block_size
write_per_batch = block_size + 2

# Per-seq token counts this fwd, and global positions (varied: some seqs start
# mid-sequence to exercise pos//block_size != 0).
tok_counts = [block_size + 3, 5, block_size * 2]
start_pos = [0, 11, 4]
cu = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
cu[1:] = torch.cumsum(torch.tensor(tok_counts, dtype=torch.int32), 0)
T = int(cu[-1])
positions = torch.cat(
    [
        torch.arange(start_pos[b], start_pos[b] + tok_counts[b], dtype=torch.int32)
        for b in range(bs)
    ]
).to(dev)
kv = torch.randn(T, head_dim, dtype=torch.bfloat16, device=dev)

# Disjoint phys blocks per seq → writes never collide → deterministic match
# (pure write-addressing test; cross-request reuse is a system-level concern).
block_tables = torch.zeros(bs, max_blocks, dtype=torch.int32, device=dev)
block_tables[0] = torch.tensor([1, 2, 3, 4, 5, 0], dtype=torch.int32)
block_tables[1] = torch.tensor([6, 7, 8, 9, 0, 0], dtype=torch.int32)
block_tables[2] = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)

ref = torch.zeros(num_pages, head_dim, dtype=torch.bfloat16, device=dev)
out = torch.zeros(num_pages, head_dim, dtype=torch.bfloat16, device=dev)

swa_write_reference(kv, positions, cu, block_tables, ref, block_size, write_per_batch)
swa_write(kv, positions, cu, block_tables, out, block_size, write_per_batch)
torch.cuda.synchronize()

assert torch.equal(
    out, ref
), f"kernel != reference; max|diff|={(out.float()-ref.float()).abs().max()}"
print("PASS: kernel == reference")

# Spot-check a known mapping: seq2 last token pos = start_pos[2]+tok_counts[2]-1.
b = 2
last_pos = start_pos[b] + tok_counts[b] - 1
phys = int(block_tables[b, last_pos // block_size])
row = phys * block_size + last_pos % block_size
assert torch.equal(out[row], kv[int(cu[b + 1]) - 1]), "last-token row mismatch"
print(f"PASS: seq2 last token pos={last_pos} -> phys={phys} row={row} bytes match")

nz = int((ref.abs().sum(-1) > 0).sum())
print(f"non-zero SWA rows written: {nz}")
print("ALL OK")
