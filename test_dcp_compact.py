# SPDX-License-Identifier: MIT
"""Unit test for the compacting DCP sparse filter (DCP/DCP_Sparse_MLA.md ch.5).

Checks, for every rank:
  1. the compacted region equals a plain-Python reference (values AND order),
  2. the per-request lengths written into out_kv_indptr are correct,
  3. the partition property the cp_lse_ag_out_rs merge depends on: the ranks'
     kept sets are disjoint and their union is exactly the global top-k,
  4. the zero-owned edge case (a rank that owns nothing for a request).

Run inside the container:
    python test_dcp_compact.py
"""

import torch

from atom.model_ops.attention_mla import triton_filter_and_convert_dcp_index

DEV = "cuda"
K = 256  # NUM_TOPK_TOKENS (must be a multiple of BLOCK_N=128)
BLOCK_SIZE = 16  # runner physical block size
W = 4  # dcp world size


def build_case(g_ctxs, max_blocks, seed):
    """Random global top-k selections + block table for the given contexts."""
    gen = torch.Generator().manual_seed(seed)
    bs = len(g_ctxs)

    qo_indptr = torch.arange(bs + 1, dtype=torch.int32)
    global_kv_indptr = torch.zeros(bs + 1, dtype=torch.int32)
    global_kv_indptr[1:] = torch.cumsum(torch.tensor(g_ctxs), 0).to(torch.int32)

    # Physical blocks are deliberately shuffled so a wrong slot formula cannot
    # accidentally match a "logical == physical" identity mapping.
    block_table = (
        torch.randperm(bs * max_blocks, generator=gen)[: bs * max_blocks]
        .reshape(bs, max_blocks)
        .to(torch.int32)
    )

    token_indices = torch.full((bs, K), -1, dtype=torch.int32)
    for b, g in enumerate(g_ctxs):
        n = min(g, K)
        # distinct global positions in [0, g), in the indexer's (arbitrary) order
        picks = torch.randperm(g, generator=gen)[:n]
        token_indices[b, :n] = picks.to(torch.int32)
    return qo_indptr, global_kv_indptr, block_table, token_indices


def reference(g_ctxs, block_table, token_indices, rank):
    """Plain-Python expected compacted slots per request."""
    out = []
    for b, g in enumerate(g_ctxs):
        n = min(g, K)
        slots = []
        for c in range(n):
            tok = int(token_indices[b, c])
            if tok < 0 or tok % W != rank:
                continue
            vbs = BLOCK_SIZE * W
            slots.append(
                int(block_table[b, tok // vbs]) * BLOCK_SIZE + (tok % vbs) // W
            )
        out.append(slots)
    return out


def run_case(name, g_ctxs, seed=0):
    bs = len(g_ctxs)
    max_blocks = max(1, (max(g_ctxs) + BLOCK_SIZE * W - 1) // (BLOCK_SIZE * W)) + 1
    qo_indptr, global_kv_indptr, block_table, token_indices = build_case(
        g_ctxs, max_blocks, seed
    )

    qo_g = qo_indptr.to(DEV)
    gkv_g = global_kv_indptr.to(DEV)
    bt_g = block_table.to(DEV)
    ti_g = token_indices.to(DEV)

    per_rank_sets = []
    for rank in range(W):
        out_buf = torch.full((bs * K,), -999, dtype=torch.int32, device=DEV)
        out_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=DEV)
        counts = torch.zeros(bs, dtype=torch.int32, device=DEV)

        triton_filter_and_convert_dcp_index(
            qo_g,
            gkv_g,
            bt_g,
            ti_g,
            rank,
            W,
            BLOCK_SIZE,
            out_kv_indptr=out_indptr,
            owned_counts=counts,
            NUM_TOPK_TOKENS=K,
            out=out_buf,
        )
        torch.cuda.synchronize()

        exp = reference(g_ctxs, block_table, token_indices, rank)
        indptr = out_indptr.cpu().tolist()

        # (2) lengths
        for b in range(bs):
            got_len = indptr[b + 1] - indptr[b]
            assert got_len == len(
                exp[b]
            ), f"[{name}] rank{rank} req{b}: length {got_len} != {len(exp[b])}"
        # (1) values and order
        for b in range(bs):
            got = out_buf[indptr[b] : indptr[b + 1]].cpu().tolist()
            assert got == exp[b], f"[{name}] rank{rank} req{b}: {got} != {exp[b]}"

        # no -1 holes anywhere in the written region
        written = out_buf[: indptr[bs]]
        assert int((written < 0).sum()) == 0, f"[{name}] rank{rank}: hole in region"

        per_rank_sets.append([indptr[b + 1] - indptr[b] for b in range(bs)])

    # (3) partition: every valid top-k token is claimed by exactly one rank.
    # Checked on COUNTS, not on slot values -- slots are per-rank local addresses
    # (each rank holds its own 1/W KV shard), so equal slot numbers across ranks
    # are expected and carry no information. Counts summing to n rules out both
    # dropped and double-claimed tokens, which is what cp_lse_ag_out_rs needs.
    for b, g in enumerate(g_ctxs):
        n = min(g, K)
        total = sum(per_rank_sets[rank][b] for rank in range(W))
        assert total == n, f"[{name}] req{b}: kept {total} of {n} top-k tokens"

    print(f"  PASS {name}: bs={bs} g_ctxs={g_ctxs}")


def main():
    print("compacting DCP filter unit test (W=%d, K=%d, block=%d)" % (W, K, BLOCK_SIZE))
    run_case("short ctx (< topk)", [13], seed=1)
    run_case("multi-request mixed", [13, 100, 7, 300], seed=2)
    run_case("ctx > topk (clipped)", [1000, 4096], seed=3)
    run_case("page boundary", [BLOCK_SIZE * W, BLOCK_SIZE * W + 1], seed=4)
    # (4) zero-owned edge case: ctx=2 with W=4 leaves ranks 2,3 with nothing.
    run_case("zero-owned ranks", [2, 1], seed=5)
    print("ALL PASS")


if __name__ == "__main__":
    main()
