# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP4 selection parity of the MiniMax-M3 indexer-CP decode on an FP4 index cache.

Not a pytest module (it needs four ranks and the real collective):

    torchrun --nproc_per_node=4 tests/model_ops/run_indexer_cp_fp4_tp4.py

Each rank runs, on the same replicated cache, block table and index-q:

* CP OFF  -- what a TP rank does today: its own head through
  ``minimax_m3_index_topk_decode`` (FP4 FlyDSL scorer, S x 1 head, whole context);
* CP ON   -- ``SparseMHAPagedAttentionImpl._cp_index_topk_decode``: all four heads over
  this rank's blocks, local top-k, ``exchange_candidates`` (the real aiter all-gather
  or all-to-all), ``merge_candidate_keys``;

and compares the selection of every row. Both go through ``make_sparse_decode_metadata``
with records from ``allocate_index_scorers``, so the work maps are the production ones.
Tie rule under test: equal scores resolve to the higher block id (``_pack_score_key``);
the comparison is exact (``torch.equal``) on the block ids AND their order, and, when
the KV layout wants it, on ``sparse_bt`` / ``sparse_ctx``.

Prints one ``RES {json}`` line per case on rank 0 and exits non-zero on any mismatch.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

TOPK, INIT, LOCAL, BLOCK = 16, 1, 2, 128
DTYPES = {"nvfp4": 9216, "fp4": 8704}


def _fill_random(dtype, pages, gen):
    """Pages of random e2m1 data and in-range scales (any data nibble is a valid code)."""
    page_bytes = DTYPES[dtype]
    buf = torch.randint(0, 256, (pages, page_bytes), device="cuda", generator=gen).to(
        torch.uint8
    )
    scales = slice(BLOCK * 64, page_bytes)
    lo, hi = (0x28, 0x50) if dtype == "nvfp4" else (120, 134)  # e4m3 / e8m0 bytes
    buf[:, scales] = torch.randint(
        lo, hi, (pages, page_bytes - BLOCK * 64), device="cuda", generator=gen
    ).to(torch.uint8)
    return buf


def _fill_written(pages, gen):
    """NVFP4 index pages as the production writer (aiter's fused norm / RoPE / NVFP4 writer)
    makes them: random qkv rows, norm weights that carry a random power of two per 16-wide
    group (a spread of e4m3 scales). The K/V cache the op also fills is a scratch."""
    import importlib.util

    from atom.model_ops.minimax_m3.nvfp4_pages import kv_page_bytes

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "nvfp4_read_tests", os.path.join(here, "..", "test_minimax_m3_nvfp4_read.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n = pages.shape[0] * BLOCK
    kv = torch.zeros(
        (pages.shape[0], kv_page_bytes(1)), dtype=torch.uint8, device="cuda"
    )
    one = torch.ones(1, device="cuda")
    slots = torch.arange(n, device="cuda", dtype=torch.int64)
    mod.write_pages(kv, pages, slots, 1, 1, one, one, gen)
    return pages


def make_case(dtype, batch, ctx, q_len, kind, seed):
    from aiter import dtypes

    gen = torch.Generator(device="cuda").manual_seed(seed)  # same on every rank
    blocks = -(-ctx // BLOCK)
    pages = batch * blocks
    if kind == "written" and dtype == "nvfp4":
        cache = _fill_written(
            torch.empty((pages, 9216), dtype=torch.uint8, device="cuda"), gen
        )
    else:
        cache = _fill_random(dtype, pages, gen)
    q = (torch.randn(batch * q_len, 4, 128, device="cuda", generator=gen) * 2).to(
        dtypes.fp8
    )
    table = torch.randperm(pages, device="cuda", generator=gen).to(torch.int32)
    table = table.view(batch, blocks).contiguous()
    if kind == "uniform":
        lens = torch.full((batch,), ctx, dtype=torch.int32, device="cuda")
    else:  # ragged: short ones leave high shards empty, one request at the full window
        lens = torch.randint(q_len, ctx + 1, (batch,), device="cuda", generator=gen)
        lens = lens.to(torch.int32)
        lens[0], lens[-1] = q_len, ctx
    return q, cache, table, lens, blocks


def scorers(dtype, ctx, batch, q_len, rank, world):
    """The step's decode metadata records: TP (own head) and CP (this shard)."""
    from atom.model_ops.minimax_m3.index_buffers import allocate_index_scorers

    cfg = SimpleNamespace(
        index_cache_dtype=dtype,
        max_model_len=ctx,
        max_num_batched_tokens=max(batch * q_len, 4096),
        enable_tbo=False,
        enable_tbo_decode=False,
    )
    tp = allocate_index_scorers(
        cfg, "cuda", max_bs=batch, max_qlen=q_len, num_idx_heads=1
    )[""]
    cp = allocate_index_scorers(
        cfg,
        "cuda",
        max_bs=batch,
        max_qlen=q_len,
        num_idx_heads=1,
        cp_world=world,
        cp_rank=rank,
    )[""]
    return tp, cp


def run_case(dtype, batch, ctx, q_len, kind, seed, rank, world):
    from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode
    from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

    q, cache, table, lens, _ = make_case(dtype, batch, ctx, q_len, kind, seed)
    tp_rec, cp_rec = scorers(dtype, ctx, batch, q_len, rank, world)
    # the page table the KV layout wants: NVFP4 KV emits none, MXFP4 KV one 128-page a
    # block (the impl's `_select_page_kwargs`)
    kv_page = {} if dtype == "nvfp4" else {"pages_per_block": 1, "block_page_stride": 1}
    slots = torch.zeros(batch * q_len, dtype=torch.int64, device="cuda")

    def meta(rec):
        return make_sparse_decode_metadata(
            seq_lens=lens,
            block_table=table,
            slot_mapping=slots,
            max_seq_len=ctx,
            num_idx_heads=1,
            max_query_len=q_len,
            index_scorer=rec,
        )

    md_tp, md_cp = meta(tp_rec), meta(cp_rec)
    assert md_cp.index_scorer.cp_world == world and md_tp.index_scorer.cp_world == 1
    off = minimax_m3_index_topk_decode(
        q[:, rank : rank + 1],
        cache,
        table,
        lens,
        ctx,
        TOPK,
        INIT,
        LOCAL,
        1,
        128**-0.5,
        max_query_len=q_len,
        n_valid_column_per_row=md_tp.n_valid_column_per_row,
        index_scorer=md_tp.index_scorer,
        emit_sparse_block_table=dtype == "fp4",
        **kv_page,
    )
    me = SimpleNamespace(
        index_cache=cache,
        index_cache_dtype=dtype,
        scale=128**-0.5,
        topk=TOPK,
        init_blocks=INIT,
        local_blocks=LOCAL,
        indexer_cp_rank=rank,
        indexer_cp_world=world,
        _select_page_kwargs=kv_page,
        kv_cache_dtype=dtype,
    )
    on = SparseMHAPagedAttentionImpl._cp_index_topk_decode(
        me, q, table, lens, ctx, q_len, md_cp.index_scorer
    )
    off = off if isinstance(off, tuple) else (off,)
    on = tuple(x for x in on if x is not None)  # NVFP4 KV: the ids, no page table
    exact = [bool(torch.equal(a, b)) for a, b in zip(on, off)]
    rows = batch * q_len
    a, b = on[0].view(rows, TOPK), off[0].view(rows, TOPK)
    set_bad = int((a.sort(1).values != b.sort(1).values).any(1).sum())
    order_bad = int((a != b).any(1).sum())
    return {
        "dtype": dtype,
        "kind": kind,
        "B": batch,
        "S": q_len,
        "ctx": ctx,
        "rows": rows,
        "set_mismatch_rows": set_bad,
        "order_mismatch_rows": order_bad,
        "all_outputs_equal": all(exact) and len(on) == len(off),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="nvfp4", choices=sorted(DTYPES))
    ap.add_argument("--batches", default="4,16,32")
    ap.add_argument("--qlens", default="1,4")
    ap.add_argument("--ctxs", default="2048,32768,131072")
    ap.add_argument("--kinds", default="uniform,ragged,written")
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    from aiter.dist.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
    initialize_model_parallel(tensor_model_parallel_size=world)
    from atom.distributed.indexer_cp import exchange_candidates  # noqa: F401

    bad = 0
    seed = 100
    for kind in args.kinds.split(","):
        for batch in map(int, args.batches.split(",")):
            for q_len in map(int, args.qlens.split(",")):
                for ctx in map(int, args.ctxs.split(",")):
                    seed += 1
                    res = run_case(
                        args.dtype, batch, ctx, q_len, kind, seed, rank, world
                    )
                    t = torch.tensor(
                        [
                            res["set_mismatch_rows"],
                            res["order_mismatch_rows"],
                            int(not res["all_outputs_equal"]),
                        ],
                        device="cuda",
                    )
                    dist.all_reduce(t)
                    res.update(
                        set_mismatch_rows=int(t[0]),
                        order_mismatch_rows=int(t[1]),
                        ranks_with_unequal_outputs=int(t[2]),
                        rows_all_ranks=res["rows"] * world,
                    )
                    bad += int(t.sum())
                    if rank == 0:
                        print("RES " + json.dumps(res), flush=True)
                    torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        print("RESULT", "FAIL" if bad else "PASS", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
