"""p1/k2-scorer: ATOM_P1_K2_SCORER=<arm> (integration branches only, default off).

Replaces the launch of amd-kernels' split4 fused scorer + top-16 (``split4.host.nvfp4_index_score_decode_rank_owned_push``
with ``topk_idx``, i.e. IndexSplit4RankOwned.score_topk up to TK_MAX_ROWS rows) by the p1k2 variant ``arm``
(p1k2.s4k2, generated from amd-kernels 3d62453 decode_score.py by bench/mk_k2.py; tokens: h<N> half-run max,
lf longest request first, wr2 batched row poll, dqb page-0 dequant under the flag poll). Same arguments, grid and
buffers; scores, ids and receive-buffer state bit-identical (k2sim / k2bench checks). The scores-only path
(topk_idx None: score_push, split4_wide) is untouched."""


def install(arm: str) -> None:
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4 import host as H

    if getattr(H, "_p1_k2_scorer", None):
        return
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4 import decode_score as ds
    from p1k2 import s4host

    orig = H.nvfp4_index_score_decode_rank_owned_push
    opts = s4host.arm_opts(arm)

    def push(iq, index_cache, dests, row_stride, dest_bytes, block_table, seq_lens, init_blocks, local_blocks, rank,
             counts, flags, iq_bufs, iq_rows, grid=0, query_len=4, sharded=True, topk_idx=None, max_seq_len=0):
        if topk_idx is None:
            return orig(iq, index_cache, dests, row_stride, dest_bytes, block_table, seq_lens, init_blocks,
                        local_blocks, rank, counts, flags, iq_bufs, iq_rows, grid=grid, query_len=query_len,
                        sharded=sharded, topk_idx=topk_idx, max_seq_len=max_seq_len)
        rows = iq.shape[0]
        assert 1 <= query_len <= ds.MAX_QL and rows % query_len == 0 and rows <= iq_rows
        assert iq.stride(1) == 1 and iq.stride(0) == ds.D
        num_reqs = rows // query_len
        assert num_reqs <= ds.MAX_REQS
        if not grid:
            grid = ds.grid_for(num_reqs, query_len)
        assert 0 < grid <= ds.SC_MAX_WG
        if num_reqs == 0:
            return
        assert dest_bytes == rows * row_stride * 4, (dest_bytes, rows, row_stride)
        assert topk_idx.dtype.itemsize == 4 and topk_idx.stride(2) == 1 and topk_idx.shape == (1, rows, 16)
        assert 0 < max_seq_len
        s4host.launch_raw(iq, index_cache, dests, block_table, seq_lens, counts, flags, iq_bufs, iq_rows, row_stride,
                          rows, init_blocks, local_blocks, rank, sharded, grid, topk_idx, max_seq_len, ql=query_len,
                          **opts)

    H.nvfp4_index_score_decode_rank_owned_push = push
    H._p1_k2_scorer = arm
