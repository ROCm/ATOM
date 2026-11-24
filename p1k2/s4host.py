"""p1/k2-scorer overlay host: launch p1k2.s4k2 (fused split4 scorer + top-16 variants) on an amd-kernels
IndexSplit4RankOwned's buffers with the arguments split4.host passes in its topk mode (score_topk), or as
_push_sim does on one GPU (push_sim_k2)."""
import functools

from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4.host import _end_bytes, _run_compiled


@functools.cache
def _kernel(ql: int, small: bool, ts: bool, lpt: bool, dqa: int, wr: int = 0, half_max: int = 0,
            unit_max: int = 0, lf: bool = False, lpt2: bool = False):
    from p1k2 import s4k2

    return s4k2.compile_score_push(ql, True, small, ts=ts, lpt=lpt, dqa=dqa, wr=wr, half_max=half_max,
                                   unit_max=unit_max, lf=lf, lpt2=lpt2)


def arm_opts(arm: str) -> dict:
    """arm tokens joined by '_': lpt, dqa, wr, h<N> (half-run max), u<N> (unit max); 'k2' = no change."""
    o = dict(lpt=False, dqa=0, wr=0, half_max=0, unit_max=0, lf=0, lpt2=False)
    for t in arm.split("_"):
        if t in ("lpt", "lpt2"):
            o[t] = True
        elif t == "lf":
            o["lf"] = 1
        elif t == "lf2":
            o["lf"] = 2
        elif t == "rot":
            o["lf"] = 3
        elif t == "rot2":
            o["lf"] = 4
        elif t == "wr":
            o["wr"] = 1
        elif t == "wr2":
            o["wr"] = 2
        elif t == "dqa":
            o["dqa"] = 1
        elif t == "dqb":
            o["dqa"] = 2
        elif t[:1] == "h" and t[1:].isdigit():
            o["half_max"] = int(t[1:])
        elif t[:1] == "u" and t[1:].isdigit():
            o["unit_max"] = int(t[1:])
        elif t[:1] == "g" and t[1:].isdigit():
            o["grid_over"] = int(t[1:])
    return o


def tk_width_for(max_seq_len):
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4 import decode_score as ds

    slots = 1 << ((-(-max_seq_len // (ds.BLK * 64))) - 1).bit_length()
    return slots * 64


def launch_raw(iq, index_cache, mark_ptrs, block_table, seq_lens, counts, flag_ptrs, iq_ptrs, max_rows, width,
               rows, init_blocks, local_blocks, rank, sharded, grid, topk_idx, max_seq_len, *, ql, ts=None,
               lpt=False, dqa=0, wr=0, half_max=0, unit_max=0, lf=False, lpt2=False, stream=None, **over):
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_prefill.host import nvfp4_e4m3_lut
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.host_stream import current_stream

    num_reqs = rows // ql
    grid = over.get("grid_over") or grid
    tk_width = tk_width_for(max_seq_len)
    assert tk_width <= width and grid >= 2 and min(grid - 1, rows) <= 128
    _run_compiled(
        _kernel(ql, num_reqs <= 32, ts is not None, lpt, dqa, wr, half_max, unit_max, lf, lpt2),
        iq.data_ptr(), index_cache.data_ptr(), nvfp4_e4m3_lut(iq.device).data_ptr(),
        *mark_ptrs,
        block_table.data_ptr(), seq_lens.data_ptr(), counts.data_ptr(),
        *flag_ptrs, *iq_ptrs,
        int(max_rows), int(index_cache.stride(0)), int(width), int(rows * width * 4),
        int(block_table.stride(0)), _end_bytes(block_table), int(num_reqs), int(init_blocks), int(local_blocks),
        int(rank), 2 if sharded else 0, int(grid),
        topk_idx.data_ptr(), int(topk_idx.stride(1)), tk_width,
        0 if ts is None else ts.data_ptr(),
        current_stream() if stream is None else stream,
    )


def score_topk(cp, iq, index_cache, block_table, seq_lens, init_blocks, local_blocks, width, topk_idx, max_seq_len,
               *, ts=None, sharded=False, grid=0, **opts):
    """Same contract as IndexSplit4RankOwned.score_topk (one launch, rows <= 128)."""
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4 import decode_score as ds

    rows = iq.shape[0]
    ql = cp.query_len
    num_reqs = rows // ql
    assert rows <= ds.TK_MAX_ROWS and rows == seq_lens.numel() * ql and width % 64 == 0
    if not grid:
        grid = ds.grid_for(num_reqs, ql)
    launch_raw(iq, index_cache, cp.mark_ptrs, block_table, seq_lens, cp.counts, cp.flag_ptrs, cp.iq_ptrs,
               cp.max_rows, width, rows, init_blocks, local_blocks, cp.rank, sharded, grid, topk_idx, max_seq_len,
               ql=ql, ts=ts, **opts)
    return grid
