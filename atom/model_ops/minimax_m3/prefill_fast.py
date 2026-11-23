# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Lean launches for the NVFP4 sparse prefill of MiniMax-M3 (``ATOM_M3_PREFILL_FAST=1``).

A prefill forward of a few thousand tokens is host bound: every sparse layer pays the Python of
the index scorer, the selector and the attention wrappers (validation, plans, views, cache
keys). Everything those launches need that is the same for every layer of a step (lengths,
tables, grids, compiled handles, the top-k buffer) is worked out once, at the step's first
sparse layer, into a ``StepPlan`` kept on the step's prefill metadata; a layer then only passes
its own pointers (index query, index pages, query, KV pages, output, global scales).

The index kernels give ATOM's scores and ids bit for bit: where ATOM selects with its Triton kernel
(below 1024 blocks) amd-kernels' scorer (with aiter's scale) and ids-only top-16, where it selects
with aiter's per-row selector aiter's scorer and that selector. For the attention, amd-kernels' fp16 kernels (``ak16``: bitwise
aiter's at the same tile; ``ak16lazy``: the lazy-rescale rule) or ``kernels.sparse_pa_prefill``
(``v2`` / ``v2lazy``), dispatched exactly as their host functions do. They read ATOM's NVFP4 page
layouts as they are.
"""

import math
import os

import numpy as np
import torch

ENABLED = os.environ.get("ATOM_M3_PREFILL_FAST", "0") == "1"
_LOG2E = np.float32(1.4426950409)
_BLK = 128
_TOPK = 16


def _stream() -> int:
    return torch._C._cuda_getCurrentRawStream(torch._C._cuda_getDevice())


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length() if n >= 1 else 1


_warned_unscaled = False


def _score_plan(sh, heads, batch, max_query_len, max_seq_len, device, sm_scale):
    """amd-kernels' scorer writing aiter's scores: fp32(max(dot) * fp32(sm_scale * LOG2E)), so the
    selectors see the same values (and the same ties) as ATOM's. An amd-kernels without the
    ``out_scale`` option writes max(dot): the same order, but two scores one ulp apart can round to
    one tie in aiter's and not here."""
    global _warned_unscaled
    from aiter.ops.flydsl.kernels.minimax_m3_index_score import LOG2E

    out_scale = float(np.float32(sm_scale * LOG2E))
    try:
        return sh._score_plan(heads, batch, max_query_len, max_seq_len, device,
                              out_scale=out_scale)
    except TypeError:
        if not _warned_unscaled:
            _warned_unscaled = True
            import logging

            logging.getLogger("atom").warning(
                "ATOM_M3_PREFILL_FAST: amd-kernels' index scorer has no out_scale; its scores "
                "are max(dot), not aiter's (ties can differ)")
        return sh._score_plan(heads, batch, max_query_len, max_seq_len, device)


def _aiter_index_route(heads, rows, batch, max_query_len, max_seq_len, init_blocks, local_blocks,
                       sm_scale, scores, n_valid, block_table, cu_seqlens_q, seq_lens, prefix_lens,
                       topk, device):
    """``minimax_m3_index_topk``'s NVFP4 prefill launches where it selects with aiter's per-row
    selector: aiter's scorer (``score_prefill_flydsl``, its geometry worked out here once) into
    the step's score workspace, then the selector. A callable ``(index_q, index_cache, stream)``,
    or None where ATOM selects with its Triton kernel (or aiter's selector declines)."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from atom.model_ops.minimax_m3 import index_topk as it

    max_block = -(-max_seq_len // _BLK)
    dev = device.index if device.index is not None else torch.cuda.current_device()
    if prefix_lens is None or not it._aiter_selector_wins(
            max_block, False, arch=get_gfx_runtime(), selector_rows=heads * rows,
            compute_units=it._compute_units(dev)):
        return None
    try:
        score = scores.logits(heads * rows, max_block).view(heads, rows, max_block)
    except (ValueError, RuntimeError):  # ATOM's path fails there too
        return None
    select, supported = it._small_k_selector()
    flat, flat_idx = score.view(heads * rows, max_block), topk.view(heads * rows, _TOPK)
    if not supported(flat, n_valid, flat_idx, _TOPK):
        return None

    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import minimax_m3_index_score_prefill as ap
    from aiter.ops.flydsl.kernels.minimax_m3_index_score import LOG2E
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

    from atom.model_ops.minimax_m3.nvfp4_pages import index_page_views

    geom = ap.geometry(max_query_len, batch, heads, max_block, device, xwide=True)
    ppw = geom["pages_per_wave"]
    st = score.stride()
    geom = {**geom, "contig_k": int(st[2] == 1 and st[1] % ppw == 0 and st[0] % ppw == 0
                                    and score.data_ptr() % (4 * ppw) == 0)}
    q_tiles = -(-max_query_len // geom["tile_q"])
    cb = ap.chunk_blocks(max_block, q_tiles, batch, heads, device, geom["waves"])
    cb = min(cb, max(1, max_block))
    cb = -(-cb // ppw) * ppw
    chunks = -(-max_block // cb)
    chunks = -(-chunks // ap.XCD_COUNT) * ap.XCD_COUNT
    launch = ap._get(geom, device, False, True, "e4m3")
    mid = (ptr_arg(score, fx.Float32), ptr_arg(block_table, fx.Int32),
           ptr_arg(cu_seqlens_q, fx.Int32), ptr_arg(seq_lens, fx.Int32),
           ptr_arg(prefix_lens, fx.Int32), heads, cb)
    tail = (*st, block_table.stride(0), float(sm_scale * LOG2E), q_tiles, batch * heads, chunks)
    sel_args = (flat, n_valid, flat_idx, _TOPK, init_blocks, local_blocks)

    def run(index_q, index_cache, stream):
        k_data, k_scale = index_page_views(index_cache)
        _run_compiled(launch, ptr_arg(index_q, fx.Float8E4M3FN), ptr_arg(k_data, fx.Uint8),
                      ptr_arg(k_scale, fx.Uint8), *mid, index_q.stride(0), index_q.stride(1),
                      k_data.stride(0), k_scale.stride(0), *tail, stream)
        select(*sel_args)

    run.score = score
    return run


class StepPlan:
    def __init__(self, *, attn: str, rows: int, num_kv_heads: int, num_heads: int, cu_seqlens_q,
                 seq_lens, block_table, n_valid, row_req_id, row_kv_lens, max_query_len,
                 max_seq_len, init_blocks, local_blocks, sm_scale, scores, device,
                 prefix_lens=None):
        from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import score_width
        from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_prefill import host as sh
        from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import host as th
        from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import prefill_kernel as tpk

        heads = num_kv_heads  # M3: one index head per KV head
        batch = cu_seqlens_q.shape[0] - 1
        self.rows, self.hkv = rows, num_kv_heads
        self.topk = torch.empty((heads, rows, _TOPK), dtype=torch.int32, device=device)

        # ---- index. Past 1024 blocks ATOM scores with aiter and selects with aiter's per-row
        # selector (`index_topk._launch_select`; its tie at the cut keeps the lower lane, not the
        # larger block): there this runs those two, faster than amd-kernels' scorer at most
        # shapes. Below, amd-kernels' scorer (aiter's scores) and ids-only top-16 (ATOM's Triton
        # selector's ties).
        self._aiter_route = _aiter_index_route(
            heads, rows, batch, max_query_len, max_seq_len, init_blocks, local_blocks, sm_scale,
            scores, n_valid, block_table, cu_seqlens_q, seq_lens, prefix_lens, self.topk, device)
        if self._aiter_route is not None:
            self._score = self._aiter_route.score
        else:
            self._amdk_index(sh, th, tpk, score_width, heads, rows, batch, max_query_len,
                             max_seq_len, init_blocks, local_blocks, sm_scale, scores, n_valid,
                             block_table, cu_seqlens_q, seq_lens, device)

        # ---- attention
        self._init_attention(attn, rows, num_kv_heads, num_heads, block_table, row_req_id,
                             row_kv_lens, sm_scale, device)

    def _amdk_index(self, sh, th, tpk, score_width, heads, rows, batch, max_query_len,
                    max_seq_len, init_blocks, local_blocks, sm_scale, scores, n_valid,
                    block_table, cu_seqlens_q, seq_lens, device):
        # scores (score_prefill.host.nvfp4_index_score_prefill), scaled as aiter's
        exe, nseg, qt, grid, shared = _score_plan(sh, heads, batch, max_query_len, max_seq_len,
                                                  device, sm_scale)
        width = score_width(max_seq_len)  # a power-of-two multiple of 64 blocks
        try:
            score = scores.logits(heads * rows, width).view(heads, rows, width)
        except (ValueError, RuntimeError):  # wider than the workspace (an unusual max_model_len)
            score = torch.empty((heads, rows, width), dtype=torch.float32, device=device)
        self._score = score
        lut = () if shared else (sh.nvfp4_e4m3_lut(device).data_ptr(),)
        bt_bytes = ((batch - 1) * block_table.stride(0) + block_table.shape[1]) * 4
        self._run_score = sh._run_compiled
        self._score_exe = exe
        self._score_mid = (*lut, score.data_ptr(), block_table.data_ptr(),
                           cu_seqlens_q.data_ptr(), seq_lens.data_ptr())
        self._score_tail = (int(block_table.stride(0)), int(score.stride(1)), int(nseg), int(qt),
                            int(rows), int(batch), int(bt_bytes), int(init_blocks),
                            int(local_blocks), int(score.stride(0)), int(grid))

        # ---- ids-only top-16 (topk.host.index_topk_prefill_indices)
        slots = _pow2_ceil((max_seq_len + _BLK * 64 - 1) // (_BLK * 64))
        assert slots <= th.MAX_SLOTS and width >= slots * 64
        tiles = tpk.PREFILL_STRIPS
        strips = next((s for s in tiles if max_seq_len <= s * _BLK * 64), tiles[-1])
        self._run_topk = th._run_compiled
        self._topk_exe = th._get_prefill_indices_kernel(strips)
        self._topk_args = (score.data_ptr(), score.stride(0), score.stride(1), slots * 64,
                           n_valid.data_ptr(), 1, self.topk.data_ptr(), self.topk.stride(0),
                           self.topk.stride(1), rows, heads)

    def _init_attention(self, attn, rows, num_kv_heads, num_heads, block_table, row_req_id,
                        row_kv_lens, sm_scale, device):
        self.attn = attn
        self._c = float(np.float32(sm_scale) * _LOG2E)
        if attn in ("ak16", "ak16lazy"):
            from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import host as ah

            lazy = attn == "ak16lazy"
            cus = ah._num_cus(device)
            n_wg = (rows + 3) // 4 * num_kv_heads
            rounds = 2 if n_wg > cus else 1
            if lazy:
                # amd-kernels' own dispatch (host.nvfp4_sparse_prefill)
                local = num_kv_heads == 1 and rows <= 1536
                tq = 1 if rows <= 256 else 2 if rows <= 1025 else 3
            else:
                # aiter's tile choice (sparse_pa_nvfp4_prefill.pick_tile): the same rows per
                # workgroup give the same page order, so the output is aiter's bit for bit
                if rows * num_kv_heads < 2 * cus:
                    tq = 1
                elif -(-rows // 4) * num_kv_heads < 2 * cus:
                    tq = 2
                else:
                    tq, rounds = 4, 2
                local = num_kv_heads == 1 and tq < 4
            lazy_local = {"lazy": True} if lazy and rows > 256 else {}
            lazy_grid = {"lazy": True} if lazy and rounds == 2 else {}
            shared_local = {"q_shared": True} if 512 < rows <= 1025 and tq == 2 else {}
            if local:
                self._attn_exe = ah._get_bf16_local(tq, False, False, True, False, **lazy_local,
                                                    **shared_local)
            else:
                self._attn_exe = ah._get_kernel("fp16", False, rounds, False, q_f16=False,
                                                sched=False, q_bf16=False, single=0, **lazy_grid)
            self._run_attn = ah._run_compiled
            self._abs_pos = (row_kv_lens - 1).contiguous()
            self._attn_mid = (self.topk.data_ptr(), block_table.data_ptr(), row_req_id.data_ptr(),
                              self._abs_pos.data_ptr())
            self._attn_tail = (rows, num_heads, num_kv_heads, self.topk.stride(0),
                               int(block_table.stride(0)))
        elif attn in ("v2", "v2lazy"):
            from atom.model_ops.minimax_m3.kernels import sparse_pa_prefill as v2

            self._attn_exe = v2._kernel(2, attn == "v2lazy", True, 2, 4, device.index)
            self._run_attn = v2._run
            self._attn_mid = (self.topk.data_ptr(), block_table.data_ptr(), row_req_id.data_ptr(),
                              row_kv_lens.data_ptr())
            self._attn_tail = (rows, num_heads, num_kv_heads, self.topk.stride(0),
                               self.topk.stride(1), int(block_table.stride(0)))
        else:
            raise ValueError(f"the fast prefill path has no attention {attn!r}")

    def index(self, index_q, index_cache) -> torch.Tensor:
        """The layer's top-16 block ids ``[Hkv, rows, 16]`` (one buffer for every layer)."""
        s = _stream()
        if self._aiter_route is not None:
            self._aiter_route(index_q, index_cache, s)
            return self.topk
        self._run_score(self._score_exe, index_q.data_ptr(), index_cache.data_ptr(),
                        *self._score_mid, int(index_cache.stride(0)), *self._score_tail, s)
        self._run_topk(self._topk_exe, *self._topk_args, s)
        return self.topk

    def attention(self, q, kv_cache, out, k_scale, v_scale) -> None:
        s = _stream()
        if self.attn.startswith("ak16"):
            self._run_attn(self._attn_exe, q.data_ptr(), kv_cache.data_ptr(), *self._attn_mid,
                           out.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(), 0,
                           *self._attn_tail, int(kv_cache.stride(0)), self._c, s)
        else:
            self._run_attn(self._attn_exe, q.data_ptr(), kv_cache.data_ptr(), *self._attn_mid,
                           out.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
                           *self._attn_tail, int(kv_cache.stride(0)), self._c, s)


def step_plan(prefill_md, **kw) -> StepPlan:
    """The step's plan, built at its first sparse layer."""
    plan = getattr(prefill_md, "_m3_fast_plan", None)
    if plan is None:
        plan = StepPlan(**kw)
        prefill_md._m3_fast_plan = plan
    return plan
