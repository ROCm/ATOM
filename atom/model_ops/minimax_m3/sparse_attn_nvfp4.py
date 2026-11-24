"""Block-sparse attention over the NVFP4 sparse KV cache (MiniMax-M3,
``ATOM_M3_SPARSE_KV_NVFP4=1``).

The cache is the NVFP4 layout of ``index_fp4`` per kv head:

- ``k_payload`` / ``v_payload`` ``[blocks, Hkv, 128, D // 2]`` uint8,
- ``k_scale`` / ``v_scale`` ``[blocks, Hkv, 128, D // 16]`` uint8 (E4M3).

A row is a decode query row or a prefill query token. It attends the top-k
128-blocks the indexer picked for it, addressed through its request's block
table, up to its own causal length. This module maps ATOM's batch onto its
per-row (request, causal length) arguments.

gfx950 runs the Gluon kernel of ``nvfp4_kernels.decode`` over the top-k ids.
gfx1250 runs aiter's FlyDSL paged decode, the operator the FP8 sparse path
runs there, with its NVFP4 K/V: like the FP8 path, every row (and kv head) is
a length-1 sequence with its own compacted table of selected pages, and the
plan's partitions merge in the kernel or the reducer. The top-k selectors emit
that table in the launch that selects (``emit_pages_per_block`` =
``NVFP4_PAGES_PER_BLOCK``); ``nvfp4_page_table`` builds it for the selectors
that do not. Like the FP8 path it can also write o_proj's per-token FP8 input
(``quantized_output``), split the same way. aiter decides which shapes it serves.
"""

import functools

import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx

from atom.model_ops.minimax_m3.nvfp4_kernels.nvfp4 import Nvfp4Cache
from atom.utils import envs

BLOCK = 128  # sparse block == cache page


@functools.cache
def _gluon_kernel() -> bool:
    """gfx950 runs the Gluon kernel over top-k ids (``nvfp4_kernels.decode``,
    CDNA4 MFMA and buffer loads); gfx1250 aiter's FlyDSL paged decode over
    ``nvfp4_page_table``."""
    return get_gfx() == "gfx950"


def reads_page_table() -> bool:
    """Whether the attention reads a page-128 table (gfx1250's paged decode),
    which the selectors then emit, or only the top-k ids (gfx950's Gluon kernel)."""
    return not _gluon_kernel()


def check_nvfp4_sparse_attention(q, k_payload, k_scale, v_payload, v_scale, topk_idx):
    """gfx950: raise unless the Gluon kernel serves these shapes (16 q heads per
    kv head, head dim 128, 128-token pages). The shapes are fixed per deployment,
    so one call per layer at its first forward is enough. gfx1250: nothing to
    check here; aiter's paged decode refuses what it does not serve."""
    if not _gluon_kernel():
        return
    from atom.model_ops.minimax_m3.nvfp4_kernels.decode import can_use_nvfp4_decode

    k = Nvfp4Cache(payload=k_payload, scales=k_scale)
    v = Nvfp4Cache(payload=v_payload, scales=v_scale)
    if not can_use_nvfp4_decode(q, k, v, topk_idx, BLOCK):
        raise ValueError(
            "ATOM_M3_SPARSE_KV_NVFP4: the NVFP4 attention kernel needs "
            f"16 q heads per kv head and head dim 128; got q {tuple(q.shape)}, "
            f"k payload {tuple(k_payload.shape)} (e.g. MiniMax-M3.1 at TP4)."
        )


@triton.jit
def _page_table_kernel(
    topk_ptr,  # [Hkv, rows, topk] logical block ids, -1 padded
    block_table_ptr,  # [reqs, max_blocks] physical page of each logical block
    row_req_ptr,  # [rows]
    row_seq_ptr,  # [rows] causal length
    table_ptr,  # out [rows * Hkv, topk] page ids of the (page, kv head) view
    ctx_ptr,  # out [rows * Hkv] tokens to attend
    topk,
    stride_t_h,
    stride_t_n,
    stride_bt,
    NUM_KV_HEADS: tl.constexpr,
    TOPK_P: tl.constexpr,
):
    row = tl.program_id(0)
    kvh = tl.program_id(1)
    seq = tl.load(row_seq_ptr + row)
    req = tl.load(row_req_ptr + row).to(tl.int64)
    off = tl.arange(0, TOPK_P)
    blk = tl.load(
        topk_ptr + kvh * stride_t_h + row * stride_t_n + off,
        mask=off < topk,
        other=-1,
    )
    # The block holding the row's last token; blocks past it hold no visible
    # token. A zero-length (padding) row selects nothing, whatever its list.
    self_blk = (tl.maximum(seq, 1) - 1) // 128
    valid = (blk >= 0) & (blk <= self_blk) & (seq > 0)
    is_tail = valid & (blk == self_blk)
    is_full = valid & (blk < self_blk)
    n_full = tl.sum(is_full.to(tl.int32), axis=0)
    has_tail = tl.sum(is_tail.to(tl.int32), axis=0) > 0
    # Full blocks first in selection order, the partial tail block last: the
    # paged decode masks a sequence's tokens past its length, which then fall
    # in the tail block only.
    slot = tl.where(is_full, tl.cumsum(is_full.to(tl.int32), axis=0) - 1, n_full)
    slot = tl.where(valid, slot, -1)
    page = tl.load(block_table_ptr + req * stride_bt + blk, mask=valid, other=0)
    page = page * NUM_KV_HEADS + kvh
    # Gather, not scatter: entry j is the page whose slot is j. Unused slots
    # hold page 0 (never read), so every slot is written and the table needs no
    # memset.
    hit = slot[None, :] == off[:, None]
    compact = tl.sum(tl.where(hit, page[None, :], 0), axis=1)
    out_row = row * NUM_KV_HEADS + kvh
    tl.store(table_ptr + out_row.to(tl.int64) * topk + off, compact, mask=off < topk)
    ctx = n_full * 128 + tl.where(has_tail, seq - self_blk * 128, 0)
    tl.store(ctx_ptr + out_row, ctx)


def nvfp4_page_table(
    topk_idx: torch.Tensor,  # [Hkv, rows, topk] int32 block ids, -1 right-padded
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    row_req: torch.Tensor,  # [rows]
    row_seq_len: torch.Tensor,  # [rows]
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """gfx1250: the paged decode's per-(row, kv head) compacted page table
    ``[rows * Hkv, topk]`` and lengths ``[rows * Hkv]``, page ``p * Hkv + h`` of
    the ``[pages * Hkv, 1, ...]`` cache view; the attention of every sparse layer
    sharing the top-k reads it. gfx950 reads the top-k ids: (None, None)."""
    if _gluon_kernel():
        return None, None
    num_kv_heads, rows, topk = topk_idx.shape
    table = torch.empty(
        rows * num_kv_heads, topk, dtype=torch.int32, device=topk_idx.device
    )
    ctx = torch.empty(rows * num_kv_heads, dtype=torch.int32, device=topk_idx.device)
    if rows:
        assert topk_idx.stride(-1) == 1 and block_table.stride(-1) == 1
        _page_table_kernel[(rows, num_kv_heads)](
            topk_idx,
            block_table,
            row_req,
            row_seq_len,
            table,
            ctx,
            topk,
            topk_idx.stride(0),
            topk_idx.stride(1),
            block_table.stride(0),
            NUM_KV_HEADS=num_kv_heads,
            TOPK_P=triton.next_power_of_2(topk),
        )
    return table, ctx


def fused_reduce_split(seqs: int, device) -> tuple[int, int | None, int, int]:
    """(partitions, workgroup budget, page groups, rows a cluster) of the
    decode rows when ``ATOM_M3_PA_FUSED_REDUCE`` merges the top-k window inside
    the paged attention instead of the PS reduce kernel.

    While every row's cluster stays resident (gfx1250: 15 clusters a 16 CUs) a
    row's pages split over a cluster of P CTAs that merge on chip; past that
    one CTA a row, whose four wave groups split its pages and merge in LDS.
    MI455X, C64 decode, CUDA graph, us/layer (fused vs the split path today):
    rows 32: 4.6 vs 5.1 (cluster8), 64: 5.1 vs 6.5 (cluster4, two wave groups),
    80: 6.3 vs 8.5, 128: 6.9 vs 8.6, 192: 8.5 vs 13.0 (cluster4), 256: 9.6 vs
    13.7 (one CTA a row)."""
    cus = torch.cuda.get_device_properties(device).multi_processor_count
    if seqs * 8 <= cus:
        return 8, 8 * seqs, 1, 1
    if seqs * 4 <= cus:
        return 4, 4 * seqs, 2, 1
    if seqs <= cus * 15 // 16:
        return 4, 4 * seqs, 1, 1
    return 1, None, 4, 1


@functools.cache
def _prefill_reuse_kernel():
    """aiter's K/V-reuse sparse prefill and its shape predicate
    (``ATOM_M3_SPARSE_PREFILL_REUSE``), or None on an aiter without it."""
    try:
        from aiter.ops.flydsl.pa_sparse_prefill import (
            pa_sparse_prefill_nvfp4,
            pa_sparse_prefill_nvfp4_supported,
        )
    except ImportError:
        return None
    return pa_sparse_prefill_nvfp4, pa_sparse_prefill_nvfp4_supported


@functools.cache
def _prefill_reuse_impl():
    """The reuse kernel's implementation keywords (``ATOM_M3_SPARSE_PREFILL_REUSE_IMPL``
    and ``_OPTS``); empty for the default v1."""
    impl = envs.ATOM_M3_SPARSE_PREFILL_REUSE_IMPL
    if impl == "v1":
        return {}
    opts = {}
    for item in filter(None, envs.ATOM_M3_SPARSE_PREFILL_REUSE_OPTS.split(",")):
        k, v = item.split("=")
        opts[k.strip()] = int(v)
    return {"impl": impl, "v2": opts}


def _flydsl_attention(
    q, k_payload, k_scale, v_payload, v_scale, table, ctx, sm_scale,
    plan_step_owner, plan_cache_owner,
    quantized_output=None, output_scale=None, quantized_direct=True,
    rows_per_request=0,
):  # fmt: skip
    """aiter's FlyDSL paged decode over the NVFP4 cache, one length-1 sequence a
    (row, kv head), planned and scratch-cached like the FP8 sparse path's.

    ``quantized_output`` / ``output_scale``: o_proj's FP8 input, written by the
    attention's epilogue instead of the BF16 output (rows past q's are a padded
    token bucket's, written inert). FP32 ``[rows, 1]`` per-token descales (one
    kv head a rank: a token's scale spans every head) are split as
    ``sparse_attn.sparse_pa_per_row`` splits the FP8 path's: prefill rows
    (``quantized_direct``) one task a row, decode rows a cluster of eight. E8M0
    ``[rows, heads * D // 32]`` MXFP8 scales keep the BF16 output's split: aiter
    writes them in every output mode, bit for bit the BF16 output quantized as
    o_proj's ``quantize_fp8`` would. A group of 32 never spans two heads, so
    with several kv heads the (row, kv head) sequences write the row-major
    values and scales through ``[rows * Hkv, group * D // 32]`` views."""
    from aiter.ops.flydsl.pa_decode import pa_decode
    from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits

    from atom.model_ops.base_attention import (
        _flydsl_plan_scratch,
        _flydsl_prepare_explicit_plan,
    )
    from atom.model_ops.minimax_m3.sparse_attn import _cluster_resident_rows

    rows, num_q_heads, head_dim = q.shape
    pages, num_kv_heads = k_payload.shape[:2]
    group = num_q_heads // num_kv_heads
    seqs = rows * num_kv_heads
    output = torch.empty_like(q)
    quant_args = {}
    mxfp8 = quantized_output is not None and output_scale.dtype != torch.float32
    if quantized_output is not None:
        assert (
            mxfp8 or num_kv_heads == 1
        ), "per-token o_proj quantization needs one kv head"
        quantized_view = quantized_output.view(-1, group, head_dim)
        quant_seqs = quantized_view.shape[0]
        pad_rows = quant_seqs - seqs
        assert pad_rows >= 0 and output_scale.shape[0] * num_kv_heads == quant_seqs
        quant_args = {
            "quantized_output": quantized_view,
            # A sequence's slice of its row's scales (MXFP8).
            "output_scale": (
                output_scale.view(quant_seqs, -1) if mxfp8 else output_scale
            ),
            # Nothing reads the BF16 output.
            "store_output": False,
            "quantized_pad_rows": pad_rows,
        }
    if seqs == 0:
        if quantized_output is not None:
            from atom.model_ops.base_attention import fill_unit_scale

            quantized_output.zero_()
            fill_unit_scale(output_scale)
        return output
    # (row, kv head) -> sequence, as the FP8 path collapses them; zero-copy
    # (.view raises rather than copy, which would break the page encoding).
    q_view = q.view(seqs, group, head_dim)
    out_view = output.view(seqs, group, head_dim)
    caches = [
        t.view(pages * num_kv_heads, 1, *t.shape[2:])
        for t in (k_payload, v_payload, k_scale, v_scale)
    ]
    k_view, v_view = caches[0], caches[1]
    ks_view, vs_view = (t.view(torch.float8_e4m3fn) for t in caches[2:])
    if (
        quantized_direct
        # The BF16 output, or o_proj's MXFP8 input (E8M0 scales); the per-token
        # FP32 FP8 output keeps the paged decode. The image's aiter pads the
        # MXFP8 bucket for one kv head only (lifted by aiter m31-pa-mh-aiter).
        and (quantized_output is None or (mxfp8 and num_kv_heads == 1))
        and envs.ATOM_M3_SPARSE_PREFILL_REUSE
        and rows >= envs.ATOM_M3_SPARSE_PREFILL_REUSE_MIN_ROWS
    ):
        reuse = _prefill_reuse_kernel()
        if reuse is not None and reuse[1](q_view, k_view, table):
            # Prefill rows: a CTA walks the union of a tile of consecutive
            # tokens' pages once (aiter `pa_sparse_prefill_nvfp4`), instead of a
            # CTA a token gathering its own 16 pages.
            mx_args = {}
            if quantized_output is not None:
                mx_args = {
                    "quantized_output": quant_args["quantized_output"],
                    "output_scale": quant_args["output_scale"],
                    "quantized_pad_rows": quant_args["quantized_pad_rows"],
                }
            reuse[0](
                out_view,
                q_view,
                k_view,
                v_view,
                ks_view,
                vs_view,
                table,
                ctx,
                sm_scale,
                kv_heads=num_kv_heads,
                tile_rows=envs.ATOM_M3_SPARSE_PREFILL_REUSE_TQ,
                **mx_args,
                **_prefill_reuse_impl(),
            )
            return output
    # The FP8 sparse path's split: a bounded top-k window per row.
    partitions = get_recommended_splits(seqs, 1)
    workgroup_budget = None
    page_groups = None
    cluster_rows = 1
    if (
        envs.ATOM_M3_PA_FUSED_REDUCE
        and not quantized_direct
        and (quantized_output is None or output_scale.dtype != torch.float32)
    ):
        # Decode rows (a full top-k window each) merge their partitions in the
        # paged attention: clusters while every row's cluster is resident, else
        # one CTA a row whose wave groups split its pages.
        partitions, workgroup_budget, page_groups, cluster_rows = fused_reduce_split(
            seqs, q.device
        )
    elif quantized_output is not None and output_scale.dtype == torch.float32:
        # The FP8 epilogue is in the one-task (direct) and cluster-of-8 output
        # modes; any other mode the plan picks quantizes after the BF16 output
        # inside aiter (`sparse_pa_per_row` explains the choice).
        if quantized_direct:
            partitions = 1
        elif seqs <= _cluster_resident_rows(q.device):
            partitions, workgroup_budget = 8, 8 * seqs
    context_bound = table.shape[1] * BLOCK
    plan = _flydsl_prepare_explicit_plan(
        q=q_view,
        k_cache=k_view,
        context_lens=ctx,
        num_seqs=seqs,
        query_length=1,
        plan_partition_cap=partitions,
        context_bound=context_bound,
        existing_plan=None,
        plan_step_owner=plan_step_owner,
        plan_cache_owner=plan_cache_owner,
        plan_workgroup_budget=workgroup_budget,
    )
    exp_sums, max_logits, partials = _flydsl_plan_scratch(
        plan, 1, group, head_dim, q.dtype, q.device, plan_cache_owner=plan_cache_owner
    )
    pa_decode(
        out_view,
        q_view,
        k_view,
        v_view,
        ctx,
        table,
        sm_scale,
        1,
        key_scale=ks_view,
        value_scale=vs_view,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=partials,
        work_plan=plan,
        max_context_length=context_bound,
        **quant_args,
        **(
            {
                "page_groups": page_groups,
                # No plan lookups; a request's rows (which share most selected
                # pages) dispatch on one XCD.
                "plan_free": True,
                "xcd_rows": rows_per_request * num_kv_heads,
                "cluster_rows": cluster_rows,
            }
            if page_groups
            else {}
        ),
    )
    return output


@torch.no_grad()
def nvfp4_sparse_attention(
    q: torch.Tensor,  # [rows, num_heads, D]
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
    v_payload: torch.Tensor,
    v_scale: torch.Tensor,
    topk_idx: torch.Tensor,  # [Hkv, rows, topk] int32 block ids, -1 right-padded
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    row_req: torch.Tensor,  # [rows] request index of each row
    row_seq_len: torch.Tensor,  # [rows] causal length each row attends to
    sm_scale: float,
    page_table: tuple[torch.Tensor, torch.Tensor] | None = None,
    plan_step_owner=None,
    plan_cache_owner=None,
    quantized_output: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    quantized_direct: bool = True,
    rows_per_request: int = 0,
) -> torch.Tensor:
    """O ``[rows, num_heads, D]`` of every row over its top-k blocks.

    ``page_table``: gfx1250, ``nvfp4_page_table`` of these arguments (or the
    selector's NVFP4 emission) when the caller already built it (it is shared
    by the sparse layers of a step); built here otherwise.
    ``plan_step_owner`` / ``plan_cache_owner``: as for the FP8 sparse path's
    ``sparse_pa_per_row``. ``quantized_output`` / ``output_scale`` /
    ``quantized_direct`` (gfx1250): see ``_flydsl_attention``; the returned BF16
    output is then unspecified.
    """
    if _gluon_kernel():
        if quantized_output is not None:
            raise NotImplementedError(
                "the gfx950 NVFP4 attention writes its BF16 output only"
            )
        from atom.model_ops.minimax_m3.nvfp4_kernels.decode import flash_decode_nvfp4

        return flash_decode_nvfp4(
            q.contiguous(),
            Nvfp4Cache(payload=k_payload, scales=k_scale),
            Nvfp4Cache(payload=v_payload, scales=v_scale),
            block_table,
            row_seq_len,
            row_req,
            BLOCK,
            topk_idx.contiguous(),
            sm_scale,
        )
    if page_table is None or page_table[0] is None:
        page_table = nvfp4_page_table(
            topk_idx.contiguous(), block_table, row_req, row_seq_len
        )
    return _flydsl_attention(
        q.contiguous(),
        k_payload,
        k_scale,
        v_payload,
        v_scale,
        *page_table,
        sm_scale,
        plan_step_owner,
        plan_cache_owner,
        quantized_output,
        output_scale,
        quantized_direct,
        rows_per_request,
    )


@triton.jit
def _decode_rows_kernel(
    seq_ptr, req_ptr, len_ptr, rows, num_reqs, Q: tl.constexpr, BLOCK: tl.constexpr
):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < rows
    req = r // Q
    live = m & (req < num_reqs)
    seq = tl.load(seq_ptr + req, mask=live, other=0).to(tl.int32)
    length = tl.where(live, tl.maximum(seq - Q + (r - req * Q) + 1, 0), 0)
    tl.store(req_ptr + r, tl.where(live, req, 0).to(tl.int32), mask=m)
    tl.store(len_ptr + r, length, mask=m)


def decode_rows(seq_lens: torch.Tensor, rows: int, max_query_len: int):
    """(row_req, row_seq_len) of a uniform spec-verify decode: rows are
    request-major, ``max_query_len`` query tokens per request, the last at the
    request's full length. Rows past ``len(seq_lens) * max_query_len`` are
    padding: request 0, length 0, so no kernel reads a block for them. One
    launch, once per forward (``SparseMHAPagedAttentionImpl._row_map``), shared
    by the index top-k and the attention of every sparse layer."""
    req = torch.empty(rows, dtype=torch.int32, device=seq_lens.device)
    seq = torch.empty(rows, dtype=torch.int32, device=seq_lens.device)
    block = 256
    _decode_rows_kernel[(triton.cdiv(rows, block),)](
        seq_lens, req, seq, rows, seq_lens.shape[0], Q=max_query_len, BLOCK=block
    )
    return req, seq


def prefill_rows(cu_seqlens_q: torch.Tensor, prefix_lens: torch.Tensor, rows: int):
    """(row_req, row_seq_len) of prefill query tokens: token i of request b sits
    at absolute position prefix_lens[b] + i. Rows past ``cu_seqlens_q[-1]`` pad a
    token-graph bucket: request 0, length 0."""
    cu = cu_seqlens_q.to(torch.int32)
    tok = torch.arange(rows, device=cu.device, dtype=torch.int32)
    live = tok < cu[-1]
    req = torch.where(
        live, torch.searchsorted(cu[1:], tok, right=True, out_int32=True), 0
    )
    seq = prefix_lens.to(torch.int32)[req] + (tok - cu[req]) + 1
    return req, torch.where(live, seq, 0)
