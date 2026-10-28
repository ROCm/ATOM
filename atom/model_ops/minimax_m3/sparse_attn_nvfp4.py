"""Block-sparse attention over the NVFP4 sparse KV cache (MiniMax-M3,
``ATOM_M3_SPARSE_KV_NVFP4=1``).

The cache is the NVFP4 layout of ``index_fp4`` per kv head:

- ``k_payload`` / ``v_payload`` ``[blocks, Hkv, 128, D // 2]`` uint8,
- ``k_scale`` / ``v_scale`` ``[blocks, Hkv, 128, D // 16]`` uint8 (E4M3).

A row is a decode query row or a prefill query token. It attends the top-k
128-blocks the indexer picked for it, addressed through its request's block
table, up to its own causal length. ``nvfp4_kernels.decode_gfx950`` does the
attention; this module maps ATOM's batch onto its per-row (request, causal
length) arguments.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.minimax_m3.nvfp4_kernels.decode_gfx950 import (
    can_use_nvfp4_decode_gfx950,
    flash_decode_nvfp4_gfx950,
)
from atom.model_ops.minimax_m3.nvfp4_kernels.nvfp4 import Nvfp4Cache

BLOCK = 128  # sparse block == cache page


def check_nvfp4_sparse_attention(q, k_payload, k_scale, v_payload, v_scale, topk_idx):
    """Raise unless the gfx950 kernel serves these shapes (16 q heads per kv head,
    head dim 128, 128-token pages, 32-bit addressable caches). The shapes are
    fixed per deployment, so one call per layer at its first forward is enough."""
    k = Nvfp4Cache(payload=k_payload, scales=k_scale)
    v = Nvfp4Cache(payload=v_payload, scales=v_scale)
    if not can_use_nvfp4_decode_gfx950(q, k, v, topk_idx, BLOCK):
        raise ValueError(
            "ATOM_M3_SPARSE_KV_NVFP4: the gfx950 NVFP4 attention kernel needs "
            f"16 q heads per kv head and head dim 128; got q {tuple(q.shape)}, "
            f"k payload {tuple(k_payload.shape)} (e.g. MiniMax-M3.1 at TP4)."
        )


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
) -> torch.Tensor:
    """O ``[rows, num_heads, D]`` of every row over its top-k blocks."""
    return flash_decode_nvfp4_gfx950(
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
    launch; computed once per layer and shared by the index top-k and the
    attention.

    shortcut: once per layer (60 launches a step), not once per step; moving it
    into the metadata builder's graph buffers would save the rest."""
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
    req = torch.where(live, torch.searchsorted(cu[1:], tok, right=True), 0)
    req = req.to(torch.int32)
    seq = prefix_lens.to(torch.int32)[req] + (tok - cu[req]) + 1
    return req, torch.where(live, seq, 0)
