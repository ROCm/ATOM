"""Block-sparse attention over the NVFP4 sparse KV cache (MiniMax-M3,
``ATOM_M3_SPARSE_KV_NVFP4=1``).

The cache is the NVFP4 layout of ``index_fp4`` per kv head:

- ``k_payload`` / ``v_payload`` ``[blocks, Hkv, 128, D // 2]`` uint8,
- ``k_scale`` / ``v_scale`` ``[blocks, Hkv, 128, D // 16]`` uint8 (E4M3).

A row is a decode query row or a prefill query token. It attends the top-k
128-blocks the indexer picked for it, addressed through its request's block
table, up to its own causal length. The kernel is SGLang's gfx950 NVFP4 decode
kernel (vendored in ``sglang_nvfp4``); this module maps ATOM's batch onto its
per-row ``(request, causal length)`` arguments.
"""

import torch
import triton
import triton.language as tl


@torch.no_grad()
def nvfp4_sparse_attention_gfx950(
    q: torch.Tensor,  # [rows, num_heads, D]
    k_payload: torch.Tensor,  # [blocks, Hkv, 128, D // 2]
    k_scale: torch.Tensor,  # [blocks, Hkv, 128, D // 16]
    v_payload: torch.Tensor,
    v_scale: torch.Tensor,
    topk_idx: torch.Tensor,  # [Hkv, rows, topk] int32 block ids, -1 right-padded
    block_table: torch.Tensor,  # [reqs, max_blocks] int32 (128-token blocks)
    row_req: torch.Tensor,  # [rows] request index of each row
    row_seq_len: torch.Tensor,  # [rows] causal length each row attends to
    sm_scale: float,
    output: torch.Tensor,  # [rows, num_heads, D] (shape only; the result is returned)
) -> torch.Tensor:
    """Rows attend their top-k 128-blocks through SGLang's gfx950 NVFP4 decode
    kernel (vendored in sglang_nvfp4, block-table addressing). Serves decode
    (rows = requests x spec rows) and prefill (rows = query tokens)."""
    from atom.model_ops.minimax_m3.sglang_nvfp4.decode_gfx950 import (
        flash_decode_nvfp4_gfx950,
    )
    from atom.model_ops.minimax_m3.sglang_nvfp4.nvfp4 import Nvfp4Cache

    o = flash_decode_nvfp4_gfx950(
        q.contiguous(),
        Nvfp4Cache(payload=k_payload, scales=k_scale),
        Nvfp4Cache(payload=v_payload, scales=v_scale),
        block_table,
        row_seq_len,
        row_req,
        128,
        topk_idx.contiguous(),
        sm_scale,
        1.0,
    )
    return o.view_as(output)


@triton.jit
def _decode_rows_kernel(
    seq_ptr, req_ptr, len_ptr, rows, Q: tl.constexpr, BLOCK: tl.constexpr
):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < rows
    req = r // Q
    seq = tl.load(seq_ptr + req, mask=m, other=0).to(tl.int32)
    tl.store(req_ptr + r, req.to(tl.int32), mask=m)
    tl.store(len_ptr + r, seq - Q + (r - req * Q) + 1, mask=m)


def decode_rows(seq_lens: torch.Tensor, rows: int):
    """(row_req, row_seq) of a uniform spec-verify decode: rows are
    request-major, ``rows // len(seq_lens)`` query tokens per request, the last
    at the request's full length. One launch (it runs per sparse layer)."""
    q = rows // seq_lens.shape[0]
    req = torch.empty(rows, dtype=torch.int32, device=seq_lens.device)
    seq = torch.empty(rows, dtype=torch.int32, device=seq_lens.device)
    block = 256
    _decode_rows_kernel[(triton.cdiv(rows, block),)](
        seq_lens, req, seq, rows, Q=q, BLOCK=block
    )
    return req, seq


def prefill_rows(cu_seqlens_q: torch.Tensor, prefix_lens: torch.Tensor, rows: int):
    """(row_req, row_seq_len) of prefill query tokens: token i of request b sits
    at absolute position prefix_lens[b] + i."""
    tok = torch.arange(rows, device=cu_seqlens_q.device, dtype=torch.int32)
    req = (torch.searchsorted(cu_seqlens_q[1:].to(torch.int32), tok, right=True)).to(
        torch.int32
    )
    seq = (
        prefix_lens.to(torch.int32)[req] + (tok - cu_seqlens_q.to(torch.int32)[req]) + 1
    )
    return req, seq
