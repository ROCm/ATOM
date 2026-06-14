# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Offline dump of decode-MLA kernel inputs for replay-based debugging.

Enabled with ``ATOM_DUMP_MLA_DECODE=1``. Each (layer, step) decode call writes a
torch ``.pt`` file that is consumed by
``op_tests/test_mla_decode_replay.py``, which replays the dump through both the
AITER ``mla_decode_fwd`` kernel and the Triton ``decode_attention_fwd``
reference (and an fp32 ground truth) and compares them.

Environment:
    MLA_DUMP_DIR            output directory (default ``~/mla_decode_dump``).
    ATOM_MLA_DUMP_MAX_STEPS per-layer step budget (default ``1``).

Only the KV blocks referenced by ``kv_indices`` are saved; block ids are
remapped to a contiguous ``0..N-1`` range so both kernels address the compacted
buffer consistently. The saved ``kv_compact`` is token-major / interleaved
(``[num_blocks, page_size, 1, kv_lora_rank + qk_rope_head_dim]``), which is what
the Triton kernel consumes directly and what the replay repacks for AITER.
"""

import os
from collections import defaultdict

import torch

# Per-layer dump counter so we only keep the first few decode steps and do not
# fill the disk on a long run.
_DUMP_COUNTS: dict[int, int] = defaultdict(int)


def _dump_dir() -> str:
    return os.path.expanduser(os.environ.get("MLA_DUMP_DIR", "~/mla_decode_dump"))


def _max_steps() -> int:
    return int(os.environ.get("ATOM_MLA_DUMP_MAX_STEPS", "1") or "1")


def _min_pages() -> int:
    """Only dump steps whose longest per-batch KV span is >= this many pages.

    The fp8 get_meta_param clamp keeps num_kv_splits>1 only when
    (pages-1)//min_block_n + 1 >= 2, i.e. pages >= 33 for nhead=128 (min_block_n=32).
    Set ATOM_MLA_DUMP_MIN_PAGES=33 to capture only steps where the >1-split path
    is actually exercised end-to-end."""
    return int(os.environ.get("ATOM_MLA_DUMP_MIN_PAGES", "0") or "0")


def _cpu(t):
    return t.detach().to("cpu") if torch.is_tensor(t) else t


def _compact_kv(kv_buffer_view, kv_indptr, kv_indices):
    """Keep only the KV blocks referenced by ``kv_indices`` and remap ids.

    kv_buffer_view: [num_blocks, page_size, 1, head_dim] (token-major)
    Returns (kv_compact_cpu, new_kv_indices_int32, valid_len).
    """
    bs = kv_indptr.numel() - 1
    kv_indptr_cpu = kv_indptr.to(torch.int64).cpu()
    valid_len = int(kv_indptr_cpu[bs].item())
    used = kv_indices.to(torch.int64).cpu()[:valid_len]
    # sorted unique block ids + inverse mapping (position of each used id in uniq)
    uniq, inverse = torch.unique(used, sorted=True, return_inverse=True)
    kv_compact = kv_buffer_view[uniq.to(kv_buffer_view.device)].detach().to("cpu")
    new_indices = inverse.to(torch.int32)
    return kv_compact, new_indices, valid_len


def dump_decode_mla(
    *,
    layer_num,
    q,
    kv_buffer_view,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_q_len,
    page_size,
    sm_scale,
    q_scale=None,
    kv_scale=None,
    num_kv_splits=None,
    context_lens=None,
    num_heads=None,
    padded_num_heads=None,
    kv_lora_rank=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    head_repeat_factor=1,
    is_sparse_mla=False,
    kv_layout="interleaved",
):
    # Never dump while a CUDA graph is being captured: the host-side .cpu()
    # syncs + file IO are illegal inside capture and the inputs are dummy.
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    if _DUMP_COUNTS[int(layer_num)] >= _max_steps():
        return
    min_pages = _min_pages()
    if min_pages > 0:
        # Longest per-batch KV span in pages; skip short-context steps where the
        # fp8 clamp collapses num_kv_splits back to 1.
        kv_indptr_cpu = kv_indptr.to(torch.int64).cpu()
        max_pages = int((kv_indptr_cpu[1:] - kv_indptr_cpu[:-1]).max().item())
        if max_pages < min_pages:
            return
    step = _DUMP_COUNTS[int(layer_num)]
    _DUMP_COUNTS[int(layer_num)] += 1

    out_dir = _dump_dir()
    os.makedirs(out_dir, exist_ok=True)
    rank = int(os.environ.get("ATOM_DP_RANK", "0") or "0")

    kv_compact, new_indices, _ = _compact_kv(kv_buffer_view, kv_indptr, kv_indices)

    payload = {
        "layer_num": int(layer_num),
        "q": _cpu(q.contiguous()),
        "kv_compact": kv_compact,
        "o_server": _cpu(o),
        "qo_indptr": _cpu(qo_indptr).to(torch.int32),
        "kv_indptr": _cpu(kv_indptr).to(torch.int32),
        "kv_indices": new_indices,
        "kv_last_page_lens": _cpu(kv_last_page_lens).to(torch.int32),
        "page_size": int(page_size),
        "max_q_len": int(max_q_len),
        "sm_scale": float(sm_scale),
        "q_scale": _cpu(q_scale),
        "kv_scale": _cpu(kv_scale),
        "num_kv_splits": int(num_kv_splits) if num_kv_splits is not None else None,
        "context_lens": _cpu(context_lens) if context_lens is not None else None,
        "num_heads": int(num_heads) if num_heads is not None else None,
        "padded_num_heads": (
            int(padded_num_heads) if padded_num_heads is not None else int(q.shape[1])
        ),
        "kv_lora_rank": int(kv_lora_rank),
        "qk_rope_head_dim": (
            int(qk_rope_head_dim) if qk_rope_head_dim is not None else None
        ),
        "v_head_dim": int(v_head_dim) if v_head_dim is not None else None,
        "head_repeat_factor": int(head_repeat_factor),
        "is_sparse_mla": bool(is_sparse_mla),
        "kv_layout": kv_layout,
        "q_dtype": str(q.dtype),
        "kv_dtype": str(kv_buffer_view.dtype),
    }

    fname = f"mla_decode_layer{int(layer_num):03d}_step{step}_rank{rank}.pt"
    path = os.path.join(out_dir, fname)
    torch.save(payload, path)
