# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Debug helper: dump decode-MLA `mla_decode_fwd` inputs/outputs.

Enabled via ``ATOM_DUMP_MLA_DECODE=1``. For the *first* decode step of each
layer, it saves the exact tensors fed to AITER's ``mla_decode_fwd`` (with the
KV cache compacted down to only the referenced blocks so dumps stay small) plus
the AITER output ``o``. The companion offline test reconstructs the Triton
reference (``decode_attention_fwd``) from the same data and compares.

Run the server with:
    ATOM_USE_TRITON_MLA=0  (so the AITER decode path runs)
    --enforce-eager        (so this python code runs every decode step instead
                            of being bypassed by a captured CUDA graph)
    ATOM_DUMP_MLA_DECODE=1
    ATOM_DUMP_MLA_DECODE_DIR=/path/to/dump
"""

import logging
import os

import torch

logger = logging.getLogger("atom")

# Layers already dumped (a layer's first decode call == first decode step).
_dumped_layers: set = set()


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _cpu(t):
    if torch.is_tensor(t):
        return t.detach().cpu()
    return t


def dump_decode_mla(
    *,
    layer_num: int,
    q: torch.Tensor,
    kv_buffer_view: torch.Tensor,  # [num_blocks, page_size, 1, qk_head_dim]
    o: torch.Tensor,  # [B, padded_num_heads, kv_lora_rank] raw kernel output
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    max_q_len: int,
    page_size: int,
    sm_scale: float,
    q_scale,
    kv_scale,
    num_kv_splits: int,
    context_lens,
    num_heads: int,
    padded_num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    head_repeat_factor: int,
    is_sparse_mla: bool,
) -> None:
    """Save one decode-MLA call to a .pt file. Never raises (best effort)."""
    if layer_num in _dumped_layers:
        return
    # Skip while a CUDA graph is being captured (the saved tensors would be
    # bogus and torch.save is illegal during capture). With --enforce-eager
    # this is never the case.
    try:
        if torch.cuda.is_current_stream_capturing():
            return
    except Exception:
        pass

    try:
        from atom.utils import envs

        out_dir = envs.ATOM_DUMP_MLA_DECODE_DIR
        os.makedirs(out_dir, exist_ok=True)

        # Compact the KV cache to only the blocks referenced by this batch and
        # remap block ids to a contiguous 0..N-1 range. kv_indptr indexes
        # positions inside kv_indices, so it stays valid after the remap.
        total_blocks = int(kv_indptr[-1].item())
        used_blocks = kv_indices[:total_blocks].to(torch.long)
        kv_compact = kv_buffer_view.index_select(0, used_blocks).contiguous()
        new_kv_indices = torch.arange(
            total_blocks, dtype=kv_indices.dtype, device=kv_indices.device
        )

        payload = {
            # tensors fed to mla_decode_fwd (compacted)
            "q": _cpu(q),
            "kv_compact": _cpu(kv_compact),  # [N, page_size, 1, qk_head_dim]
            "o_aiter": _cpu(o),  # reference output produced in-server by AITER
            "qo_indptr": _cpu(qo_indptr),
            "kv_indptr": _cpu(kv_indptr),
            "kv_indices": _cpu(new_kv_indices),  # remapped to compact buffer
            "kv_last_page_lens": _cpu(kv_last_page_lens),
            "context_lens": _cpu(context_lens),
            # scalars / scales
            "max_q_len": int(max_q_len),
            "page_size": int(page_size),
            "sm_scale": float(sm_scale),
            "q_scale": _cpu(q_scale),
            "kv_scale": _cpu(kv_scale),
            "num_kv_splits": int(num_kv_splits),
            # shape metadata
            "num_heads": int(num_heads),
            "padded_num_heads": int(padded_num_heads),
            "kv_lora_rank": int(kv_lora_rank),
            "qk_rope_head_dim": int(qk_rope_head_dim),
            "v_head_dim": int(v_head_dim),
            "head_repeat_factor": int(head_repeat_factor),
            "is_sparse_mla": bool(is_sparse_mla),
            "q_dtype": str(q.dtype),
            "kv_dtype": str(kv_buffer_view.dtype),
            "o_dtype": str(o.dtype),
            "layer_num": int(layer_num),
            "rank": _rank(),
        }

        rank = _rank()
        fname = f"mla_decode_layer{int(layer_num):03d}_step0_rank{rank}.pt"
        path = os.path.join(out_dir, fname)
        torch.save(payload, path)
        _dumped_layers.add(layer_num)
        logger.info(
            "[mla_dump] saved %s (bs=%d, total_blocks=%d, q=%s, kv=%s)",
            path,
            qo_indptr.numel() - 1,
            total_blocks,
            tuple(q.shape),
            tuple(kv_compact.shape),
        )
    except Exception as e:  # never break inference because of dumping
        logger.warning("[mla_dump] failed for layer %s: %s", layer_num, e)
