"""Draft extend metadata — draft pool, DRAFT_EXTEND_V2 bs×K fill after verify."""

from __future__ import annotations

import os

import numpy as np
import torch

from atom.plugin.sglang.glm52_mtp.common import (
    get_extend_lens_cpu,
    get_extend_prefix_lens_cpu,
    get_seq_lens_cpu,
    is_draft_extend_mode,
)
from atom.plugin.sglang.glm52_mtp.multi_token import build_mtp_multi_token_decode_metadata


def draft_extend_k_only() -> bool:
    """Debug opt-in: attend only the current K-token extend chunk (legacy workaround)."""
    return os.environ.get("ATOM_GLM52_DRAFT_EXTEND_K_ONLY", "0") in (
        "1",
        "true",
        "True",
    )


def resolve_draft_extend_lens(
    forward_batch,
    positions: torch.Tensor,
    bs: int,
    draft_token_num: int,
):
    """Return prefix and total KV lengths for DRAFT_EXTEND_V2."""
    draft_token_num = int(draft_token_num)
    if draft_extend_k_only():
        prefix_lens = np.zeros(bs, dtype=np.int32)
        context_lens = np.full(bs, draft_token_num, dtype=np.int32)
        return prefix_lens, context_lens

    seq_lens = get_seq_lens_cpu(forward_batch, bs)
    position_rows = positions.detach().cpu().numpy().astype(np.int32)
    required = bs * draft_token_num
    if position_rows.size >= required:
        prefix_lens = position_rows[:required:draft_token_num].astype(np.int32)
    else:
        prefix_lens = get_extend_prefix_lens_cpu(forward_batch, bs)
        if prefix_lens is None:
            prefix_lens = np.maximum(seq_lens - draft_token_num, 0).astype(np.int32)
        else:
            prefix_lens = prefix_lens.astype(np.int32)

    context_lens = (prefix_lens + draft_token_num).astype(np.int32)
    context_lens = np.maximum(context_lens, seq_lens).astype(np.int32)
    return prefix_lens.astype(np.int32), context_lens.astype(np.int32)


def draft_extend_token_num(forward_batch, positions: torch.Tensor, bs: int) -> int:
    extend_lens = get_extend_lens_cpu(forward_batch, positions, bs)
    if extend_lens.size:
        return int(extend_lens.max(initial=1))
    tokens_per_req = getattr(
        getattr(forward_batch, "spec_info", None), "num_tokens_per_req", None
    )
    if tokens_per_req is not None:
        return int(tokens_per_req)
    if bs > 0 and int(positions.numel()) >= bs:
        return max(1, int(positions.numel()) // bs)
    return 1


def should_use_mtp_draft_extend_decode_path(forward_batch) -> bool:
    """Use decode-style draft_extend metadata (native propose i=0 semantics)."""
    override = os.environ.get("ATOM_GLM52_DRAFT_EXTEND_PATH", "").lower()
    if override in ("prefill", "prefill_prefix"):
        return False
    if override in ("decode",):
        return True
    if is_draft_extend_mode(forward_batch):
        return True
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return True
    if int(getattr(forward_batch, "_graph_cache_bs", 0) or 0) > 0:
        return True
    return False


def build_mtp_draft_extend_decode_metadata(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
):
    """Build decode-style metadata for SGLang DRAFT_EXTEND_V2 (draft step i=0)."""
    bs = int(forward_batch.batch_size)
    draft_token_num = draft_extend_token_num(forward_batch, positions, bs)
    if draft_token_num <= 0:
        raise RuntimeError("GLM-5.2 DSA draft_extend requires draft_token_num")
    return build_mtp_multi_token_decode_metadata(
        forward_batch,
        positions,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
        atom_config=atom_config,
        draft_token_num=draft_token_num,
        resolve_lens_fn=resolve_draft_extend_lens,
    )
