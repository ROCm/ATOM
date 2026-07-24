"""Target verify metadata — target pool, bs×K query rows."""

from __future__ import annotations

import os

import numpy as np
import torch

from atom.plugin.sglang.glm52_mtp.multi_token import build_mtp_multi_token_decode_metadata


def resolve_target_verify_lens(
    forward_batch,
    positions: torch.Tensor,
    bs: int,
    draft_token_num: int,
):
    """Return committed prefix lengths and total KV lengths for target_verify."""
    del forward_batch
    position_rows = positions.detach().cpu().numpy().astype(np.int32)
    required = bs * draft_token_num
    if position_rows.size < required:
        raise RuntimeError(
            "GLM-5.2 DSA target_verify positions are shorter than "
            f"bs*draft_token_num: positions={position_rows.size}, "
            f"bs={bs}, draft_token_num={draft_token_num}"
        )
    prefix_lens = position_rows[:required:draft_token_num].astype(np.int32)
    context_lens = prefix_lens + draft_token_num
    return prefix_lens, context_lens


def should_use_mtp_verify_prefill_path(
    forward_batch,
    positions: torch.Tensor,
    atom_config,
) -> bool:
    """Choose prefill vs decode metadata for eager target_verify."""
    del forward_batch, positions, atom_config
    override = os.environ.get("ATOM_GLM52_TV_VERIFY_PATH", "").lower()
    if override in ("prefill", "prefill_prefix"):
        return True
    if override in ("decode",):
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    return False


def build_mtp_verify_decode_metadata(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
):
    """Build ATOM-native decode-style metadata for SGLang target_verify."""
    bs = int(forward_batch.batch_size)
    draft_token_num = int(
        getattr(getattr(forward_batch, "spec_info", None), "draft_token_num", 0) or 0
    )
    if draft_token_num <= 0:
        raise RuntimeError("GLM-5.2 DSA target_verify requires draft_token_num")
    return build_mtp_multi_token_decode_metadata(
        forward_batch,
        positions,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
        atom_config=atom_config,
        draft_token_num=draft_token_num,
        resolve_lens_fn=resolve_target_verify_lens,
    )
