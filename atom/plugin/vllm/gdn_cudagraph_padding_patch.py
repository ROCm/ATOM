"""Patch vLLM GDN metadata for FULL-cudagraph padded decode batches.

vLLM's FULL CUDAGraph replay can pad decode batches with zero-length requests.
The stock GDN metadata builder treats those padded rows as real recurrent-state
updates and fills graph tails with ``NULL_BLOCK_ID == 0``. Slot 0 is a valid
state slot, so padded rows can pollute recurrent state and degrade accuracy.

Keep the fix in the ATOM plugin: use ``PAD_SLOT_ID == -1`` for graph padding and
compact non-spec decode metadata to the rows whose query length is positive.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import torch

logger = logging.getLogger("atom")

_GDN_CUDAGRAPH_PADDING_PATCH_APPLIED = False


def _get_common_attn_metadata(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "common_attn_metadata" in kwargs:
        return kwargs["common_attn_metadata"]
    if len(args) >= 2:
        return args[1]
    return None


def _compact_full_graph_decode_metadata(
    builder: Any,
    common_attn_metadata: Any,
    attn_metadata: Any,
    gdn_attn_mod: Any,
    pad_slot_id: int,
) -> bool:
    if common_attn_metadata is None:
        return False
    if not getattr(builder, "use_full_cuda_graph", False):
        return False
    if (
        attn_metadata.num_prefills != 0
        or attn_metadata.num_spec_decodes != 0
        or attn_metadata.num_decodes <= 0
    ):
        return False

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    if query_start_loc_cpu is None or query_start_loc_cpu.numel() <= 1:
        return False

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    real_decode_mask_cpu = query_lens_cpu > 0
    real_num_decodes = int(real_decode_mask_cpu.sum().item())
    if real_num_decodes == attn_metadata.num_decodes:
        return False

    batch_size = int(common_attn_metadata.num_actual_tokens)
    if batch_size > builder.decode_cudagraph_max_bs:
        return False

    query_start_loc = common_attn_metadata.query_start_loc
    block_table_tensor = gdn_attn_mod.mamba_get_block_table_tensor(
        common_attn_metadata.block_table_tensor,
        common_attn_metadata.seq_lens,
        builder.kv_cache_spec,
        builder.vllm_config.cache_config.mamba_cache_mode,
    )

    state_indices = builder.non_spec_state_indices_tensor[:batch_size]
    if real_num_decodes > 0:
        real_decode_mask = real_decode_mask_cpu.to(
            query_start_loc.device, non_blocking=True
        )
        state_indices[:real_num_decodes].copy_(
            block_table_tensor[real_decode_mask, 0], non_blocking=True
        )
    state_indices[real_num_decodes:].fill_(pad_slot_id)

    compact_query_start_loc_cpu = torch.zeros(real_num_decodes + 1, dtype=torch.int32)
    if real_num_decodes > 0:
        torch.cumsum(
            query_lens_cpu[real_decode_mask_cpu].to(torch.int32),
            dim=0,
            out=compact_query_start_loc_cpu[1:],
        )

    query_start_loc_buf = builder.non_spec_query_start_loc[: batch_size + 1]
    query_start_loc_buf[: real_num_decodes + 1].copy_(
        compact_query_start_loc_cpu.to(query_start_loc.device, non_blocking=True),
        non_blocking=True,
    )
    terminal = query_start_loc_buf[real_num_decodes]
    query_start_loc_buf[real_num_decodes + 1 :].fill_(terminal)

    attn_metadata.num_decodes = real_num_decodes
    attn_metadata.num_decode_tokens = int(
        query_lens_cpu[real_decode_mask_cpu].sum().item()
    )
    attn_metadata.non_spec_state_indices_tensor = state_indices
    attn_metadata.non_spec_query_start_loc = query_start_loc_buf
    return True


def apply_vllm_gdn_cudagraph_padding_patch() -> bool:
    """Patch vLLM's GDN metadata builder without modifying vLLM sources."""
    global _GDN_CUDAGRAPH_PADDING_PATCH_APPLIED
    if _GDN_CUDAGRAPH_PADDING_PATCH_APPLIED:
        return False

    try:
        from vllm.v1.attention.backends import gdn_attn
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    except Exception as e:  # pragma: no cover - import guard
        logger.debug(
            "ATOM vLLM GDN cudagraph padding patch: GDN backend unavailable "
            "(%s), skip",
            e,
        )
        return False

    # ``NULL_BLOCK_ID == 0`` is a valid recurrent-state slot. Make graph padding
    # use the same negative sentinel that cache writers and GDN kernels skip.
    if hasattr(gdn_attn, "NULL_BLOCK_ID"):
        gdn_attn.NULL_BLOCK_ID = PAD_SLOT_ID

    builder_cls = getattr(gdn_attn, "GDNAttentionMetadataBuilder", None)
    original_build = getattr(builder_cls, "build", None)
    if original_build is None:
        return False
    if getattr(original_build, "_atom_gdn_cudagraph_padding_patched", False):
        _GDN_CUDAGRAPH_PADDING_PATCH_APPLIED = True
        return False

    @functools.wraps(original_build)
    def wrapped_build(self, *args, **kwargs):
        attn_metadata = original_build(self, *args, **kwargs)
        common_attn_metadata = _get_common_attn_metadata(args, kwargs)
        _compact_full_graph_decode_metadata(
            self, common_attn_metadata, attn_metadata, gdn_attn, PAD_SLOT_ID
        )
        return attn_metadata

    wrapped_build._atom_gdn_cudagraph_padding_patched = True  # type: ignore[attr-defined]
    builder_cls.build = wrapped_build
    _GDN_CUDAGRAPH_PADDING_PATCH_APPLIED = True
    logger.info(
        "ATOM plugin: patched vLLM GDN metadata for FULL-cudagraph padded "
        "decode batches (padding sentinel=-1, compact real decode rows)."
    )
    return True
