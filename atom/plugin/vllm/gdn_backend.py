"""ATOM vLLM GDN attention backend overrides.

vLLM still owns construction of CommonAttentionMetadata.  ATOM only replaces the
GDN backend-specific metadata builder so GDN fixes can live in the plugin
without monkeypatching vLLM classes in place.
"""

from __future__ import annotations

import logging

import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID, mamba_get_block_table_tensor

logger = logging.getLogger("atom")

_GDN_BACKEND_REGISTERED = False


def _get_trailing_decode_layout(
    query_start_loc_cpu: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    """Return query lengths and the real prefix size of a padded decode batch.

    vLLM FULL graphs pad requests at the end of ``query_start_loc_cpu`` with
    zero-length rows.  Keep this validation on CPU: constructing a GPU boolean
    mask and using it for advanced indexing is not CUDA-graph safe on ROCm.
    """
    if query_start_loc_cpu.numel() != batch_size + 1:
        raise RuntimeError(
            "vLLM GDN metadata layout changed: query_start_loc_cpu has "
            f"{query_start_loc_cpu.numel()} entries for {batch_size} requests."
        )

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    is_real = query_lens_cpu > 0
    real_num_decodes = int(is_real.sum().item())
    if bool(is_real[real_num_decodes:].any()):
        raise RuntimeError(
            "vLLM FULL-CUDAGraph GDN padding is no longer trailing; ATOM's "
            "request-indexed decode metadata adapter must be updated."
        )
    if real_num_decodes > 0 and not torch.all(query_lens_cpu[:real_num_decodes] == 1):
        raise RuntimeError(
            "vLLM FULL-CUDAGraph GDN decode rows must contain exactly one token."
        )
    return query_lens_cpu, real_num_decodes


class AtomGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    """ATOM GDN metadata builder.

    Inherits vLLM's GDN builder so runner-side ``isinstance`` checks for spec
    decode continue to work. The post-build compaction rebuilds request-indexed
    metadata for FULL-cudagraph padded decode rows.
    """

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        attn_metadata = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
        )
        self._compact_full_graph_decode_metadata(common_attn_metadata, attn_metadata)
        return attn_metadata

    def _compact_full_graph_decode_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        if not getattr(self, "use_full_cuda_graph", False):
            return
        if (
            attn_metadata.num_prefills != 0
            or attn_metadata.num_spec_decodes != 0
            or attn_metadata.num_decodes <= 0
        ):
            return

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        if query_start_loc_cpu is None or query_start_loc_cpu.numel() <= 1:
            return

        batch_size = int(common_attn_metadata.num_reqs)
        query_lens_cpu, real_num_decodes = _get_trailing_decode_layout(
            query_start_loc_cpu,
            batch_size,
        )
        if real_num_decodes == attn_metadata.num_decodes:
            return

        if batch_size > self.decode_cudagraph_max_bs:
            return

        query_start_loc = common_attn_metadata.query_start_loc
        block_table_tensor = mamba_get_block_table_tensor(
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        if (
            block_table_tensor.ndim != 2
            or block_table_tensor.shape[0] < batch_size
            or block_table_tensor.shape[1] < 1
        ):
            raise RuntimeError(
                "vLLM GDN block-table layout changed: expected at least "
                f"({batch_size}, 1), got {tuple(block_table_tensor.shape)}."
            )

        state_indices = self.non_spec_state_indices_tensor[:batch_size]
        if real_num_decodes > 0:
            state_indices[:real_num_decodes].copy_(
                block_table_tensor[:real_num_decodes, 0], non_blocking=True
            )
        state_indices[real_num_decodes:].fill_(PAD_SLOT_ID)

        compact_query_start_loc_cpu = torch.zeros(
            real_num_decodes + 1, dtype=torch.int32
        )
        if real_num_decodes > 0:
            torch.cumsum(
                query_lens_cpu[:real_num_decodes].to(torch.int32),
                dim=0,
                out=compact_query_start_loc_cpu[1:],
            )

        query_start_loc_buf = self.non_spec_query_start_loc[: batch_size + 1]
        query_start_loc_buf[: real_num_decodes + 1].copy_(
            compact_query_start_loc_cpu.to(query_start_loc.device, non_blocking=True),
            non_blocking=True,
        )
        terminal = query_start_loc_buf[real_num_decodes]
        query_start_loc_buf[real_num_decodes + 1 :].fill_(terminal)

        attn_metadata.num_decodes = real_num_decodes
        attn_metadata.num_decode_tokens = int(
            query_lens_cpu[:real_num_decodes].sum().item()
        )
        attn_metadata.non_spec_state_indices_tensor = state_indices
        attn_metadata.non_spec_query_start_loc = query_start_loc_buf


class AtomGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AtomGDNAttentionMetadataBuilder]:
        return AtomGDNAttentionMetadataBuilder


def register_gdn_attention_backend() -> None:
    global _GDN_BACKEND_REGISTERED
    if _GDN_BACKEND_REGISTERED:
        return

    register_backend(
        MambaAttentionBackendEnum.GDN_ATTN,
        f"{AtomGDNAttentionBackend.__module__}.{AtomGDNAttentionBackend.__qualname__}",
        is_mamba=True,
    )
    _GDN_BACKEND_REGISTERED = True
    logger.info(
        "ATOM plugin: registered GDN attention backend override with ATOM "
        "metadata builder."
    )
