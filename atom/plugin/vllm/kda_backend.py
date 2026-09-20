"""Kimi-K3 KDA metadata adapter for the ATOM vLLM plugin."""

from __future__ import annotations

import torch

# vLLM 0.26 keeps the platform-neutral KDA metadata implementation beside its
# NVIDIA kernel. Its Triton metadata staging also supports ROCm.
from vllm.models.kimi_k3.nvidia.kda_metadata import (
    KimiK3KDAAttentionBackend,
    KimiK3KDAMetadata,
    KimiK3KDAMetadataBuilder,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    mamba_get_block_table_tensor,
)

# ATOM's own KDA metadata names a separate slot to READ the incoming state from,
# for the one forward where a prefix-cache hit forks off a checkpoint. vLLM's
# block manager has no such fork, and None tells the shared Kimi-K3 KDA forward
# (which reads this unconditionally) to stay on the write slot.
KimiK3KDAMetadata.non_spec_state_indices_in_tensor = None


class AtomKimiK3KDAMetadataBuilder(KimiK3KDAMetadataBuilder):
    """Adapt vLLM's KDA metadata to ATOM's request-indexed decode kernel."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # vLLM clamps every backend's reorder_batch_threshold to 1 under DCP
        # unless the backend declares it handles varlen decode there. That is a
        # statement about a *paged* cache being round-robin sharded, and KDA has
        # none: its state is a per-request recurrent tensor that DCP leaves
        # replicated -- only the MLA latent cache is sharded. So the clamp buys
        # nothing here and costs a lot, reclassifying the speculative verify
        # batch as prefill and moving KDA off its fused spec-decode path onto
        # the chunked one, which is also the path that cannot be captured.
        # Restore the threshold this builder picks when DCP is off.
        self._init_reorder_batch_threshold(
            1, self.use_spec_decode, supports_dcp_with_varlen=True
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> KimiK3KDAMetadata:
        metadata = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
        )
        self._adapt_full_graph_decode_metadata(common_attn_metadata, metadata)
        return metadata

    def _adapt_full_graph_decode_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        metadata: KimiK3KDAMetadata,
    ) -> None:
        if not getattr(self, "use_full_cuda_graph", False):
            return
        if (
            metadata.num_prefills != 0
            or metadata.num_spec_decodes != 0
            or metadata.num_decodes <= 0
        ):
            return

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        if query_start_loc_cpu is None or query_start_loc_cpu.numel() <= 1:
            return

        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        real_decode_mask_cpu = query_lens_cpu > 0
        real_num_decodes = int(real_decode_mask_cpu.sum().item())
        if real_num_decodes == metadata.num_decodes:
            return

        batch_size = int(common_attn_metadata.num_reqs)
        if batch_size > self.decode_cudagraph_max_bs:
            return

        query_start_loc = common_attn_metadata.query_start_loc
        block_table_tensor = mamba_get_block_table_tensor(
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        state_indices = self.non_spec_state_indices_tensor[:batch_size]
        if real_num_decodes > 0:
            real_decode_mask = real_decode_mask_cpu.to(
                query_start_loc.device, non_blocking=True
            )
            state_indices[:real_num_decodes].copy_(
                block_table_tensor[real_decode_mask, 0], non_blocking=True
            )
        state_indices[real_num_decodes:].fill_(PAD_SLOT_ID)

        compact_query_start_loc_cpu = torch.zeros(
            real_num_decodes + 1, dtype=torch.int32
        )
        if real_num_decodes > 0:
            torch.cumsum(
                query_lens_cpu[real_decode_mask_cpu].to(torch.int32),
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

        metadata.num_decodes = real_num_decodes
        metadata.num_decode_tokens = int(
            query_lens_cpu[real_decode_mask_cpu].sum().item()
        )
        metadata.non_spec_state_indices_tensor = state_indices
        metadata.non_spec_query_start_loc = query_start_loc_buf


class AtomKimiK3KDAAttentionBackend(KimiK3KDAAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AtomKimiK3KDAMetadataBuilder]:
        return AtomKimiK3KDAMetadataBuilder
