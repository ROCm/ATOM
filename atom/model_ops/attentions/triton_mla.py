# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging

import torch
from aiter.ops.triton.attention.mla_decode import csr_to_dense_block_table

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mla import MLAAttention
from atom.utils import envs
from atom.utils.forward_context import AttentionMetaData

from .aiter_mla import AiterMLAMetadataBuilder, MLAChunkContextMetadata
from .backends import AttentionBackend

logger = logging.getLogger("atom")


class TritonMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_TRITON_MLA"

    @staticmethod
    def get_builder_cls() -> type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["MLAAttention"]:
        return MLAAttention


class TritonMLAMetadataBuilder(AiterMLAMetadataBuilder):

    def __init__(self, model_runner):
        super().__init__(model_runner)

        hf = model_runner.config.hf_config
        kv_lora_rank = hf.kv_lora_rank
        num_kv_splits = 4
        triton_mla_buffers = {
            "triton_block_table": torch.zeros(
                self.max_bs,
                self.max_num_blocks_per_seq,
                dtype=torch.int32,
                device=self.device,
            ),
            "triton_attn_logits": torch.empty(
                self.max_bs,
                self.padded_num_attention_heads,
                num_kv_splits,
                kv_lora_rank + 1,
                dtype=torch.float32,
                device=self.device,
            ),
            "triton_lse": torch.empty(
                self.max_bs,
                self.padded_num_attention_heads,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        self.model_runner.forward_vars.update(triton_mla_buffers)

    def set_mla_persistent_worker_buffers(
        self, bs, max_q_len, only_update=False, num_reject_tokens=None, **kwargs
    ):
        # Triton MLA does not use aiter persistent worker buffers.
        # `**kwargs` absorbs arguments the aiter caller grows over time (e.g.
        # `is_cp_round_robin`); this override returns nothing, so ignoring them
        # is correct, and pinning the exact signature only breaks the Triton
        # path whenever the aiter side adds a keyword.
        return {}

    def _cut_decode_buffers(
        self,
        rows: int,
        max_seqlen_k: int,
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
    ) -> dict:
        """The three buffers `attention_mla`'s Triton branch reads, cut to `rows`.

        One implementation for both entry points, because the defect was that
        only `prepare_decode` cut them: a mid-step then ran its own wider batch
        against the verify forward's cut. Costs a zero plus a densify per
        mid-step, which that path did not pay before.
        """
        var = self.model_runner.forward_vars
        block_table = var["triton_block_table"][:rows, :max_seqlen_k]
        block_table.zero_()
        csr_to_dense_block_table(kv_indices, kv_indptr, block_table, max_seqlen_k, rows)
        return {
            "triton_block_table": block_table,
            "triton_attn_logits": var["triton_attn_logits"][:rows],
            "triton_lse": var["triton_lse"][:rows],
        }

    def prepare_decode(
        self,
        batch: ScheduledBatch,
        running_bs: int,
        running_tokens: int,
        max_seqlen_q: int,
    ):
        attn_metadata, positions = super().prepare_decode(
            batch, running_bs, running_tokens, max_seqlen_q
        )
        for name, buf in self._cut_decode_buffers(
            batch.total_seqs_num_decode,
            attn_metadata.max_seqlen_k,
            attn_metadata.kv_indices,
            attn_metadata.kv_indptr,
        ).items():
            setattr(attn_metadata, name, buf)
        return attn_metadata, positions

    def prepare_mtp_decode(
        self,
        bs: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        positions: torch.Tensor,
        only_update: bool = False,
        num_reject_tokens: torch.Tensor | None = None,
        **fused,
    ):
        """Re-cut the Triton decode buffers off the `kv_indptr` the base rebuilt.

        `**fused` rather than the base's three keyword-only arguments spelled
        out again: this class inherits `fuse_mtp_decode_position_update`, so
        `EagleProposer` passes them, and an override that restated them could
        fall behind the base without anything noticing until the call raised.
        """
        result = super().prepare_mtp_decode(
            bs,
            max_seqlen_q,
            max_seqlen_k,
            positions,
            only_update,
            num_reject_tokens,
            **fused,
        )
        running_bs = int(positions.shape[-1])  # rows; see the base contract
        var = self.model_runner.forward_vars
        return result | self._cut_decode_buffers(
            running_bs,
            max_seqlen_k,
            var["kv_indices"].gpu,
            var["kv_indptr"].gpu[: running_bs + 1],
        )

    def prepare_prefill(self, batch: ScheduledBatch, running_bs: int):
        attn_metadata, positions = super().prepare_prefill(batch, running_bs)

        if envs.ATOM_USE_TRITON_MLA_SHUFFLE_KV and attn_metadata.has_cached:
            # The shuffled cached-prefix gather (gather_kv_b_proj with
            # shuffled_kv_cache=True) reads block_size-token blocks, so it needs
            # block-granular CSR indices (logical block ids) instead of the
            # token-granular kv_indices used by the plain layout. Build them from
            # the full per-seq context (cached + just-written new tokens).
            bs = batch.total_seqs_num_prefill
            block_size = self.model_runner.block_size
            # All GPU: derive block counts from the (already on-device) full
            # context lengths and pack the dense logical block table — populated
            # by super().prepare_prefill for has_cached — into CSR via a masked
            # select (row-major == per-seq CSR order).
            var = self.model_runner.forward_vars
            ctx = attn_metadata.context_lens[:bs]  # int32 [bs], full context
            block_counts = (ctx + (block_size - 1)) // block_size  # [bs]

            indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
            indptr[1:] = torch.cumsum(block_counts, dim=0).to(torch.int32)

            block_tables = var["block_tables"].gpu[:bs]  # [bs, max_blocks] logical
            col = torch.arange(block_tables.shape[1], device=self.device)
            mask = col[None, :] < block_counts[:, None]
            indices = block_tables[mask].to(torch.int32)

            attn_metadata.shuffle_kv_block_indptr = indptr
            attn_metadata.shuffle_kv_block_indices = indices
            # If super() decided to chunk the cached prefix (total_kv >
            # attn_prefill_chunk_size), rebuild block-aligned chunk metadata so
            # the per-chunk gather can read the shuffled blocks. Otherwise the
            # single-pass gather above is used (mla_chunk_meta stays None).
            if (
                hasattr(attn_metadata, "mla_chunk_meta")
                and attn_metadata.mla_chunk_meta is not None
            ):
                attn_metadata.mla_chunk_meta = self._build_mla_chunk_meta_shuffle(
                    attn_metadata, bs
                )

        return attn_metadata, positions

    def _build_mla_chunk_meta_shuffle(self, attn_metadata, bs: int):
        """Block-aligned variant of ``_build_mla_chunk_meta`` for the shuffled
        KV layout.

        The shuffled gather reads whole ``block_size``-token blocks, so chunks
        are split along the *block* axis (≤ ``attn_prefill_chunk_size //
        block_size`` blocks per chunk) rather than the token axis. Each chunk
        carries block-granular CSR (``shuffle_kv_block_indptr/indices``) plus
        token-granular ``cu_seqlens_k`` (output positions / flash_attn lens).

        Built entirely on-device from the GPU ``num_cached_tokens`` and the
        dense logical block table (populated by super().prepare_prefill).
        """
        device = self.device
        block_size = self.model_runner.block_size
        chunk_blocks = max(1, self.attn_prefill_chunk_size // block_size)

        cached = attn_metadata.num_cached_tokens[:bs].to(torch.int64)  # [bs]
        per_seq_blocks = (cached + (block_size - 1)) // block_size  # [bs]
        total_blocks = int(per_seq_blocks.sum().item())
        if total_blocks == 0:
            return None
        num_chunks = (total_blocks + chunk_blocks - 1) // chunk_blocks

        # Global logical block list in per-seq CSR order (leading cached blocks
        # of each seq), via a masked select on the dense block table.
        block_tables = self.model_runner.forward_vars["block_tables"].gpu[:bs]
        col = torch.arange(block_tables.shape[1], device=device)
        global_blocks = block_tables[col[None, :] < per_seq_blocks[:, None]].to(
            torch.int32
        )  # [total_blocks]
        blk_offsets = torch.zeros(bs + 1, dtype=torch.int64, device=device)
        blk_offsets[1:] = torch.cumsum(per_seq_blocks, dim=0)

        blk_indptr_list: list[torch.Tensor] = []
        blk_indices_list: list[torch.Tensor] = []
        cu_seqlens_k_list: list[torch.Tensor] = []
        chunk_total_tokens: list[torch.Tensor] = []
        chunk_max_seqlen_k: list[torch.Tensor] = []

        for c in range(num_chunks):
            gb_start = c * chunk_blocks
            gb_end = min(gb_start + chunk_blocks, total_blocks)
            # Per-seq local block range covered by this chunk.
            seq_lo = blk_offsets[:bs].clamp(gb_start, gb_end)
            seq_hi = blk_offsets[1 : bs + 1].clamp(gb_start, gb_end)
            per_seq_chunk_blocks = (seq_hi - seq_lo).to(torch.int32)
            local_lo = seq_lo - blk_offsets[:bs]  # first local block in chunk
            local_hi = seq_hi - blk_offsets[:bs]  # last+1 local block in chunk
            # Token count: full blocks * block_size, clamped to cached_len for
            # the seq's final (partial) block.
            per_seq_chunk_tokens = (
                (torch.minimum(local_hi * block_size, cached) - local_lo * block_size)
                .clamp_min(0)
                .to(torch.int32)
            )

            blk_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
            blk_indptr[1:] = torch.cumsum(per_seq_chunk_blocks, dim=0)
            cu_k = torch.zeros(bs + 1, dtype=torch.int32, device=device)
            cu_k[1:] = torch.cumsum(per_seq_chunk_tokens, dim=0)

            blk_indptr_list.append(blk_indptr)
            blk_indices_list.append(global_blocks[gb_start:gb_end])
            cu_seqlens_k_list.append(cu_k)
            chunk_total_tokens.append(cu_k[-1])
            chunk_max_seqlen_k.append(per_seq_chunk_tokens.max())

        # Single host sync for the python-int scalars the forward consumes.
        total_tokens_list = torch.stack(chunk_total_tokens).tolist()
        max_seqlen_k_list = torch.stack(chunk_max_seqlen_k).tolist()

        return MLAChunkContextMetadata(
            kv_indptr=blk_indptr_list,  # unused by shuffle gather; kept for parity
            kv_indices=blk_indices_list,
            cu_seqlens_k=cu_seqlens_k_list,
            total_tokens=total_tokens_list,
            max_seqlen_k=max_seqlen_k_list,
            num_chunks=num_chunks,
            k_workspace=self.k_chunk_workspace,
            v_workspace=self.v_chunk_workspace,
            shuffle_kv_block_indptr=blk_indptr_list,
            shuffle_kv_block_indices=blk_indices_list,
        )

    def build_for_cudagraph_capture(self, bs: int) -> AttentionMetaData:
        attn_metadata, context = super().build_for_cudagraph_capture(bs)

        var = self.model_runner.forward_vars
        attn_metadata.triton_block_table = var["triton_block_table"][:bs]
        attn_metadata.triton_attn_logits = var["triton_attn_logits"][:bs]
        attn_metadata.triton_lse = var["triton_lse"][:bs]

        return attn_metadata, context
