from dataclasses import dataclass

import torch


@dataclass
class MiniMaxM3SparsePrefillMetadata:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int
    total_kv_blocks: int
    # Per-query-token request id and absolute position, layer-INVARIANT (depend
    # only on cu_seqlens_q + prefix_lens). Built ONCE in prepare_prefill and
    # reused across all sparse layers by the ASM prefill block-table builder, so
    # the per-layer build needs no host sync. None on the non-ASM path.
    query_req_id: torch.Tensor | None = None  # [total_q] int32
    query_abs_pos: torch.Tensor | None = None  # [total_q] int32
    # Per-token CSR for pa_fwd_asm prefill (each token a length-1 segment):
    # arange(total_q + 1). Layer-invariant, built once in prepare_prefill.
    per_token_qo_indptr: torch.Tensor | None = None  # [total_q+1] int32


@dataclass
class MiniMaxM3SparseDecodeMetadata:
    seq_lens: torch.Tensor
    block_table: torch.Tensor


@dataclass
class MiniMaxM3SparseMetadata:
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    prefill: MiniMaxM3SparsePrefillMetadata | None = None
    decode: MiniMaxM3SparseDecodeMetadata | None = None
