from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Context:
    is_prefill: bool = False
    batch_size: int = 0
    graph_bs: int = 0
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    min_seqlen_q: int = 0
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    dropout_p: float = 0.0

    max_q_len: Optional[int] = None
    kv_indptr: Optional[torch.Tensor] = None
    kv_indices: Optional[torch.Tensor] = None
    kv_last_page_lens: Optional[torch.Tensor] = None

_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill: bool,
    batch_size: int,
    graph_bs=0,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    min_seqlen_q=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    dropout_p=0.0,
    max_q_len=None,
    kv_indptr=None,
    kv_indices=None,
    kv_last_page_lens=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        batch_size,
        graph_bs,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q,
        slot_mapping,
        context_lens,
        block_tables,
        dropout_p,
        max_q_len,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
