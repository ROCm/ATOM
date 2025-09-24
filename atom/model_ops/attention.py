import aiter
import torch
import triton
import triton.language as tl
from aiter.paged_attn import PagedAttention
from torch import nn

from atom.utils.context import get_context

# from flash_attn import flash_attn_with_kvcache


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        kv_cache_dtype="bf16",
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_cache_dtype = kv_cache_dtype
        self.max_model_len = 0
        self.k_scale = self.v_scale = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            if self.kv_cache_dtype == "fp8":
                aiter.reshape_and_cache_with_pertoken_quant(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    self.k_scale,
                    self.v_scale,
                    context.slot_mapping,
                    asm_layout=True,
                )
            else:
                aiter.reshape_and_cache(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    context.slot_mapping,
                    kv_cache_dtype="auto",
                    k_scale=None,
                    v_scale=None,
                    asm_layout=True,
                )

        if context.is_prefill:
            # if context.block_tables is not None:  # prefix cache
            #     k, v = k_cache, v_cache
            o = aiter.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                min_seqlen_q=context.min_seqlen_q,
                dropout_p=context.dropout_p,
                softmax_scale=self.scale,
                causal=True,
            )
        else:  # decode
            o = aiter.pa_fwd_asm(
                q,
                k_cache,
                v_cache,
                context.block_tables,
                context.context_lens,
                context.block_tables.stride(0),
                K_QScale=self.k_scale,
                V_QScale=self.v_scale,
                out_=None,
                high_precision=0,
            )

            # o = PagedAttention.forward_decode(
            #     q,
            #     k_cache,
            #     v_cache,
            #     context.block_tables,
            #     context.context_lens,
            #     max_seq_len=self.max_model_len,
            #     kv_cache_dtype="auto",
            #     num_kv_heads=self.num_kv_heads,
            #     scale=self.scale,
            #     alibi_slopes=None,
            #     k_scale=self.k_scale,
            #     v_scale=self.v_scale,
            # )

            # o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
            #                             cache_seqlens=context.context_lens, block_table=context.block_tables,
            #                             softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
