from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import aiter
import torch
from aiter import dtypes, fused_qk_norm_rope_cache_quant_shuffle
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache
from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits
from atom.config import get_current_atom_config
from atom.model_ops.attention_mla import MLAModules
from atom.model_ops.attention_mha import (
    SparseMHAPagedAttentionImpl as AtomSparseMHAPagedAttentionImpl,
)
from atom.model_ops.base_attention import (
    cp_mha_gather_cache,
    run_pa_decode_gluon,
    run_pa_fwd_asm,
)
from atom.model_ops.minimax_m3.sparse_attn import (
    SPARSE_BLOCK_SIZE,
)
from atom.plugin.vllm.attention.backend import (
    AiterMhaBackendForVllm,
    SparseMHAIndexerBackend,
    SparseMHAPagedAttentionBackend,
)
from atom.plugin.vllm.attention.layer_common import (
    _register_vllm_static_forward_context,
)
from atom.utils import envs
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

if TYPE_CHECKING:
    from atom.plugin.vllm.attention.metadata import (
        AiterMhaMetadataForVllm,
    )

ATOM_USE_GLUON_PA_DECODE = envs.ATOM_USE_GLUON_PA_DECODE

# the dispatch rule is based on the kernel benchmark result
#  of gluon/asm pa with model-specific shapes
_GLUON_PA_DECODE_BS_MAPPING = {
    "qwen3_moe": 64,
    "glm4_moe": 32,
    "minimax_m2": 16,
}
_NO_PS_FIXED_SPLITS = 64
_MINIMAX_M3_MODEL_TYPES = {"minimax_m3", "minimax_m3_text", "minimax_m3_vl"}


def _init_vllm_mha_layer_state(
    layer,
    *,
    layer_name: str,
    kv_cache_dtype: str,
    calculate_kv_scales: bool,
    quant_config,
) -> None:
    from vllm.model_executor.layers.attention.attention import _init_kv_cache_quant
    from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

    atom_config = get_current_atom_config()
    vllm_config = atom_config.plugin_config.vllm_config

    layer.layer_name = layer_name
    layer.kv_cache_dtype = kv_cache_dtype
    layer.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
        kv_cache_dtype, vllm_config.model_config
    )
    layer.calculate_kv_scales = calculate_kv_scales
    layer.quant_config = quant_config
    layer.kv_cache = torch.tensor([])

    _init_kv_cache_quant(layer, quant_config, layer_name)


def _set_default_mha_scales(layer) -> None:
    from vllm.model_executor.layers.attention.attention import set_default_quant_scales

    set_default_quant_scales(layer, register_buffer=False)
    if hasattr(layer, "_o_scale_float"):
        layer._o_scale_float = None


class AttentionForVllmMHA(nn.Module, AttentionLayerBase):
    @classmethod
    def create(cls, *args, **kwargs):
        from atom.model_ops.attention_mha import (
            SparseMHAPagedAttentionImpl as AtomSparseMHAPagedAttentionImpl,
        )

        impl_cls = kwargs.pop("impl_cls", None)
        if impl_cls is AtomSparseMHAPagedAttentionImpl:
            return SparseMHAPagedAttentionImplForVllm(*args, **kwargs)
        return cls(*args, **kwargs)

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        alibi_slopes: list[float] = None,
        kv_cache_dtype="bf16",
        layer_num=0,
        use_mla: bool = False,
        mla_modules: Optional[MLAModules] = None,
        sinks: Optional[nn.Parameter] = None,
        per_layer_sliding_window: Optional[int] = None,
        rotary_emb: Optional[torch.nn.Module] = None,
        prefix: Optional[str] = None,
        q_norm: Optional[torch.nn.Module] = None,
        k_norm: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        from vllm.v1.attention.backend import AttentionType

        atom_config = get_current_atom_config()
        cache_config = atom_config.plugin_config.vllm_cache_config
        quant_config = atom_config.plugin_config.vllm_quant_config

        layer_name = prefix if prefix is not None else f"MHA_{layer_num}"
        cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else kv_cache_dtype
        )
        calculate_kv_scales = (
            cache_config.calculate_kv_scales if cache_config is not None else False
        )

        self.head_size_v = head_dim
        self.attn_type = AttentionType.DECODER
        self.attn_backend = AiterMhaBackendForVllm
        self.has_sink = sinks is not None
        self.dtype = torch.get_default_dtype()

        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_size = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.k_cache = self.v_cache = torch.tensor([])
        self.max_model_len = 0
        self.k_scale = self.v_scale = None
        self.device = "cuda:" + str(torch.cuda.current_device())
        self.layer_num = layer_num
        self.kv_scale_float = (
            torch.finfo(torch.float8_e4m3fn).max / torch.finfo(aiter.dtypes.fp8).max
            if cache_dtype == "fp8"
            else 1.0
        )
        self.kv_scale = torch.tensor(self.kv_scale_float, dtype=torch.float32)
        self.per_token_quant = True
        self.sinks = sinks
        self.sliding_window = (
            per_layer_sliding_window if per_layer_sliding_window is not None else -1
        )
        self.rotary_emb = rotary_emb
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.use_flash_layout = False
        self.supports_quant_query_input = False

        hf_config = atom_config.hf_config
        self.model_type = getattr(hf_config, "model_type", "")

        _init_vllm_mha_layer_state(
            self,
            layer_name=layer_name,
            kv_cache_dtype=cache_dtype,
            calculate_kv_scales=calculate_kv_scales,
            quant_config=quant_config,
        )

        _register_vllm_static_forward_context(self)

    @property
    def impl(self):
        return self

    def get_attn_backend(self):
        return self.attn_backend

    def process_weights_after_loading(
        self, act_dtype: torch.dtype = torch.bfloat16
    ) -> None:
        _set_default_mha_scales(self)

    def calc_kv_scales(self, query, key, value):
        self._q_scale.copy_(torch.abs(query).max() / self.q_range)
        self._k_scale.copy_(torch.abs(key).max() / self.k_range)
        self._v_scale.copy_(torch.abs(value).max() / self.v_range)
        self._q_scale_float = self._q_scale.item()
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        self.calculate_kv_scales = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor = None,
        q_scale: Optional[torch.Tensor] = None,
        qkv: torch.Tensor = None,
        **kwargs,
    ):
        if self.calculate_kv_scales and key is not None and value is not None:
            self.calc_kv_scales(query, key, value)
        return torch.ops.aiter.atom_vllm_mha_attention(
            query,
            key,
            value,
            self.kv_cache,
            self.layer_name,
            positions,
            q_scale,
            qkv,
        )

    def rope_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qkv: torch.Tensor,
        position: torch.Tensor,
        attention_metadata: "AiterMhaMetadataForVllm",
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        flash_layout: bool = False,
    ):
        num_blocks, block_size, num_kv_heads, head_size = k_cache.shape
        x = 16 // k_cache.element_size()

        if not flash_layout:
            new_key_cache = k_cache.view(
                num_blocks, num_kv_heads, head_size // x, block_size, x
            )
            new_value_cache = v_cache.view(
                num_blocks, num_kv_heads, block_size // x, head_size, x
            )
        else:
            new_key_cache = k_cache
            new_value_cache = v_cache

        # if flash kv_cache layout, the shape of kv_cache is:
        #
        # key_cache:   [num_blocks, block_size, num_kv_heads, head_size]
        # value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        #
        # if not, the shape is:
        #
        # key_cache:   [num_blocks, num_kv_heads, head_size // x, block_size, x]
        # value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        #
        # and the origin kv cache layout in fwd_args is not flash

        attn_metadata = attention_metadata
        slot_mapping = attn_metadata.slot_mapping[: q.shape[0]]

        use_triton_attn = self.sliding_window != -1 or self.head_dim != 128
        # use_triton_attn = True
        self.use_triton_attn = use_triton_attn

        if (
            self.rotary_emb is not None
            and self.q_norm is not None
            and self.k_norm is not None
        ):
            from atom.model_ops.layernorm import GemmaRMSNorm

            if isinstance(self.q_norm, GemmaRMSNorm):
                # GemmaRMSNorm (1+w) path — use the Triton fused kernel
                from atom.model_ops.triton_fused_qkv_norm_rope_cache import (
                    triton_fused_norm_rope_cache,
                )

                q_size = self.num_heads * self.head_dim
                kv_size = self.num_kv_heads * self.head_dim
                q = q.view(-1, q_size)
                k = k.view(-1, kv_size)
                v = v.view(-1, kv_size)

                q, k = triton_fused_norm_rope_cache(
                    q,
                    k,
                    v,
                    position,
                    q_norm=self.q_norm,
                    k_norm=self.k_norm,
                    rotary_emb=self.rotary_emb,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    k_cache=new_key_cache,
                    v_cache=new_value_cache,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    slot_mapping=slot_mapping,
                    kv_cache_dtype=self.kv_cache_dtype,
                )
                # Reshape q, k for attention: [T, num_heads, head_dim]
                q = q.view(-1, self.num_heads, self.head_dim)
                k = k.view(-1, self.num_kv_heads, self.head_dim)
                v = v.view(-1, self.num_kv_heads, self.head_dim)
            else:
                # Standard RMSNorm — use existing aiter kernel
                fused_qk_norm_rope_cache_quant_shuffle(
                    q=q,
                    k=k,
                    v=v,
                    num_heads_q=self.num_heads,
                    num_heads_k=self.num_kv_heads,
                    num_heads_v=self.num_kv_heads,
                    head_dim=self.head_dim,
                    eps=self.q_norm.eps,
                    qw=self.q_norm.weight,
                    kw=self.k_norm.weight,
                    cos_sin_cache=self.rotary_emb.cos_sin_cache,
                    is_neox_style=self.rotary_emb.is_neox_style,
                    pos_ids=position,
                    k_cache=new_key_cache,
                    v_cache=new_value_cache,
                    slot_mapping=slot_mapping,
                    kv_cache_dtype=(
                        "auto" if self.kv_cache_dtype == "bf16" else self.kv_cache_dtype
                    ),
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
        elif use_triton_attn and self.rotary_emb is not None:
            k_scale = v_scale = self.per_tensor_scale
            self.per_token_quant = False
            q, k, _k_cache, _v_cache = fused_qk_rope_reshape_and_cache(
                q,
                k,
                v,
                new_key_cache,
                new_value_cache,
                slot_mapping,
                position,
                self.rotary_emb.cos_cache,
                self.rotary_emb.sin_cache,
                k_scale,
                v_scale,
                self.rotary_emb.is_neox_style,
                flash_layout=flash_layout,
                apply_scale=self.kv_cache_dtype.startswith("fp8"),
                offs=None,
                q_out=q,
                k_out=k,
                output_zeros=False,
            )
        else:
            # for asm paged attention
            asm_layout = True
            if use_triton_attn:
                asm_layout = False
            if self.rotary_emb is not None:
                assert position is not None
                q, k = self.rotary_emb(position, q, k)
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)
            new_value_cache = new_value_cache.view(
                num_blocks, num_kv_heads, head_size, block_size
            )
            if self.kv_cache_dtype == "fp8":
                aiter.reshape_and_cache_with_pertoken_quant(
                    k,
                    v,
                    new_key_cache,
                    new_value_cache,
                    k_scale,
                    v_scale,
                    slot_mapping,
                    asm_layout=asm_layout,
                )
            else:
                aiter.reshape_and_cache(
                    k,
                    v,
                    new_key_cache,
                    new_value_cache,
                    slot_mapping,
                    kv_cache_dtype="auto",
                    k_scale=None,
                    v_scale=None,
                    asm_layout=True,
                )

        return q, k, v, k_cache, v_cache, k_scale, v_scale

    def paged_attention_triton(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        num_decodes: int,
        out: torch.Tensor,
        attn_metadata: "AiterMhaMetadataForVllm",
        ps: bool = True,
    ):
        # q.shape[0] == num_decodes * max_query_len for MTP (one row per decode
        # token, query_len > 1). For non-MTP it equals num_decodes (query_len = 1).
        # pa_decode_gluon handles multi-token causal masking internally when
        # `query_length > 1` is passed; intermediate buffers must be sized
        # `num_decodes` (not q.shape[0]) and `query_group_size` must include
        # the max_qlen multiplier — mirroring server-mode `paged_attention_triton`.
        _, num_q_heads_total, head_size = q.shape
        num_blocks, num_kv_heads, _, block_size, _ = k_cache.shape
        decode_metadata = attn_metadata.decode_metadata
        max_qlen = decode_metadata.max_query_len if decode_metadata is not None else 1
        assert num_q_heads_total % num_kv_heads == 0

        seq_lens = attn_metadata.seq_lens[:num_decodes]
        block_tables = attn_metadata.block_table[:num_decodes]

        query_group_size = max_qlen * (num_q_heads_total // num_kv_heads)
        context_partition_size = 256

        use_ps = True
        if use_ps:
            max_context_partition_num = get_recommended_splits(
                num_decodes, num_kv_heads
            )
        else:
            max_context_partition_num = _NO_PS_FIXED_SPLITS

        if self.sliding_window > 0:
            max_context_partition_num = 1
            context_partition_size = 128

        intermediate_shape = (
            num_decodes,
            num_kv_heads,
            max_context_partition_num,
            query_group_size,
        )
        compute_type = (
            torch.bfloat16 if self.kv_cache_dtype == "bf16" else aiter.dtypes.fp8
        )
        exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device=q.device)
        max_logits = torch.empty(
            intermediate_shape, dtype=torch.float32, device=q.device
        )
        temporary_output = torch.empty(
            *intermediate_shape,
            head_size,
            dtype=q.dtype,
            device=q.device,
        )

        if k_scale is not None and k_scale.numel() > 1:
            k_scale = k_scale.unsqueeze(-1)
            v_scale = v_scale.unsqueeze(-1)

        # Kernel takes natural q layout [batch * query_length, num_q_heads, head_size].
        # Internally it derives batch_size = q.shape[0] // query_length and reshapes
        # to [batch, query_length, num_kv_heads, group, head_size]. See
        # aiter/aiter/ops/triton/gluon/pa_decode_gluon.py:5371-5377 and 5542-5544.
        run_pa_decode_gluon(
            output=out,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            context_lens=seq_lens,
            block_tables=block_tables,
            softmax_scale=self.scale,
            max_seqlen_q=max_qlen,  # query_length handles multi-token causal mask.
            max_context_partition_num=max_context_partition_num,
            context_partition_size=context_partition_size,
            compute_type=compute_type,
            q_scale=None,
            k_scale=None if self.kv_cache_dtype == "bf16" else k_scale,
            v_scale=None if self.kv_cache_dtype == "bf16" else v_scale,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=temporary_output,
            alibi_slopes=None,
            sinks=self.sinks,
            sliding_window=self.sliding_window,
            ps=use_ps,
        )
        return out

    def paged_attention_asm(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        num_decodes: int,
        attn_metadata: "AiterMhaMetadataForVllm",
        out: torch.Tensor,
    ):
        decode_metadata = attn_metadata.decode_metadata
        max_qlen = decode_metadata.max_query_len if decode_metadata is not None else 1
        qo_indptr = (
            decode_metadata.query_start_loc if decode_metadata is not None else None
        )
        run_pa_fwd_asm(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=attn_metadata.block_table[:num_decodes],
            context_lens=attn_metadata.seq_lens[:num_decodes],
            k_scale=k_scale,
            v_scale=v_scale,
            out=out,
            qo_indptr=qo_indptr,
            max_qlen=max_qlen,
            high_precision=0,
        )

        return

    def extend_for_sliding_window(
        self,
        attn_metadata: "AiterMhaMetadataForVllm",
        query: torch.Tensor,
        key_cache,
        value_cache,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        block_table: torch.Tensor,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
    ):
        assert attn_metadata.extend_metadata is not None
        assert attn_metadata.extend_metadata.chunk_context_metadata is not None
        chunked_metadata = attn_metadata.extend_metadata.chunk_context_metadata
        swa_metadata = chunked_metadata.swa_metadata
        assert swa_metadata is not None
        swa_cu_seqlens = swa_metadata.swa_cu_seqlens
        swa_seq_starts = swa_metadata.swa_seq_starts
        swa_token_to_batch = swa_metadata.swa_token_to_batch
        swa_max_seqlens = swa_metadata.swa_max_seqlens
        swa_total_tokens = swa_metadata.swa_total_tokens
        key_fetched, value_fetched = (
            swa_metadata.swa_workspace[0],
            swa_metadata.swa_workspace[1],
        )

        cp_mha_gather_cache(
            key_cache=key_cache,
            value_cache=value_cache,
            key=key_fetched,
            value=value_fetched,
            block_tables=block_table,
            k_scales=k_scale,
            v_scales=v_scale,
            cu_seqlens_kv=swa_cu_seqlens,
            token_to_batch=swa_token_to_batch,
            seq_starts=swa_seq_starts,
            dequant=self.kv_cache_dtype.startswith("fp8"),
            kv_cache_layout="SHUFFLE",
            total_tokens=swa_total_tokens,
            per_token_quant=self.per_token_quant,
        )

        sliding_window = (
            (self.sliding_window, 0, 0)
            if self.sliding_window is not None
            else (-1, -1, 0)
        )
        aiter.flash_attn_varlen_func(
            q=query,
            k=key_fetched,
            v=value_fetched,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=swa_cu_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=swa_max_seqlens,
            min_seqlen_q=1,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            window_size=sliding_window,
            alibi_slopes=self.alibi_slopes,
            sink_ptr=self.sinks,
            return_lse=False,
            out=output,
        )

    def extend_forward(
        self,
        attn_metadata: "AiterMhaMetadataForVllm",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        min_seqlen_q: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
    ):
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

        if self.sliding_window != -1:
            self.extend_for_sliding_window(
                attn_metadata,
                query,
                key_cache,
                value_cache,
                output,
                cu_seqlens_q,
                max_seqlen_q,
                block_table,
                k_scale,
                v_scale,
            )
            return
        out, lse = aiter.flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            min_seqlen_q=min_seqlen_q,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            sink_ptr=self.sinks,
            alibi_slopes=self.alibi_slopes,
            return_lse=True,
        )
        assert attn_metadata.extend_metadata is not None
        chunk_context_metadata = attn_metadata.extend_metadata.chunk_context_metadata
        num_chunks = chunk_context_metadata.num_chunks
        workspace = chunk_context_metadata.workspace
        cu_seqlens_kv = chunk_context_metadata.cu_seq_lens_chunk
        max_seqlens = chunk_context_metadata.max_seq_lens
        chunk_starts = chunk_context_metadata.chunk_starts
        token_to_batch = chunk_context_metadata.token_to_batch
        total_token_per_batch = chunk_context_metadata.total_token_per_batch
        key_fetched, value_fetched = workspace[0], workspace[1]
        chunked_output = None
        chunked_lse = None
        for chunk_idx in range(num_chunks):
            cp_mha_gather_cache(
                key_cache=key_cache,
                value_cache=value_cache,
                key=key_fetched,
                value=value_fetched,
                block_tables=block_table,
                k_scales=k_scale,
                v_scales=v_scale,
                cu_seqlens_kv=cu_seqlens_kv[chunk_idx],
                token_to_batch=token_to_batch[chunk_idx],
                seq_starts=chunk_starts[chunk_idx],
                dequant=self.kv_cache_dtype.startswith("fp8"),
                kv_cache_layout="SHUFFLE",
                total_tokens=total_token_per_batch[chunk_idx],
                per_token_quant=self.per_token_quant,
            )

            suf_out, suf_lse = aiter.flash_attn_varlen_func(
                q=query,
                k=key_fetched,
                v=value_fetched,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv[chunk_idx],
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlens[chunk_idx],
                min_seqlen_q=min_seqlen_q,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
                window_size=(-1, -1, 0),
                sink_ptr=self.sinks,
                alibi_slopes=self.alibi_slopes,
                return_lse=True,
            )

            if chunked_output is None:
                chunked_output = suf_out
                chunked_lse = suf_lse
            else:
                tmp_output = torch.empty_like(out)
                tmp_lse = torch.empty_like(lse)
                merge_attn_states(
                    output=tmp_output,
                    output_lse=tmp_lse,
                    prefix_output=chunked_output,
                    prefix_lse=chunked_lse,
                    suffix_output=suf_out,
                    suffix_lse=suf_lse,
                )
                chunked_output = tmp_output
                chunked_lse = tmp_lse

        merge_attn_states(
            output=output,
            prefix_output=chunked_output,
            prefix_lse=chunked_lse,
            suffix_output=out,
            suffix_lse=lse,
        )

    def _dispatch_decode_backend(self, num_decodes):
        if self.model_type in _MINIMAX_M3_MODEL_TYPES:
            # The ASM paged-attention decode path can misaddress high physical
            # KV block ids seen by MiniMax-M3 after sustained serving.
            return self.paged_attention_triton

        # use asm pa for models without setting gluon pa decode bs
        gluon_pa_decode_bs = _GLUON_PA_DECODE_BS_MAPPING.get(self.model_type, -1)
        if self.use_triton_attn:
            return self.paged_attention_triton
        else:
            if ATOM_USE_GLUON_PA_DECODE and num_decodes <= gluon_pa_decode_bs:
                return self.paged_attention_triton
            else:
                return self.paged_attention_asm

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AiterMhaMetadataForVllm" = None,
        position: torch.Tensor = None,
        q_scale: torch.Tensor = None,
        qkv: torch.Tensor = None,
        output: torch.Tensor = None,
    ):
        # create the output here, it use query shape
        num_tokens = query.shape[0]
        output_dtype = query.dtype
        output_shape = torch.Size((num_tokens, self.num_heads * self.head_size))
        output = torch.empty(output_shape, dtype=output_dtype, device=query.device)

        # dummy run will skip attention in cuda graph capture phase
        if attn_metadata is None:
            return output.fill_(0)

        # vLLM's compiled unified_attention custom op does not pass positions into
        # impl.forward. ATOMModelBase stashes them on vLLM ForwardContext as
        # additional_kwargs["atom_positions"] (see atom/plugin/vllm/model_wrapper.py).
        if position is None:
            from vllm.forward_context import (
                get_forward_context as get_vllm_forward_context,
                is_forward_context_available,
            )

            if is_forward_context_available():
                position = get_vllm_forward_context().additional_kwargs.get(
                    "atom_positions"
                )
        if position is None:
            sfc = get_current_atom_config().compilation_config.static_forward_context
            position = sfc.get("positions")
        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_kv_heads, self.head_dim)
        value = value.view(-1, self.num_kv_heads, self.head_dim)
        output = output.view(-1, self.num_heads, self.head_dim)

        num_actual_tokens = attn_metadata.num_actual_tokens
        k_cache, v_cache = kv_cache.unbind(0)
        num_blocks, block_size, num_kv_heads, _ = k_cache.shape

        if self.kv_cache_dtype == "fp8":
            target_dtype = dtypes.d_dtypes[self.kv_cache_dtype]
            k_cache = k_cache.view(target_dtype)
            v_cache = v_cache.view(target_dtype)

        # create kv scale according to the num_blocks
        # usually it is created when cuda graph capture for decode phase
        if self.kv_cache_dtype == "fp8":
            if self.k_scale is None or self.v_scale is None:
                # origin kv_scale is per tensor scale of value one.
                self.per_tensor_scale = self.kv_scale
                self.kv_scale = torch.zeros(
                    2,
                    num_blocks,
                    num_kv_heads,
                    block_size,
                    dtype=dtypes.fp32,
                    device=self.device,
                )
                self.k_scale = self.kv_scale[0]
                self.v_scale = self.kv_scale[1]

        # as vLLM cuda graph capture padding mechanism, here split the qkvo with
        # the actual tokens
        query = query[:num_actual_tokens]
        # vLLM can call plugin attention without fused qkv/position tensors for
        # some dense-model paths (for example Llama). Slice them only when present.
        if qkv is not None:
            qkv = qkv[:num_actual_tokens]
        if position is not None:
            position = position[:num_actual_tokens]
        if key is not None:
            key = key[:num_actual_tokens]
        if value is not None:
            value = value[:num_actual_tokens]
        output_actual_tokens = output[:num_actual_tokens]
        # rope and cache flush fusion. ATOM always use shuffle layout for kv cache
        result = self.rope_cache(
            q=query,
            k=key,
            v=value,
            qkv=qkv,
            position=position,
            attention_metadata=attn_metadata,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            flash_layout=False,
        )
        query, key, value, k_cache, v_cache, k_scale, v_scale = result

        num_decodes = attn_metadata.num_decodes
        num_prefills = attn_metadata.num_prefills
        num_extends = attn_metadata.num_extends

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_extend_tokens = attn_metadata.num_extend_tokens

        num_blocks, block_size, num_kv_heads, head_size = k_cache.shape
        x = 16 // k_cache.element_size()
        new_key_cache = k_cache.view(
            num_blocks, num_kv_heads, head_size // x, block_size, x
        )
        new_value_cache = v_cache.view(
            num_blocks, num_kv_heads, block_size // x, head_size, x
        )
        # calculate for prefills
        if num_prefills > 0:
            assert attn_metadata.prefill_metadata is not None

            # prefill part is after decode and extend
            prefill_query = query[num_decode_tokens + num_extend_tokens :]
            prefill_key = key[num_decode_tokens + num_extend_tokens :]
            prefill_value = value[num_decode_tokens + num_extend_tokens :]

            sliding_window = (
                (self.sliding_window, 0, 0)
                if self.sliding_window is not None
                else (-1, -1, 0)
            )

            aiter.flash_attn_varlen_func(
                q=prefill_query,
                k=prefill_key,
                v=prefill_value,
                cu_seqlens_q=attn_metadata.prefill_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.prefill_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.prefill_metadata.max_query_len,
                max_seqlen_k=attn_metadata.prefill_metadata.max_seq_len,
                min_seqlen_q=1,
                dropout_p=attn_metadata.dropout_p,
                softmax_scale=self.scale,
                causal=True,
                window_size=sliding_window,
                alibi_slopes=None,
                sink_ptr=self.sinks,
                out=output_actual_tokens[num_decode_tokens + num_extend_tokens :],
            )

        # calculate for extends
        if num_extends > 0:
            assert attn_metadata.extend_metadata is not None
            extend_tokens_slice = slice(
                num_decode_tokens, num_decode_tokens + num_extend_tokens
            )
            extend_reqs_slice = slice(num_decodes, num_decodes + num_extends)
            extend_querys = query[extend_tokens_slice]
            extend_keys = key[extend_tokens_slice]
            extend_values = value[extend_tokens_slice]
            extend_outputs = output[extend_tokens_slice]
            extend_block_table = attn_metadata.block_table[extend_reqs_slice]
            extend_slot_mapping = attn_metadata.slot_mapping[extend_tokens_slice]
            self.extend_forward(
                attn_metadata=attn_metadata,
                query=extend_querys,
                key=extend_keys,
                value=extend_values,
                key_cache=new_key_cache,
                value_cache=new_value_cache,
                output=extend_outputs,
                cu_seqlens_q=attn_metadata.extend_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.extend_metadata.max_query_len,
                max_seqlen_k=attn_metadata.extend_metadata.max_seq_len,
                min_seqlen_q=1,
                block_table=extend_block_table,
                slot_mapping=extend_slot_mapping,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        # calculate for decodes
        if num_decodes > 0:
            assert attn_metadata.decode_metadata is not None

            decode_backend_func = self._dispatch_decode_backend(num_decodes)
            decode_backend_func(
                q=query[:num_decode_tokens],
                k_cache=new_key_cache,
                v_cache=new_value_cache,
                k_scale=k_scale,
                v_scale=v_scale,
                num_decodes=num_decodes,
                out=output_actual_tokens[:num_decode_tokens],
                attn_metadata=attn_metadata,
            )

        output = output.view(-1, self.num_heads * self.head_dim)

        return output

    def get_kv_cache_spec(self, vllm_config):
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        assert self.attn_type == AttentionType.DECODER
        block_size = vllm_config.cache_config.block_size
        if self.sliding_window is not None:
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_torch_dtype,
                sliding_window=self.sliding_window,
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
        )


class SparseMHAIndexerCache(nn.Module, AttentionLayerBase):
    """Key-only index cache owned by MiniMax-M3 sparse attention."""

    def __init__(
        self,
        *,
        layer_name: str,
        head_dim: int,
        kv_cache_dtype: str,
    ) -> None:
        from vllm.v1.attention.backend import AttentionType
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

        super().__init__()
        atom_config = get_current_atom_config()
        vllm_config = atom_config.plugin_config.vllm_config
        self.layer_name = layer_name
        self.prefix = layer_name
        self.attn_type = AttentionType.DECODER
        self.attn_backend = SparseMHAIndexerBackend
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )
        self.num_kv_heads = 1
        self.head_size = head_dim
        self.head_size_v = head_dim
        self.sliding_window = -1
        self.kv_cache = torch.tensor([])
        _register_vllm_static_forward_context(self)

    @property
    def impl(self):
        return self

    def get_attn_backend(self):
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config):
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
        )


AttentionLayerBase.register(SparseMHAIndexerCache)


class SparseMHAPagedAttentionImplForVllm(
    AttentionForVllmMHA, AtomSparseMHAPagedAttentionImpl
):
    """MiniMax-M3 sparse MHA adapter for vLLM plugin mode."""

    def __init__(
        self,
        *args,
        index_q_norm: Optional[torch.nn.Module] = None,
        index_k_norm: Optional[torch.nn.Module] = None,
        index_rotary_emb: Optional[torch.nn.Module] = None,
        index_q_size: int = 0,
        index_head_dim: int = 0,
        topk: int = 0,
        init_blocks: int = 0,
        local_blocks: int = 0,
        skip_index_topk: bool = False,
        sparse_layer_ordinal: int = -1,
        impl_cls=None,
        **kwargs,
    ) -> None:
        del impl_cls
        AttentionForVllmMHA.__init__(self, *args, **kwargs)
        if self.head_dim != 128:
            raise ValueError("MiniMax-M3 sparse attention requires head_dim == 128.")
        if index_q_norm is None or index_k_norm is None:
            raise ValueError("MiniMax-M3 sparse attention requires index norms.")
        if index_head_dim <= 0 or index_q_size <= 0:
            raise ValueError("MiniMax-M3 sparse attention requires index dimensions.")

        self.attn_backend = SparseMHAPagedAttentionBackend
        self.index_q_norm = index_q_norm
        self.index_k_norm = index_k_norm
        self.index_rotary_emb = (
            index_rotary_emb if index_rotary_emb is not None else self.rotary_emb
        )
        self.index_q_size = index_q_size
        self.index_head_dim = index_head_dim
        self.num_idx_heads = self.num_kv_heads
        self.topk = topk
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.skip_index_topk = skip_index_topk
        self.sparse_layer_ordinal = sparse_layer_ordinal
        self.index_cache: Optional[torch.Tensor] = None
        self.index_topk_cache_state: Optional[dict] = None
        self._index_q_cache_key_info: Optional[tuple] = None
        self._index_q: Optional[torch.Tensor] = None

        index_prefix = f"{self.layer_name}.index_cache"
        static_forward_context = (
            get_current_atom_config().plugin_config.vllm_config.compilation_config.static_forward_context
        )
        existing_index_layer = static_forward_context.get(index_prefix)
        if existing_index_layer is None:
            self.index_cache_layer = SparseMHAIndexerCache(
                layer_name=index_prefix,
                head_dim=self.index_head_dim,
                kv_cache_dtype="auto",
            )
        elif isinstance(existing_index_layer, SparseMHAIndexerCache):
            self.index_cache_layer = existing_index_layer
        else:
            raise ValueError(f"Duplicate layer name: {index_prefix}")

    def _page16_shuffle_kv_cache(self, kv_cache: torch.Tensor):
        if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
            raise ValueError(
                "MiniMax-M3 sparse adapter expects vLLM KV cache shape "
                "(2, num_blocks, block_size, num_kv_heads, head_dim)."
            )
        _kv, num_blocks, block_size, num_kv_heads, head_dim = kv_cache.shape
        if block_size != SPARSE_BLOCK_SIZE:
            raise ValueError(
                f"MiniMax-M3 sparse block size must be {SPARSE_BLOCK_SIZE}."
            )
        if num_kv_heads != self.num_kv_heads or head_dim != self.head_dim:
            raise ValueError(
                "MiniMax-M3 sparse KV cache shape does not match layer config: "
                f"cache=({num_kv_heads}, {head_dim}) "
                f"layer=({self.num_kv_heads}, {self.head_dim})."
            )
        key_cache, value_cache = kv_cache.unbind(0)
        if self.kv_cache_dtype == "fp8":
            target_dtype = dtypes.d_dtypes[self.kv_cache_dtype]
            key_cache = key_cache.view(target_dtype)
            value_cache = value_cache.view(target_dtype)
            if self.k_scale is None or self.v_scale is None:
                self.kv_scale = torch.zeros(
                    2,
                    num_blocks,
                    num_kv_heads,
                    block_size,
                    dtype=dtypes.fp32,
                    device=kv_cache.device,
                )
                self.k_scale = self.kv_scale[0]
                self.v_scale = self.kv_scale[1]

        x = 16 // key_cache.element_size()
        key_cache = key_cache.view(
            num_blocks, num_kv_heads, head_dim // x, block_size, x
        )
        value_cache = value_cache.view(
            num_blocks, num_kv_heads, block_size // x, head_dim, x
        )
        k_scale = v_scale = None
        if self.k_scale is not None and self.v_scale is not None:
            k_scale = self.k_scale.view(num_blocks, num_kv_heads, block_size)
            v_scale = self.v_scale.view(num_blocks, num_kv_heads, block_size)
        return self._to_page16_shuffle(
            key_cache,
            value_cache,
            k_scale,
            v_scale,
        )

    def _rope_index_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qkv: torch.Tensor,
        position: torch.Tensor,
        main_metadata,
        index_metadata,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
    ):
        del q, k, v
        if qkv is None:
            raise ValueError("MiniMax-M3 sparse adapter requires packed qkv input.")
        index_cache = self.index_cache_layer.kv_cache
        if index_cache.numel() == 0:
            raise RuntimeError("MiniMax-M3 index cache is not bound by vLLM.")

        qkv = qkv.contiguous()
        num_tokens = qkv.shape[0]
        q_out = torch.empty(
            (num_tokens, self.num_heads * self.head_dim),
            dtype=qkv.dtype,
            device=qkv.device,
        )
        index_q = torch.empty(
            (num_tokens, self.index_q_size), dtype=qkv.dtype, device=qkv.device
        )
        from atom.models.minimax_m3 import _minimax_m3_cos_sin_cache

        cos_sin_cache = _minimax_m3_cos_sin_cache(self.rotary_emb, qkv)
        is_fp8 = self.kv_cache_dtype == "fp8"
        kv_cache_dtype = self.kv_cache_dtype if is_fp8 else "auto"
        aiter.fused_qknorm_idxrqknorm(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin_cache,
            position,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
            self.num_idx_heads,
            slot_mapping=main_metadata.slot_mapping,
            kv_cache_k=k_cache,
            kv_cache_v=v_cache,
            index_cache=index_cache,
            block_size=k_cache.shape[3],
            q_out=q_out,
            index_q_out=index_q,
            index_slot_mapping=index_metadata.slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale if is_fp8 else None,
            v_scale=v_scale if is_fp8 else None,
            asm_layout=True,
        )
        self._index_q = index_q.view(-1, self.num_idx_heads, self.index_head_dim)
        q_view = q_out.view(-1, self.num_heads, self.head_dim)
        return q_view

    def _metadata_pair(self, main_metadata):
        index_metadata = None
        try:
            from vllm.forward_context import (
                get_forward_context as get_vllm_forward_context,
                is_forward_context_available,
            )

            if is_forward_context_available():
                all_metadata = get_vllm_forward_context().attn_metadata
                if isinstance(all_metadata, dict):
                    main_metadata = all_metadata.get(self.layer_name, main_metadata)
                    index_metadata = all_metadata.get(self.index_cache_layer.layer_name)
        except Exception:
            index_metadata = None
        if index_metadata is None:
            index_metadata = main_metadata
        return main_metadata, index_metadata

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata=None,
        position: torch.Tensor = None,
        q_scale: Optional[torch.Tensor] = None,
        qkv: torch.Tensor = None,
        output: torch.Tensor = None,
    ) -> torch.Tensor:
        del q_scale
        num_tokens = query.shape[0]
        if output is None:
            output = torch.empty(
                (num_tokens, self.num_heads * self.head_dim),
                dtype=query.dtype,
                device=query.device,
            )
        if attn_metadata is None:
            return output.fill_(0)
        if position is None:
            from vllm.forward_context import (
                get_forward_context as get_vllm_forward_context,
                is_forward_context_available,
            )

            if is_forward_context_available():
                position = get_vllm_forward_context().additional_kwargs.get(
                    "atom_positions"
                )
        if position is None:
            sfc = get_current_atom_config().compilation_config.static_forward_context
            position = sfc.get("positions")
        if position is None:
            raise RuntimeError("MiniMax-M3 sparse attention requires positions.")

        main_metadata, index_metadata = self._metadata_pair(attn_metadata)

        k_cache, v_cache, k_scale, v_scale = self._page16_shuffle_kv_cache(kv_cache)
        q = self._rope_index_cache(
            query,
            key,
            value,
            qkv,
            position,
            main_metadata,
            index_metadata,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
        )
        index_q = self._index_q
        assert index_q is not None

        actual_tokens = min(
            getattr(main_metadata, "num_actual_tokens", q.shape[0]), q.shape[0]
        )
        num_prefill_tokens = main_metadata.num_prefill_tokens
        output_view = output.view(-1, self.num_heads, self.head_dim)
        self.index_cache = self.index_cache_layer.kv_cache
        fwd_ctx = SimpleNamespace(attn_metadata=main_metadata)

        q_actual = q[:actual_tokens]
        key_actual = key[:actual_tokens] if key is not None else key
        value_actual = value[:actual_tokens] if value is not None else value
        self._index_q = index_q[:actual_tokens]
        output_actual = output_view[:actual_tokens]
        if actual_tokens < output_view.shape[0]:
            output_view[actual_tokens:].zero_()

        if actual_tokens == 0:
            pass
        elif num_prefill_tokens > 0:
            AtomSparseMHAPagedAttentionImpl._sparse_prefill(
                self,
                q_actual,
                key_actual,
                value_actual,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                fwd_ctx,
                output=output_actual,
            )
        elif main_metadata.num_decodes > 0:
            AtomSparseMHAPagedAttentionImpl._sparse_decode(
                self,
                q_actual,
                key_actual,
                value_actual,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                fwd_ctx,
                output=output_actual,
            )

        self._index_q = None
        return output.view(-1, self.num_heads * self.head_dim)


AttentionLayerBase.register(SparseMHAPagedAttentionImplForVllm)
