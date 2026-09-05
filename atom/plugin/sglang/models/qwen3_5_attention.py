"""Native ATOM attention bridge for Qwen3.5 in SGLang plugin mode."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
    is_in_tc_piecewise_cuda_graph,
)

from atom.config import KVCacheTensor, get_current_atom_config
from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.model_ops.base_attention import BaseAttention, LinearAttention
from atom.plugin.sglang.patches.prefill_compile_only_patch import (
    is_compile_only_prefill_active,
)
from atom.utils import envs, mark_spliting_op


def _use_stable_piecewise_output() -> bool:
    """Return whether real prefill tc_piecewise requires a stable output."""

    # Compile-only prefill enters the tc_piecewise replay session solely to
    # dispatch its compiled callable; it does not capture/replay CUDA Graphs.
    # The explicit mode check prevents that shared session flag from selecting
    # graph-stable mutating ops when ordinary returning ops are safe.
    return is_in_tc_piecewise_cuda_graph() and not is_compile_only_prefill_active()


def _resolve_tc_piecewise_real_tokens(
    padded_num_tokens: int,
) -> tuple[Any | None, int]:
    """Return the active prefill batch and its unpadded token count."""

    context = get_tc_piecewise_forward_context()
    forward_batch = getattr(context, "forward_batch", None)
    if forward_batch is None:
        return None, padded_num_tokens

    real_num_tokens = getattr(context, "raw_num_tokens", None)
    if real_num_tokens is None:
        real_num_tokens = getattr(forward_batch, "num_token_non_padded_cpu", None)
    if real_num_tokens is None:
        return forward_batch, padded_num_tokens

    real_num_tokens = max(0, min(int(real_num_tokens), padded_num_tokens))
    return forward_batch, real_num_tokens


def _slice_padded_token_tensor(
    tensor: torch.Tensor | None,
    real_num_tokens: int,
    padded_num_tokens: int,
) -> torch.Tensor | None:
    """Narrow tensors whose leading dimension is the padded token axis."""

    if tensor is not None and tensor.ndim > 0 and tensor.shape[0] == padded_num_tokens:
        return tensor[:real_num_tokens]
    return tensor


def _get_attention_layer(layer_name: str) -> Any:
    """Resolve an ATOM attention layer registered for compiled execution."""

    return get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]


def _attention_with_stable_output_fake(
    query: torch.Tensor,
    q_scale: torch.Tensor | None,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor | None,
    layer_name: str,
    qkv: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    return None


@mark_spliting_op(
    is_custom=True,
    gen_fake=_attention_with_stable_output_fake,
    mutates_args=["output"],
)
def sglang_qwen35_attention_with_stable_output(
    query: torch.Tensor,
    q_scale: torch.Tensor | None,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor | None,
    layer_name: str,
    qkv: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    """Run MHA at a real prefill tc_piecewise CUDA Graph split boundary.

    The preallocated output keeps the following captured segment's input
    address stable. Compile-only prefill and decode use native returning ops.
    """

    padded_num_tokens = query.shape[0]
    forward_batch, real_num_tokens = _resolve_tc_piecewise_real_tokens(
        padded_num_tokens
    )
    attention = _get_attention_layer(layer_name)
    original_out_cache_loc = (
        getattr(forward_batch, "out_cache_loc", None)
        if forward_batch is not None
        else None
    )
    if original_out_cache_loc is not None:
        forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]
    try:
        result = attention.impl.forward(
            query=query[:real_num_tokens],
            key=key[:real_num_tokens],
            value=value[:real_num_tokens],
            position=_slice_padded_token_tensor(
                positions, real_num_tokens, padded_num_tokens
            ),
            q_scale=_slice_padded_token_tensor(
                q_scale, real_num_tokens, padded_num_tokens
            ),
            qkv=_slice_padded_token_tensor(qkv, real_num_tokens, padded_num_tokens),
        )
    finally:
        if original_out_cache_loc is not None:
            forward_batch.out_cache_loc = original_out_cache_loc

    output[:real_num_tokens].copy_(result)
    output[real_num_tokens:].zero_()


def _linear_attention_with_stable_output_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


@mark_spliting_op(
    is_custom=True,
    gen_fake=_linear_attention_with_stable_output_fake,
    mutates_args=["output"],
)
def sglang_qwen35_linear_attention_with_stable_output(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Run GDN at a real prefill tc_piecewise CUDA Graph split boundary.

    Writing into the preallocated output keeps the following captured segment's
    input address stable. Compile-only prefill and decode use returning ops.
    """

    padded_num_tokens = mixed_qkv.shape[0]
    _, real_num_tokens = _resolve_tc_piecewise_real_tokens(padded_num_tokens)
    attention = _get_attention_layer(layer_name)
    attention.impl.forward(
        mixed_qkv[:real_num_tokens],
        b[:real_num_tokens],
        a[:real_num_tokens],
        output[:real_num_tokens],
        layer_name,
    )
    output[real_num_tokens:].zero_()


def _use_sglang_asm_cache_layout(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    use_triton_attn: bool,
) -> bool:
    """Select SHUFFLE only when SGLang's external cache can represent it."""
    if v_cache.dim() == 5:
        return True
    x = 16 // k_cache.element_size()
    page_size = int(v_cache.shape[-1])
    return not use_triton_attn and page_size >= x and page_size % x == 0


class SGLangATOMQwen35PagedAttentionImpl(PagedAttentionImpl):
    """Paged attention policy for SGLang-owned KV cache storage."""

    def _use_asm_cache_layout(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        use_triton_attn: bool,
    ) -> bool:
        # ASM V-cache groups tokens in x-wide lanes. SGLang's 4-D page-size=1
        # pool cannot represent [block, head, page/x, dim, x], so it must keep
        # the standard [block, head, dim, page] layout.
        return _use_sglang_asm_cache_layout(
            k_cache, v_cache, use_triton_attn=use_triton_attn
        )


class SGLangATOMQwen35Attention(BaseAttention):
    """ATOM's native dense-MHA frontend with SGLang-owned cache storage."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        kv_cache_dtype: str = "bf16",
        layer_num: int = 0,
        use_mla: bool = False,
        mla_modules: Any = None,
        sinks: Any = None,
        per_layer_sliding_window: int | None = None,
        rotary_emb: Any = None,
        prefix: str | None = None,
        q_norm: Any = None,
        k_norm: Any = None,
        **kwargs: Any,
    ) -> None:
        if use_mla:
            raise ValueError("Qwen3.5 full attention must use dense MHA")
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=False,
            mla_modules=mla_modules,
            sinks=sinks,
            per_layer_sliding_window=per_layer_sliding_window,
            rotary_emb=rotary_emb,
            prefix=prefix,
            q_norm=q_norm,
            k_norm=k_norm,
            **kwargs,
        )

        atom_config = get_current_atom_config()
        cache_dtype = "fp8" if str(kv_cache_dtype).startswith("fp8") else kv_cache_dtype
        self.use_mla = False
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.scale = scale
        self.num_kv_heads = int(num_kv_heads)
        self.kv_cache_dtype = cache_dtype
        self.layer_num = int(layer_num)
        self.output_dtype = atom_config.torch_dtype
        # SGLang's layer discovery and graph setup expect this public name.
        self.layer_id = int(self.layer_num)
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = None
        self.impl = SGLangATOMQwen35PagedAttentionImpl(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            kv_cache_dtype=cache_dtype,
            layer_num=layer_num,
            mla_modules=mla_modules,
            sinks=sinks,
            sliding_window=per_layer_sliding_window,
            rotary_emb=rotary_emb,
            dtype=atom_config.torch_dtype,
            q_norm=q_norm,
            k_norm=k_norm,
            **kwargs,
        )
        self.layer_name = prefix if prefix is not None else f"MHA_{layer_num}"
        static_context = atom_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer: {self.layer_name}")
        static_context[self.layer_name] = self

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor | None = None,
        q_scale: torch.Tensor | None = None,
        qkv: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if _use_stable_piecewise_output():
            # This allocation belongs to the preceding compiled segment. The
            # split op mutates it in place, so the following CUDA graph always
            # consumes the same address used during capture.
            output = torch.empty(
                query.shape,
                dtype=self.output_dtype,
                device=query.device,
            )
            torch.ops.aiter.sglang_qwen35_attention_with_stable_output(
                query,
                q_scale,
                key,
                value,
                positions,
                self.layer_name,
                qkv,
                output,
            )
            return output

        return torch.ops.aiter.unified_attention_with_output_base(
            query,
            q_scale,
            key,
            value,
            positions,
            self.layer_name,
            False,
            qkv,
        )


class SGLangATOMQwen35LinearAttention(LinearAttention):
    """ATOM GDN frontend with stable output for real prefill tc_piecewise."""

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> torch.Tensor:
        if _use_stable_piecewise_output():
            torch.ops.aiter.sglang_qwen35_linear_attention_with_stable_output(
                mixed_qkv,
                b,
                a,
                core_attn_out,
                self.layer_name,
            )
            return core_attn_out

        return torch.ops.aiter.linear_attention_with_output_base(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.layer_name,
        )


@contextmanager
def qwen35_native_attention_construction() -> Iterator[None]:
    """Construct Qwen3.5 full-attention layers with native ATOM MHA."""

    from atom.model_ops import base_attention
    from atom.models import qwen3_next

    previous_attention = base_attention.Attention
    previous_base_linear_attention = base_attention.LinearAttention
    previous_qwen_linear_attention = qwen3_next.LinearAttention
    base_attention.Attention = SGLangATOMQwen35Attention
    base_attention.LinearAttention = SGLangATOMQwen35LinearAttention
    qwen3_next.LinearAttention = SGLangATOMQwen35LinearAttention
    try:
        yield
    finally:
        base_attention.Attention = previous_attention
        base_attention.LinearAttention = previous_base_linear_attention
        qwen3_next.LinearAttention = previous_qwen_linear_attention


def _full_attention_backend(forward_batch: Any) -> Any:
    from atom.plugin.sglang.attention_backend.backend_resolver import (
        resolve_attn_backend,
    )

    backend = resolve_attn_backend(forward_batch)
    return getattr(backend, "full_attn_backend", backend)


def _mha_pools(forward_batch: Any) -> tuple[Any, Any]:
    backend = _full_attention_backend(forward_batch)
    token_pool = getattr(forward_batch, "token_to_kv_pool", None) or getattr(
        backend, "token_to_kv_pool", None
    )
    req_pool = getattr(forward_batch, "req_to_token_pool", None) or getattr(
        backend, "req_to_token_pool", None
    )
    if token_pool is None or req_pool is None:
        raise RuntimeError("Qwen3.5 SGLang full-attention pools are unavailable")
    return token_pool, req_pool


def _native_attention_layers() -> list[SGLangATOMQwen35Attention]:
    static_context = get_current_atom_config().compilation_config.static_forward_context
    return [
        layer
        for layer in static_context.values()
        if isinstance(layer, SGLangATOMQwen35Attention)
    ]


def _fp8_cache_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


@dataclass(frozen=True)
class _Qwen35CacheViewBinding:
    token_pool: Any
    linear_backend: Any
    mamba_pool: Any
    mamba_map: Any
    kv_cache_data: dict[str, KVCacheTensor]


def bind_qwen35_cache_views(forward_batch: Any) -> dict[str, KVCacheTensor]:
    """Bind SGLang storage as ATOM-compatible K/V cache views.

    The K view is vectorized as ``[block, head, dim/x, page, x]`` for both
    layouts. When the page is x-aligned, V uses Native ATOM's SHUFFLE view
    ``[block, head, page/x, dim, x]``. Smaller pages use the standard NHD view
    ``[block, head, dim, page]``. All views are zero-copy aliases of SGLang's
    token pool.
    """

    token_pool, _ = _mha_pools(forward_batch)
    page_size = int(getattr(token_pool, "page_size", 1))
    kv_cache_data: dict[str, KVCacheTensor] = {}

    for attention in _native_attention_layers():
        layer_id = int(attention.layer_num)
        k_buffer, v_buffer = token_pool.get_kv_buffer(layer_id)
        num_slots, num_kv_heads, head_dim = k_buffer.shape
        if num_slots % page_size != 0:
            raise RuntimeError(
                f"Qwen3.5 KV slots ({num_slots}) are not divisible by page_size "
                f"({page_size})"
            )
        num_blocks = num_slots // page_size
        x = 16 // k_buffer.element_size()
        k_cache = k_buffer.view(num_blocks, num_kv_heads, head_dim // x, page_size, x)
        if page_size >= x and page_size % x == 0:
            v_cache = v_buffer.view(
                num_blocks, num_kv_heads, page_size // x, head_dim, x
            )
        else:
            v_cache = v_buffer.view(num_blocks, num_kv_heads, head_dim, page_size)
        k_scale = v_scale = None
        if _fp8_cache_dtype(k_buffer.dtype):
            scale_shape = (num_blocks, num_kv_heads, page_size)
            for name in ("k_scale", "v_scale"):
                scale = getattr(attention, name, None)
                if (
                    scale is None
                    or tuple(scale.shape) != scale_shape
                    or scale.device != k_buffer.device
                ):
                    scale = torch.ones(
                        scale_shape, dtype=torch.float32, device=k_buffer.device
                    )
                    setattr(attention, name, scale)
            k_scale, v_scale = attention.k_scale, attention.v_scale

        attention.k_cache = k_cache
        attention.v_cache = v_cache
        attention.impl.use_flash_layout = False
        kv_cache_data[f"layer_{layer_id}"] = KVCacheTensor(
            layer_num=layer_id,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    if not kv_cache_data:
        raise RuntimeError("Qwen3.5 native ATOM full-attention layers were not found")
    return kv_cache_data


def _cpu_lens(forward_batch: Any, name: str, fallback: torch.Tensor) -> list[int]:
    value = getattr(forward_batch, name, None)
    if value is None:
        value = fallback.detach().cpu()
    if torch.is_tensor(value):
        value = value.tolist()
    return [int(item) for item in value]


def _block_tables(
    forward_batch: Any,
    req_pool: Any,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    max_seq_len: int,
    extend_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = int(forward_batch.batch_size)
    max_blocks = max(1, (max_seq_len + page_size - 1) // page_size)
    token_table = req_pool.req_to_token[
        forward_batch.req_pool_indices[:batch_size], : max_blocks * page_size
    ].clone()
    if extend_lens is not None:
        extend_lens = extend_lens[:batch_size].to(
            device=token_table.device, dtype=torch.long
        )
        prefix_lens = (
            seq_lens[:batch_size].to(device=token_table.device, dtype=torch.long)
            - extend_lens
        )
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if torch.is_tensor(out_cache_loc) and out_cache_loc.numel() > 0:
            columns = torch.arange(token_table.shape[1], device=token_table.device)
            relative_positions = columns.unsqueeze(0) - prefix_lens.unsqueeze(1)
            extend_mask = (relative_positions >= 0) & (
                relative_positions < extend_lens.unsqueeze(1)
            )
            query_offsets = torch.cumsum(extend_lens, dim=0) - extend_lens
            source_indices = query_offsets.unsqueeze(1) + relative_positions
            source_indices = source_indices.clamp(
                min=0, max=int(out_cache_loc.numel()) - 1
            ).to(torch.long)
            source_slots = out_cache_loc.gather(0, source_indices.reshape(-1)).view_as(
                source_indices
            )
            token_table = torch.where(extend_mask, source_slots, token_table)
    return token_table[:, ::page_size].to(dtype=torch.int32).contiguous() // page_size


def _frontier_slot_mapping(
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Map each live decode row to its current physical KV slot."""

    logical_slots = (context_lens.to(dtype=torch.long) - 1).clamp_min(0)
    block_indices = torch.div(
        logical_slots, page_size, rounding_mode="floor"
    ).unsqueeze(1)
    block_ids = (
        block_tables[: context_lens.numel()]
        .to(device=context_lens.device, dtype=torch.long)
        .gather(1, block_indices)
        .squeeze(1)
    )
    physical_slots = block_ids * page_size + torch.remainder(logical_slots, page_size)
    return torch.where(
        context_lens > 0,
        physical_slots,
        torch.full_like(physical_slots, -1),
    )


def build_qwen35_attention_metadata(
    forward_batch: Any,
    positions: torch.Tensor,
    *,
    token_pool: Any,
    req_pool: Any,
) -> Any:
    """Translate SGLang scheduling metadata to ATOM dense-MHA metadata."""

    from atom.utils.forward_context import AttentionMetaData, AttnState

    _, real_num_tokens = _resolve_tc_piecewise_real_tokens(positions.shape[0])
    positions = positions[:real_num_tokens]
    batch_size = int(forward_batch.batch_size)
    seq_lens = forward_batch.seq_lens[:batch_size].to(dtype=torch.int32)
    page_size = int(getattr(token_pool, "page_size", 1))
    mode = forward_batch.forward_mode

    if mode.is_decode_or_idle():
        backend_metadata = getattr(
            _full_attention_backend(forward_batch), "forward_metadata", None
        )
        context_lens = getattr(backend_metadata, "kv_lens", None)
        if context_lens is None:
            context_lens = seq_lens
        else:
            context_lens = context_lens[:batch_size].to(dtype=torch.int32)
        page_table = getattr(backend_metadata, "page_table", None)
        if page_table is None:
            max_seq_len = max(
                _cpu_lens(forward_batch, "seq_lens_cpu", context_lens), default=0
            )
            page_table = _block_tables(
                forward_batch,
                req_pool,
                context_lens,
                page_size=page_size,
                max_seq_len=max_seq_len,
            )
        else:
            # The backend installs a fixed-width page table before CUDA graph
            # warmup. Derive the bound from that tensor for both warmup and
            # capture so torch.compile does not specialize on stream-capture
            # state and attempt a new Dynamo path inside HIP graph capture.
            max_seq_len = int(page_table.shape[1]) * page_size
        page_table = page_table[:batch_size]
        return AttentionMetaData(
            cu_seqlens_q=torch.arange(
                batch_size + 1, dtype=torch.int32, device=positions.device
            ),
            max_seqlen_q=1,
            max_seqlen_k=max_seq_len,
            min_seqlen_q=1,
            slot_mapping=_frontier_slot_mapping(page_table, context_lens, page_size),
            context_lens=context_lens,
            block_tables=page_table,
            state=AttnState.DECODE,
        )

    extend_lens = getattr(forward_batch, "extend_seq_lens", None)
    if extend_lens is None:
        extend_lens = torch.as_tensor(
            _cpu_lens(forward_batch, "extend_seq_lens_cpu", seq_lens),
            dtype=torch.int32,
            device=positions.device,
        )
    else:
        extend_lens = extend_lens[:batch_size].to(
            device=positions.device, dtype=torch.int32
        )
    seq_lens_cpu = _cpu_lens(forward_batch, "seq_lens_cpu", seq_lens)
    extend_lens_cpu = _cpu_lens(forward_batch, "extend_seq_lens_cpu", extend_lens)[
        :batch_size
    ]
    prefix_lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if prefix_lens_cpu is None:
        prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
        prefix_lens_cpu = (
            None
            if prefix_lens is None
            else _cpu_lens(forward_batch, "extend_prefix_lens", prefix_lens)
        )
    if prefix_lens_cpu is None:
        prefix_lens_cpu = [
            seq_len - extend_len
            for seq_len, extend_len in zip(seq_lens_cpu, extend_lens_cpu)
        ]
    else:
        prefix_lens_cpu = [int(length) for length in prefix_lens_cpu[:batch_size]]
    has_cached = any(length > 0 for length in prefix_lens_cpu)
    max_q_len = max(extend_lens_cpu, default=0)
    max_k_len = max(seq_lens_cpu, default=0) if has_cached else max_q_len

    cu_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=positions.device)
    torch.cumsum(extend_lens, dim=0, out=cu_q[1:])
    if has_cached:
        cu_k = torch.zeros_like(cu_q)
        torch.cumsum(seq_lens, dim=0, out=cu_k[1:])
        block_tables = _block_tables(
            forward_batch,
            req_pool,
            seq_lens,
            page_size=page_size,
            max_seq_len=max_k_len,
            extend_lens=extend_lens,
        )
    else:
        cu_k = cu_q
        if envs.ATOM_USE_UNIFIED_ATTN:
            block_tables = _block_tables(
                forward_batch,
                req_pool,
                seq_lens,
                page_size=page_size,
                max_seq_len=max_k_len,
                extend_lens=extend_lens,
            )
        else:
            block_tables = cu_k[:-1, None] + torch.arange(
                max_k_len, dtype=torch.int32, device=positions.device
            )

    # ATOM MHA consumes context_lens as the total usable KV length for each
    # request (seqused_k), including during cold prefill.  Prefix lengths belong
    # in num_cached_tokens/seq_starts only.  Passing prefix_lens here makes a
    # cold prefill advertise zero valid keys and corrupts every full-attention
    # layer's output.
    context_lens = seq_lens.to(device=positions.device, dtype=torch.int32)
    metadata = AttentionMetaData(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q_len,
        max_seqlen_k=max_k_len,
        # Match CommonAttentionBuilder: AITER treats zero as the conservative
        # varlen bound. Passing the actual minimum selects a fixed-length
        # optimization that is not valid for every SGLang extend batch.
        min_seqlen_q=0,
        slot_mapping=forward_batch.out_cache_loc[: positions.shape[0]],
        context_lens=context_lens,
        block_tables=block_tables,
        has_cached=has_cached,
        total_kv=sum(seq_lens_cpu) if has_cached else int(positions.shape[0]),
        num_cached_tokens=(
            torch.as_tensor(prefix_lens_cpu, dtype=torch.int32, device=positions.device)
            if has_cached
            else None
        ),
        seq_starts=(
            torch.zeros(batch_size, dtype=torch.int32, device=positions.device)
            if has_cached
            else None
        ),
        state=AttnState.PREFILL_PREFIX if has_cached else AttnState.PREFILL_NATIVE,
    )
    return metadata


def build_qwen35_forward_metadata(
    forward_batch_or_metadata: Any,
    positions: torch.Tensor,
) -> Any:
    """Build the complete Qwen3.5 MHA + GDN metadata for ATOM runtime."""

    from atom.plugin.sglang.attention_backend.attention_gdn import (
        SGLangGDNForwardContext,
    )
    from atom.plugin.sglang.runtime import SGLangForwardBatchMetadata

    metadata = SGLangForwardBatchMetadata.build(forward_batch_or_metadata)
    if metadata is None or metadata.forward_batch is None:
        raise RuntimeError("Qwen3.5 requires SGLang forward metadata")
    forward_batch = metadata.forward_batch
    token_pool, req_pool = _mha_pools(forward_batch)

    attn_metadata = build_qwen35_attention_metadata(
        forward_batch,
        positions,
        token_pool=token_pool,
        req_pool=req_pool,
    )
    if not metadata.save_kv_cache and attn_metadata.slot_mapping is not None:
        attn_metadata.slot_mapping = torch.full_like(attn_metadata.slot_mapping, -1)

    attn_backend = SGLangGDNForwardContext._resolve_attn_backend(forward_batch)
    linear_backend = SGLangGDNForwardContext._linear_attn_backend(attn_backend)
    attn_metadata.gdn_metadata = SGLangGDNForwardContext._build_gdn_metadata(
        forward_batch, linear_backend
    )
    if attn_metadata.gdn_metadata is None:
        raise RuntimeError("Qwen3.5 GDN metadata is unavailable")
    return attn_metadata


def install_qwen35_cache_views(
    forward_batch: Any, *, cache_owner: Any | None = None
) -> None:
    """Install Qwen3.5 MHA + GDN cache views for subsequent forwards."""

    from atom.plugin.sglang.attention_backend.attention_gdn import (
        SGLangGDNForwardContext,
    )
    from atom.plugin.sglang.attention_backend.backend_resolver import (
        resolve_mamba_req_pool,
    )
    from atom.utils.forward_context import (
        get_forward_context,
        set_kv_cache_data,
    )

    attn_backend = SGLangGDNForwardContext._resolve_attn_backend(forward_batch)
    linear_backend = SGLangGDNForwardContext._linear_attn_backend(attn_backend)
    token_pool, _ = _mha_pools(forward_batch)
    mamba_pool = resolve_mamba_req_pool(forward_batch, linear_backend)
    mamba_map = getattr(mamba_pool, "mamba_map", None)

    cached = (
        getattr(cache_owner, "_atom_qwen35_cache_view_binding", None)
        if cache_owner is not None
        else None
    )
    if (
        cached is not None
        and cached.token_pool is token_pool
        and cached.linear_backend is linear_backend
        and cached.mamba_pool is mamba_pool
        and cached.mamba_map is mamba_map
    ):
        kv_cache_data = cached.kv_cache_data
    else:
        kv_cache_data = bind_qwen35_cache_views(forward_batch)
        kv_cache_data.update(
            SGLangGDNForwardContext._build_kv_cache_tensors(
                forward_batch, linear_backend
            )
        )
        if cache_owner is not None:
            cache_owner._atom_qwen35_cache_view_binding = _Qwen35CacheViewBinding(
                token_pool=token_pool,
                linear_backend=linear_backend,
                mamba_pool=mamba_pool,
                mamba_map=mamba_map,
                kv_cache_data=kv_cache_data,
            )

    set_kv_cache_data(kv_cache_data)
    # Generic SGLangPluginRuntime installs the ForwardContext before invoking
    # bind_cache_views. Refresh that live object as well as the persistent store.
    get_forward_context().kv_cache_data = kv_cache_data
