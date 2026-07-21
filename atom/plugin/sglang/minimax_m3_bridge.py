"""Bridge SGLang ForwardBatch metadata to ATOM MiniMax-M3 sparse attention."""

from __future__ import annotations

from typing import Any

import torch

from atom.model_ops.minimax_m3.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    make_sparse_decode_metadata,
    make_sparse_prefill_metadata,
)


def is_minimax_m3_config(config: Any) -> bool:
    archs = getattr(config, "architectures", None) or []
    model_type = str(getattr(config, "model_type", "")).lower()
    return any("MiniMaxM3" in str(arch) for arch in archs) or "minimax_m3" in model_type


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _m3_sparse_cfg(config: Any) -> dict:
    return getattr(_text_config(config), "sparse_attention_config", None) or {}


def _m3_sparse_layer_ids(config: Any) -> list[int]:
    freq = _m3_sparse_cfg(config).get("sparse_attention_freq", []) or []
    return [i for i, enabled in enumerate(freq) if enabled]


def _m3_index_dim(config: Any) -> int:
    index_dim = _m3_sparse_cfg(config).get("sparse_index_dim", None)
    if index_dim is None:
        raise RuntimeError(
            "MiniMax-M3 sparse_attention_config.sparse_index_dim missing"
        )
    return int(index_dim)


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        candidate
        for candidate in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2", None),
        )
        if candidate is not None
    }


def _resolve_m3_index_cache_dtype(fallback: torch.dtype) -> torch.dtype:
    try:
        from atom.config import get_current_atom_config

        index_cache_dtype = getattr(get_current_atom_config(), 'index_cache_dtype', None)
    except Exception:
        index_cache_dtype = None

    if str(index_cache_dtype).startswith('fp8'):
        from aiter import dtypes

        return dtypes.d_dtypes['fp8']
    if index_cache_dtype == 'bf16':
        return torch.bfloat16
    return fallback


class ATOMMiniMaxM3SGLangKVPool:
    """Add MiniMax-M3 index-K cache to an older SGLang MHA KV pool."""

    is_atom_minimax_m3_pool = True

    def __init__(
        self,
        main_pool,
        hf_config: Any,
        index_dtype: torch.dtype,
        *,
        use_fp8_scales: bool,
    ) -> None:
        self.main_pool = main_pool
        self.size = int(main_pool.size)
        self.page_size = int(main_pool.page_size)
        self.dtype = main_pool.dtype
        self.device = main_pool.device
        self.layer_num = main_pool.layer_num
        self.start_layer = main_pool.start_layer
        self.end_layer = main_pool.end_layer
        self.head_num = main_pool.head_num
        self.head_dim = main_pool.head_dim
        self.layer_transfer_counter = getattr(main_pool, "layer_transfer_counter", None)
        self._local_layers = list(range(self.start_layer, self.end_layer))
        self._layer_mapping = {
            layer_id: idx for idx, layer_id in enumerate(self._local_layers)
        }
        self._sparse_layers = [
            lid
            for lid in _m3_sparse_layer_ids(hf_config)
            if self.start_layer <= lid < self.end_layer
        ]
        self._sparse_layer_mapping = {
            layer_id: idx for idx, layer_id in enumerate(self._sparse_layers)
        }
        index_dim = _m3_index_dim(hf_config)
        self.index_k_buffer = [
            torch.empty(
                (self.size + self.page_size, 1, index_dim),
                dtype=index_dtype,
                device=self.device,
            )
            for _ in self._sparse_layers
        ]
        self.k_scale_buffer = []
        self.v_scale_buffer = []
        if use_fp8_scales:
            num_blocks = (self.size + self.page_size) // self.page_size
            self.k_scale_buffer = [
                torch.zeros(
                    (num_blocks, self.head_num, self.page_size),
                    dtype=torch.float32,
                    device=self.device,
                )
                for _ in self._local_layers
            ]
            self.v_scale_buffer = [
                torch.zeros(
                    (num_blocks, self.head_num, self.page_size),
                    dtype=torch.float32,
                    device=self.device,
                )
                for _ in self._local_layers
            ]
        self.mem_usage = (
            float(getattr(main_pool, "mem_usage", 0.0))
            + sum(t.nbytes for t in self.index_k_buffer) / (1 << 30)
            + sum(t.nbytes for t in self.k_scale_buffer) / (1 << 30)
            + sum(t.nbytes for t in self.v_scale_buffer) / (1 << 30)
        )

    def __getattr__(self, name: str):
        return getattr(self.main_pool, name)

    def get_key_buffer(self, layer_id: int):
        return self.main_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        return self.main_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.main_pool.get_kv_buffer(layer_id)

    def set_kv_buffer(self, *args, **kwargs) -> None:
        return self.main_pool.set_kv_buffer(*args, **kwargs)

    def get_kv_size_bytes(self):
        k_bytes, v_bytes = self.main_pool.get_kv_size_bytes()
        return (
            k_bytes
            + sum(t.nbytes for t in self.index_k_buffer)
            + sum(t.nbytes for t in self.k_scale_buffer),
            v_bytes + sum(t.nbytes for t in self.v_scale_buffer),
        )

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        mapped = self._sparse_layer_mapping.get(int(layer_id))
        if mapped is None:
            raise ValueError(f"MiniMax-M3 layer {layer_id} has no index-K cache")
        return self.index_k_buffer[mapped]

    def get_kv_scale_buffer(
        self, layer_id: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        mapped = self._layer_mapping.get(int(layer_id))
        if mapped is None or not self.k_scale_buffer:
            return None, None
        return self.k_scale_buffer[mapped], self.v_scale_buffer[mapped]


def install_minimax_m3_pool_patch() -> None:
    """Patch older SGLang builds that lack MiniMaxSparseKVPool support."""

    import sglang.srt.model_executor.model_runner_kv_cache_mixin as mixin

    if getattr(mixin.ModelRunnerKVCacheMixin, "_atom_minimax_m3_pool_patched", False):
        return

    original_resolve = mixin.ModelRunnerKVCacheMixin._resolve_memory_pool_config
    original_init_pools = mixin.ModelRunnerKVCacheMixin._init_pools

    def _is_m3_runner(runner) -> bool:
        return is_minimax_m3_config(getattr(runner.model_config, "hf_config", None))

    def _local_kv_heads(runner) -> int:
        try:
            from sglang.srt.layers.dp_attention import get_attention_tp_size

            return int(runner.model_config.get_num_kv_heads(get_attention_tp_size()))
        except Exception:
            hf_config = _text_config(runner.model_config.hf_config)
            tp_size = max(1, int(getattr(runner, "tp_size", 1)))
            return max(1, int(getattr(hf_config, "num_key_value_heads", 1)) // tp_size)

    def _resolve_memory_pool_config(self, pre_model_load_memory: int):
        config = original_resolve(self, pre_model_load_memory)
        if not _is_m3_runner(self):
            return config

        hf_config = _text_config(self.model_config.hf_config)
        kv_dtype = self.kv_cache_dtype
        use_fp8_scales = _is_fp8_dtype(self.kv_cache_dtype) or str(
            self.kv_cache_dtype
        ).startswith("fp8")
        index_dtype = _resolve_m3_index_cache_dtype(
            getattr(self, 'dtype', getattr(self, 'torch_dtype', torch.bfloat16))
        )
        num_layers = int(
            getattr(self, "num_effective_layers", hf_config.num_hidden_layers)
        )
        main_bytes = (
            2
            * num_layers
            * _local_kv_heads(self)
            * int(hf_config.head_dim)
            * _dtype_size(kv_dtype)
        )
        index_bytes = (
            len(_m3_sparse_layer_ids(self.model_config.hf_config))
            * _m3_index_dim(self.model_config.hf_config)
            * _dtype_size(index_dtype)
        )
        scale_bytes = 0
        if use_fp8_scales:
            scale_bytes = (
                2 * num_layers * _local_kv_heads(self) * _dtype_size(torch.float32)
            )
        extra_bytes = index_bytes + scale_bytes
        if main_bytes <= 0 or extra_bytes <= 0:
            return config

        old_tokens = int(config.max_total_num_tokens)
        new_tokens = (old_tokens * main_bytes) // (main_bytes + extra_bytes)
        page_size = int(self.server_args.page_size)
        new_tokens = max(page_size, (new_tokens // page_size) * page_size)
        if new_tokens < old_tokens:
            config.max_total_num_tokens = new_tokens
            config.max_running_requests = self._resolve_max_num_reqs(new_tokens)
        return config

    def _init_pools(self):
        original_init_pools(self)
        if not _is_m3_runner(self):
            return
        pool = getattr(self, "token_to_kv_pool", None)
        if pool is None or hasattr(pool, "get_index_k_buffer"):
            return
        use_fp8_scales = _is_fp8_dtype(self.kv_cache_dtype) or str(
            self.kv_cache_dtype
        ).startswith("fp8")
        index_dtype = _resolve_m3_index_cache_dtype(
            getattr(self, 'dtype', getattr(self, 'torch_dtype', torch.bfloat16))
        )
        self.token_to_kv_pool = ATOMMiniMaxM3SGLangKVPool(
            pool,
            self.model_config.hf_config,
            index_dtype,
            use_fp8_scales=use_fp8_scales,
        )

    mixin.ModelRunnerKVCacheMixin._resolve_memory_pool_config = (
        _resolve_memory_pool_config
    )
    mixin.ModelRunnerKVCacheMixin._init_pools = _init_pools
    mixin.ModelRunnerKVCacheMixin._atom_minimax_m3_pool_patched = True


def maybe_get_minimax_m3_pools_from_sglang_batch(forward_batch=None):
    if forward_batch is None:
        return None, None
    token_to_kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
    req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
    if token_to_kv_pool is None or req_to_token_pool is None:
        return None, None
    return token_to_kv_pool, req_to_token_pool


def _page_size(token_to_kv_pool) -> int:
    page_size = int(getattr(token_to_kv_pool, "page_size", 1))
    if page_size != SPARSE_BLOCK_SIZE:
        raise ValueError(
            "MiniMax-M3 native sparse attention requires SGLang page size "
            f"{SPARSE_BLOCK_SIZE}, got {page_size}. Launch with --page-size "
            f"{SPARSE_BLOCK_SIZE}."
        )
    return page_size


def _is_stream_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _seq_lens(forward_batch, bs: int) -> torch.Tensor:
    return forward_batch.seq_lens[:bs].to(dtype=torch.int32)


def _extend_lens(forward_batch, positions: torch.Tensor, bs: int) -> torch.Tensor:
    extend_lens = getattr(forward_batch, "extend_seq_lens", None)
    if extend_lens is not None:
        return extend_lens[:bs].to(device=positions.device, dtype=torch.int32)

    extend_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_lens_cpu is not None:
        return torch.as_tensor(
            extend_lens_cpu[:bs], dtype=torch.int32, device=positions.device
        )

    tokens_per_req = getattr(
        getattr(forward_batch, "spec_info", None), "num_tokens_per_req", None
    )
    if tokens_per_req is None:
        tokens_per_req = max(1, int(positions.numel()) // max(1, bs))
    return torch.full(
        (bs,), int(tokens_per_req), dtype=torch.int32, device=positions.device
    )


def _build_block_table(
    forward_batch,
    req_to_token_pool,
    *,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor | None,
    page_size: int,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    bs = int(forward_batch.batch_size)
    if max_seq_len is None:
        # This path is prefill/eager only. Decode graph capture must pass a static
        # max_seq_len because GPU scalar reads (`.item()`) are illegal in capture.
        max_seq_len = int(seq_lens.max().item()) if bs else 0
    max_blocks = max(1, (max_seq_len + page_size - 1) // page_size)
    req_pool_indices = forward_batch.req_pool_indices[:bs]
    token_table = req_to_token_pool.req_to_token[
        req_pool_indices, : max_blocks * page_size
    ].clone()

    if extend_lens is not None:
        prefix_lens = seq_lens - extend_lens
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is not None:
            offset = 0
            for req_idx in range(bs):
                prefix_len = int(prefix_lens[req_idx].item())
                query_len = int(extend_lens[req_idx].item())
                if query_len > 0:
                    token_table[req_idx, prefix_len : prefix_len + query_len] = (
                        out_cache_loc[offset : offset + query_len]
                    )
                offset += query_len

    return (
        (token_table[:, : max_blocks * page_size : page_size] // page_size)
        .to(dtype=torch.int32)
        .contiguous()
    )


def build_atom_minimax_m3_attention_metadata_from_sglang(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
):
    from atom.utils.forward_context import AttentionMetaData

    page_size = _page_size(token_to_kv_pool)
    bs = int(forward_batch.batch_size)
    seq_lens = _seq_lens(forward_batch, bs)
    is_prefill = bool(
        getattr(forward_batch.forward_mode, "is_prefill", lambda: False)()
    )
    max_context_len = int(req_to_token_pool.req_to_token.shape[1])

    if is_prefill:
        extend_lens = _extend_lens(forward_batch, positions, bs)
        cu_q = torch.empty(bs + 1, dtype=torch.int32, device=positions.device)
        cu_q[0] = 0
        torch.cumsum(extend_lens, dim=0, out=cu_q[1:])
        total_tokens = int(cu_q[-1].item()) if bs else 0
        block_table = _build_block_table(
            forward_batch,
            req_to_token_pool,
            seq_lens=seq_lens,
            extend_lens=extend_lens,
            page_size=page_size,
        )
        max_query_len = int(extend_lens.max().item()) if bs else 0
        max_seq_len = int(seq_lens.max().item()) if bs else 0
        slot_mapping = getattr(forward_batch, "out_cache_loc", None)
        slot_mapping = (
            slot_mapping[:total_tokens]
            if torch.is_tensor(slot_mapping)
            else torch.empty(0, dtype=torch.int64, device=positions.device)
        )
        sparse_md = make_sparse_prefill_metadata(
            cu_seqlens_q=cu_q,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            num_prefills=bs,
            num_prefill_tokens=total_tokens,
        )
        sparse_md.prefill.qo_indptr = torch.arange(
            total_tokens + 1,
            dtype=torch.int32,
            device=positions.device,
        )
        md = AttentionMetaData(
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_q,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_seq_len,
            slot_mapping=slot_mapping,
            context_lens=seq_lens,
            block_tables=block_table,
        )
        md.sparse_attention_metadata = sparse_md
        return md

    tokens_per_req = max(1, int(positions.numel()) // max(1, bs)) if bs else 1
    # Decode CUDA graph capture cannot synchronize on seq_lens.max(). Use the
    # static req_to_token table capacity, matching the fixed-shape graph contract.
    max_seq_len = (
        max_context_len
        if _is_stream_capturing()
        else (int(seq_lens.max().item()) if bs else 0)
    )
    block_table = _build_block_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=seq_lens,
        extend_lens=None,
        page_size=page_size,
        max_seq_len=max_seq_len,
    )
    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
    if torch.is_tensor(out_cache_loc):
        slot_mapping = out_cache_loc[: bs * tokens_per_req]
    else:
        scratch = max(0, int(getattr(token_to_kv_pool, "size", 1)) - 1)
        slot_mapping = torch.full(
            (bs * tokens_per_req,),
            scratch,
            dtype=torch.int64,
            device=positions.device,
        )
    sparse_md = make_sparse_decode_metadata(
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        max_seq_len=max_seq_len,
        max_query_len=tokens_per_req,
    )
    cu_q = torch.arange(
        0,
        bs * tokens_per_req + 1,
        tokens_per_req,
        dtype=torch.int32,
        device=positions.device,
    )
    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        max_seqlen_q=tokens_per_req,
        max_seqlen_k=max_seq_len,
        slot_mapping=slot_mapping,
        context_lens=seq_lens,
        block_tables=block_table,
    )
    md.sparse_attention_metadata = sparse_md
    return md


def _iter_m3_attention_layers(model):
    from atom.models.minimax_m3 import MiniMaxM3Attention, MiniMaxM3SparseAttention

    for module in model.modules():
        if isinstance(module, (MiniMaxM3Attention, MiniMaxM3SparseAttention)):
            attn = getattr(module, "attn", None)
            impl = getattr(attn, "impl", None)
            if impl is not None:
                yield module, attn, impl, isinstance(module, MiniMaxM3SparseAttention)


def _get_index_cache_view(
    token_to_kv_pool,
    layer_id: int,
    *,
    k_buffer: torch.Tensor,
    index_dim: int,
) -> torch.Tensor:
    """Return SGLang MiniMaxSparseKVPool's existing index-K cache view."""

    if not hasattr(token_to_kv_pool, "get_index_k_buffer"):
        raise RuntimeError(
            "MiniMax-M3 native sparse attention requires SGLang MiniMaxSparseKVPool "
            "with get_index_k_buffer(); refusing to allocate a side index cache."
        )
    page_size = _page_size(token_to_kv_pool)
    num_slots = int(k_buffer.shape[0])
    num_blocks = max(1, num_slots // page_size)
    expected = (num_blocks, page_size, int(index_dim))
    index_buffer = token_to_kv_pool.get_index_k_buffer(layer_id)
    return index_buffer[: num_blocks * page_size].view(*expected)


def bind_minimax_m3_sparse_cache_views(model, token_to_kv_pool) -> bool:
    if token_to_kv_pool is None or not hasattr(token_to_kv_pool, "get_kv_buffer"):
        return False

    from atom.config import KVCacheTensor
    from atom.utils.forward_context import get_forward_context, set_kv_cache_data

    page_size = _page_size(token_to_kv_pool)
    kv_cache_data = {}
    bound = False
    for _, attn, impl, is_sparse_layer in _iter_m3_attention_layers(model):
        layer_id = int(impl.layer_num)
        k_buffer, v_buffer = token_to_kv_pool.get_kv_buffer(layer_id)
        num_slots, num_kv_heads, head_dim = k_buffer.shape
        num_blocks = max(1, int(num_slots) // page_size)
        live_slots = num_blocks * page_size
        x = 16 // k_buffer.element_size()
        k_cache = k_buffer[:live_slots].view(
            num_blocks, num_kv_heads, head_dim // x, page_size, x
        )
        v_cache = v_buffer[:live_slots].view(
            num_blocks, num_kv_heads, page_size // x, head_dim, x
        )
        if hasattr(token_to_kv_pool, "get_kv_scale_buffer"):
            k_scale, v_scale = token_to_kv_pool.get_kv_scale_buffer(layer_id)
        else:
            k_scale = None
            v_scale = None
        if not is_sparse_layer:
            pass
        elif getattr(impl, "skip_index_topk", False):
            # Shared-index sparse layers reuse the previous full-index layer's
            # top-k result and never write/read their own index-K cache.
            impl.index_cache = None
        else:
            impl.index_cache = _get_index_cache_view(
                token_to_kv_pool,
                layer_id,
                k_buffer=k_buffer,
                index_dim=impl.index_head_dim,
            )
        impl.max_model_len = int(getattr(token_to_kv_pool, "size", live_slots))
        if is_sparse_layer:
            impl.index_topk_cache_state = getattr(
                token_to_kv_pool, "_atom_minimax_m3_topk_cache_state", None
            )
            if impl.index_topk_cache_state is None:
                impl.index_topk_cache_state = {}
                setattr(
                    token_to_kv_pool,
                    "_atom_minimax_m3_topk_cache_state",
                    impl.index_topk_cache_state,
                )
        kv_cache_data[f"layer_{layer_id}"] = KVCacheTensor(
            layer_num=layer_id,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        attn.k_cache = k_cache
        attn.v_cache = v_cache
        attn.k_scale = k_scale
        attn.v_scale = v_scale
        bound = True

    if not bound:
        return False

    set_kv_cache_data(kv_cache_data)
    get_forward_context().kv_cache_data = kv_cache_data
    return True
