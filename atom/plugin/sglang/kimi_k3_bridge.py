"""Native ATOM attention bridge for Kimi-K3 on SGLang 0.5.19+."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch

# Kimi-K3 full attention uses true MLA.  Each cache entry stores the latent KV
# plus the rotary lane: kv_lora_rank (512) + qk_rope_head_dim (64).
KIMI_K3_MLA_CACHE_ENTRY_DIM = 576
logger = logging.getLogger(__name__)
# Provenance key for RuntimeContext.override — process-scoped, not id(owner).
_KIMI_K3_MEM_FRACTION_OVERRIDE = "atom-kimi-k3-mem-fraction"


def is_kimi_k3_config(config: Any) -> bool:
    archs = getattr(config, "architectures", None) or []
    return any("KimiK3ForConditionalGeneration" in str(arch) for arch in archs)


def _is_kimi_k3_owner(owner: Any) -> bool:
    model_config = getattr(owner, "model_config", None)
    return is_kimi_k3_config(getattr(model_config, "hf_config", None))


def _kimi_k3_mem_fraction_already_restored(ctx: Any) -> bool:
    # 0.5.20: overrides_log is a method (not a property). Prefer the public
    # API; fall back to the private list for older trees / plain mocks.
    log = getattr(ctx, "overrides_log", None)
    if callable(log):
        entries = log()
    elif log is not None:
        entries = log
    else:
        entries = getattr(ctx, "_overrides_log", None) or ()
    for source, fields in entries:
        if source == _KIMI_K3_MEM_FRACTION_OVERRIDE and "mem_fraction_static" in fields:
            return True
    return False


def _restore_kimi_k3_mem_fraction(owner: Any) -> None:
    """Undo SGLang AITER long-context 0.85 mem_fraction reserve for K3.

    Mirror upstream ``attention_hook``: only undo when the *resolved* backend
    is aiter, context_len > 8192, and the 0.85 scale was actually applied
    (not skipped via ``SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION``). Write
    through ``RuntimeContext.override``; dedupe by override source name.
    """

    model_config = getattr(owner, "model_config", None)
    context_len = int(getattr(model_config, "context_len", 0) or 0)
    try:
        from sglang.srt.runtime_context import get_context, get_exec, get_schedule

        ctx = get_context()
        # Resolved leaf — not server_args.attention_backend (operator raw input).
        attention_backend = str(
            getattr(getattr(get_exec(), "kernel", None), "attention_backend", "") or ""
        )
        current = float(get_schedule().mem_fraction_static)
        server_args = ctx.server_args
    except Exception as exc:  # noqa: BLE001 - first call may precede publish
        logger.warning(
            "Kimi-K3 mem_fraction restore skipped: schedule/exec unread (%s)",
            exc,
        )
        return

    if _kimi_k3_mem_fraction_already_restored(ctx):
        return

    if attention_backend != "aiter" or context_len <= 8192:
        return

    explicit_mem_fraction = (getattr(server_args, "_raw_input", None) or {}).get(
        "mem_fraction_static"
    ) is not None
    try:
        # Must import ``envs`` (the Envs instance), not the environ module.
        from sglang.srt.environ import envs
    except ImportError:
        honor_explicit = False
    else:
        honor_explicit = bool(envs.SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION.get())

    if explicit_mem_fraction and honor_explicit:
        # Upstream did not apply the 0.85 scale; do not divide it back out.
        return

    restored = current / 0.85
    if restored > 1.0:
        return

    ctx.override(_KIMI_K3_MEM_FRACTION_OVERRIDE, mem_fraction_static=restored)
    logger.info(
        "Kimi-K3 restored mem_fraction_static %.4f -> %.4f after "
        "SGLang AITER long-context reserve",
        current,
        restored,
    )


def install_kimi_k3_pool_patch() -> None:
    """Allocate K3 full-attention KV with ATOM's true-MLA cache ABI.

    SGLang moved pool construction from ``ModelRunnerKVCacheMixin`` into
    ``KVCacheConfigurator`` and replaced the ``ModelRunner.kimi_linear_config``
    property with ``hybrid_arch.kimi_linear_config(model_config)``. Patch those
    surfaces for 0.5.19+.

    Important: several SGLang modules already did
    ``from sglang.srt.configs.hybrid_arch import kimi_linear_config`` before
    prepare_model runs. Rebinding only ``hybrid_arch.kimi_linear_config``
    leaves those local names on the original function, so also rebind the
    known importers.
    """

    import sys

    from sglang.srt.configs import hybrid_arch
    from sglang.srt.configs.kimi_linear import KimiLinearConfig
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    cls = KVCacheConfigurator
    if getattr(cls, "_atom_kimi_k3_pool_patched", False):
        return

    original_kimi_linear_config = hybrid_arch.kimi_linear_config
    original_resolve = cls._resolve_memory_pool_config
    original_init_pools = cls._init_pools

    def _kimi_linear_config(model_config):
        config = original_kimi_linear_config(model_config)
        if config is not None:
            return config
        if not is_kimi_k3_config(getattr(model_config, "hf_config", None)):
            return None

        text_config = getattr(model_config, "hf_text_config", None)
        if text_config is None:
            text_config = getattr(
                getattr(model_config, "hf_config", None), "text_config", None
            )
        if isinstance(text_config, KimiLinearConfig):
            return text_config
        if getattr(text_config, "model_type", None) != "kimi_linear":
            return None

        cache_name = "_atom_kimi_k3_linear_config"
        cached = getattr(model_config, cache_name, None)
        if cached is None:
            cached = KimiLinearConfig(**text_config.to_dict())
            setattr(model_config, cache_name, cached)
        return cached

    def _resolve_memory_pool_config(self, pre_model_load_memory: int):
        if not _is_kimi_k3_owner(self):
            return original_resolve(self, pre_model_load_memory)

        _restore_kimi_k3_mem_fraction(self)
        config = original_resolve(self, pre_model_load_memory)

        old_k = int(getattr(self.model_config, "head_dim", 0))
        old_v = int(getattr(self.model_config, "v_head_dim", old_k))
        old_row = old_k + old_v
        # SGLang's hybrid pool exposes paired K/V buffers.  ATOM consumes only
        # the K buffer as the 576-wide MLA latent cache, while the V buffer is
        # retained solely to satisfy SGLang's pool interface.
        native_row = 2 * KIMI_K3_MLA_CACHE_ENTRY_DIM
        if old_row > 0 and old_row != native_row:
            page_size = int(getattr(self, "page_size", 0) or self.server_args.page_size)
            tokens = int(config.max_total_num_tokens) * old_row // native_row
            config.max_total_num_tokens = max(
                page_size, (tokens // page_size) * page_size
            )
            config.max_running_requests = self.resolve_max_num_reqs(
                config.max_total_num_tokens
            )
        return config

    def _init_pools(
        self,
        *,
        sizes,
        req_to_token_pool,
        token_to_kv_pool_allocator,
    ):
        if not _is_kimi_k3_owner(self):
            return original_init_pools(
                self,
                sizes=sizes,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )

        old_head_dim = self.model_config.head_dim
        old_v_head_dim = self.model_config.v_head_dim
        self.model_config.head_dim = KIMI_K3_MLA_CACHE_ENTRY_DIM
        self.model_config.v_head_dim = KIMI_K3_MLA_CACHE_ENTRY_DIM
        try:
            pools = original_init_pools(
                self,
                sizes=sizes,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )
        finally:
            self.model_config.head_dim = old_head_dim
            self.model_config.v_head_dim = old_v_head_dim

        pool = pools.token_to_kv_pool
        full_pool = getattr(pool, "full_kv_pool", pool)
        if full_pool is None:
            raise RuntimeError("Kimi-K3 SGLang full-attention KV pool is missing")
        if (
            int(full_pool.head_dim) != KIMI_K3_MLA_CACHE_ENTRY_DIM
            or int(full_pool.v_head_dim) != KIMI_K3_MLA_CACHE_ENTRY_DIM
        ):
            raise RuntimeError(
                "Kimi-K3 KV pool ABI mismatch: "
                f"K={full_pool.head_dim}, V={full_pool.v_head_dim}, "
                "expected "
                f"{KIMI_K3_MLA_CACHE_ENTRY_DIM}/{KIMI_K3_MLA_CACHE_ENTRY_DIM}"
            )
        req_pool = pools.req_to_token_pool
        if req_pool is None or not hasattr(req_pool, "get_mamba_indices"):
            raise RuntimeError("Kimi-K3 HybridReqToTokenPool is missing")
        pool._atom_kimi_k3_req_pool = req_pool
        if full_pool is not pool:
            full_pool._atom_kimi_k3_req_pool = req_pool
        logger.info(
            "Kimi-K3 attention owner=ATOM, KV owner=SGLang, "
            "layout=MLA/NHD, latent_dim=576"
        )
        return pools

    hybrid_arch.kimi_linear_config = _kimi_linear_config
    # Rebind already-imported local names. prepare_model runs after SGLang has
    # imported these modules, so module-attribute patch alone is a no-op there.
    rebound = []
    for mod_name in (
        "sglang.srt.mem_cache.kv_cache_configurator",
        "sglang.srt.mem_cache.kv_cache_builder",
        "sglang.srt.layers.attention.attention_registry",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "kimi_linear_config"):
            mod.kimi_linear_config = _kimi_linear_config
            rebound.append(mod_name)
    cls._resolve_memory_pool_config = _resolve_memory_pool_config
    cls._init_pools = _init_pools
    cls._atom_kimi_k3_pool_patched = True
    logger.info(
        "Kimi-K3 patched hybrid_arch.kimi_linear_config; rebound imports: %s",
        rebound or "(none yet imported)",
    )


@contextmanager
def kimi_k3_native_attention_construction():
    """Construct K3 full-attention layers with native ATOM attention."""

    from atom.models import kimi_k3

    previous = kimi_k3.Attention
    kimi_k3.Attention = SGLangATOMKimiK3Attention
    try:
        yield
    finally:
        kimi_k3.Attention = previous


class SGLangATOMKimiK3Attention(torch.nn.Module):
    """Thin frontend preserving ATOM's native true-MLA execution path."""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        kv_cache_dtype="bf16",
        layer_num=0,
        use_mla=False,
        prefix: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        from atom.config import get_current_atom_config
        from atom.model_ops.attention_mla import MLAAttention

        if (
            int(head_dim) != KIMI_K3_MLA_CACHE_ENTRY_DIM
            or not use_mla
            or int(num_kv_heads) != 1
        ):
            raise RuntimeError(
                f"Unexpected Kimi-K3 full-attention contract: head_dim={head_dim}, "
                f"num_kv_heads={num_kv_heads}, use_mla={use_mla}"
            )
        atom_config = get_current_atom_config()
        cache_dtype = "fp8" if str(kv_cache_dtype).startswith("fp8") else kv_cache_dtype
        self.layer_num = int(layer_num)
        self.layer_name = prefix or f"KIMI_K3_MLA_{layer_num}"
        self.impl = MLAAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            kv_cache_dtype=cache_dtype,
            layer_num=layer_num,
            dtype=atom_config.torch_dtype,
            mla_modules=kwargs.pop("mla_modules"),
            **kwargs,
        )
        atom_config.compilation_config.static_forward_context[self.layer_name] = self

    def forward(self, query, key, value, positions=None, **kwargs):
        del kwargs
        return torch.ops.aiter.unified_attention_with_output_base(
            query,
            None,
            key,
            value,
            positions,
            self.layer_name,
            True,
            None,
        )

    def process_weights_after_loading(self):
        return self.impl.process_weights_after_loading()


def _iter_kimi_full_attention(model: Any):
    from atom.models.kimi_k3 import KimiFullAttention

    for module in model.modules():
        if isinstance(module, KimiFullAttention):
            attn = getattr(module, "attn", None)
            if not isinstance(attn, SGLangATOMKimiK3Attention):
                raise TypeError(
                    "Kimi-K3 full attention did not construct the native ATOM frontend"
                )
            yield attn


def maybe_get_kimi_k3_pools(forward_batch: Any):
    token_pool = getattr(forward_batch, "token_to_kv_pool", None)
    req_pool = getattr(token_pool, "_atom_kimi_k3_req_pool", None)
    if req_pool is None:
        req_pool = getattr(forward_batch, "req_to_token_pool", None)
    if token_pool is not None and req_pool is not None:
        return token_pool, req_pool

    try:
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            has_forward_context,
        )

        backend = get_attn_backend() if has_forward_context() else None
    except Exception:  # noqa: BLE001 - forward context is optional
        backend = None
    if backend is not None:
        if token_pool is None:
            token_pool = getattr(backend, "_atom_token_to_kv_pool", None)
            if token_pool is None:
                token_pool = getattr(backend, "token_to_kv_pool", None)
        if req_pool is None:
            req_pool = getattr(token_pool, "_atom_kimi_k3_req_pool", None)
        if req_pool is None:
            req_pool = getattr(backend, "_atom_req_to_token_pool", None)
        if req_pool is None:
            req_pool = getattr(backend, "req_to_token_pool", None)
    return token_pool, req_pool


def _is_stream_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except (AssertionError, RuntimeError):
        return False


def kimi_k3_query_dtype() -> torch.dtype:
    """Match Q's fused MLA representation to the SGLang KV-cache dtype."""

    from atom.config import get_current_atom_config
    from atom.plugin.sglang.models.kv_cache_utils import is_fp8_kv_cache_dtype

    atom_config = get_current_atom_config()
    if is_fp8_kv_cache_dtype(getattr(atom_config, "kv_cache_dtype", "bf16")):
        # AITER's FP8 MLA decode kernels require both Q and the latent KV
        # cache to use the FP8 ABI. The fused Q+RoPE+cache op quantizes Q with
        # its existing q_scale when metadata.dtype_q selects this dtype.
        return torch.float8_e4m3fn
    return atom_config.torch_dtype


def _seq_lens(forward_batch: Any, batch_size: int) -> torch.Tensor:
    return forward_batch.seq_lens[:batch_size].to(dtype=torch.int32)


def _extend_lens(
    forward_batch: Any, positions: torch.Tensor, batch_size: int
) -> torch.Tensor:
    extend_lens = getattr(forward_batch, "extend_seq_lens", None)
    if extend_lens is not None:
        return extend_lens[:batch_size].to(device=positions.device, dtype=torch.int32)

    extend_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_lens_cpu is not None:
        return torch.as_tensor(
            extend_lens_cpu[:batch_size],
            dtype=torch.int32,
            device=positions.device,
        )

    tokens_per_req = getattr(
        getattr(forward_batch, "spec_info", None), "num_tokens_per_req", None
    )
    if tokens_per_req is None:
        tokens_per_req = max(1, int(positions.numel()) // max(1, batch_size))
    return torch.full(
        (batch_size,),
        int(tokens_per_req),
        dtype=torch.int32,
        device=positions.device,
    )


def _build_block_table(
    forward_batch: Any,
    req_to_token_pool: Any,
    *,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor | None,
    page_size: int,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    batch_size = int(forward_batch.batch_size)
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item()) if batch_size else 0
    max_blocks = max(1, (max_seq_len + page_size - 1) // page_size)
    req_pool_indices = forward_batch.req_pool_indices[:batch_size]
    token_table = req_to_token_pool.req_to_token[
        req_pool_indices, : max_blocks * page_size
    ].clone()

    if extend_lens is not None:
        prefix_lens = seq_lens - extend_lens
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is not None:
            offset = 0
            for req_idx in range(batch_size):
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


def _attach_sglang_mla_metadata(metadata: Any) -> Any:
    """Reuse the active SGLang backend's graph-stable MLA decode buffers."""

    try:
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            has_forward_context,
        )

        backend = get_attn_backend() if has_forward_context() else None
        forward_metadata = getattr(backend, "forward_metadata", None)
    except Exception:  # noqa: BLE001 - optional during eager unit tests
        forward_metadata = None

    if forward_metadata is None:
        return metadata

    metadata.kv_indptr = getattr(forward_metadata, "kv_indptr", None)
    metadata.kv_indices = getattr(forward_metadata, "kv_indices", None)
    metadata.kv_last_page_lens = getattr(forward_metadata, "kv_last_page_len", None)
    metadata.work_meta_data = getattr(forward_metadata, "work_meta_data", None)
    metadata.work_info_set = getattr(forward_metadata, "work_info_set", None)
    metadata.work_indptr = getattr(forward_metadata, "work_indptr", None)
    metadata.reduce_indptr = getattr(forward_metadata, "reduce_indptr", None)
    metadata.reduce_final_map = getattr(forward_metadata, "reduce_final_map", None)
    metadata.reduce_partial_map = getattr(forward_metadata, "reduce_partial_map", None)
    metadata.num_kv_splits = getattr(forward_metadata, "num_kv_splits", None)
    return metadata


def bind_kimi_k3_cache_views(model: Any, token_to_kv_pool: Any) -> bool:
    if token_to_kv_pool is None or not hasattr(token_to_kv_pool, "get_kv_buffer"):
        return False

    from atom.config import KVCacheTensor
    from atom.utils.forward_context import get_forward_context, set_kv_cache_data

    page_size = int(token_to_kv_pool.page_size)
    if page_size != 128:
        raise RuntimeError(f"Kimi-K3 requires page_size=128, got {page_size}")

    kv_cache_data = dict(getattr(get_forward_context(), "kv_cache_data", None) or {})
    for attn in _iter_kimi_full_attention(model):
        k_buffer, _ = token_to_kv_pool.get_kv_buffer(attn.layer_num)
        if (
            k_buffer.ndim != 3
            or int(k_buffer.shape[1]) < 1
            or int(k_buffer.shape[2]) != KIMI_K3_MLA_CACHE_ENTRY_DIM
        ):
            raise RuntimeError(
                "Kimi-K3 SGLang pool must expose [slots, kv_heads>=1, 576] "
                "MLA K cache, "
                f"got K={tuple(k_buffer.shape)}"
            )
        # SGLang's hybrid pool retains one logical KV head per full-attention
        # head (12 for K3).  True MLA needs one shared latent lane; using the
        # first lane preserves SGLang's slot ownership without duplicating the
        # 576-wide cache in every logical head.
        k_cache = k_buffer[:, :1, :]
        attn.impl.kv_cache = k_cache
        kv_cache_data[f"layer_{attn.layer_num}"] = KVCacheTensor(
            layer_num=attn.layer_num,
            k_cache=k_cache,
            v_cache=None,
            k_scale=None,
            v_scale=None,
        )

    set_kv_cache_data(kv_cache_data)
    get_forward_context().kv_cache_data = kv_cache_data
    return bool(kv_cache_data)


def build_kimi_k3_attention_metadata(
    forward_batch: Any,
    positions: torch.Tensor,
    *,
    token_to_kv_pool: Any,
    req_to_token_pool: Any,
):
    """Translate the current SGLang batch into native ATOM paged-MHA metadata."""

    from atom.utils.forward_context import AttentionMetaData, AttnState

    page_size = int(token_to_kv_pool.page_size)
    try:
        dtype_q = kimi_k3_query_dtype()
    except AssertionError:
        # Metadata-only unit tests run outside an initialized ATOM runtime.
        dtype_q = torch.bfloat16
    bs = int(forward_batch.batch_size)
    seq_lens = _seq_lens(forward_batch, bs)
    is_prefill = bool(forward_batch.forward_mode.is_prefill())

    if is_prefill:
        extend_lens = _extend_lens(forward_batch, positions, bs)
        cu_q = torch.zeros(bs + 1, dtype=torch.int32, device=positions.device)
        torch.cumsum(extend_lens, dim=0, out=cu_q[1:])
        total_tokens = int(positions.shape[0])
        block_tables = _build_block_table(
            forward_batch,
            req_to_token_pool,
            seq_lens=seq_lens,
            extend_lens=extend_lens,
            page_size=page_size,
        )
        slot_mapping = forward_batch.out_cache_loc[:total_tokens]
        metadata = AttentionMetaData(
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_q,
            max_seqlen_q=int(extend_lens.max().item()) if bs else 0,
            max_seqlen_k=int(seq_lens.max().item()) if bs else 0,
            min_seqlen_q=int(extend_lens.min().item()) if bs else 0,
            total_kv=int(seq_lens.sum().item()),
            has_cached=False,
            dropout_p=0.0,
            slot_mapping=slot_mapping,
            context_lens=seq_lens,
            block_tables=block_tables,
            state=AttnState.PREFILL_NATIVE,
        )
        metadata.dtype_q = dtype_q
        return _attach_sglang_mla_metadata(metadata)

    max_seq_len = (
        int(req_to_token_pool.req_to_token.shape[1])
        if _is_stream_capturing()
        else (int(seq_lens.max().item()) if bs else 0)
    )
    total_kv = (
        bs * max_seq_len if _is_stream_capturing() else int(seq_lens.sum().item())
    )
    block_tables = _build_block_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=seq_lens,
        extend_lens=None,
        page_size=page_size,
        max_seq_len=max_seq_len,
    )
    slot_mapping = forward_batch.out_cache_loc[:bs]
    metadata = AttentionMetaData(
        cu_seqlens_q=torch.arange(
            0, bs + 1, dtype=torch.int32, device=positions.device
        ),
        max_seqlen_q=1,
        max_seqlen_k=max_seq_len,
        min_seqlen_q=1,
        # CSR metadata, rather than this bookkeeping value, drives MLA
        # decode. Use a static upper bound while CUDA graph capture forbids
        # device-to-host scalar synchronization.
        total_kv=total_kv,
        has_cached=False,
        dropout_p=0.0,
        slot_mapping=slot_mapping,
        context_lens=seq_lens,
        block_tables=block_tables,
        state=AttnState.DECODE,
    )
    metadata.dtype_q = dtype_q
    return _attach_sglang_mla_metadata(metadata)
