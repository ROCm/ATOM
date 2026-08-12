"""Native ATOM attention bridge for Kimi-K3 on SGLang 0.5.15."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch

# Kimi-K3 full attention uses true MLA.  Each cache entry stores the latent KV
# plus the rotary lane: kv_lora_rank (512) + qk_rope_head_dim (64).
KIMI_K3_MLA_CACHE_ENTRY_DIM = 576
logger = logging.getLogger(__name__)

_UNSET = object()


@contextmanager
def _forced_mla_kv_pool(configurator: Any):
    """Build the K3 full-attention KV pool as a *true* MLA pool.

    ATOM reads one 576-wide latent per token per layer and never reads a V
    buffer (``bind_kimi_k3_cache_views`` binds ``k_buffer[:, :1, :]`` and sets
    ``v_cache=None``). ``MLATokenToKVPool`` allocates exactly that shape --
    ``[slots, 1, kv_lora_rank + qk_rope_head_dim]`` -- whereas the MHA fallback
    also allocates an identically sized V buffer that is pure waste.

    sglang only takes that branch when ``use_mla_backend`` is set, which it
    derives from an architecture whitelist that misses K3 (see
    the MLA-whitelist note below). Flip it for the *pool build only*: attention
    backend selection, which ATOM owns, must stay on its current path.
    """
    mc = configurator.model_config
    text_config = getattr(mc, "hf_text_config", None)
    dims = {
        "kv_lora_rank": getattr(text_config, "kv_lora_rank", None) or 512,
        "qk_rope_head_dim": getattr(text_config, "qk_rope_head_dim", None) or 64,
    }
    saved_flag = configurator.use_mla_backend
    saved_dims = {name: getattr(mc, name, _UNSET) for name in dims}
    for name, value in dims.items():
        if not getattr(mc, name, None):
            setattr(mc, name, value)
    configurator.use_mla_backend = True
    try:
        yield
    finally:
        configurator.use_mla_backend = saved_flag
        for name, value in saved_dims.items():
            if value is _UNSET:
                try:
                    delattr(mc, name)
                except Exception:  # noqa: BLE001, S110 - config may be slotted
                    pass
            else:
                setattr(mc, name, value)


def is_kimi_k3_config(config: Any) -> bool:
    archs = getattr(config, "architectures", None) or []
    return any("KimiK3ForConditionalGeneration" in str(arch) for arch in archs)


def is_kimi_k3_dspark_config(config: Any) -> bool:
    """Detect the standalone Kimi-K3 DSpark draft config (arch K3DSparkModel)."""
    archs = getattr(config, "architectures", None) or []
    if any("K3DSparkModel" in str(arch) for arch in archs):
        return True
    return getattr(config, "model_type", None) == "k3_dspark"


def _is_kimi_k3_runner(runner: Any) -> bool:
    return is_kimi_k3_config(getattr(runner.model_config, "hf_config", None))


def _restore_kimi_k3_mem_fraction(runner: Any) -> None:
    if getattr(runner, "_atom_kimi_k3_mem_fraction_restored", False):
        return
    server_args = runner.server_args
    context_len = int(getattr(runner.model_config, "context_len", 0) or 0)
    attention_backend = str(getattr(server_args, "attention_backend", ""))
    current = float(
        getattr(runner, "mem_fraction_static", server_args.mem_fraction_static)
    )
    if attention_backend == "aiter" and context_len > 8192:
        restored = current / 0.85
        if restored <= 1.0:
            runner.mem_fraction_static = restored
            server_args.mem_fraction_static = restored
            logger.info(
                "Kimi-K3 restored mem_fraction_static %.4f -> %.4f after "
                "SGLang AITER long-context reserve",
                current,
                restored,
            )
    runner._atom_kimi_k3_mem_fraction_restored = True


def _restore_kimi_k3_mem_fraction_server_args(
    server_args: Any, model_config: Any
) -> None:
    """sglang-main variant of ``_restore_kimi_k3_mem_fraction``.

    The KV-pool code moved off the ModelRunner mixin onto a slotted
    ``KVCacheConfigurator``, so the guard/restore must live on ``server_args``
    (persistent across the configurator's fresh instances) instead of the runner.
    """
    if getattr(server_args, "_atom_kimi_k3_mem_fraction_restored", False):
        return
    context_len = int(getattr(model_config, "context_len", 0) or 0)
    attention_backend = str(getattr(server_args, "attention_backend", ""))
    current = float(server_args.mem_fraction_static)
    if attention_backend == "aiter" and context_len > 8192:
        restored = current / 0.85
        if restored <= 1.0:
            server_args.mem_fraction_static = restored
            logger.info(
                "Kimi-K3 restored mem_fraction_static %.4f -> %.4f after SGLang "
                "AITER long-context reserve (sglang-main)",
                current,
                restored,
            )
    try:
        server_args._atom_kimi_k3_mem_fraction_restored = True
    except Exception:  # noqa: BLE001, S110 - best-effort guard
        pass


def _patch_kimi_linear_config_detection() -> None:
    """Make sglang-main detect Kimi-K3 as a KimiLinear hybrid.

    sglang-main's ``kimi_linear_config()`` only matches a real
    ``KimiLinearConfig`` instance; this image's K3 checkpoint ships a generic
    config whose ``hf_text_config.model_type == "kimi_linear"``. Synthesize a
    ``KimiLinearConfig`` from that text config so the hybrid KDA/MLA pool and
    attention routing engage -- the sglang-main equivalent of the old
    ``ModelRunner.kimi_linear_config`` property override.
    """
    import importlib

    from sglang.srt.configs import hybrid_arch
    from sglang.srt.configs.kimi_linear import KimiLinearConfig

    if getattr(hybrid_arch, "_atom_kimi_k3_detect_patched", False):
        return
    original = hybrid_arch.kimi_linear_config

    def _kimi_linear_config(model_config: Any):
        config = original(model_config)
        if config is not None:
            return config
        if not is_kimi_k3_config(getattr(model_config, "hf_config", None)):
            return None
        text_config = getattr(model_config, "hf_text_config", None)
        if isinstance(text_config, KimiLinearConfig):
            return text_config
        if getattr(text_config, "model_type", None) != "kimi_linear":
            return None
        cache = getattr(model_config, "_atom_kimi_k3_linear_config", None)
        if cache is None:
            cache = KimiLinearConfig(**text_config.to_dict())
            try:
                model_config._atom_kimi_k3_linear_config = cache
            except Exception:  # noqa: BLE001, S110 - config may be slotted
                pass
        return cache

    # Rebind in every module that captured the symbol via `from ... import`.
    hybrid_arch.kimi_linear_config = _kimi_linear_config
    for mod_name in (
        "sglang.srt.layers.attention.attention_registry",
        "sglang.srt.layers.attention.triton_backend",
        "sglang.srt.mem_cache.kv_cache_builder",
    ):
        try:
            module = importlib.import_module(mod_name)
            if hasattr(module, "kimi_linear_config"):
                module.kimi_linear_config = _kimi_linear_config
        except Exception:  # noqa: BLE001 - module optional across versions
            logger.debug("Kimi-K3: could not rebind kimi_linear_config in %s", mod_name)
    hybrid_arch._atom_kimi_k3_detect_patched = True


def _install_kimi_k3_pool_patch_sglang_main() -> bool:
    """Port of the K3 KV-pool patch onto sglang-main's ``KVCacheConfigurator``.

    Returns True when the sglang-main API is present and patched; False on a
    genuine 0.5.15 runtime so the caller can fall back to the legacy mixin path.
    """
    try:
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    except Exception:  # noqa: BLE001 - 0.5.15 has no configurator
        return False

    _patch_kimi_linear_config_detection()

    if getattr(KVCacheConfigurator, "_atom_kimi_k3_pool_patched", False):
        return True

    original_resolve = KVCacheConfigurator._resolve_memory_pool_config
    original_init = KVCacheConfigurator._init_pools

    def _is_k3(self) -> bool:
        return is_kimi_k3_config(getattr(self.model_config, "hf_config", None))

    def _resolve_memory_pool_config(self, pre_model_load_memory: int):
        if not _is_k3(self):
            return original_resolve(self, pre_model_load_memory)
        _restore_kimi_k3_mem_fraction_server_args(self.server_args, self.model_config)
        # True-MLA pool: the budget sglang computes here already matches the real
        # [slots, 1, 576] allocation, so no row-width compensation is needed.
        with _forced_mla_kv_pool(self):
            config = original_resolve(self, pre_model_load_memory)

        # sglang sizes the KV pool from free memory alone. A true-MLA token is
        # ~24x cheaper than the old paired-K/V, 12-lane layout, so that formula
        # would spend *all* the reclaimed memory on more tokens and leave no
        # headroom for prefill activations (observed with an intermediate fix:
        # available_gpu_mem 6.25GB -> 1.95GB, then a HIP OOM on the first
        # chunked prefill). The scheduler can never address more than
        # ``max_running_requests * context_len`` tokens, so cap there and leave
        # the remainder free.
        max_reqs = getattr(self.server_args, "max_running_requests", None)
        ctx_len = getattr(self.model_config, "context_len", None)
        if max_reqs and ctx_len:
            page_size = int(self.server_args.page_size)
            cap = (int(max_reqs) * int(ctx_len) // page_size) * page_size
            if cap > 0 and config.max_total_num_tokens > cap:
                config.max_total_num_tokens = cap
                config.max_running_requests = self.resolve_max_num_reqs(cap)
        return config

    def _init_pools(self, *, sizes, req_to_token_pool, token_to_kv_pool_allocator):
        if not _is_k3(self):
            # The standalone Kimi-K3 DSpark draft owns a SIBLING MLA pool. ATOM's
            # K3DSparkMLAAttention writes the 576-wide latent in bf16, but sglang
            # has no MLA knowledge for arch K3DSparkModel and would allocate a
            # generic MHA pool ([slots, kv_heads, hidden/num_heads], fp8). Force
            # the 576-wide bf16 latent layout, mirroring the target pool patch.
            if is_kimi_k3_dspark_config(getattr(self.model_config, "hf_config", None)):
                import torch as _torch

                mc = self.model_config
                htc = getattr(mc, "hf_text_config", None)
                saved_dtype = self.kv_cache_dtype
                saved_head = getattr(mc, "head_dim", None)
                saved_v = getattr(mc, "v_head_dim", None)
                saved_htc_v = (
                    getattr(htc, "v_head_dim", None) if htc is not None else None
                )
                self.kv_cache_dtype = _torch.bfloat16
                mc.head_dim = KIMI_K3_MLA_CACHE_ENTRY_DIM
                try:
                    mc.v_head_dim = KIMI_K3_MLA_CACHE_ENTRY_DIM
                except Exception:  # noqa: BLE001, S110 - optional attr
                    pass
                if htc is not None:
                    try:
                        htc.v_head_dim = KIMI_K3_MLA_CACHE_ENTRY_DIM
                    except Exception:  # noqa: BLE001, S110 - optional attr
                        pass
                try:
                    with _forced_mla_kv_pool(self):
                        return original_init(
                            self,
                            sizes=sizes,
                            req_to_token_pool=req_to_token_pool,
                            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                        )
                finally:
                    self.kv_cache_dtype = saved_dtype
                    mc.head_dim = saved_head
                    if saved_v is not None:
                        mc.v_head_dim = saved_v
                    if htc is not None and saved_htc_v is not None:
                        htc.v_head_dim = saved_htc_v
            return original_init(
                self,
                sizes=sizes,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )
        with _forced_mla_kv_pool(self):
            pools = original_init(
                self,
                sizes=sizes,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )

        pool = pools.token_to_kv_pool
        full_pool = getattr(pool, "full_kv_pool", pool)
        if full_pool is None:
            raise RuntimeError("Kimi-K3 SGLang full-attention KV pool is missing")
        # True-MLA pool: one [slots, 1, 576] latent buffer per layer, no V.
        latent_dim = int(getattr(full_pool, "kv_cache_dim", -1))
        if latent_dim != KIMI_K3_MLA_CACHE_ENTRY_DIM:
            raise RuntimeError(
                "Kimi-K3 KV pool ABI mismatch: expected a single MLA latent of "
                f"{KIMI_K3_MLA_CACHE_ENTRY_DIM}, got kv_cache_dim={latent_dim} "
                f"(pool={type(full_pool).__name__})"
            )
        req_pool = pools.req_to_token_pool
        if req_pool is None or not hasattr(req_pool, "get_mamba_indices"):
            raise RuntimeError("Kimi-K3 HybridReqToTokenPool is missing")
        pool._atom_kimi_k3_req_pool = req_pool
        if full_pool is not pool:
            full_pool._atom_kimi_k3_req_pool = req_pool
        logger.info(
            "Kimi-K3 attention owner=ATOM, KV owner=SGLang, "
            "layout=MLA/NHD, latent_dim=576 (sglang-main)"
        )
        return pools

    KVCacheConfigurator._resolve_memory_pool_config = _resolve_memory_pool_config
    KVCacheConfigurator._init_pools = _init_pools
    KVCacheConfigurator._atom_kimi_k3_pool_patched = True
    return True


def install_kimi_k3_pool_patch() -> None:
    """Allocate K3 full-attention KV with ATOM's true-MLA cache ABI."""

    # sglang-main relocated the KV-pool code onto KVCacheConfigurator and turned
    # kimi_linear_config into a standalone function; prefer that path.
    if _install_kimi_k3_pool_patch_sglang_main():
        return

    import sglang.srt.model_executor.model_runner_kv_cache_mixin as mixin
    from sglang.srt.configs.kimi_linear import KimiLinearConfig
    from sglang.srt.model_executor.model_runner import ModelRunner

    cls = mixin.ModelRunnerKVCacheMixin
    if getattr(cls, "_atom_kimi_k3_pool_patched", False):
        return

    original_kimi_property = ModelRunner.kimi_linear_config
    original_resolve = cls._resolve_memory_pool_config
    original_init_pools = cls._init_pools

    def _kimi_linear_config(self):
        config = original_kimi_property.__get__(self, type(self))
        if config is not None:
            return config
        text_config = getattr(self.model_config, "hf_text_config", None)
        if not _is_kimi_k3_runner(self):
            return None
        if isinstance(text_config, KimiLinearConfig):
            return text_config
        if getattr(text_config, "model_type", None) != "kimi_linear":
            return None

        cache_name = "_atom_kimi_k3_linear_config"
        config = getattr(self, cache_name, None)
        if config is None:
            config = KimiLinearConfig(**text_config.to_dict())
            setattr(self, cache_name, config)
        return config

    def _resolve_memory_pool_config(self, pre_model_load_memory: int):
        if not _is_kimi_k3_runner(self):
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
            page_size = int(self.server_args.page_size)
            tokens = int(config.max_total_num_tokens) * old_row // native_row
            config.max_total_num_tokens = max(
                page_size, (tokens // page_size) * page_size
            )
            config.max_running_requests = self._resolve_max_num_reqs(
                config.max_total_num_tokens
            )
        return config

    def _init_pools(self):
        if not _is_kimi_k3_runner(self):
            return original_init_pools(self)

        old_head_dim = self.model_config.head_dim
        old_v_head_dim = self.model_config.v_head_dim
        self.model_config.head_dim = KIMI_K3_MLA_CACHE_ENTRY_DIM
        self.model_config.v_head_dim = KIMI_K3_MLA_CACHE_ENTRY_DIM
        try:
            original_init_pools(self)
        finally:
            self.model_config.head_dim = old_head_dim
            self.model_config.v_head_dim = old_v_head_dim

        pool = getattr(self, "token_to_kv_pool", None)
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
        req_pool = getattr(self, "req_to_token_pool", None)
        if req_pool is None or not hasattr(req_pool, "get_mamba_indices"):
            raise RuntimeError("Kimi-K3 HybridReqToTokenPool is missing")
        pool._atom_kimi_k3_req_pool = req_pool
        if full_pool is not pool:
            full_pool._atom_kimi_k3_req_pool = req_pool
        logger.info(
            "Kimi-K3 attention owner=ATOM, KV owner=SGLang, "
            "layout=MLA/NHD, latent_dim=576"
        )

    ModelRunner.kimi_linear_config = property(_kimi_linear_config)
    cls._resolve_memory_pool_config = _resolve_memory_pool_config
    cls._init_pools = _init_pools
    cls._atom_kimi_k3_pool_patched = True


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

    block_table = (
        (token_table[:, : max_blocks * page_size : page_size] // page_size)
        .to(dtype=torch.int32)
        .contiguous()
    )
    # ``token_table`` is a captured intermediate feeding the slice/floor-div that
    # produces ``block_table``; pin both during capture so a later smaller-bs
    # capture cannot reuse their memory and clobber the first-captured (max-bs)
    # graph on replay.
    from atom.plugin.sglang.kimi_k3_spec_verify import keepalive_if_capturing

    keepalive_if_capturing(token_table, block_table)
    return block_table


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


def _build_kimi_k3_verify_metadata(
    forward_batch: Any,
    positions: torch.Tensor,
    *,
    req_to_token_pool: Any,
    seq_lens: torch.Tensor,
    page_size: int,
    dtype_q: torch.dtype,
    bs: int,
):
    """Build MLA attention metadata for DSpark TARGET_VERIFY.

    The T draft tokens per request must attend to the committed prefix KV plus
    the earlier draft tokens (causal). Routing verify as a fresh
    ``PREFILL_NATIVE`` (``has_cached=False``) makes ATOM's MLA drop the prefix
    (``_forward_prefill_mha`` only self-attends the T new tokens -> the few
    full-attention layers lose the prompt context and the output drifts).

    We instead reuse the paged decode path (``state=DECODE`` -> ``is_prefill=False``
    -> ``_forward_decode``/``mla_decode_fwd`` with an intra-block causal mask for
    ``max_q_len>1``). It reads the full paged context via the MLA backend's own
    verify ``forward_metadata`` (kv_indptr/kv_indices/kv_last_page_len/work_*),
    the same proven indices the K3 decode path uses, so the T draft tokens attend
    the committed prefix + earlier drafts. ``cu_seqlens_q`` = T queries per req.
    """
    from atom.plugin.sglang.attention_backend.attention_gdn import (
        SGLangGDNForwardContext,
    )
    from atom.utils.forward_context import AttentionMetaData, AttnState

    # The MLA (full-attention) backend already built the complete paged verify
    # metadata during init_forward_metadata(TARGET_VERIFY) -- kv_indptr/kv_indices/
    # qo_indptr AND the persistent scheduler buffers (work_metadata/work_indptr/
    # work_info_set/reduce_*), because _use_mla_ps_kernel is on for fp8 KV. These
    # live in the backend's graph-stable buffers (managed by its capture/replay
    # hooks), so reusing them makes this path CUDA-graph-safe: we only reference
    # tensors + read shapes/constants here (no .item(), no allocation).
    attn_backend = SGLangGDNForwardContext._resolve_attn_backend(forward_batch)
    full_attn_backend = getattr(attn_backend, "full_attn_backend", attn_backend)
    fm = getattr(full_attn_backend, "forward_metadata", None)
    if fm is None:
        raise RuntimeError(
            "Kimi-K3 target-verify: MLA backend forward_metadata missing"
        )
    kv_indptr = getattr(fm, "kv_indptr", None)
    kv_indices = getattr(fm, "kv_indices", None)
    qo_indptr = getattr(fm, "qo_indptr", None)
    if kv_indptr is None or kv_indices is None or qo_indptr is None:
        raise RuntimeError("Kimi-K3 target-verify: MLA paged metadata missing")

    device = positions.device
    total_tokens = int(positions.shape[0])
    draft_num = int(forward_batch.spec_info.draft_token_num)
    # Static upper bound (shape, not value -> no device sync during capture).
    max_seqlen_k = int(full_attn_backend.req_to_token.shape[1])

    kv_last_page_lens = getattr(fm, "kv_last_page_len", None)
    if kv_last_page_lens is None:
        kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)
    context_lens = getattr(fm, "kv_lens", None)
    if context_lens is None:
        context_lens = kv_indptr[1 : bs + 1] - kv_indptr[:bs]

    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
    slot_mapping = (
        out_cache_loc[:total_tokens]
        if out_cache_loc is not None
        else torch.arange(total_tokens, dtype=torch.int64, device=device)
    )

    metadata = AttentionMetaData(
        cu_seqlens_q=qo_indptr,
        max_seqlen_q=draft_num,
        max_seqlen_k=max_seqlen_k,
        min_seqlen_q=draft_num,
        total_kv=bs * max_seqlen_k,
        has_cached=False,
        dropout_p=0.0,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        state=AttnState.DECODE,
    )
    metadata.dtype_q = dtype_q
    # Persistent scheduler buffers (note sglang uses `work_metadata`; ATOM's MLA
    # decode kernel reads `work_meta_data`). These are graph-stable backend buffers.
    metadata.work_meta_data = getattr(fm, "work_metadata", None)
    metadata.work_indptr = getattr(fm, "work_indptr", None)
    metadata.work_info_set = getattr(fm, "work_info_set", None)
    metadata.reduce_indptr = getattr(fm, "reduce_indptr", None)
    metadata.reduce_final_map = getattr(fm, "reduce_final_map", None)
    metadata.reduce_partial_map = getattr(fm, "reduce_partial_map", None)
    metadata.num_kv_splits = getattr(fm, "num_kv_splits", None)
    # Pin any capture-time transients (context_lens / kv_last_page_lens built here
    # when the backend didn't expose them) so a later smaller-bs capture cannot
    # reuse their memory and clobber the first-captured (max-bs) verify graph.
    from atom.plugin.sglang.kimi_k3_spec_verify import keepalive_if_capturing

    keepalive_if_capturing(metadata, context_lens, kv_last_page_lens)
    return metadata


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

    is_target_verify = bool(
        getattr(forward_batch.forward_mode, "is_target_verify", lambda: False)()
    )
    if is_target_verify and bs:
        return _build_kimi_k3_verify_metadata(
            forward_batch,
            positions,
            req_to_token_pool=req_to_token_pool,
            seq_lens=seq_lens,
            page_size=page_size,
            dtype_q=dtype_q,
            bs=bs,
        )

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
