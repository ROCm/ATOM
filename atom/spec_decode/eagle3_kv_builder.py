import logging

import torch
from aiter import dtypes

from atom.config import KVCacheTensor

logger = logging.getLogger("atom")


class Eagle3DraftBuilder:
    """KV cache subsystem for a draft that owns its pool alongside a target
    whose pool it cannot share.

    Two draft flavors are handled, selected from the draft config:

    * **MHA** (default) — an Eagle3-style draft with a `[2, L, blocks,
      block_size, kv_heads, head_dim]` K/V-split cache. Used when the target
      is MLA (which the MHA draft cannot borrow).
    * **MLA** (`kv_lora_rank` present, e.g. a Kimi-K3 DSpark draft) — a single
      `kv_lora_rank + qk_rope_head_dim`-wide latent cache per layer, mirroring
      `aiter_mla.build_kv_cache_tensor`. Used when the target is a `kimi_linear`
      hybrid (MHA@192 full-attn + linear-attn), which allocates NO MLA pool for
      the draft to borrow — so the draft must own its own 576-wide latent cache.

    Implements the same subset of `AttentionMetadataBuilder` hooks that
    ModelRunner consults during KV pool sizing and per-module binding —
    `compute_block_bytes`, `allocate_kv_cache_tensors`, and
    `build_kv_cache_tensor` — so the draft's independent cache fits the
    post-#659 builder protocol without leaking into the target's builder. The
    draft does NOT drive prepare_decode/prepare_prefill; it piggybacks on the
    target builder's metadata flow during propose.
    """

    def __init__(self, model_runner, draft_hf):
        self.model_runner = model_runner
        self.draft_hf = draft_hf
        self.block_size = model_runner.block_size
        self.num_layers = draft_hf.num_hidden_layers
        self._next_layer_id = 0  # consumed by build_kv_cache_tensor
        self.num_blocks = 0  # set in allocate_kv_cache_tensors

        # An MLA draft stores a single compressed latent (kv_lora_rank) plus the
        # decoupled RoPE key (qk_rope_head_dim) — no per-head K/V split — so it
        # needs a fundamentally different cache layout from the MHA draft.
        self.kv_lora_rank = getattr(draft_hf, "kv_lora_rank", None)
        self.is_mla = self.kv_lora_rank is not None
        if self.is_mla:
            self.qk_rope_head_dim = draft_hf.qk_rope_head_dim
            self.mla_dim = self.kv_lora_rank + self.qk_rope_head_dim
            # Latent cache is single-headed; keep the attrs defined so the
            # shared helpers (get_kv_transfer_tensors) stay layout-agnostic.
            self.num_kv_heads = 1
            self.head_dim = self.mla_dim
        else:
            self.num_kv_heads = draft_hf.num_key_value_heads // model_runner.world_size
            self.head_dim = draft_hf.head_dim

    def compute_block_bytes(self) -> int:
        """Per-block bytes for the draft's independent KV cache."""
        kv_dtype_size = dtypes.d_dtypes[
            self.model_runner.config.kv_cache_dtype
        ].itemsize
        if self.is_mla:
            # Single latent cache: no K/V split, one "head" of width mla_dim,
            # no per-head dequant scale (MLA reads the latent directly). The pool
            # is allocated bf16 regardless of the target's kv_cache_dtype (see
            # allocate_kv_cache_tensors), so budget it at bf16's itemsize --
            # using the target's fp8 itemsize here under-counts the draft's
            # per-block cost 2x, inflating num_kvcache_blocks until the real
            # bf16 allocation overshoots the memory budget.
            return self.num_layers * self.block_size * self.mla_dim * dtypes.bf16.itemsize
        bb = (
            2
            * self.num_layers
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * kv_dtype_size
        )
        if self.model_runner.config.kv_cache_dtype == "fp8":
            # fp8 KV cache needs an extra per-(layer, block, kv_head) scale
            # tensor (one fp32 per element) to dequantize fp8 → bf16 at
            # attention time. Reserve that space alongside the cache.
            bb += (
                2
                * self.num_layers
                * self.block_size
                * self.num_kv_heads
                * dtypes.fp32.itemsize
            )
        return bb

    def allocate_kv_cache_tensors(self, num_kv_heads, num_draft_layers) -> dict:
        """Allocate the draft's independent KV pool under namespaced keys so it
        does not collide with the target builder's `kv_cache` / `kv_scale`.

        MHA: `[2, L, blocks, block_size, kv_heads, head_dim]` cache + fp32 scale.
        MLA: a single `[L, blocks, block_size, mla_dim]` latent cache (no scale).
        """
        runner = self.model_runner
        config = runner.config
        # Draft's block budget scales with the target pool: same total token
        # capacity, just paged at the draft's own block size.
        self.num_blocks = (
            config.num_kvcache_blocks * runner.block_size // self.block_size
        )
        if self.is_mla:
            # bf16 regardless of --kv_cache_dtype. This is a SIBLING pool, so
            # nothing forces it to match the target's cache dtype, and the MLA
            # draft carries no fp8 scales: _build_mla_kv_cache_tensor binds it
            # with k_scale=None, and the draft model hardcodes the matching
            # "bf16" into its Attention. An fp8 latent pool here would need a
            # calibrated scale the draft checkpoint does not ship.
            cache = torch.zeros(
                self.num_layers,
                self.num_blocks,
                self.block_size,
                self.mla_dim,
                dtype=dtypes.d_dtypes["bf16"],
                device="cuda",
            )
            logger.info(
                "Allocated DSpark MLA draft KV cache: %s bf16 (target KV is %s)",
                tuple(cache.shape),
                config.kv_cache_dtype,
            )
            return {"eagle3_kv_cache": cache}
        cache = torch.zeros(
            2,
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            dtype=dtypes.d_dtypes[config.kv_cache_dtype],
            device="cuda",
        )
        scale = torch.zeros(
            2,
            self.num_layers,
            self.num_blocks,
            self.num_kv_heads,
            self.block_size,
            dtype=dtypes.fp32,
            device="cuda",
        )
        logger.info(f"Allocated Eagle3 draft KV cache: {cache.shape}")
        return {"eagle3_kv_cache": cache, "eagle3_kv_scale": scale}

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Bind one draft attention module to its slice of the independent draft
        KV cache. Returns None for modules this builder does not own (wrong
        attention flavor, or not an attention at all) so ModelRunner falls
        through to the target builder.
        """
        if not (hasattr(module, "base_attention") and hasattr(module, "use_mla")):
            return None
        if self.is_mla:
            if not module.use_mla:
                return None
            return self._build_mla_kv_cache_tensor(layer_id, module)
        if module.use_mla:
            return None
        runner = self.model_runner
        idx = self._next_layer_id
        self._next_layer_id += 1
        cache = runner.eagle3_kv_cache
        x = 16 // cache.element_size()
        k_cache = cache[0, idx].view(
            self.num_blocks,
            self.num_kv_heads,
            self.head_dim // x,
            self.block_size,
            x,
        )
        v_cache = cache[1, idx].view(
            self.num_blocks,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        )
        module.max_model_len = runner.config.max_model_len
        if runner.config.kv_cache_dtype == "fp8":
            module.k_scale = runner.eagle3_kv_scale[0, idx]
            module.v_scale = runner.eagle3_kv_scale[1, idx]
        module.k_cache = k_cache
        module.v_cache = v_cache
        return KVCacheTensor(
            layer_num=layer_id,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=getattr(module, "k_scale", None),
            v_scale=getattr(module, "v_scale", None),
        )

    def _build_mla_kv_cache_tensor(self, layer_id: int, module):
        """Bind one MLA draft attention module to its `(N*block_size, 1,
        mla_dim)` latent slice, matching `aiter_mla.build_kv_cache_tensor` so
        `concat_and_cache_mla` sees `kv_cache.size(2) == kv_lora_rank +
        qk_rope_head_dim`.
        """
        runner = self.model_runner
        idx = self._next_layer_id
        self._next_layer_id += 1
        kv_cache = runner.eagle3_kv_cache[idx].view(
            self.num_blocks * self.block_size,
            1,
            self.mla_dim,
        )
        module.max_model_len = runner.config.max_model_len
        module.kv_cache = kv_cache
        return KVCacheTensor(
            layer_num=layer_id,
            k_cache=kv_cache,
            v_cache=None,
            k_scale=None,
            v_scale=None,
        )

    def get_kv_transfer_tensors(self) -> list:
        from atom.kv_transfer.disaggregation.types import KVTransferRegion

        runner = self.model_runner
        if not hasattr(runner, "eagle3_kv_cache"):
            return []

        regions: list[KVTransferRegion] = []
        cache = runner.eagle3_kv_cache
        if self.is_mla:
            # Single latent cache indexed by layer; no K/V split, no scale.
            for layer_id in range(self.num_layers):
                t = cache[layer_id]
                regions.append(
                    KVTransferRegion(
                        base_addr=t.data_ptr(),
                        total_bytes=t.numel() * t.element_size(),
                        unit_bytes=t.stride(0) * t.element_size(),
                    )
                )
            return regions
        for layer_id in range(self.num_layers):
            for kv in range(2):
                t = cache[kv, layer_id]
                regions.append(
                    KVTransferRegion(
                        base_addr=t.data_ptr(),
                        total_bytes=t.numel() * t.element_size(),
                        unit_bytes=t.stride(0) * t.element_size(),
                    )
                )
        scale = getattr(runner, "eagle3_kv_scale", None)
        if (
            self.model_runner.config.kv_cache_dtype == "fp8"
            and scale is not None
            and scale.numel() > 0
        ):
            for layer_id in range(self.num_layers):
                for kv in range(2):
                    t = scale[kv, layer_id]
                    regions.append(
                        KVTransferRegion(
                            base_addr=t.data_ptr(),
                            total_bytes=t.numel() * t.element_size(),
                            unit_bytes=t.stride(0) * t.element_size(),
                        )
                    )
        return regions
