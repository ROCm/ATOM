import logging

import torch
from aiter import dtypes

from atom.config import KVCacheTensor
from atom.model_ops.attentions.sub_pool_spec import SubPoolSpec, page_pool
from atom.utils import envs

logger = logging.getLogger("atom")


class Eagle3DraftBuilder:
    """KV cache subsystem for an Eagle3 MHA draft alongside a non-MHA target.

    Implements the same subset of `AttentionMetadataBuilder` hooks that
    ModelRunner consults during KV pool sizing and per-module binding —
    `sub_pool_specs`, `allocate_kv_cache_tensors`, and
    `build_kv_cache_tensor` — so the draft's independent cache fits the
    post-#659 builder protocol without leaking into the target's builder. The
    draft does NOT drive prepare_decode/prepare_prefill; it piggybacks on the
    target builder's metadata flow during propose.

    Scoped to MHA drafts on purpose. An MLA draft's cache is the same 576-wide
    latent an MLA target's own layers use, so it binds into the target's pool
    as extra rows instead (both EagleProposer and DSparkProposer fork on
    `kv_lora_rank` and only reach here for the MHA flavor).
    """

    def __init__(self, model_runner, draft_hf):
        self.model_runner = model_runner
        self.draft_hf = draft_hf
        self.block_size = model_runner.block_size
        self.num_layers = draft_hf.num_hidden_layers
        self._next_layer_id = 0  # consumed by build_kv_cache_tensor
        self.num_blocks = 0  # set in allocate_kv_cache_tensors
        self.num_kv_heads = draft_hf.num_key_value_heads // model_runner.world_size
        self.head_dim = draft_hf.head_dim

    def sub_pool_specs(self) -> list[SubPoolSpec]:
        """The draft's independent KV cache.

        `page_pool` puts it in the same entry class as the target builder's
        pool: the draft KV rides the target's block ids, so the two
        contributions sum into one per-block cost rather than forming a
        second pool.
        """
        kv_dtype_size = dtypes.d_dtypes[
            self.model_runner.config.kv_cache_dtype
        ].itemsize
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
        return [page_pool(bb)]

    def allocate_kv_cache_tensors(self, num_kv_heads, num_draft_layers) -> dict:
        """Allocate the draft's independent KV pool under namespaced keys so it
        does not collide with the target builder's `kv_cache` / `kv_scale`.

        Layout is `[2, L, blocks, block_size, kv_heads, head_dim]` plus an fp32
        scale tensor when the cache dtype is fp8.
        """
        runner = self.model_runner
        config = runner.config
        # Draft's block budget scales with the target pool: same total token
        # capacity, just paged at the draft's own block size.
        self.num_blocks = (
            config.num_kvcache_blocks * runner.block_size // self.block_size
        )
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

    @staticmethod
    def _use_flash_layout(impl) -> bool:
        """Whether to hand this draft module a flash (4D) view of the pool.

        Mirrors the branch condition in `rope_cache`: of its three writers only
        `fused_qk_rope_reshape_and_cache` emits a 4D VHD V, which no prefill
        reader consumes. Those modules get flash instead (same kernel, one flag
        flipped). The other two already write a SHUFFLE V that both prefill and
        decode read, so a flash pool would only break them.
        """
        if impl is None or getattr(impl, "rotary_emb", None) is None:
            return False
        # rope_cache's first branch re-views V to SHUFFLE itself.
        if (
            getattr(impl, "q_norm", None) is not None
            and getattr(impl, "k_norm", None) is not None
        ):
            return False
        # rope_cache's `use_triton_attn`.
        return bool(
            envs.ATOM_FORCE_ATTN_TRITON
            or impl.sliding_window != -1
            or impl.head_dim != 128
        )

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Bind one draft attention module to its slice of the independent draft
        KV cache. Returns None for modules this builder does not own (wrong
        attention flavor, or not an attention at all) so ModelRunner falls
        through to the target builder.
        """
        if not (hasattr(module, "base_attention") and hasattr(module, "use_mla")):
            return None
        if module.use_mla:
            return None
        runner = self.model_runner
        idx = self._next_layer_id
        self._next_layer_id += 1
        cache = runner.eagle3_kv_cache
        impl = getattr(module, "impl", None)
        flash = self._use_flash_layout(impl)
        if flash:
            # The pool is allocated [.., num_blocks, block_size, kv_heads,
            # head_dim] -- already the flash layout, so take it unviewed.
            k_cache = cache[0, idx]
            v_cache = cache[1, idx]
        else:
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
        if impl is not None:
            impl.use_flash_layout = flash
        # Which layout a draft pool got is otherwise invisible: both are views of
        # the same allocation, so the shapes are the only thing that tells them
        # apart, and reading the wrong one corrupts attention without faulting.
        logger.info(
            "Eagle3 draft layer %d KV layout: %s (K %s, V %s)",
            layer_id,
            "flash" if flash else "SHUFFLE-K/VHD-V",
            tuple(k_cache.shape),
            tuple(v_cache.shape),
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

    def get_kv_transfer_tensors(self) -> list:
        from atom.kv_transfer.disaggregation.types import KVTransferRegion

        runner = self.model_runner
        if not hasattr(runner, "eagle3_kv_cache"):
            return []

        regions: list[KVTransferRegion] = []
        cache = runner.eagle3_kv_cache
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
