import functools

import torch
from atom.plugin.vllm.platform import disable_vllm_plugin_attention


def set_default_quant_scales(
    layer: torch.nn.Module, register_buffer: bool = False
) -> None:
    """Sets default quantization scales for the layer."""
    if register_buffer:
        layer.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_q_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_prob_scale", torch.tensor(1.0, dtype=torch.float32))
    else:
        layer._k_scale.fill_(1.0)
        layer._v_scale.fill_(1.0)
        layer._q_scale.fill_(1.0)
        layer._prob_scale.fill_(1.0)

    # We also keep q/k/v_scale on host (cpu) memory for attention
    # backends that require the scales to be on host instead of on device.
    # e.g. Flashinfer
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0
    layer._prob_scale_float = 1.0


def _patch_vllm_mla_attention_process_weights_after_loading(mla_attention_cls) -> None:
    """
    We patch the process_weights_after_loading method one reason is that
    orig_process_weights_after_loading need a act_dtype parameter,
    but in atom, we don't have this parameter. if disable_vllm_plugin_attention,
    we will fallback to original vllm attention backend.
    """
    orig_process_weights_after_loading = mla_attention_cls.process_weights_after_loading
    if getattr(
        orig_process_weights_after_loading,
        "_atom_mla_process_weights_after_loading_patched",
        False,
    ):
        return

    @functools.wraps(orig_process_weights_after_loading)
    def _process_weights_after_loading(self, act_dtype: torch.dtype = torch.bfloat16):
        if disable_vllm_plugin_attention:
            return orig_process_weights_after_loading(self, act_dtype)

        if hasattr(self.impl, "process_weights_after_loading"):
            self.impl.process_weights_after_loading()

        set_default_quant_scales(self, register_buffer=False)

    setattr(
        _process_weights_after_loading,
        "_atom_mla_process_weights_after_loading_patched",
        True,
    )
    mla_attention_cls.process_weights_after_loading = _process_weights_after_loading


def _patch_vllm_mla_attention_forward_impl(mla_attention_cls) -> None:
    """
    We patch the forward_impl method is to make qk rope and kv cache update
    can be fused in attention forward pass.
    """
    orig_forward_impl = mla_attention_cls.forward_impl
    if getattr(orig_forward_impl, "_atom_mla_forward_impl_patched", False):
        return

    @functools.wraps(orig_forward_impl)
    def _forward_impl(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if disable_vllm_plugin_attention:
            return orig_forward_impl(
                self,
                q,
                k_c_normed,
                k_pe,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        if hasattr(self.impl, "forward_impl_plugin_mode"):
            return self.impl.forward_impl_plugin_mode(
                self,
                q,
                k_c_normed,
                k_pe,
                kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        return orig_forward_impl(
            self,
            q,
            k_c_normed,
            k_pe,
            kv_cache,
            attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    setattr(_forward_impl, "_atom_mla_forward_impl_patched", True)
    mla_attention_cls.forward_impl = _forward_impl


def patch_vllm_mla_attention() -> None:
    try:
        from vllm.attention.layer import MLAAttention
    except ImportError:
        from vllm.model_executor.layers.attention import MLAAttention

    _patch_vllm_mla_attention_process_weights_after_loading(MLAAttention)
    _patch_vllm_mla_attention_forward_impl(MLAAttention)
