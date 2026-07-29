"""ATOM vLLM-Omni platform integration.

This module contains the vLLM-Omni `OmniPlatform` subclass used in ATOM's
vLLM-Omni plugin mode. Overrides both AR and diffusion attention backend
selection to use ATOM implementations.
"""

import logging

from atom.utils import envs

logger = logging.getLogger("atom")
# This flag is used to enable the vLLM-Omni plugin mode.
disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN
disable_vllm_plugin_attention = envs.ATOM_DISABLE_VLLM_PLUGIN_ATTENTION

if not disable_vllm_plugin:
    from vllm_omni.platforms.rocm.platform import RocmOmniPlatform

    class ATOMOmniPlatform(RocmOmniPlatform):
        @classmethod
        def get_attn_backend_cls(
            cls, selected_backend, attn_selector_config, num_heads
        ) -> str:
            if disable_vllm_plugin_attention:
                logger.info("Fallback to original vLLM attention backend")
                return super().get_attn_backend_cls(
                    selected_backend, attn_selector_config, num_heads
                )

            logger.info("Use atom attention backend")
            if attn_selector_config.use_mla:
                return "atom.model_ops.attentions.aiter_mla.AiterMLABackend"
            return "atom.model_ops.attentions.aiter_attention.AiterBackend"

        @classmethod
        def get_diffusion_attn_backend_cls(
            cls, selected_backend: str | None, head_size: int
        ) -> str:
            if disable_vllm_plugin_attention:
                logger.info(
                    "Fallback to original vLLM-Omni diffusion attention backend"
                )
                return super().get_diffusion_attn_backend_cls(
                    selected_backend, head_size
                )

            # Respect env var override for non-FLASH_ATTN backends
            # (TORCH_SDPA, SAGE_ATTN, etc.)
            if (
                selected_backend is not None
                and selected_backend.upper() != "FLASH_ATTN"
            ):
                return super().get_diffusion_attn_backend_cls(
                    selected_backend, head_size
                )

            logger.info("Use atom diffusion attention backend")
            return (
                "atom.plugin.vllm_omni.diffusion.attention_backend.flash_attn"
                ".AiterFlashAttentionBackend"
            )

else:
    ATOMOmniPlatform = None
