"""ATOM vLLM platform integration.

This module contains the vLLM `Platform` subclass used in ATOM's vLLM plugin
mode. Keep platform behavior here so `register.py` can focus on registration
and wiring only.
"""

import logging
import os

logger = logging.getLogger("atom")

# This flag is used to enable the vLLM plugin mode.
disable_vllm_plugin = os.getenv("ATOM_DISABLE_VLLM_PLUGIN", "0").lower() == "1"
disable_vllm_plugin_attention = (
    os.getenv("ATOM_DISABLE_VLLM_PLUGIN_ATTENTION", "0").lower() == "1"
)

if not disable_vllm_plugin:
    from vllm.platforms.rocm import RocmPlatform

    class ATOMPlatform(RocmPlatform):
        # For multi-modality models, to make AiterBackend supported by ViT,
        # get_supported_vit_attn_backends may need to be overridden here
        @classmethod
        def get_attn_backend_cls(cls, selected_backend, attn_selector_config) -> str:
            if disable_vllm_plugin_attention:
                logger.info("Fallback to original vLLM attention backend")
                return super().get_attn_backend_cls(
                    selected_backend, attn_selector_config
                )

            logger.info("Use atom attention backend")
            return "atom.model_ops.attentions.aiter_attention.AiterBackend"
else:
    ATOMPlatform = None
