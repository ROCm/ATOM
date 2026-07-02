"""vLLM-Omni plugin integration for ATOM."""

from .register import register_omni_model, register_omni_platform

__all__ = ["register_omni_platform", "register_omni_model"]
