"""vLLM plugin integration for ATOM."""

from .register import register_model, register_platform

__all__ = ["register_platform", "register_model"]
