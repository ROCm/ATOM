try:
    from .base_model_wrapper import ATOMGlm5Moe, ATOMQwen35Moe
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("rtp_llm"):
        raise
    ATOMGlm5Moe = None
    ATOMQwen35Moe = None
else:
    try:
        from atom.models.deepseek_v2 import GlmMoeDsaForCausalLM
        from atom.plugin.register import _ATOM_SUPPORTED_MODELS
    except ImportError:
        # Unit tests may stub partial module trees and intentionally skip
        # full model imports. Keep wrapper symbols importable in that case.
        pass
    else:
        _ATOM_SUPPORTED_MODELS.setdefault("GlmMoeDsaForCausalLM", GlmMoeDsaForCausalLM)

__all__ = ["ATOMGlm5Moe", "ATOMQwen35Moe"]
