try:
    from .base_model_wrapper import ATOMGlm5Moe, ATOMQwen35Moe, ATOMQwen4Exp
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("rtp_llm"):
        raise
    ATOMGlm5Moe = None
    ATOMQwen35Moe = None
    ATOMQwen4Exp = None
else:
    try:
        from atom.plugin.register import _ATOM_SUPPORTED_MODELS
    except ImportError:
        # Unit tests may stub partial module trees and intentionally skip
        # full model imports. Keep wrapper symbols importable in that case.
        pass
    else:
        try:
            from atom.models.deepseek_v2 import GlmMoeDsaForCausalLM

            _ATOM_SUPPORTED_MODELS.setdefault(
                "GlmMoeDsaForCausalLM", GlmMoeDsaForCausalLM
            )
        except Exception:
            pass
        try:
            from atom.models.qwen4_exp import Qwen4ExpForConditionalGeneration

            _ATOM_SUPPORTED_MODELS.setdefault(
                "Qwen4ExpForConditionalGeneration",
                Qwen4ExpForConditionalGeneration,
            )
        except Exception:
            pass

__all__ = ["ATOMGlm5Moe", "ATOMQwen35Moe", "ATOMQwen4Exp"]
