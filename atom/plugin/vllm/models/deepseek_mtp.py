"""vLLM-specific DeepSeek MTP model extensions.

DEPRECATED -- this module no longer adds anything to the native model and is
safe to delete. It used to carry ``get_recycle_hidden``, which re-applied
``shared_head.norm`` so vLLM could recycle the post-final-norm hidden into the
next MTP step. That norm now runs at the end of
``DeepSeekMultiTokenPredictorLayer.forward``, so the model's own output is
already the state to recycle and the override became a duplicate of
``SharedHead.norm``.

``_ATOM_MODEL_REGISTRY`` maps ``DeepSeekMTPModel`` straight to
``atom.models.deepseek_mtp:DeepSeekMTP`` again, matching how every other MTP
arch is registered. The re-export below only keeps stale imports working.
"""

from atom.models.deepseek_mtp import DeepSeekMTP

__all__ = ["DeepSeekMTP"]
