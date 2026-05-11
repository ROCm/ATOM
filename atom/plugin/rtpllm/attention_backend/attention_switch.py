import logging

from atom.plugin.prepare import is_rtpllm

logger = logging.getLogger("atom.plugin.rtpllm.attention_backend.attention_switch")

_PATCHED = False
_ORIGINAL_ATTENTION_CLS = None


def apply_attention_mha_rtpllm_patch() -> None:
    """Switch ATOM Attention to RTP-style adapter for rtpllm plugin mode."""

    global _PATCHED, _ORIGINAL_ATTENTION_CLS
    if _PATCHED:
        return

    import atom.model_ops as ops
    from .rtp_full_attention import RTPFullAttention

    if _ORIGINAL_ATTENTION_CLS is None:
        _ORIGINAL_ATTENTION_CLS = getattr(ops, "Attention", None)

    if is_rtpllm():
        ops.RTPFullAttention = RTPFullAttention
        ops.Attention = RTPFullAttention
        logger.info(
            "Applied RTP-LLM attention patch: atom.model_ops.Attention -> RTPFullAttention."
        )

    _PATCHED = True
