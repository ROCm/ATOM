import logging

from atom.plugin.prepare import is_rtpllm

logger = logging.getLogger("atom.plugin.rtpllm.attention_backend.attention_switch")

_PATCHED = False


def apply_attention_mha_rtpllm_patch() -> None:
    """Switch ATOM Attention to RTP-style adapter for rtpllm plugin mode."""

    global _PATCHED
    if _PATCHED:
        return

    import atom.model_ops as ops
    from .rtp_full_attention import RTPFullAttention

    if not is_rtpllm():
        return

    ops.Attention = RTPFullAttention
    logger.info(
        "Applied RTP-LLM attention patch: atom.model_ops.Attention -> RTPFullAttention."
    )
    _PATCHED = True
