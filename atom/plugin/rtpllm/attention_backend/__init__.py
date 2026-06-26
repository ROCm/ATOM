from .rtp_mla_attention import RTPMLAAttention, apply_attention_mla_rtpllm_patch
from .rtp_sparse_mla_backend import RTPSparseMlaBackend


def __getattr__(name):
    if name == "AttentionForRTPLLM":
        from .rtp_full_attention import AttentionForRTPLLM

        return AttentionForRTPLLM
    if name == "RTPFullAttention":
        from .rtp_full_attention import RTPFullAttention

        return RTPFullAttention
    if name == "RTPAttention":
        from .rtp_full_attention import RTPFullAttention

        return RTPFullAttention
    if name == "apply_attention_gdn_rtpllm_patch":
        from .attention_gdn import apply_attention_gdn_rtpllm_patch

        return apply_attention_gdn_rtpllm_patch
    if name == "apply_attention_mha_rtpllm_patch":
        from .attention_switch import apply_attention_mha_rtpllm_patch

        return apply_attention_mha_rtpllm_patch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AttentionForRTPLLM",
    "RTPFullAttention",
    "RTPMLAAttention",
    "RTPSparseMlaBackend",
    "apply_attention_gdn_rtpllm_patch",
    "apply_attention_mha_rtpllm_patch",
    "apply_attention_mla_rtpllm_patch",
]
