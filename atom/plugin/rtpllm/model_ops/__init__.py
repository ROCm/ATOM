from .attention_gdn import apply_attention_gdn_rtpllm_patch
from .attention_switch import apply_attention_mha_rtpllm_patch

__all__ = ["apply_attention_gdn_rtpllm_patch", "apply_attention_mha_rtpllm_patch"]
