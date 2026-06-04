"""ATOM vLLM platform integration."""

from atom.utils import envs

# This flag is used to enable the vLLM plugin mode.
disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN

if not disable_vllm_plugin:
    from vllm.platforms.rocm import RocmPlatform

    class ATOMPlatform(RocmPlatform):
        """ATOM platform wrapper.

        Attention backend selection is owned by ATOM's vLLM attention layers
        (`AttentionForVllm*`). We intentionally do not override
        `get_attn_backend_cls()` here, so any fallback vLLM standard attention
        keeps ROCmPlatform's native backend selection.
        """

else:
    ATOMPlatform = None
