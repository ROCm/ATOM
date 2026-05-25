# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Type

from atom.plugin.vllm.attention_backend.attention_gdn import GatedDeltaNet


class _PluginGatedDeltaNet(GatedDeltaNet):
    """Plugin-path GatedDeltaNet with vk layout hard-wired on.

    vLLM's attention-backend protocol resolves `get_impl_cls()` to a class
    and then constructs it with its own argument list — it doesn't accept
    pre-bound kwargs from us. To express the plugin-path constraint that
    ssm_state must be written in vk (`[V, K]`-per-head) layout — required
    by the decode kernels this path uses (fused_sigmoid_gating_delta_rule_update
    and flydsl_gdr_decode) — we subclass GatedDeltaNet and force the flag
    in __init__ regardless of what vLLM passes through **kwargs.

    See atom/plugin/vllm/attention_backend/attention_gdn.py for the
    layout semantics and why kv layout would corrupt this path.
    """

    def __init__(self, *args, **kwargs):
        # Force vk regardless of caller. Drop any caller-supplied value to
        # avoid silent override.
        kwargs.pop("use_vk_layout", None)
        super().__init__(*args, use_vk_layout=True, **kwargs)


class GDNAttentionBackend:
    @staticmethod
    def get_name() -> str:
        return "ROCM_GDN_ATTENTION"

    @staticmethod
    def get_impl_cls() -> Type["GatedDeltaNet"]:
        return _PluginGatedDeltaNet
