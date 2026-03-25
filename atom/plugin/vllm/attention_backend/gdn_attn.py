# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Type

from atom.plugin.vllm.attention_backend.attention_gdn import GatedDeltaNet


class GDNAttentionBackend:
    @staticmethod
    def get_name() -> str:
        return "ROCM_GDN_ATTENTION"

    @staticmethod
    def get_impl_cls() -> Type["GatedDeltaNet"]:
        return GatedDeltaNet
