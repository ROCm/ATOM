# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K3's two mono decode kernels and the pieces they share.

``attention.kda`` is K1 (the attention seam through the KDA recurrence) and
``layer`` is K2 (``o_proj`` and its all-reduce, the MLP seam, the latent MoE);
``attention.back`` and ``stages.moe`` hold K2's two halves' stages, which
``layer`` places in one launch, and ``stages.gemv`` the bf16 GEMV both use.

Ported from vLLM PR #60683 (``vllm/models/kimi_k3/amd/mono``), whose device
layer was itself a fork of ``atom/mono``; the port reads those primitives from
``atom.mono`` directly and takes ATOM's mailbox discipline (zero and fence at a
step's start) in place of the fork's device epoch counter.
"""
