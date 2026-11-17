# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) ``tilert/pd_vllm``'s DeepSeek-V4.1 receive side.
"""The decode host's side of the TileRT PD wire for DeepSeek-V4.1-Flash: the control-plane messages and the landed
buffer's layout (``wire``), the conversion of a landed request into the decode caches (``convert``), the receive
path (``receive``: the control plane, its slot, the RDMA-registered buffer), a request's decode over a device
interface (``session``) and the router-facing HTTP host (``server``).
"""
