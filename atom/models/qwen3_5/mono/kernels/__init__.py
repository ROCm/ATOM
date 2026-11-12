# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.5 mono layer kernel: ``layer`` builds the one launch of a decoder layer (Gated
DeltaNet or gated full attention, then the MoE) from the stages (``gdn``, ``attn`` or, under
``ATOM_PA_FLYDSL``, ``attn_flydsl``, ``mixer``, ``ffn``) over the helpers ``common`` binds to it
(``helpers``, ``ops``)."""
