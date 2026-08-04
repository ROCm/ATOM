# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MI308-only SGLang external package hooks."""

from __future__ import annotations

import os

from atom.plugin.sglang import models as _base_models

# SGLang discovers external model classes by scanning ``package.__path__``.
# Reuse the shared adapter package path so MI308 gets the same registrations,
# while keeping the MI308 block-layout hook isolated to this external package.
__path__ = list(_base_models.__path__)


def _install_mi308_dsv4_block_layout() -> None:
    model_name = os.environ.get("SGLANG_MODEL_NAME", "").upper()
    if not model_name.startswith("MI308 DEEPSEEK-V4-FLASH"):
        return

    import atom.models.deepseek_v4 as dsv4
    from atom.plugin.sglang.deepseek_v4_bridge import ATOM_DEEPSEEK_V4_BLOCK_SIZE

    # The MI308 validation recipe uses the SGLang proxy KV pool's historical
    # 128-token compressed-KV pages. Keep the model-side compressor math aligned
    # without changing the shared DSV4 SGLang adapter that MI355 also imports.
    dsv4._V4_BLOCK_SIZE = ATOM_DEEPSEEK_V4_BLOCK_SIZE


_install_mi308_dsv4_block_layout()
