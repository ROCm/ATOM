#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""OBSOLETE — use ``install_swiglu_fp8()`` via ``import atomrl.lumen.models.megatron`` instead.

Set ``LUMEN_FUSED_SWIGLU_QUANT=1`` when needed. The Lumen import path patches
``SwiGLUFunction`` at runtime in ``lumen/patches/runtime/megatron_import.py``;
this disk script is only for Megatron-only workflows outside Lumen.

Patch fused_bias_swiglu.py to use ROCm-compatible FP8 dtype.

Megatron's SwiGLUFunction uses ``torch.float8_e4m3fn`` for FP8 activation
storage, but MI300X requires ``torch.float8_e4m3fnuz``.  This patch replaces
the hard-coded dtype with a runtime check.

Usage:
    python patch_swiglu_fp8_dtype.py /path/to/megatron_lm
"""

import sys
from pathlib import Path


def patch(megatron_root: str) -> None:
    fpath = (
        Path(megatron_root) / "megatron" / "core" / "fusions" / "fused_bias_swiglu.py"
    )
    if not fpath.exists():
        print(f"[patch_swiglu_fp8_dtype] {fpath} not found — skipping")
        return

    src = fpath.read_text()

    marker = "# patched-swiglu-fp8-dtype"
    if marker in src:
        print("[patch_swiglu_fp8_dtype] already applied — skipping")
        return

    helper = (
        f"\n{marker}\n"
        "def _get_fp8_e4m3_dtype():\n"
        "    import torch\n"
        "    if hasattr(torch, 'float8_e4m3fnuz'):\n"
        "        t = torch.zeros(1, device='cpu')\n"
        "        try:\n"
        "            t.to(torch.float8_e4m3fn)\n"
        "            return torch.float8_e4m3fn\n"
        "        except (RuntimeError, Exception):\n"
        "            return torch.float8_e4m3fnuz\n"
        "    return torch.float8_e4m3fn\n"
        "_FP8_E4M3 = _get_fp8_e4m3_dtype()\n"
    )

    old = "import torch\nimport torch.nn.functional as F"
    if old not in src:
        print("[patch_swiglu_fp8_dtype] anchor not found — skipping")
        return

    src = src.replace(old, old + helper, 1)

    src = src.replace(
        "input.to(torch.float8_e4m3fn) if fp8_input_store",
        "input.to(_FP8_E4M3) if fp8_input_store",
    )

    fpath.write_text(src)
    count = src.count("_FP8_E4M3")
    print(
        f"[patch_swiglu_fp8_dtype] patched {fpath} ({count} occurrences of _FP8_E4M3)"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <megatron_root>")
        sys.exit(1)
    patch(sys.argv[1])
