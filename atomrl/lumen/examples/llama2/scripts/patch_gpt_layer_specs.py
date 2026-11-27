#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Deprecated alias — use ``examples/dsv4/patch_megatron_source.py --tag llama`` instead.

This script applied FusedLayerNorm + RMSNorm fixes directly. The logic now lives
in ``lumen/patches/source/llama.py`` (PatchPhase.SOURCE).
"""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path


def _ensure_lumen_stub() -> None:
    lumen = sys.modules.get("atomrl.lumen")
    if lumen is not None and getattr(lumen, "__path__", None):
        return
    # This file is examples/<model>/scripts/*.py, so parents[3] is the Lumen
    # tree. That tree is <checkout>/atomrl/lumen on the host and /workspace/Lumen
    # when an example launcher bind-mounts it. Accept both.
    root = Path(__file__).resolve().parents[3]
    candidates = [
        root.parent / "atomrl" / "lumen",
        root.parent / "ATOM" / "atomrl" / "atomrl" / "lumen",
    ]
    lumen_dir = next(
        (path for path in candidates if (path / "patches" / "cli.py").is_file()),
        None,
    )
    if lumen_dir is None:
        raise SystemExit(
            "could not find atomrl.lumen next to "
            + str(root)
            + "; looked in "
            + ", ".join(str(path) for path in candidates)
        )
    stub = types.ModuleType("atomrl.lumen")
    stub.__path__ = [str(lumen_dir)]
    sys.modules["atomrl.lumen"] = stub


def main(argv: list[str] | None = None) -> int:
    warnings.warn(
        "patch_gpt_layer_specs.py is deprecated; use "
        "'PYTHONPATH=<ATOM>/atomrl python3 examples/dsv4/patch_megatron_source.py <megatron-root> --tag llama'",
        DeprecationWarning,
        stacklevel=2,
    )
    _ensure_lumen_stub()
    from atomrl.lumen.patches.cli import main as cli_main

    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print(f"Usage: {sys.argv[0]} <megatron-root>")
        return 1
    return cli_main([*argv, "--tag", "llama"])


if __name__ == "__main__":
    raise SystemExit(main())
