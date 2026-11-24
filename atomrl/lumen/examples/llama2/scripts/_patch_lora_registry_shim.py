#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Deprecated — use ``examples/dsv4/patch_megatron_source.py <megatron-root> --tag llama,lora``."""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path


def _cli_main(argv: list[str]) -> int:
    _ensure_lumen_stub()
    from atomrl.lumen.patches.cli import main as registry_main

    return registry_main([*argv, "--tag", "lora"])


def _ensure_lumen_stub() -> None:
    lumen = sys.modules.get("atomrl.lumen")
    if lumen is not None and getattr(lumen, "__path__", None):
        return
    root = Path(__file__).resolve().parents[3]
    stub = types.ModuleType("atomrl.lumen")
    stub.__path__ = [str(root.parent / "atomrl" / "lumen")]
    sys.modules["atomrl.lumen"] = stub


def main(argv: list[str] | None = None) -> int:
    warnings.warn(
        f"{Path(__file__).name} is deprecated; use "
        "'PYTHONPATH=<ATOM>/atomrl python3 examples/dsv4/patch_megatron_source.py <megatron-root> --tag llama,lora'",
        DeprecationWarning,
        stacklevel=2,
    )
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print(f"Usage: {sys.argv[0]} <megatron-root>")
        return 1
    return _cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
