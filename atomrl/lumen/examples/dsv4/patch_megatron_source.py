#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Apply Lumen SOURCE patches to a Megatron-LM checkout (torch-less host entry).

Works on hosts without PyTorch (e.g. ``prepare_rocm_megatron.sh``) by stubbing
the ``atomrl.lumen`` package before loading the patch registry.

Examples::

    # Apply all default SOURCE patches (DSV4 + ROCm platform)
    PYTHONPATH=~/ATOM/atomrl python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM

    # List patches
    PYTHONPATH=~/ATOM/atomrl python3 examples/dsv4/patch_megatron_source.py --list
    PYTHONPATH=~/ATOM/atomrl python3 examples/dsv4/patch_megatron_source.py --list --tag dsv4
    PYTHONPATH=~/ATOM/atomrl python3 examples/dsv4/patch_megatron_source.py --list --tag rocm

    # Dry-run (no file changes)
    PYTHONPATH=~/ATOM/atomrl python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --dry-run --tag llama

    # Multiple tags in one flag (OR): DSV4 + standalone ROCm platform patches
    PYTHONPATH=~/ATOM/atomrl python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --tag dsv4,rocm

    # LLaMA + LoRA in one command (OR)
    PYTHONPATH=~/ATOM/atomrl python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM --tag llama,lora

    python3 examples/dsv4/patch_megatron_source.py /path/to/Megatron-LM
    python3 examples/dsv4/patch_megatron_source.py --list --tag dsv4
    python3 examples/dsv4/patch_megatron_source.py --list --tag dsv4,rocm

See ``examples/dsv4/PATCHES.md`` for IMPORT / CONFIG_BUILD / TRAINING runtime patches.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _ensure_lumen_package_stub() -> None:
    """Avoid ``lumen/__init__.py`` (torch) when only patching Megatron source."""
    lumen = sys.modules.get("atomrl.lumen")
    if lumen is not None and getattr(lumen, "__path__", None):
        return
    # examples/<model>/run_*.sh mounts this tree at /workspace/Lumen, so the
    # Lumen directory is not always <checkout>/atomrl/lumen. Accept both.
    root = Path(__file__).resolve().parents[2]
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
    _ensure_lumen_package_stub()
    from atomrl.lumen.patches.cli import main as cli_main

    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print(f"Usage: {sys.argv[0]} <megatron-root>")
        print(f"       {sys.argv[0]} --list [--tag dsv4] [--tag rocm]")
        return 1
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
