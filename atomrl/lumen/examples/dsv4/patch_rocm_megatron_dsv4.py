#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Deprecated alias for :mod:`patch_megatron_source`.

Use ``examples/dsv4/patch_megatron_source.py`` or ``python3 -m atomrl.lumen.patches``.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path


def _load_main():
    path = Path(__file__).resolve().parent / "patch_megatron_source.py"
    spec = importlib.util.spec_from_file_location("patch_megatron_source", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.main


def _deprecated_main(argv=None):
    warnings.warn(
        "patch_rocm_megatron_dsv4.py is deprecated; use patch_megatron_source.py "
        "or python3 -m atomrl.lumen.patches",
        DeprecationWarning,
        stacklevel=2,
    )
    return _load_main()(argv)


if __name__ == "__main__":
    raise SystemExit(_deprecated_main(sys.argv[1:]))
