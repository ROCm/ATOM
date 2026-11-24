#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Deprecated — use ``examples/dsv4/patch_megatron_source.py <megatron-root> --tag lora``."""

from __future__ import annotations

import sys

from _patch_lora_registry_shim import main


def main_legacy():
    megatron_root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/megatron_lm"
    return main([megatron_root])


if __name__ == "__main__":
    raise SystemExit(main_legacy())
