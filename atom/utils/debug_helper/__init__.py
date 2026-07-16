# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM debug helper — env-gated dump / compare / monkey-patch primitives.

Modules
-------
- ``dump``       Forward / weight / sampler dump hooks. All env-gated.
- ``compare``    cos_max + slot_split + warmup-vs-prefill picker; CLI:
                 ``python -m atom.utils.debug_helper.compare <subcommand>``.
- ``ref_patch``  Monkey-patch helpers for instrumenting read-only references.

Companion skill: ``.claude/skills/dump-bisect-debug.md`` for the methodology.

All controlling env vars are registered in ``atom/utils/envs.py`` "Debug Dump"
section. Set the ``*_DIR`` vars to enable; everything else has sensible
defaults. Unset = no-op.
"""

from atom.utils.debug_helper.compare import (
    COS_ALGO_DIFF,
    COS_BIT_EQUAL,
    COS_BUG,
    COS_NUM_DRIFT,
    byte_equal_pct,
    compare_slots,
    cos_max,
    flag_for,
    pick_prefill_call,
    schema_diff,
    slot_split,
)
from atom.utils.debug_helper.dump import (
    install_block_forward_hooks,
    maybe_dump_weights_and_exit,
    maybe_log_topk,
)
from atom.utils.debug_helper.ref_patch import (
    patch_block_forward,
    patch_method,
    patch_module_dump,
)

__all__ = [
    # dump
    "install_block_forward_hooks",
    "maybe_dump_weights_and_exit",
    "maybe_log_topk",
    # compare primitives
    "cos_max",
    "flag_for",
    "byte_equal_pct",
    "slot_split",
    "compare_slots",
    "pick_prefill_call",
    "schema_diff",
    "COS_BIT_EQUAL",
    "COS_NUM_DRIFT",
    "COS_ALGO_DIFF",
    "COS_BUG",
    # ref patching
    "patch_method",
    "patch_block_forward",
    "patch_module_dump",
]
