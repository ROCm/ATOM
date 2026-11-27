# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The engine step hooks KV transfer installs (`EngineCore.step_hooks`).

A step hook has two methods: `before_schedule()`, called ahead of every
`schedule()` of the engine loop, and `on_forward_done(seconds)`, called after
every real forward with its wall time. The engine knows nothing else of them.
"""

from __future__ import annotations


def kv_transfer_step_hooks(engine, config) -> tuple:
    """The step hooks for an engine with KV transfer configured."""
    from atom.kv_transfer.offload.load_wait import load_wait_hook

    hooks = (load_wait_hook(engine, config),)
    return tuple(hook for hook in hooks if hook is not None)
