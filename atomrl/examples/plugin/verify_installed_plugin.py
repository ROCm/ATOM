# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZhangDanyang-AMD
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Verify one pinned host framework against the installed AtomRL adapter."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect


def verify_verl() -> None:
    importlib.import_module("verl")
    from atomrl.plugin.verl.register import register
    from verl.workers.rollout.base import _ROLLOUT_REGISTRY

    assert importlib.metadata.version("verl") == "0.9.1"
    register()
    assert ("atom", "async") in _ROLLOUT_REGISTRY
    entry_points = importlib.metadata.entry_points(group="verl.plugins")
    assert any(
        entry_point.value == "atomrl.plugin.verl.register:register"
        for entry_point in entry_points
    )


def verify_cli_framework(framework: str) -> None:
    importlib.import_module(framework)
    importlib.import_module(f"{framework}.rollout.base_types")
    importlib.import_module(f"{framework}.utils.types")
    hooks = importlib.import_module(f"atomrl.plugin.{framework}.hooks")
    lifecycle = importlib.import_module(f"atomrl.plugin.{framework}.lifecycle")

    assert list(inspect.signature(hooks.custom_megatron_init).parameters) == ["args"]
    assert list(inspect.signature(hooks.generate_rollout).parameters) == [
        "args",
        "rollout_id",
        "data_source",
        "evaluation",
    ]
    assert callable(hooks.custom_update_weight_post_write)
    assert callable(lifecycle.setup_atom_manager)
    assert callable(lifecycle.teardown_atom_manager)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("framework", choices=("verl", "vime", "miles", "slime"))
    framework = parser.parse_args().framework
    if framework == "verl":
        verify_verl()
    else:
        verify_cli_framework(framework)
    print(f"{framework} plugin verification passed")


if __name__ == "__main__":
    main()
