# SPDX-License-Identifier: MIT
"""Environment variables of DeepSeek-V4.1's radix (prefix) cache on the SGLang
plugin. They configure the plugin alone, so they are read here rather than in
``atom.utils.envs``, with the same parsing; each is read when accessed
(``deepseek_v41_envs.ATOM_V41_RADIX``), so a ``monkeypatch.setenv`` in a test
takes effect. Documented in ``recipes/atom_sglang/DeepSeek-V4.1-Flash.md``.
"""

import os
from collections.abc import Callable
from typing import Any


def _flag_env(name: str, default: str = "0") -> bool:
    # As `atom.utils.envs._flag_env`: stripped, and empty reads as off.
    raw = os.getenv(name, default).strip().lower()
    return bool(raw) and raw not in ("0", "false", "no", "off")


def _int_env(name: str, default: int) -> int:
    # As `atom.utils.envs._int_env`: unset or empty reads as ``default``;
    # anything else must be an integer.
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


environment_variables: dict[str, Callable[[], Any]] = {
    # Opt in to radix (prefix) cache for DeepSeek-V4.1 on the SGLang plugin.
    # Off by default until the fused replay below is validated on GPU.
    "ATOM_V41_RADIX": lambda: _flag_env("ATOM_V41_RADIX"),
    # DeepSeek-V4.1 on the SGLang plugin with radix cache on: the rows below a
    # prefix hit recomputed into the request's fresh STATE slot before its
    # extend (`deepseek_v41_radix`). Even, at least the sliding window.
    "ATOM_V41_RADIX_REPLAY_ROWS": lambda: _int_env("ATOM_V41_RADIX_REPLAY_ROWS", 128),
    # Where those rows run: "fused" (default) prepends them to the request's
    # own span in the step's forward; "side" runs them in a separate forward
    # before it (the first implementation, kept for A/B).
    "ATOM_V41_RADIX_REPLAY_MODE": lambda: (
        os.getenv("ATOM_V41_RADIX_REPLAY_MODE", "fused").strip().lower()
    ),
    # Diagnostics for the same: per replay (side forward or fused step), a
    # checksum of every shared PAGE below each hit before and after it, and a
    # TP check of the step's replay row count.
    "ATOM_V41_RADIX_DEBUG": lambda: _flag_env("ATOM_V41_RADIX_DEBUG"),
    # Test hook for the same: the first N steps that would fuse their replay
    # take the side-forward fallback (the path a step over the builder takes).
    "ATOM_V41_RADIX_TEST_SIDE_STEPS": lambda: _int_env(
        "ATOM_V41_RADIX_TEST_SIDE_STEPS", 0
    ),
}


def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
