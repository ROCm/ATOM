# SPDX-License-Identifier: MIT
"""Environment variables of DeepSeek-V4.1 on the SGLang plugin.

They configure the plugin alone, so they are registered here rather than in
``atom.utils.envs``, and parsed by its ``_flag_env``; each is read when
accessed (``deepseek_v41_envs.ATOM_SGLANG_V41_DECODER_REPLAY``), so a
``monkeypatch.setenv`` in a test takes effect. Documented in
``recipes/atom_sglang/DeepSeek-V4.1-Flash.md``.
"""

from collections.abc import Callable
from typing import Any

from atom.utils.envs import _flag_env

environment_variables: dict[str, Callable[[], Any]] = {
    # Decoder SWA bounded replay (a prefill runs V4.1's decoder on each
    # request's last window rows only): on, as natively; 0 / false / no / off,
    # or an empty value, turn it off, as native's
    # --no-decoder-swa-bounded-replay does.
    "ATOM_SGLANG_V41_DECODER_REPLAY": lambda: _flag_env(
        "ATOM_SGLANG_V41_DECODER_REPLAY", "1"
    ),
    # Radix (prefix) cache for DeepSeek-V4.1 on the SGLang plugin
    # (`deepseek_v41_radix`): on whenever SGLang's radix cache is. 0 / false /
    # no / off, or an empty value, turn SGLang's radix cache off at launch
    # (`deepseek_v41_radix.v41_radix_launch_override`).
    "ATOM_SGLANG_V41_RADIX": lambda: _flag_env("ATOM_SGLANG_V41_RADIX", "1"),
    # Performance runs only: let a DSpark launch run SGLang's
    # SGLANG_SIMULATE_ACC_LEN above 1 (a fixed mean accept length; the text
    # is then not the model's). Off: such a launch is refused.
    "ATOM_SGLANG_V41_DSPARK_ALLOW_SIMULATED_ACC": lambda: _flag_env(
        "ATOM_SGLANG_V41_DSPARK_ALLOW_SIMULATED_ACC"
    ),
}


def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
