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
    # A DSpark decode round launches its draft block before the host waits
    # for the previous round's published lengths
    # (`dspark_deferred_seq_lens_patch`); 0 keeps SGLang's wait. Read at
    # install.
    "ATOM_SGLANG_V41_DSPARK_DEFERRED_SEQ_LENS": lambda: _flag_env(
        "ATOM_SGLANG_V41_DSPARK_DEFERRED_SEQ_LENS", "1"
    ),
    # A DSpark round's PAGE rows its verify would read back are copied before
    # the draft launch (`dspark_verify_rows_prefetch_patch`); 0 reads them
    # back behind the draft block. Read at install.
    "ATOM_SGLANG_V41_DSPARK_PREFETCH_ROWS": lambda: _flag_env(
        "ATOM_SGLANG_V41_DSPARK_PREFETCH_ROWS", "1"
    ),
    # An arriving EXTEND takes its PAGE rows from a copy queued where SGLang
    # allocated them (`v41_extend_rows_at_alloc_patch`); 0 reads them back at
    # staging, behind the step in flight. Read at install.
    "ATOM_SGLANG_V41_EXTEND_ROWS_AT_ALLOC": lambda: _flag_env(
        "ATOM_SGLANG_V41_EXTEND_ROWS_AT_ALLOC", "1"
    ),
    # Prefill token graphs (`v41_prefill_token_graph_patch`), as native
    # records them; 0 runs every prefill eagerly.
    "ATOM_SGLANG_V41_PREFILL_TOKEN_GRAPHS": lambda: _flag_env(
        "ATOM_SGLANG_V41_PREFILL_TOKEN_GRAPHS", "1"
    ),
    # One eager prefill at the chunk budget once SGLang's graphs are done
    # (`deepseek_v41_bridge.warm_deepseek_v41_eager_prefill`, native's
    # `warmup_model`), with or without graphs; 0 skips it.
    "ATOM_SGLANG_V41_PREFILL_WARMUP": lambda: _flag_env(
        "ATOM_SGLANG_V41_PREFILL_WARMUP", "1"
    ),
    # All-greedy sampling picks with aiter's exact top-1, native's kernel,
    # instead of torch.argmax (`v41_greedy_sampler_patch`); 0 keeps SGLang's
    # sampler. Read at install.
    "ATOM_SGLANG_V41_FAST_GREEDY": lambda: _flag_env(
        "ATOM_SGLANG_V41_FAST_GREEDY", "1"
    ),
}


def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
