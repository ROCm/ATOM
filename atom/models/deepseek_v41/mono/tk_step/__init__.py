# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek V4.1 decode steps served by the TileRT-derived TP8 decode step (``mono.hi.tk.TkRank``: the recorded
mode-5 DSpark step's 118 launches a rank, one HIP graph a parity) behind ATOM's step override
(``atom/mono/runtime/step_override.py``), so requests keep ATOM's own server, scheduler and PD.

A TkRank holds the request's state itself (TileRT-layout caches, its logs, its drafts): it admits a request
(``set_sampling`` / ``admit_kernel`` / the request's KV converted from ATOM's caches, ``kv``), then serves every
decode step of it (``runner``); a served request never returns to the production path, whose caches no longer hold
its KV (``dispatch``). Contract and design: runs/20261115-flydsl-tilert/m7/STEP_OVERRIDE.md.

Switched by ``ATOM_DSV41_TK_STEP`` (``1``; default ``0``) on the one deployment it is written for (``refusal``).
"""

from atom.utils import envs

TP = 8  # the recorded plan's TP
SPEC_TOKENS = 5
ROWS = SPEC_TOKENS + 1  # one request's verify: the anchor and its 5 drafts
MAX_SEQ_LEN = 1 << 20  # the plan's context (its kernargs and workspaces)


def enabled() -> bool:
    return envs.ATOM_DSV41_TK_STEP


def refusal(atom_config) -> str | None:
    """Why this deployment cannot run the TileRT-path step, or None."""
    from atom.models.deepseek_v41.mono.kernels.dims import Dims
    from atom.mono.runtime.deployment import common_refusals, first_refusal

    spec = atom_config.speculative_config
    method = None if spec is None else spec.method
    checks = common_refusals(atom_config, Dims, (TP,)) + [
        (method == "dspark", f"speculative method {method}"),
        (
            spec is not None and spec.num_speculative_tokens == SPEC_TOKENS,
            "DSpark block size",
        ),
        (not atom_config.dspark.confidence_schedule, "DSpark confidence schedule"),
        (not atom_config.dspark.ragged, "DSpark ragged verify"),
        # one request at a time (the plan's step is one request's 6 rows), and a request it served cannot fall
        # back: the production caches do not hold the KV it decoded
        (atom_config.max_num_seqs == 1, f"max_num_seqs {atom_config.max_num_seqs}"),
        (
            atom_config.max_model_len <= MAX_SEQ_LEN,
            f"max_len {atom_config.max_model_len}",
        ),
        # the KV of the tokens it decodes never reaches ATOM's paged cache: a cached prefix over them would be
        # read back by a later request
        (not atom_config.enable_prefix_caching, "prefix caching"),
        # the step override is offered on graph decode steps only (ModelRunner.run_model)
        (not atom_config.enforce_eager, "enforce_eager"),
    ]
    return first_refusal(checks)
