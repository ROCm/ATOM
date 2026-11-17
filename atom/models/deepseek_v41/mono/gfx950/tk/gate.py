# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek V4.1 decode steps served by the TileRT-derived TP8 decode step (``host.TkRank``: the recorded
mode-5 DSpark step's 118 launches a rank, one HIP graph a parity) behind ATOM's step override
(``atom/mono/runtime/step_override.py``), so requests keep ATOM's own server, scheduler and PD.

A TkRank holds the request's state itself (TileRT-layout caches, its logs, its drafts): it admits a request
(``set_sampling`` / ``admit_kernel`` / the request's KV converted from ATOM's caches, ``kv``), then serves every
decode step of it (``runner``); a served request never returns to the production path, whose caches no longer hold
its KV (``dispatch``). Contract and design: runs/20261115-flydsl-tilert/m7/STEP_OVERRIDE.md.

Selected like gfx1250's single-launch step, with no switch of its own (``mono.selection``): under
the mono master switch (``ATOM_MONO_ENABLE``) on a gfx950 device (``enabled``), on the one deployment it is written for
(``refusal``), V4.1's decode steps take it; an explicit ``ATOM_DSV41_MONO_HI=1`` keeps the hi path instead, and
anywhere else the hi / mono selection runs unchanged.
"""

import torch

from atom.utils import envs

TP = 8  # the recorded plan's TP
SPEC_TOKENS = 5
ROWS = SPEC_TOKENS + 1  # one request's verify: the anchor and its 5 drafts
MAX_SEQ_LEN = 1 << 20  # the plan's context (its kernargs and workspaces)
BLOCK = 128  # ATOM's KV block (``kv.BLOCK``)


def _arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.gcnArchName.split(":")[0]


def enabled() -> bool:
    """``ATOM_MONO_ENABLE`` on a gfx950 device (``gfx1250.enabled``'s rule for its own arch)."""
    return bool(envs.ATOM_MONO_ENABLE) and _arch() == "gfx950"


def refusal(atom_config) -> str | None:
    """Why this deployment cannot run the TileRT-path step, or None."""
    from atom.models.deepseek_v41.mono.kernels.dims import Dims
    from atom.mono.runtime.deployment import common_refusals, first_refusal

    # (a field missing from ``atom_config`` is a deployment this is not written for)
    spec = getattr(atom_config, "speculative_config", None)
    method = None if spec is None else spec.method
    dspark = getattr(atom_config, "dspark", None)
    seqs = getattr(atom_config, "max_num_seqs", None)
    max_len = getattr(atom_config, "max_model_len", None)
    kv_dtype = getattr(atom_config, "kv_cache_dtype", None)
    index_dtype = getattr(atom_config, "index_cache_dtype", None)
    block = getattr(atom_config, "kv_cache_block_size", None)
    checks = common_refusals(atom_config, Dims, (TP,)) + [
        (method == "dspark", f"speculative method {method}"),
        (
            spec is not None and spec.num_speculative_tokens == SPEC_TOKENS,
            "DSpark block size",
        ),
        (
            dspark is not None and not dspark.confidence_schedule,
            "DSpark confidence schedule",
        ),
        (dspark is not None and not dspark.ragged, "DSpark ragged verify"),
        # one request at a time (the plan's step is one request's 6 rows), and a request it served cannot fall
        # back: the production caches do not hold the KV it decoded
        (seqs == 1, f"max_num_seqs {seqs}"),
        # the caches its KV conversion reads at admission (``kv.from_cache``): BF16 KV, an FP8 index, 128-row blocks
        (
            kv_dtype == "bf16",
            f"--kv_cache_dtype {kv_dtype} (the TileRT-path step reads bf16)",
        ),
        (
            index_dtype == "fp8",
            f"--index-cache-dtype {index_dtype} (the TileRT-path step reads fp8)",
        ),
        (block == BLOCK, f"--block-size {block} (the TileRT-path step reads {BLOCK})"),
        (max_len is not None and max_len <= MAX_SEQ_LEN, f"max_len {max_len}"),
        # the KV of the tokens it decodes never reaches ATOM's paged cache: a cached prefix over them would be
        # read back by a later request
        (not getattr(atom_config, "enable_prefix_caching", True), "prefix caching"),
        # the step override is offered on graph decode steps only (ModelRunner.run_model)
        (not getattr(atom_config, "enforce_eager", False), "enforce_eager"),
    ]
    return first_refusal(checks)
