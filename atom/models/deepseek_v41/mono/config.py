# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The deployments V4.1 mono serves, and why any other is refused."""

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.mono.runtime.deployment import common_refusals, first_refusal
from atom.utils import envs

# the TP sizes the kernels are built for (``kernels.dims``): TP2 and TP4 run on
# GPU; TP8's builds compile and fit, but the original path does not start there
SUPPORTED_TP = (2, 4, 8)
# a DSpark verify: every request's anchor token and its 5 drafted ones
SPEC_TOKENS = 5
# the requests a step serves at most: the DSpark draft's blocks a pass, and
MAX_REQUESTS = 8
# the target's rows a step (every row carries its own metadata, however many
# requests they are): the buffers' width, eight verifies
MAX_ROWS = MAX_REQUESTS * (SPEC_TOKENS + 1)
# a token's keys at most: the original decode's 64 splits of one 16-key tile
# each, which the mono attention reproduces split by split
ATTENTION_KEYS_MAX = 64 * 16
# (top-k, candidate blocks, rows a block, index heads, index head dim)
INDEXER_SHAPE = (ip.TOPK, ip.TOPK_BLOCKS, ip.BLOCK_ROWS, ip.HEADS, ip.DIM)


def config_refusal(atom_config) -> str | None:
    """Why this deployment cannot use V4.1 mono, or None."""
    spec = atom_config.speculative_config
    hf = atom_config.hf_config
    keys = hf.sliding_window + hf.index_topk
    checks = [(envs.ATOM_MONO_ENABLE, "switched off")]
    checks += common_refusals(atom_config, Dims, SUPPORTED_TP) + [
        (
            atom_config.kv_cache_dtype == "bf16",
            f"kv cache {atom_config.kv_cache_dtype}",
        ),
        (
            atom_config.index_cache_dtype in ("fp8", "fp4"),
            f"index cache {atom_config.index_cache_dtype}",
        ),
        (
            keys <= ATTENTION_KEYS_MAX,
            f"{keys} attention keys a token > {ATTENTION_KEYS_MAX}",
        ),
        # the indexer kernels' constants (``index_plan``)
        (
            (
                hf.index_topk,
                hf.candidate_topk_blocks,
                hf.candidate_block_size,
                hf.index_n_heads,
                hf.index_head_dim,
            )
            == INDEXER_SHAPE,
            "indexer shape",
        ),
        (spec is not None and spec.method == "dspark", "not DSpark"),
        (
            spec is not None and spec.num_speculative_tokens == SPEC_TOKENS,
            f"speculative tokens != {SPEC_TOKENS}",
        ),
        (
            not (atom_config.dspark.confidence_schedule or atom_config.dspark.ragged),
            "DSpark confidence schedule / ragged",
        ),
        # a side-stream compressor would run beside the persistent kernels and
        # contend for their CUs
        (envs.ATOM_DSV41_SIDE_STREAMS == 0, "ATOM_DSV41_SIDE_STREAMS"),
    ]
    return first_refusal(checks)
