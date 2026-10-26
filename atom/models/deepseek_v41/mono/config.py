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
# DP attention ranks (attention TP 1, the MoE expert parallel over them,
# ``kernels.dims``): the builds measured
SUPPORTED_DPA = (2,)
# the rows a step at most, the kernels' widest build (``kernels.attn_pre``): past
# it TP2's shared expert outgrows one unit a CTA (72 tasks x 4 token tiles > 256
# CTAs) and K1 / K2 spill VGPRs. Every row carries its own metadata, so a
# target step is routed by its rows alone, however many requests they are and
# however many rows each holds
MAX_ROWS = 48
# the MoE's rows at most (``kernels.moe``): DP attention runs every rank's
# step through one MoE, up to MAX_ROWS a rank over two ranks
MAX_MOE_ROWS = 96
# the DSpark draft's blocks a pass at most
MAX_REQUESTS = 8
# a token's keys at most: the original decode's 64 splits of one 16-key tile
# each, which the mono attention reproduces split by split
ATTENTION_KEYS_MAX = 64 * 16
# (top-k, candidate blocks, rows a block, index heads, index head dim)
INDEXER_SHAPE = (ip.TOPK, ip.TOPK_BLOCKS, ip.BLOCK_ROWS, ip.HEADS, ip.DIM)


def config_refusal(atom_config) -> str | None:
    """Why this deployment cannot use V4.1 mono, or None."""
    spec = atom_config.speculative_config
    method = None if spec is None else spec.method
    dspark = method == "dspark"
    hf = atom_config.hf_config
    keys = hf.sliding_window + hf.index_topk
    checks = [(envs.ATOM_MONO_ENABLE, "switched off")]
    checks += common_refusals(atom_config, Dims, SUPPORTED_TP, SUPPORTED_DPA) + [
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
        # no drafter, or DSpark, whose aux taps K1 writes: another drafter's
        # hooks on the target layers would never fire under mono
        (dspark or spec is None, f"speculative method {method}"),
        # a request's verify (its anchor and drafted tokens) fits one step
        (
            not dspark or spec.num_speculative_tokens + 1 <= MAX_ROWS,
            f"a verify of more than {MAX_ROWS} rows",
        ),
        # a side-stream compressor would run beside the persistent kernels and
        # contend for their CUs
        (envs.ATOM_DSV41_SIDE_STREAMS == 0, "ATOM_DSV41_SIDE_STREAMS"),
    ]
    return first_refusal(checks)
