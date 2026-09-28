# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which MiniMax-M3 forwards take the mono path.

The configuration is checked once; a step is then routed to mono only when it is
a decode of 1..MAX_TOKENS tokens (a cudagraph's pad rows counted; a speculative
verify's q tokens of a request count as q).
Every per-step input is TP-uniform: the kernels all-reduce in-kernel, so ranks
that disagreed would wait on each other forever (a per-rank flag such as
``is_dummy_run`` must never gate this).
"""

import logging

import torch

from atom.models.minimax_m3.mono.check import (
    forward_checked,
    forward_probed,
    trace_logits,
)
from atom.models.minimax_m3.mono.config import (
    MAX_CONTEXT,
    MAX_TOKENS,
    TP,
    MonoUnsupported,
)
from atom.models.minimax_m3.mono.runner import MonoDecodeRunner
from atom.plugin.prepare import is_sglang, is_vllm
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def _config_refusal(atom_config, text_config) -> str | None:
    """Why this deployment cannot use mono, or None."""
    checks = (
        (
            atom_config.tensor_parallel_size == TP,
            f"TP {atom_config.tensor_parallel_size} != {TP}",
        ),
        (atom_config.parallel_config.data_parallel_size == 1, "DP > 1"),
        (atom_config.pipeline_parallel_size == 1, "PP > 1"),
        (not (is_vllm() or is_sglang()), "plugin mode"),
        (not getattr(text_config, "use_index_cache", False), "use_index_cache"),
        (atom_config.kv_cache_dtype == "fp8", f"kv cache {atom_config.kv_cache_dtype}"),
        (
            (atom_config.index_cache_dtype or atom_config.kv_cache_dtype) == "fp8",
            "index cache not fp8",
        ),
        (
            atom_config.kv_cache_block_size == 128,
            f"block size {atom_config.kv_cache_block_size}",
        ),
        (
            atom_config.max_model_len <= MAX_CONTEXT,
            f"max_model_len {atom_config.max_model_len} > {MAX_CONTEXT}",
        ),
        (not (atom_config.enable_tbo or atom_config.enable_tbo_decode), "TBO"),
        (
            not atom_config.compilation_config.cudagraph_mode.requires_piecewise_compilation(),
            "piecewise cudagraph",
        ),
    )
    for ok, why in checks:
        if not ok:
            return why
    return None


class MonoDecode:
    """Routes supported decode steps of one ``MiniMaxM3SparseForCausalLM``."""

    def __init__(self, causal_lm, atom_config, text_config) -> None:
        self._lm = causal_lm
        self._runner: MonoDecodeRunner | None = None
        self._enabled = False
        if not envs.ATOM_MONO_ENABLE:
            return
        why = _config_refusal(atom_config, text_config)
        if why is not None:
            logger.info("MiniMax-M3 mono decode off: %s", why)
            return
        self._enabled = True

    def supports(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors,
        inputs_embeds,
    ) -> bool:
        if not self._enabled:
            return False
        n = input_ids.numel()  # a cudagraph's pad rows included
        if intermediate_tensors is not None or inputs_embeds is not None:
            return False
        if not 1 <= n <= MAX_TOKENS:
            return False
        # the kernels read every token's position and slot as an int64's low word
        if positions.dtype != torch.int64:
            return False
        fwd = get_forward_context()
        if (
            fwd.context is None
            or fwd.context.is_prefill
            or fwd.ubatch_slices is not None
        ):
            return False
        md = fwd.attn_metadata
        sparse_md = getattr(md, "sparse_attention_metadata", None)
        if sparse_md is None or sparse_md.decode is None or sparse_md.num_prefills:
            return False
        decode = sparse_md.decode
        # q query tokens per request (speculative verify: q > 1), expanded to a
        # row per token by token_rows
        q = decode.max_query_len
        if n % q:
            return False
        bt = decode.block_table
        if bt.dtype != torch.int32 or decode.seq_lens.dtype != torch.int32:
            return False
        # a row per request (eager decode cuts the tables to the real batch while
        # the ids stay padded) and rows bt.shape[1] apart
        if (
            bt.shape[0] < n // q
            or decode.seq_lens.shape[0] < n // q
            or bt.stride(0) != bt.shape[1]
        ):
            return False
        slots = sparse_md.slot_mapping
        if slots.dtype != torch.int64 or slots.numel() < n:
            return False
        if self._runner is None:
            # created on the first eligible step, which is a warmup forward before
            # any graph capture (allocations and the peer handshake cannot be captured)
            if torch.cuda.is_current_stream_capturing():
                return False
            try:
                self._runner = MonoDecodeRunner(self._lm)
            except MonoUnsupported as why:
                logger.warning("MiniMax-M3 mono decode off: %s", why)
                self._enabled = False
                return False
            logger.info("MiniMax-M3 mono decode on")
        return True

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        sums = None
        if envs.ATOM_MONO_CHECK:
            out = forward_checked(self._runner, input_ids, positions)
        elif envs.ATOM_MONO_TRACE:
            sums = {}
            out = forward_probed(self._runner, input_ids, positions, sums)
        else:
            out = self._runner.forward(input_ids, positions)
        if envs.ATOM_MONO_TRACE:
            trace_logits(
                self._lm, self._runner.rank, input_ids, positions, out,
                envs.ATOM_MONO_TRACE, sums,
            )  # fmt: skip
        return out
