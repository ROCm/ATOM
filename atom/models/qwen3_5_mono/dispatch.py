# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which Qwen3.5 forwards take the mono path: decode steps of 1..MAX_TOKENS tokens on
a TP4 deployment that matches the kernels, with ``ATOM_QWEN35_MONO=1``.

Every check is TP-uniform (the kernels all-reduce in-kernel: ranks that disagreed
would wait on each other forever). ``ATOM_QWEN35_MONO_CHECK=1`` runs every layer both
ways and logs the differences (eager steps only)."""

import logging
import os

import torch

from atom.models.qwen3_5_mono.config import MAX_TOKENS, MTP_MAX_REQS, SPEC, TP

logger = logging.getLogger("atom")


def _refusal(atom_config):
    checks = [
        (
            atom_config.tensor_parallel_size == TP,
            f"TP {atom_config.tensor_parallel_size}",
        ),
        (not atom_config.enable_expert_parallel, "expert parallel"),
        (atom_config.parallel_config.data_parallel_size == 1, "DP > 1"),
        (
            atom_config.kv_cache_dtype in ("bf16", "auto"),
            f"kv cache {atom_config.kv_cache_dtype}",
        ),
        (
            atom_config.kv_cache_block_size == 16,
            f"block size {atom_config.kv_cache_block_size}",
        ),
        (
            atom_config.max_model_len <= 16384,
            f"max_model_len {atom_config.max_model_len}",
        ),
        (
            _spec_tokens(atom_config) in (0, SPEC - 1),
            f"{_spec_tokens(atom_config)} draft tokens",
        ),
    ]
    return next((why for ok, why in checks if not ok), None)


def _spec_tokens(atom_config):
    """Draft tokens a step (0: no speculative decoding)."""
    sc = getattr(atom_config, "speculative_config", None)
    if sc is None:
        return 0
    return int(getattr(sc, "num_speculative_tokens", 0) or 0)


def _refusal_tp1(atom_config):
    """TP1 (kernels_tp1.py): MTP verify steps of 4 or 5 tokens a request on
    ReplaySSM state (checked a step: the builder's metadata), any context length."""
    checks = [
        (not atom_config.enable_expert_parallel, "expert parallel"),
        (atom_config.parallel_config.data_parallel_size == 1, "DP > 1"),
        (
            atom_config.kv_cache_dtype in ("bf16", "auto"),
            f"kv cache {atom_config.kv_cache_dtype}",
        ),
        (
            atom_config.kv_cache_block_size == 16,
            f"block size {atom_config.kv_cache_block_size}",
        ),
        (
            _spec_tokens(atom_config) in (3, 4),
            f"{_spec_tokens(atom_config)} draft tokens",
        ),
    ]
    return next((why for ok, why in checks if not ok), None)


class QwenMonoDecode:
    def __init__(self, language_model, atom_config):
        self._lm = language_model
        self._runner = None
        self.enabled = os.environ.get("ATOM_QWEN35_MONO", "0") == "1"
        self.check = os.environ.get("ATOM_QWEN35_MONO_CHECK", "0") == "1"
        # MTP: the verify steps (SPEC tokens a request) take the mono path. TP4
        # (kernels_mtp.py) writes per-draft state slots and needs ReplaySSM off
        # (ATOM_ENABLE_REPLAYSSM=0); TP1 (kernels_tp1.py) runs on ReplaySSM
        self.spec = _spec_tokens(atom_config) > 0
        self.tp1 = atom_config.tensor_parallel_size == 1
        self.tokens = _spec_tokens(atom_config) + 1
        if self.enabled:
            why = _refusal_tp1(atom_config) if self.tp1 else _refusal(atom_config)
            if why is not None:
                logger.info("Qwen3.5 mono decode off: %s", why)
                self.enabled = False

    def supports(
        self, input_ids, positions, intermediate_tensors, inputs_embeds
    ) -> bool:
        if not self.enabled or intermediate_tensors is not None:
            return False
        from atom.utils.forward_context import get_forward_context

        # the runner is built on the first forward (the eager prefill warmup every rank
        # runs), so every rank has it before any graph capture: a rank that met its
        # first decode-shaped forward inside a capture would take ATOM's path there
        # while the others wait on it in-kernel
        if self._runner is None and not torch.cuda.is_current_stream_capturing():
            from atom.models.qwen3_5_mono.runner import MonoRunner, MonoUnsupported

            try:
                if self.tp1:
                    from atom.models.qwen3_5_mono.runner_tp1 import MonoRunnerTP1

                    self._runner = MonoRunnerTP1(self._lm, self.tokens)
                else:
                    self._runner = MonoRunner(self._lm, spec=self.spec)
            except MonoUnsupported as err:
                logger.warning("Qwen3.5 mono decode off: %s", err)
                self.enabled = False
                return False
        fwd = get_forward_context()
        if fwd.context is None or fwd.context.is_prefill:
            # timing builds: a prefill is the next Python step after a stretch of graph
            # replays; their last decode step's stamps are still in the buffer
            r = self._runner
            if (
                r is not None
                and r.timing
                and r.steps > 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                r.log_timeline(-1)
            return False
        n = input_ids.numel()
        md = fwd.attn_metadata
        gm = getattr(md, "gdn_metadata", None)
        if gm is None:
            return False
        if self.tp1:
            from atom.models.qwen3_5_mono.config_tp1 import TOK

            S = self.tokens
            if (
                not getattr(gm, "replayssm", False)
                or gm.spec_state_indices_tensor is None
                or gm.num_prefills != 0
                or gm.num_decodes != 0
                or getattr(md, "max_seqlen_q", 0) != S
                or n % S != 0
                or not S <= n <= TOK
                or gm.num_spec_decodes > n // S  # graph padding: slot -1 rows
            ):
                return False
        elif self.spec:
            if getattr(gm, "replayssm", False) or gm.spec_state_indices_tensor is None:
                return False
            if (
                gm.num_prefills != 0
                or gm.num_decodes != 0
                or getattr(md, "max_seqlen_q", 0) != SPEC
            ):
                return False
            if n % SPEC != 0 or not 1 <= n // SPEC <= MTP_MAX_REQS:
                return False
            si = gm.spec_state_indices_tensor
            if (
                si.dtype != torch.int32
                or si.dim() != 2
                or si.shape[1] != SPEC
                or si.stride(1) != 1
                or si.stride(0) != SPEC
            ):
                return False
        else:
            if not 1 <= n <= MAX_TOKENS or getattr(md, "max_seqlen_q", 1) != 1:
                return False
        if (
            positions.dtype != torch.int64
            or positions.dim() != 2
            or positions.shape[0] != 3
            or positions.stride(1) != 1
        ):
            return False
        return self._runner is not None

    def forward(self, h_embed, positions):
        if self.check and not torch.cuda.is_current_stream_capturing():
            return self._runner.forward_checked(h_embed, positions)
        return self._runner.forward(h_embed, positions)
