# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Route DFLASH draft greedy sampling through ATOM's TP-sharded lm_head.

SGLang's ``DFlashWorkerV2._greedy_sample_from_vocab_parallel_head`` decides how
to reduce the target lm_head to one token id per draft position:

    if not hasattr(lm_head, "shard_indices"):
        logits = torch.matmul(hs, weight.T)
        out = torch.argmax(logits, dim=-1)      # local index used as a token id
        return out
    ...                                          # SGLang's own TP reduction

The implicit assumption is "no ``shard_indices`` means the head is not
vocab-parallel". ATOM's ``ParallelLMHead`` breaks it: the head *is* sharded
(``vocab_start_idx = num_embeddings // tp_size * tp_rank``) but carries ATOM's
own sharding ABI rather than SGLang's ``VocabParallelEmbeddingShardIndices``.
At TP8 the stock path therefore turns a per-rank local vocab index into a
global token id, and DFLASH drafts garbage.

ATOM's head already exposes exactly the reduction DFLASH needs:
``compute_argmax_token()`` takes the per-rank argmax and all-gathers only the
``[N, 2]`` (value, global-id) pairs instead of the ``[N, vocab]`` logits.
This patch makes the worker prefer it when present.

Why a monkey patch rather than teaching ATOM to publish ``shard_indices``:
publishing it would silently reroute *every* ATOM model's DFLASH sampling
through SGLang's implementation (``torch.matmul`` instead of ATOM's tuned GEMM,
and SGLang's path ignores ``lm_head.bias``). Patching one method keeps the
blast radius at the single model family that opts in, which matches how the
plugin already handles SGLang divergences -- see
``deepseek_v4_bridge.install_deepseek_v4_proxy_pool_patch`` and
``runtime/load_config_patch.apply_load_config_patch``.

The patched method is byte-identical between the pinned SGLang branch and
official ``v0.5.15.post1``, so the patch is not tied to the experimental
branch. It degrades to the stock implementation whenever the head does not
provide ``compute_argmax_token``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_atom_dflash_external_lm_head_patch"


def install_dflash_lm_head_patch() -> None:
    """Prefer ``lm_head.compute_argmax_token`` in DFLASH draft greedy sampling.

    Idempotent, and a no-op when SGLang has no DFLASH worker (older releases)
    or when the method it patches has been renamed.
    """
    try:
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
    except ImportError:
        logger.debug("SGLang has no DFLASH worker; skipping ATOM lm_head patch.")
        return

    if getattr(DFlashWorkerV2, _PATCH_FLAG, False):
        return

    original = getattr(DFlashWorkerV2, "_greedy_sample_from_vocab_parallel_head", None)
    if original is None:
        logger.warning(
            "SGLang DFlashWorkerV2 has no _greedy_sample_from_vocab_parallel_head; "
            "ATOM cannot route DFLASH draft sampling through its lm_head. "
            "TP>1 draft tokens would be wrong, so DFLASH must not be used."
        )
        return

    def patched(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head: Any,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        compute_argmax_token = getattr(lm_head, "compute_argmax_token", None)
        if not callable(compute_argmax_token):
            return original(
                self,
                hidden_states=hidden_states,
                lm_head=lm_head,
                chunk_size=chunk_size,
            )

        num_tokens = int(hidden_states.shape[0])
        if num_tokens == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        weight_dtype = lm_head.weight.dtype
        out_tokens = torch.empty(
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )
        step = max(int(chunk_size), 1)
        for start in range(0, num_tokens, step):
            end = min(num_tokens, start + step)
            hs = hidden_states[start:end]
            if hs.dtype != weight_dtype:
                hs = hs.to(weight_dtype)
            token_ids = compute_argmax_token(hs)
            if token_ids.shape != (end - start,):
                raise ValueError(
                    "ATOM lm_head.compute_argmax_token returned an invalid shape: "
                    f"expected {(end - start,)}, got {tuple(token_ids.shape)}."
                )
            out_tokens[start:end].copy_(token_ids.to(torch.long))
        return out_tokens

    patched.__name__ = original.__name__
    patched.__qualname__ = original.__qualname__
    patched.__doc__ = original.__doc__
    setattr(DFlashWorkerV2, "_greedy_sample_from_vocab_parallel_head", patched)
    setattr(DFlashWorkerV2, _PATCH_FLAG, True)
    setattr(DFlashWorkerV2, "_atom_dflash_original_greedy_sample", original)
    logger.info(
        "ATOM patched SGLang DFLASH draft greedy sampling to use the external "
        "lm_head's compute_argmax_token()."
    )
