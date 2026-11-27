# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 on the ATOM plugin, SGLang 0.5.22: an all-greedy sampling
step picks its tokens with aiter's exact top-1, the kernel native ATOM's
sampler runs, instead of ``torch.argmax``.

SGLang's ``Sampler.forward`` takes an all-greedy batch (every request at
temperature 0) to ``torch.argmax(logits, -1)`` over the gathered fp32 logits
(``SGLANG_USE_AITER`` unset, as in this plugin's launch). On MI308X (gfx942)
PyTorch's reduction runs that argmax over the 129280-entry V4.1 vocabulary in
~47 us for one row, ~86 us for two and ~198 us for four, and from eight rows
on at 49-65 us (a CUDA-graph micro benchmark). It runs after the decode
graph, on the stream the next step's graph waits on, so every reference-config decode step (no speculation) pays it
in full. Native ATOM's all-greedy sampler is ``aiter.topk_select(logits, 1,
tie="low")``: 8-25 us for 1-64 rows.

``tie="low"`` is torch.argmax's tie rule (the lowest token id among equal
maxima), and ``topk_select`` ranks NaN first, as torch.argmax returns a
NaN's index, so the picked ids are the same; they are returned as int64, the
dtype ``torch.argmax`` gives, so nothing downstream sees a change.

Taken only on SGLang's own torch.argmax branch with nothing else to produce:
all-greedy, no logprobs (neither returned nor from separate logprob logits),
no sampling-mask capture, ``SGLANG_USE_AITER``'s greedy path not selected, and
a CUDA [rows, vocab] logits tensor with unit last stride. Every other call --
and the speculative paths, which pick their tokens elsewhere -- runs SGLang's
function unchanged. On the fast path the same NaN probe, logit preprocessing
(custom logit processors, NaN sanitising) and TP token sync run as upstream.

Scope: the DeepSeek-V4.1 ATOM model path (installed when the V4.1 model is
prepared). Switch: ``ATOM_SGLANG_V41_FAST_GREEDY=0`` (`deepseek_v41_envs`)
keeps SGLang's sampler, which every other call runs anyway, so no code stays
alive for the switch alone. Shape guard: ``Sampler.forward(self,
logits_output, sampling_info, return_logprob, top_logprobs_nums,
token_ids_logprobs, positions, *, logprob_logits)``; anything else keeps
SGLang's sampler and logs why. The plugin runs on SGLang 0.5.22 only
(`sglang_version_guard` refuses any other at launch, before this installs).
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps

from atom.plugin.sglang import deepseek_v41_envs

logger = logging.getLogger("atom")

_EXPECTED_PARAMS = [
    "self",
    "logits_output",
    "sampling_info",
    "return_logprob",
    "top_logprobs_nums",
    "token_ids_logprobs",
    "positions",
    "logprob_logits",
]
# Module attributes the fast path reads at call time (sglang/srt/layers/sampler.py).
_MODULE_ATTRS = ("_use_aiter", "_disable_aiter_greedy_sample", "maybe_detect_nan")
_LOGIT_DTYPES = ("float32", "bfloat16", "float16")


def _upstream_shape_error(module) -> str | None:
    sampler = getattr(module, "Sampler", None)
    if sampler is None:
        return "sglang.srt.layers.sampler.Sampler is missing"
    missing = [a for a in _MODULE_ATTRS if not hasattr(module, a)]
    missing += [
        m
        for m in ("_preprocess_logits", "_sync_token_ids_across_tp")
        if not hasattr(sampler, m)
    ]
    if missing:
        return f"sampler module lacks {missing}"
    params = list(inspect.signature(inspect.unwrap(sampler.forward)).parameters)
    if params != _EXPECTED_PARAMS:
        return f"Sampler.forward takes {params}, not {_EXPECTED_PARAMS}"
    return None


def greedy_token_ids(logits):
    """Row-wise argmax with torch.argmax's lowest-id tie rule, as int64:
    native ATOM's all-greedy kernel (``atom.model_ops.sampler``)."""
    import torch
    from aiter import topk_select

    ids = topk_select(logits, 1, tie="low")[1].view(-1)
    return ids if ids.dtype == torch.int64 else ids.to(torch.int64)


def takes_fast_greedy(
    module, logits, sampling_info, return_logprob, logprob_logits
) -> bool:
    """Whether this call is SGLang's plain ``torch.argmax`` branch with
    nothing else to produce (see the module docstring)."""
    if not sampling_info.is_all_greedy:
        return False
    if return_logprob or logprob_logits is not None:
        return False
    if sampling_info.sampling_mask_batch_indices is not None:
        return False
    if module._use_aiter and not module._disable_aiter_greedy_sample:
        return False  # SGLang's own aiter greedy path: leave it to SGLang
    if logits is None:
        return False
    return (
        logits.device.type == "cuda"
        and logits.dim() == 2
        and logits.shape[0] > 0
        and logits.stride(-1) == 1
        and str(logits.dtype).rsplit(".", 1)[-1] in _LOGIT_DTYPES
    )


def make_fast_greedy_forward(original, module, pick=greedy_token_ids):
    """``Sampler.forward`` that serves the plain all-greedy call with ``pick``
    and hands every other call to ``original``."""

    @wraps(original)
    def forward(
        self,
        logits_output,
        sampling_info,
        return_logprob,
        top_logprobs_nums,
        token_ids_logprobs,
        positions,
        *,
        logprob_logits=None,
    ):
        logits = logits_output.next_token_logits
        if not takes_fast_greedy(
            module, logits, sampling_info, return_logprob, logprob_logits
        ):
            return original(
                self,
                logits_output,
                sampling_info,
                return_logprob,
                top_logprobs_nums,
                token_ids_logprobs,
                positions,
                logprob_logits=logprob_logits,
            )
        # Upstream's order: async NaN probe, preprocessing (custom logit
        # processors, NaN sanitising), the pick, the optional TP token sync.
        module.maybe_detect_nan(logits, "sampler: next_token_logits")
        logits = self._preprocess_logits(logits, sampling_info)
        batch_next_token_ids = pick(logits)
        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)
        return batch_next_token_ids

    forward._atom_v41_fast_greedy = True
    return forward


def install_v41_greedy_sampler_patch() -> bool:
    """Route SGLang's plain all-greedy sampling through ``greedy_token_ids``
    (idempotent). Returns whether it is active."""
    if not deepseek_v41_envs.ATOM_SGLANG_V41_FAST_GREEDY:
        logger.info(
            "ATOM DeepSeek-V4.1: all-greedy sampling keeps torch.argmax "
            "(ATOM_SGLANG_V41_FAST_GREEDY=0)"
        )
        return False
    from sglang.srt.layers import sampler as sampler_module

    current = sampler_module.Sampler.forward
    if getattr(current, "_atom_v41_fast_greedy", False):
        return True
    reason = _upstream_shape_error(sampler_module)
    if reason is not None:
        logger.warning(
            "ATOM DeepSeek-V4.1: SGLang sampler not patched (%s); all-greedy "
            "sampling keeps torch.argmax",
            reason,
        )
        return False
    sampler_module.Sampler.forward = make_fast_greedy_forward(current, sampler_module)
    logger.info(
        "ATOM DeepSeek-V4.1: all-greedy sampling picks with aiter "
        "topk_select(k=1, tie=low) instead of torch.argmax"
    )
    return True
