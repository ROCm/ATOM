# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 DSpark on the ATOM plugin, SGLang 0.5.22: a sampled verify of
a greedy draft (``SGLANG_DSPARK_USE_GREEDY_DRAFT=1``) accepts by the target's
own top-p draw, as native ATOM's DSpark verify does.

A DSpark round whose batch holds a sampled request accepts eagerly, outside
the verify CUDA graph. With a greedy draft SGLang still runs its rejection
sampler (``dspark_verify.accept_draft_tokens``): a dense one-hot draft
distribution ``[bs, gamma, V]`` (``zeros`` + ``scatter_``), the top-p target
distribution of every verify row (``softmax`` + a top-p renorm, which on
this plugin takes ``dspark_topp_verify_probs_patch``'s exact pivot, a
``torch.topk`` of 1280 values a row), two ``torch.rand`` and
``chain_speculative_sampling_triton``. On MI308X TP4 that is about 0.6 ms of
GPU time a round at one request and 1.6 ms at 32 (profiled).

With a one-hot draft ``q = delta(d)`` the rejection rule accepts ``d`` with
probability ``p(d)`` and otherwise draws from ``p`` without ``d``,
renormalised; the bonus position draws from ``p``. Drawing ``t ~ p`` at every
verify row and accepting while ``t == d`` gives the same joint distribution:
``P(accept) = p(d)``, and a rejected row's ``t`` is a draw from ``p`` given
``t != d``. That is native ATOM's verify (``Sampler`` +
``rejection_greedy_sample_kernel``), so this patch runs it with native's own
kernels: ``softmax(logits / T)`` in fp32, aiter's
``top_p_sampling_from_probs`` per verify row, then SGLang's greedy accept
rule (``compute_dflash_correct_drafts_and_bonus``) on the drawn ids instead
of the argmax. Outputs follow the target's top-p distribution either way
(aiter's top-p, native's semantics); acceptance follows native's rule
exactly. The kernel takes its philox seed and offset from each rank's
default CUDA generator, as SGLang's own ``torch.rand`` draws do
(``deterministic=True`` fixes its reduction order, not the draw); ranks
agree because the accept outcome goes through SGLang's ``build_accept_outs``
unchanged, whose TP sync makes every rank take rank 0's ``correct_len`` /
``bonus``.

Scope: the patched ``accept_draft_tokens`` takes this path only for a greedy
draft block, a batch with no greedy request (SGLang encodes those as
``top_k = 1``), top-p (``need_top_p_sampling``) without top-k or min-p, no
block verification, CUDA ``[bs * draft_token_num, V]`` logits and one
temperature / top-p per request, never under graph capture. Everything else
(sampled drafts, mixed batches, top-k / min-p, temperature-only requests)
runs SGLang's function.

Shape guard: the source shape of ``accept_draft_tokens`` this reads (its
greedy-draft one-hot branch and its greedy accept rule); where it differs,
SGLang's function stays and one line logs why. The plugin runs on SGLang
0.5.22 only (`sglang_version_guard` refuses any other at launch, before this
installs).
"""

from __future__ import annotations

import logging
from functools import wraps

from atom.plugin.sglang.patches.sglang_version_guard import (
    missing_source_needles,
    sglang_version,
)

logger = logging.getLogger("atom")

_MARK = "_atom_v41_draw_verify"

# What the replaced function does for a greedy draft: the draft distribution
# is the one-hot of the drafted ids, an all-greedy batch takes the argmax
# rule, a batch with a greedy row mixes the two.
_ACCEPT_NEEDLES = (
    "all_greedy = sampling_info is None or sampling_info.is_all_greedy",
    "if draft_block.greedy_draft:",
    "draft_probs.scatter_(-1, draft_tokens.unsqueeze(-1), 1.0)",
    "if not sampling_info.is_any_greedy:",
    "AcceptSampling.execute(",
    "SelectMixedAccept.execute(",
)
_GREEDY_NEEDLES = (
    "compute_dflash_correct_drafts_and_bonus(",
    "cap_trim_lens = torch.zeros_like(correct_len)",
    "CapCorrectLen.execute(",
)


def _upstream_shape_error(dspark_verify, dspark_accept) -> str | None:
    fn = getattr(dspark_verify, "accept_draft_tokens", None)
    if fn is None:
        return "dspark_verify.accept_draft_tokens is missing"
    missing = missing_source_needles(fn, _ACCEPT_NEEDLES)
    if missing is None:
        return "accept_draft_tokens has no source"
    if missing:
        return f"accept_draft_tokens no longer reads {missing}"
    for name in ("compute_dflash_correct_drafts_and_bonus", "CapCorrectLen"):
        if not hasattr(dspark_accept, name):
            return f"dspark_accept lacks {name}"
    missing = missing_source_needles(dspark_accept.accept_greedy, _GREEDY_NEEDLES)
    if missing is None:
        return "dspark_accept.accept_greedy has no source"
    if missing:
        return f"dspark_accept.accept_greedy no longer reads {missing}"
    return None


def takes_draw_verify(
    *, candidates, target_logits, draft_block, sampling_info, block_verification
) -> bool:
    """Whether this accept is a greedy draft's top-p-only sampled verify the
    draw rule covers (see the module docstring). Host checks only."""
    import torch

    if sampling_info is None or not draft_block.greedy_draft:
        return False
    if block_verification or torch.cuda.is_current_stream_capturing():
        return False
    if sampling_info.is_all_greedy or sampling_info.is_any_greedy:
        return False
    if (
        sampling_info.need_top_k_sampling
        or sampling_info.need_min_p_sampling
        or not sampling_info.need_top_p_sampling
    ):
        return False
    x = target_logits
    if x is None or x.device.type != "cuda" or x.dim() != 2 or candidates.dim() != 2:
        return False
    bs, width = int(candidates.shape[0]), int(candidates.shape[1])
    if int(x.shape[0]) != bs * width:
        return False
    return (
        int(sampling_info.top_ps.numel()) == bs
        and int(sampling_info.temperatures.numel()) == bs
    )


def draw_target_tokens(target_logits, temperatures, top_ps, rows_per_req: int):
    """One top-p draw per verify row, as native ATOM's sampler draws a sampled
    request's token: ``softmax(logits / T)`` in fp32, then aiter's
    ``top_p_sampling_from_probs`` with the request's top_p (its
    ``deterministic`` flag fixes the reduction order; the draw comes from the
    default CUDA generator). Returns int64 ids ``[rows]``."""
    import aiter.ops.sampling  # noqa: F401  (registers torch.ops.aiter.*)
    import torch

    temps = torch.repeat_interleave(
        temperatures.reshape(-1).to(torch.float32), rows_per_req
    ).unsqueeze(-1)
    probs = (target_logits / temps).softmax(dim=-1, dtype=torch.float32).contiguous()
    ps = torch.repeat_interleave(
        top_ps.reshape(-1).to(torch.float32), rows_per_req
    ).contiguous()
    ids = torch.ops.aiter.top_p_sampling_from_probs(
        probs, None, ps, 0.0, deterministic=True
    )
    return ids.to(torch.int64)


def draw_accept(
    *, candidates, target_logits, sampling_info, cutoff_verify_lens, dspark_accept
):
    """``(correct_len, bonus, cap_trim_lens)`` of the draw rule, in SGLang's
    greedy accept's dtypes and layout (``accept_greedy`` on drawn ids)."""
    import torch

    bs, width = int(candidates.shape[0]), int(candidates.shape[1])
    predict = draw_target_tokens(
        target_logits, sampling_info.temperatures, sampling_info.top_ps, width
    ).view(bs, width)
    correct_len, bonus = dspark_accept.compute_dflash_correct_drafts_and_bonus(
        candidates=candidates, target_predict=predict
    )
    cap_trim_lens = torch.zeros_like(correct_len)
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = dspark_accept.CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
        rows = torch.arange(bs, device=predict.device)
        bonus = predict[rows, correct_len.to(torch.long)].to(torch.int64)
    return correct_len, bonus, cap_trim_lens


def make_draw_accept(original, dspark_accept):
    @wraps(original)
    def accept_draft_tokens(
        *,
        candidates,
        target_logits,
        draft_block,
        sampling_info,
        draft_input,
        gamma,
        verify_num_draft_tokens,
        cutoff_layout=None,
        fused_argmax=False,
        block_verification=False,
    ):
        if not takes_draw_verify(
            candidates=candidates,
            target_logits=target_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            block_verification=block_verification,
        ):
            return original(
                candidates=candidates,
                target_logits=target_logits,
                draft_block=draft_block,
                sampling_info=sampling_info,
                draft_input=draft_input,
                gamma=gamma,
                verify_num_draft_tokens=verify_num_draft_tokens,
                cutoff_layout=cutoff_layout,
                fused_argmax=fused_argmax,
                block_verification=block_verification,
            )
        return draw_accept(
            candidates=candidates,
            target_logits=target_logits,
            sampling_info=sampling_info,
            cutoff_verify_lens=(
                None if cutoff_layout is None else cutoff_layout.verify_lens
            ),
            dspark_accept=dspark_accept,
        )

    setattr(accept_draft_tokens, _MARK, True)
    return accept_draft_tokens


def install_dspark_draw_verify_patch() -> bool:
    """Install (idempotent). Returns whether the draw rule is active."""
    from sglang.kernels.ops.speculative.dspark import dspark_accept
    from sglang.srt.speculative.dspark_components import dspark_verify

    if getattr(dspark_verify.accept_draft_tokens, _MARK, False):
        return True
    reason = _upstream_shape_error(dspark_verify, dspark_accept)
    if reason is not None:
        logger.warning(
            "ATOM DeepSeek-V4.1 DSpark: draw-rule verify NOT installed; a greedy "
            "draft's sampled verify keeps SGLang's rejection sampler (%s)",
            reason,
        )
        return False
    dspark_verify.accept_draft_tokens = make_draw_accept(
        dspark_verify.accept_draft_tokens, dspark_accept
    )
    logger.info(
        "ATOM DeepSeek-V4.1 DSpark: a greedy draft's top-p sampled verify accepts "
        "by the target's own draw, as native ATOM (dspark_verify.accept_draft_tokens; "
        "sglang %s)",
        sglang_version(),
    )
    return True
