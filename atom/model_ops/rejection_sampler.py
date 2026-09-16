from typing import Optional

import torch
import triton
import triton.language as tl
from atom.utils import envs
from atom.utils.forward_context import SpecDecodeMetadata
from torch import nn

ATOM_ENABLE_RELAXED_MTP = envs.ATOM_ENABLE_RELAXED_MTP
if ATOM_ENABLE_RELAXED_MTP:
    RELAXED_TOP_N = 10
    RELAXED_DELTA = 0.6
else:
    RELAXED_TOP_N = 1
    RELAXED_DELTA = 0.0

# --- Synthetic (forced) acceptance for spec-decode benchmarking ---
# Enabled via --spec-decode-acceptance-length / --spec-decode-acceptance-rate
# (SpeculativeConfig.synthetic_acceptance_rates, already resolved to per-position
# rates by the config). When set, the rejection sampler ignores the real
# draft/target comparison and force-accepts draft tokens so the measured mean
# acceptance length converges to the configured value. Purely a benchmarking /
# bring-up knob (e.g. while an MTP/EAGLE head is still training, or to replay a
# published acceptance-length curve). Mirrors vLLM's "synthetic"
# rejection_sample_method. See ROCm/ATOM#555.
#
# Cache {(rates, device): conditional-rate tensor} — the kernel walks positions
# sequentially, so it needs P(accept i | accepted through i-1) rather than the
# unconditional rates the config carries.
_SYNTHETIC_COND_CACHE: dict[tuple[tuple[float, ...], torch.device], torch.Tensor] = {}
# Base seed for the per-step synthetic RNG. The forced accept/reject draw MUST be
# identical on every TP rank: sampled_tokens is broadcast from rank 0 while
# num_bonus_tokens stays local, so a per-rank torch.rand would desync the two and
# make the anchor gather read a rejected (-1) column -> the draft model then
# embeds an invalid id (HSA out-of-bounds). A dedicated device generator re-seeded
# from the step counter gives that: same Philox stream, so bit-identical uniforms
# on every rank, and isolated from whatever else consumes the global RNG.
_SYNTHETIC_RNG_BASE_SEED = 0x5EED
# One generator per device, kept alive so the draw costs a kernel and nothing else.
_SYNTHETIC_GENERATORS: dict[torch.device, torch.Generator] = {}


def _synthetic_generator(device: torch.device) -> torch.Generator:
    generator = _SYNTHETIC_GENERATORS.get(device)
    if generator is None:
        generator = torch.Generator(device=device)
        _SYNTHETIC_GENERATORS[device] = generator
    return generator


def acceptance_length_to_rates(length: float, n: int) -> list[float]:
    """Mean acceptance length -> per-position *unconditional* acceptance rates.

    Entry ``i`` is the marginal probability that the first ``i+1`` draft tokens
    are all accepted, so the mean number of accepted draft tokens is the sum of
    the list and the mean acceptance length is ``1 + sum`` (the ``1`` being the
    target's own guaranteed token).

    The schedule is the minimum-variance one: accept ``floor(length - 1)``
    positions outright and the next with the leftover fraction. Any schedule
    summing to ``length - 1`` hits the requested mean, but the spread differs
    wildly between them, and spread is not a free parameter here -- it drives
    ITL tails and how the batch drains. This is the schedule vLLM resolves
    ``synthetic_acceptance_length`` to and the one SGLang's ``match-expected``
    draws, so a forced-length run stays comparable across all three engines.
    """
    num_accepted_drafts = length - 1
    num_full = int(num_accepted_drafts)
    return (
        [1.0] * num_full + [num_accepted_drafts - num_full] + [0.0] * (n - num_full - 1)
    )[:n]


def _get_synthetic_cond_rates(
    rates: tuple[float, ...], device: torch.device
) -> torch.Tensor:
    """Unconditional per-position rates -> conditional, as a device tensor.

    The kernel stops at the first rejection, so at position ``i`` it needs
    ``P(accept i | accepted through i-1) = rates[i] / rates[i-1]``, not the
    unconditional ``rates[i]``.
    """
    key = (rates, device)
    cached = _SYNTHETIC_COND_CACHE.get(key)
    if cached is not None:
        return cached
    cond: list[float] = []
    prev = 1.0
    for rate in rates:
        cond.append(rate / prev if prev > 0.0 else 0.0)
        prev = rate
    tensor = torch.tensor(cond, dtype=torch.float32, device=device)
    _SYNTHETIC_COND_CACHE[key] = tensor
    return tensor


class RejectionSampler(nn.Module):
    def __init__(self, synthetic_acceptance_rates: list[float] | None = None):
        super().__init__()
        # Debug/benchmark override: force an acceptance-length curve (see module
        # docstring). None => normal draft/target rejection sampling. The config
        # has already validated and resolved whichever knob the user set into
        # per-position unconditional rates.
        self.synthetic_acceptance_rates = (
            tuple(synthetic_acceptance_rates)
            if synthetic_acceptance_rates is not None
            else None
        )
        # Per-step seed for the synthetic RNG, advanced once per forward. Stays in
        # lockstep across TP ranks (SPMD), so the forced accept/reject pattern is
        # identical on every rank while still varying step to step.
        self._synthetic_step = 0

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        target_logits: torch.Tensor,
        # [batch_size, 1]
        bonus_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure target_logits is contiguous. For greedy sampling, we can use
        # logits directly (argmax is the same for logits and probs), but we
        # need to ensure it's contiguous to satisfy the assertion in rejection_sample.
        target_logits = target_logits.contiguous()

        # Validate shapes match expectations
        expected_num_tokens = len(metadata.draft_token_ids)
        if target_logits.shape[0] != expected_num_tokens:
            raise ValueError(
                f"target_logits shape mismatch: expected first dimension to be "
                f"{expected_num_tokens} (len(draft_token_ids)), but got {target_logits.shape[0]}"
            )

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            # metadata.num_draft_tokens_np,
            metadata.num_spec_steps,
            metadata.cu_num_draft_tokens,
            None,
            target_logits,
            bonus_token_ids,
            synthetic_acceptance_rates=self.synthetic_acceptance_rates,
            synthetic_step=self._synthetic_step,
        )
        if self.synthetic_acceptance_rates is not None:
            self._synthetic_step += 1
        return output_token_ids


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # # [batch_size]
    # num_draft_tokens: list[int],
    num_spec_steps: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    # Debug override: per-position unconditional acceptance rates, one entry per
    # speculative position (None => normal draft/target path).
    synthetic_acceptance_rates: tuple[float, ...] | None = None,
    # Per-step seed for the (rank-consistent) synthetic RNG; ignored otherwise.
    synthetic_step: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(cu_num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # Create output buffer. Each kernel program writes positions
    # [0 .. num_draft_tokens] for its request and fills the unwritten tail
    # [num_draft_tokens+1 .. num_spec_steps] with the -1 truncation sentinel
    # itself (variable-length verification / DSpark Phase 2), so modify triton kernel
    output_token_ids = torch.empty(
        (batch_size, num_spec_steps + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    num_bonus_tokens = torch.empty(batch_size, dtype=torch.int32, device=device)

    if synthetic_acceptance_rates is not None:
        # Synthetic path: force a target acceptance length independent of the
        # real draft/target agreement. Draft tokens are accepted with the
        # configured per-position probability; on the first rejection we emit the
        # target argmax as the correction token and stop (same output layout /
        # num_bonus_tokens semantics as the greedy path).
        cond_rates = _get_synthetic_cond_rates(synthetic_acceptance_rates, device)
        target_argmax = target_probs.argmax(dim=-1)
        # Rank-consistent uniforms: a dedicated device generator re-seeded from the
        # step counter draws the same Philox stream on every TP rank / GPU, so the
        # accept/reject pattern — and hence num_bonus_tokens — matches the
        # broadcast sampled_tokens. A per-rank unseeded torch.rand would desync
        # them and make the anchor gather read a -1 column (draft embeds an
        # invalid id -> crash).
        #
        # These stay on the device on purpose. Drawing on the host and copying in
        # costs a pageable H2D, which PyTorch issues as memcpy + stream sync: the
        # caller then blocks until the whole target forward has drained, right
        # between verify and propose, so the draft model's kernels cannot be
        # dispatched while the target is still running.
        generator = _synthetic_generator(device)
        generator.manual_seed(_SYNTHETIC_RNG_BASE_SEED + int(synthetic_step))
        uniform = torch.rand(
            num_tokens, dtype=torch.float32, device=device, generator=generator
        )
        rejection_synthetic_sample_kernel[(batch_size,)](
            output_token_ids,
            num_bonus_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            uniform,
            cond_rates,
            num_spec_steps,
            num_warps=1,
        )
    elif RELAXED_TOP_N <= 1:
        # Strict greedy path: draft must exactly match target argmax
        target_argmax = target_probs.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            num_bonus_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            num_spec_steps,
            num_warps=1,
        )
    else:
        # Relaxed acceptance path: accept if draft is among top-N
        # candidates with prob >= (top1_prob - delta)
        probs = target_probs.softmax(dim=-1, dtype=torch.float32)
        topn_probs, topn_ids = torch.topk(probs, RELAXED_TOP_N, dim=-1)

        top1_probs = topn_probs[:, 0:1]
        valid_mask = topn_probs >= (top1_probs - RELAXED_DELTA)
        topn_ids[~valid_mask] = -1
        topn_ids = topn_ids.to(torch.int32).contiguous()

        rejection_relaxed_sample_kernel[(batch_size,)](
            output_token_ids,
            num_bonus_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            topn_ids,
            bonus_token_ids,
            num_spec_steps,
            RELAXED_TOP_N,
            num_warps=1,
        )

    return output_token_ids, num_bonus_tokens


@triton.jit(do_not_specialize=["num_spec_steps"])
# TODO use the same sampler as main model
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    num_bonus_tokens_ptr,
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    num_spec_steps,
):
    req_idx = tl.program_id(0)

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    num_bonus_token = -1
    INVALID_TOKEN: tl.constexpr = -1
    for pos in range(num_draft_tokens):
        if rejected:
            target_argmax_id = INVALID_TOKEN
        else:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            target_argmax_id = tl.cast(target_argmax_id, tl.int32)
            if draft_token_id != target_argmax_id:
                # rejected = False
                rejected = True
            num_bonus_token += 1
        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
            target_argmax_id,
        )

    if rejected:
        bonus_token_id = INVALID_TOKEN
    else:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        num_bonus_token += 1
    tl.store(
        output_token_ids_ptr + req_idx * (num_spec_steps + 1) + num_draft_tokens,
        bonus_token_id,
    )
    # Fill the unwritten tail [num_draft_tokens+1 .. num_spec_steps] with the
    # -1 sentinel so downstream first-`-1` truncation is correct with
    # variable-length verification (output buffer is torch.empty).
    for pos in range(num_draft_tokens + 1, num_spec_steps + 1):
        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
            INVALID_TOKEN,
        )
    tl.store(num_bonus_tokens_ptr + req_idx, num_bonus_token)


@triton.jit(do_not_specialize=["num_spec_steps"])
def rejection_synthetic_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    num_bonus_tokens_ptr,
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    uniform_ptr,  # [num_tokens] — per-position U(0, 1) samples
    cond_rates_ptr,  # [num_spec_steps] — P(accept pos | accepted through pos-1)
    num_spec_steps,
):
    req_idx = tl.program_id(0)

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    num_bonus_token = -1
    INVALID_TOKEN: tl.constexpr = -1
    for pos in range(num_draft_tokens):
        if rejected:
            output_id = INVALID_TOKEN
        else:
            u = tl.load(uniform_ptr + start_idx + pos)
            acceptance_rate = tl.load(cond_rates_ptr + pos)
            if u < acceptance_rate:
                # Force-accept: emit the draft's own proposed token.
                output_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                output_id = tl.cast(output_id, tl.int32)
            else:
                # Reject: emit the target correction token and stop accepting.
                output_id = tl.load(target_argmax_ptr + start_idx + pos)
                output_id = tl.cast(output_id, tl.int32)
                rejected = True
            num_bonus_token += 1
        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
            output_id,
        )

    if rejected:
        bonus_token_id = INVALID_TOKEN
    else:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        num_bonus_token += 1
    tl.store(
        output_token_ids_ptr + req_idx * (num_spec_steps + 1) + num_draft_tokens,
        bonus_token_id,
    )
    # Fill the unwritten tail [num_draft_tokens+1 .. num_spec_steps] with the
    # -1 sentinel so downstream first-`-1` truncation is correct with
    # variable-length verification (output buffer is torch.empty).
    for pos in range(num_draft_tokens + 1, num_spec_steps + 1):
        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
            INVALID_TOKEN,
        )
    tl.store(num_bonus_tokens_ptr + req_idx, num_bonus_token)


@triton.jit(do_not_specialize=["num_spec_steps", "top_n"])
def rejection_relaxed_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    num_bonus_tokens_ptr,
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    topn_ids_ptr,  # [num_tokens, top_n] — candidate token ids, -1 = invalid
    bonus_token_ids_ptr,  # [batch_size]
    num_spec_steps,
    top_n,
):
    req_idx = tl.program_id(0)

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    num_bonus_token = -1
    INVALID_TOKEN: tl.constexpr = -1

    for pos in range(num_draft_tokens):
        if rejected:
            output_id = INVALID_TOKEN
        else:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)

            base_offset = (start_idx + pos) * top_n
            top1_id = tl.load(topn_ids_ptr + base_offset)

            found = False
            for k in range(top_n):
                candidate_id = tl.load(topn_ids_ptr + base_offset + k)
                if candidate_id == draft_token_id:
                    found = True

            if found:
                output_id = draft_token_id
            else:
                output_id = top1_id
                rejected = True

            num_bonus_token += 1

        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
            output_id,
        )

    if rejected:
        bonus_token_id = INVALID_TOKEN
    else:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        num_bonus_token += 1
    tl.store(
        output_token_ids_ptr + req_idx * (num_spec_steps + 1) + num_draft_tokens,
        bonus_token_id,
    )
    # Fill the unwritten tail [num_draft_tokens+1 .. num_spec_steps] with the
    # -1 sentinel so downstream first-`-1` truncation is correct with
    # variable-length verification (output buffer is torch.empty).
    for pos in range(num_draft_tokens + 1, num_spec_steps + 1):
        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
            INVALID_TOKEN,
        )
    tl.store(num_bonus_tokens_ptr + req_idx, num_bonus_token)
