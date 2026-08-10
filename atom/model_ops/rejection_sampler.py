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

# --- Synthetic (forced) acceptance rate for spec-decode debugging ---
# Enabled via --spec-decode-acceptance-rate (SpeculativeConfig.synthetic_
# acceptance_rate). When set (float in [0, 1]), the rejection sampler ignores the
# real draft/target comparison and force-accepts each draft token with a
# position-dependent probability calibrated so the measured mean acceptance rate
# converges to the configured value. Purely a benchmarking / bring-up knob (e.g.
# while an MTP/EAGLE head is still training). Mirrors vLLM's "synthetic"
# rejection_sample_method. See ROCm/ATOM#555.
#
# Lowest per-position conditional-acceptance decay (empirically ~what a
# well-tuned draft model exhibits); bounds the search so the base rate stays
# <= 1 while still reaching the requested mean. Matches vLLM's constant.
MIN_ACCEPTANCE_DECAY_FACTOR = 0.85
# Cache {(rate, num_spec_steps): (base_rate, decay_factor)} — params depend on
# both the target rate and the runtime step count.
_SYNTHETIC_PARAMS_CACHE: dict[tuple[float, int], tuple[float, float]] = {}
# Base seed for the per-step synthetic RNG. The forced accept/reject draw MUST be
# identical on every TP rank: sampled_tokens is broadcast from rank 0 while
# num_bonus_tokens stays local, so a per-rank torch.rand would desync the two and
# make the anchor gather read a rejected (-1) column -> the draft model then
# embeds an invalid id (HSA out-of-bounds). We therefore derive the uniforms from
# a fixed CPU-seeded generator (identical across ranks / GPUs, and reproducible).
_SYNTHETIC_RNG_BASE_SEED = 0x5EED


def compute_synthetic_rejection_sampler_params(
    p_avg: float, n: int, tol: float = 1e-9
) -> tuple[float, float]:
    """Derive (base_acceptance_rate, decay_factor) for a target mean rate.

    With ``n`` speculative positions the conditional acceptance probability at
    position ``i`` is ``base_rate * decay_factor**i``. The joint probability of
    accepting through position ``i`` is therefore
    ``base_rate**(i+1) * decay_factor**(i*(i+1)/2)`` and the mean of those joint
    probabilities across the ``n`` positions equals ``p_avg`` (== ATOM's
    ``accepted_draft_tokens / total_draft_tokens`` acceptance rate).
    """

    def mean_joint_prob(a_0: float, gamma: float, n: int) -> float:
        total = 0.0
        for i in range(n):
            total += a_0 ** (i + 1) * gamma ** (i * (i + 1) // 2)
        return total / n

    def min_valid_decay_factor(p: float, n: int) -> float:
        low, high = MIN_ACCEPTANCE_DECAY_FACTOR, 1.0
        # Even with a base rate of 1, a large decay is required for big p; find
        # the smallest decay that can still reach p so base_rate stays <= 1.
        if mean_joint_prob(1.0, low, n) >= p:
            return low
        while (high - low) > tol:
            mid = (low + high) / 2
            if mean_joint_prob(1.0, mid, n) >= p:
                high = mid
            else:
                low = mid
        return high

    def compute_base_acceptance_rate(p_avg: float, gamma: float, n: int) -> float:
        if p_avg <= 0.0:
            return 0.0
        if p_avg >= 1.0:
            return 1.0
        low, high = 0.0, 1.0
        while (high - low) > tol:
            mid = (low + high) / 2
            if mean_joint_prob(mid, gamma, n) >= p_avg:
                high = mid
            else:
                low = mid
        return high

    decay_factor = min_valid_decay_factor(p_avg, n)
    base_rate = compute_base_acceptance_rate(p_avg, decay_factor, n)
    return base_rate, decay_factor


def _get_synthetic_params(rate: float, num_spec_steps: int) -> tuple[float, float]:
    key = (rate, num_spec_steps)
    if key not in _SYNTHETIC_PARAMS_CACHE:
        _SYNTHETIC_PARAMS_CACHE[key] = compute_synthetic_rejection_sampler_params(
            rate, num_spec_steps
        )
    return _SYNTHETIC_PARAMS_CACHE[key]


class RejectionSampler(nn.Module):
    def __init__(self, synthetic_acceptance_rate: float | None = None):
        super().__init__()
        # Debug/benchmark override: force a fixed acceptance rate (see module
        # docstring). None => normal draft/target rejection sampling.
        if synthetic_acceptance_rate is not None and not (
            0.0 <= synthetic_acceptance_rate <= 1.0
        ):
            raise ValueError(
                "synthetic_acceptance_rate must be in [0, 1], "
                f"but got {synthetic_acceptance_rate}"
            )
        self.synthetic_acceptance_rate = synthetic_acceptance_rate
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
            synthetic_acceptance_rate=self.synthetic_acceptance_rate,
            synthetic_step=self._synthetic_step,
        )
        if self.synthetic_acceptance_rate is not None:
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
    # Debug override: forced acceptance rate in [0, 1] (None => normal path).
    synthetic_acceptance_rate: float | None = None,
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

    if synthetic_acceptance_rate is not None:
        # Synthetic path: force a target acceptance rate independent of the real
        # draft/target agreement. Draft tokens are accepted with a
        # position-decaying probability; on the first rejection we emit the
        # target argmax as the correction token and stop (same output layout /
        # num_bonus_tokens semantics as the greedy path).
        base_rate, decay_factor = _get_synthetic_params(
            synthetic_acceptance_rate, num_spec_steps
        )
        target_argmax = target_probs.argmax(dim=-1)
        # Rank-consistent uniforms: draw on CPU from a fixed per-step seed (same on
        # every TP rank / GPU) so the accept/reject pattern — and hence
        # num_bonus_tokens — is identical across ranks and matches the broadcast
        # sampled_tokens. A per-rank torch.rand would desync them and make the
        # anchor gather read a -1 column (draft embeds an invalid id -> crash).
        cpu_generator = torch.Generator()
        cpu_generator.manual_seed(_SYNTHETIC_RNG_BASE_SEED + int(synthetic_step))
        uniform = torch.rand(
            num_tokens, dtype=torch.float32, generator=cpu_generator
        ).to(device)
        rejection_synthetic_sample_kernel[(batch_size,)](
            output_token_ids,
            num_bonus_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            uniform,
            base_rate,
            decay_factor,
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
    base_acceptance_rate,
    decay_factor,
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
    # Per-position conditional acceptance probability, decaying geometrically.
    acceptance_rate = base_acceptance_rate
    for pos in range(num_draft_tokens):
        if rejected:
            output_id = INVALID_TOKEN
        else:
            u = tl.load(uniform_ptr + start_idx + pos)
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
            acceptance_rate = acceptance_rate * decay_factor
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
