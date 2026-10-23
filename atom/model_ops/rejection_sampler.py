import torch
import triton
import triton.language as tl
from aiter import topk_select
from torch import nn

from atom.utils import envs
from atom.utils.forward_context import SpecDecodeMetadata

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
# Base seed for the probability-ratio path's accept coins. Same discipline and
# same reason: `coin * q < p` decides the verdict, which stays rank-local while
# sampled_tokens is broadcast from rank 0. Distinct from the synthetic base so
# the two cannot alias each other's stream.
_RATIO_RNG_BASE_SEED = 0x11A7
# One generator per (device, purpose), kept alive so a draw costs only a
# kernel.
_SYNTHETIC_GENERATORS: dict[torch.device, torch.Generator] = {}
_RATIO_GENERATORS: dict[torch.device, torch.Generator] = {}


def _synthetic_generator(device: torch.device) -> torch.Generator:
    generator = _SYNTHETIC_GENERATORS.get(device)
    if generator is None:
        generator = torch.Generator(device=device)
        _SYNTHETIC_GENERATORS[device] = generator
    return generator


def _ratio_generator(device: torch.device) -> torch.Generator:
    generator = _RATIO_GENERATORS.get(device)
    if generator is None:
        generator = torch.Generator(device=device)
        _RATIO_GENERATORS[device] = generator
    return generator


# Vocab tile the residual's two full-vocab passes walk in. A constant rather
# than something autotuned per shape: a different tile changes the fp32
# summation order of `norm_sum`, moving the CDF boundary by an ulp.
_RATIO_BLOCK_V = 4096


@triton.jit
def _softmax_temp_partial_kernel(
    logits_ptr,  # [num_rows, vocab] bf16/fp16/fp32
    temp_ptr,  # [num_rows] fp32, already expanded per row
    part_max_ptr,  # [n_tiles, num_rows] fp32
    part_sum_ptr,  # [n_tiles, num_rows] fp32
    num_rows,
    vocab,
    logits_row_stride,
    BLOCK_V: tl.constexpr,
):
    """One (row, V tile): this tile's max and its sum of ``exp(x - tile_max)``."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    temp = tl.load(temp_ptr + row).to(tl.float32)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    vmask = offs < vocab
    x = tl.load(
        logits_ptr + row.to(tl.int64) * logits_row_stride + offs,
        mask=vmask,
        other=-float("inf"),
    ).to(tl.float32)
    x = x / temp
    tile_max = tl.max(x, axis=0)
    # A fully masked tile is -inf, and exp(-inf - -inf) is NaN, so mask the sum
    # rather than letting the exponential zero it.
    tile_sum = tl.sum(tl.where(vmask, tl.exp(x - tile_max), 0.0), axis=0)
    tl.store(part_max_ptr + tile * num_rows + row, tile_max)
    tl.store(part_sum_ptr + tile * num_rows + row, tile_sum)


@triton.jit
def _softmax_temp_combine_kernel(
    part_max_ptr,
    part_sum_ptr,
    row_max_ptr,
    row_inv_z_ptr,
    num_rows,
    n_tiles,
    BLOCK_TILES: tl.constexpr,
):
    """One row: global max and 1/Z from the per-tile pairs."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < n_tiles
    maxes = tl.load(
        part_max_ptr + offs * num_rows + row, mask=mask, other=-float("inf")
    )
    sums = tl.load(part_sum_ptr + offs * num_rows + row, mask=mask, other=0.0)
    global_max = tl.max(maxes, axis=0)
    z = tl.sum(tl.where(mask, sums * tl.exp(maxes - global_max), 0.0), axis=0)
    tl.store(row_max_ptr + row, global_max)
    tl.store(row_inv_z_ptr + row, tl.where(z > 0.0, 1.0 / z, 0.0))


@triton.jit
def _softmax_temp_store_kernel(
    logits_ptr,
    temp_ptr,
    row_max_ptr,
    row_inv_z_ptr,
    out_ptr,  # [num_rows, vocab] fp32
    vocab,
    logits_row_stride,
    BLOCK_V: tl.constexpr,
):
    """One (row, V tile): ``exp(x - row_max) / Z``."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    temp = tl.load(temp_ptr + row).to(tl.float32)
    row_max = tl.load(row_max_ptr + row)
    inv_z = tl.load(row_inv_z_ptr + row)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    vmask = offs < vocab
    x = tl.load(
        logits_ptr + row.to(tl.int64) * logits_row_stride + offs,
        mask=vmask,
        other=-float("inf"),
    ).to(tl.float32)
    x = x / temp
    tl.store(
        out_ptr + row.to(tl.int64) * vocab + offs,
        tl.exp(x - row_max) * inv_z,
        mask=vmask,
    )


def _ratio_target_probs(
    target_logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    temperatures: torch.Tensor,  # [batch_size] fp32, pre-clamped to SAMPLER_EPS
) -> torch.Tensor:
    """``p = softmax(target_logits / T)`` in fp32, one temperature per request.

    The one tensor the verify side has to materialize that it did not before.
    ATOM's unfiltered verify draw goes through aiter's fused
    `mixed_sample_outer_exponential` (sampler.py), which takes raw logits and
    never forms probabilities -- only the top-k/top-p branch builds an fp32
    `[rows, V]`, and that branch is refused here, so there is no existing tensor
    to borrow.

    fp32 and not bf16: `p - q` is a difference of two near-equal numbers on
    exactly the rows the residual cares about, and bf16's 8 mantissa bits leave
    that difference with almost no significant ones.

    Tiled across programs rather than one program per row: the row-per-program
    shape has only `num_rows` programs, which does not fill the device at the
    batch sizes this runs at. Combining the per-tile (max, sumexp) is the same
    max-subtracted softmax with the reduction reassociated.
    """
    rows, vocab = target_logits.shape
    # Expand [batch_size] -> [num_tokens] the way
    # `Sampler.sample_verification_tokens` does, so a row's temperature is the
    # same number in both.
    request_indices = torch.searchsorted(
        cu_num_draft_tokens,
        torch.arange(
            rows, device=target_logits.device, dtype=cu_num_draft_tokens.dtype
        ),
        right=True,
    )
    per_row = temperatures.to(torch.float32)[request_indices].contiguous()
    device = target_logits.device
    out = torch.empty((rows, vocab), dtype=torch.float32, device=device)
    if not rows:
        return out
    n_tiles = triton.cdiv(vocab, _RATIO_BLOCK_V)
    f32 = {"dtype": torch.float32, "device": device}
    part_max = torch.empty((n_tiles, rows), **f32)
    part_sum = torch.empty((n_tiles, rows), **f32)
    row_max = torch.empty((rows,), **f32)
    row_inv_z = torch.empty((rows,), **f32)
    _softmax_temp_partial_kernel[(rows, n_tiles)](
        target_logits,
        per_row,
        part_max,
        part_sum,
        rows,
        vocab,
        target_logits.stride(0),
        BLOCK_V=_RATIO_BLOCK_V,
        num_warps=8,
    )
    _softmax_temp_combine_kernel[(rows,)](
        part_max,
        part_sum,
        row_max,
        row_inv_z,
        rows,
        n_tiles,
        BLOCK_TILES=triton.next_power_of_2(n_tiles),
        num_warps=4,
    )
    _softmax_temp_store_kernel[(rows, n_tiles)](
        target_logits,
        per_row,
        row_max,
        row_inv_z,
        out,
        vocab,
        target_logits.stride(0),
        BLOCK_V=_RATIO_BLOCK_V,
        num_warps=8,
    )
    return out


def _assert_unfiltered(top_ks, top_ps) -> None:
    """Internal invariant: no caller reaches the ratio path with a filter on.

    ``ModelRunner.postprocess`` drops ``draft_probs`` for a filtered batch, so
    this should be unreachable -- it is kept because the thing it guards is
    silent: the draft drew from an UNFILTERED ``softmax(logits / T)``, and a
    ratio test against a filtered target ``p`` would bias the emitted
    distribution with nothing in the output to show for it.

    ``top_ks`` is ATOM's ``-1 means disabled`` convention and ``top_ps`` its
    ``1.0 means disabled`` one (see ``Sampler.forward``); either may also arrive
    as None or as a uniform python scalar.
    """

    def _active_k(value) -> bool:
        if value is None:
            return False
        if isinstance(value, torch.Tensor):
            return bool((value > 0).any().item())
        return value > 0

    def _active_p(value) -> bool:
        if value is None:
            return False
        if isinstance(value, torch.Tensor):
            return bool((value < 1.0).any().item())
        return value < 1.0

    if _active_k(top_ks) or _active_p(top_ps):
        raise ValueError(
            "probability-ratio speculative acceptance was handed a top-k/top-p "
            "batch: the draft drew from an UNFILTERED softmax(logits / T), so "
            "the ratio test against a filtered target p would change the "
            "emitted distribution without saying so. The caller must pass "
            "draft_probs=None for a filtered batch, which falls back to "
            "exact-match acceptance and is still lossless."
        )


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
        # Same, for the probability-ratio accept coins. Advanced on every
        # forward that takes that path, so no two steps replay a stream.
        self._ratio_step = 0

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        target_logits: torch.Tensor,
        # [batch_size, 1]
        bonus_token_ids: torch.Tensor,
        *,
        target_token_ids: torch.Tensor | None = None,
        # ---- probability-ratio path ----------------------------------------
        # [num_draft_rows, vocab_size] fp32: the q each draft token was drawn
        # from. None selects exact-match acceptance.
        draft_probs: torch.Tensor | None = None,
        # [batch_size] int32: first q row of each request, from the drafter's
        # req_id -> slot map. Negative means "no live q"; see the kernel.
        draft_probs_row_base: torch.Tensor | None = None,
        # [batch_size] fp32, pre-clamped. Required with draft_probs.
        temperatures: torch.Tensor | None = None,
        # The caller drops draft_probs for a filtered batch, so these are
        # disabled whenever the ratio path is live.
        top_ks: torch.Tensor | int | None = None,
        top_ps: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if draft_probs is not None:
            _assert_unfiltered(top_ks, top_ps)
        # Ensure target_logits is contiguous. For greedy sampling, we can use
        # logits directly (argmax is the same for logits and probs), but we
        # need to ensure it's contiguous to satisfy the assertion in rejection_sample.
        target_logits = target_logits.contiguous()

        output_token_ids = rejection_sample(
            metadata.input_ids,
            metadata.target_logits_indices,
            # metadata.num_draft_tokens_np,
            metadata.num_spec_steps,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            metadata.bonus_logits_indices,
            synthetic_acceptance_rates=self.synthetic_acceptance_rates,
            synthetic_step=self._synthetic_step,
            target_token_ids=target_token_ids,
            temperatures=temperatures,
            draft_probs_row_base=draft_probs_row_base,
            ratio_step=self._ratio_step,
        )
        if self.synthetic_acceptance_rates is not None:
            self._synthetic_step += 1
        if draft_probs is not None and self.synthetic_acceptance_rates is None:
            self._ratio_step += 1
        return output_token_ids


def rejection_sample(
    # the step's input ids; draft r is input_ids[target_logits_indices[r] + 1]
    input_ids: torch.Tensor,
    # [num_tokens] -- the row of the step's logits each target_probs row is
    target_logits_indices: torch.Tensor,
    # # [batch_size]
    # num_draft_tokens: list[int],
    num_spec_steps: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_draft_rows, vocab_size] fp32, or None. The q each draft token was
    # drawn from; None keeps the exact-match acceptance below. Rows are reached
    # through `draft_probs_row_base`, not `start_idx`.
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    # [batch_size] -- each request's last row, the one its bonus is sampled at
    bonus_logits_indices: torch.Tensor,
    # Debug override: per-position unconditional acceptance rates, one entry per
    # speculative position (None => normal draft/target path).
    synthetic_acceptance_rates: tuple[float, ...] | None = None,
    # Per-step seed for the (rank-consistent) synthetic RNG; ignored otherwise.
    synthetic_step: int = 0,
    target_token_ids: torch.Tensor | None = None,
    # ---- probability-ratio path (keyword-only, so the existing positional
    # callers keep meaning what they meant) ----------------------------------
    *,
    # [batch_size] fp32, pre-clamped to SAMPLER_EPS. The ratio rule compares
    # against p = softmax(target_logits / T), and `target_probs` arrives raw.
    temperatures: torch.Tensor | None = None,
    # [batch_size] int32: this request's first row in `draft_probs`, or a
    # negative sentinel when it has no live q. None means the flat ragged
    # layout, i.e. row == the request's `start_idx`.
    draft_probs_row_base: torch.Tensor | None = None,
    # Per-step seed for the (rank-consistent) accept coins.
    ratio_step: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert input_ids.ndim == 1 and target_logits_indices.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(cu_num_draft_tokens)
    num_tokens = target_logits_indices.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert input_ids.is_contiguous() and target_logits_indices.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)
    if draft_probs is not None:
        assert draft_probs.shape[-1] == vocab_size, (
            f"draft q covers {draft_probs.shape[-1]} ids but the target scored "
            f"{vocab_size}; the ratio rule indexes both rows with the same id"
        )
        assert temperatures is not None, (
            "the probability-ratio path needs the target temperatures: p is "
            "softmax(target_logits / T), not softmax(target_logits)"
        )
        # The ratio branch below sits above the relaxed one, so the relaxed
        # path would simply never run. The two are not combinable: relaxed
        # acceptance is deliberately not distribution-preserving, while the
        # ratio rule's entire claim is that it is.
        #
        # An assert, not a raise: the pair is refused when any drafter is
        # constructed (`Drafter.__init__`), so reaching here means that gate
        # was bypassed. Re-checking on the verify path cost a per-step
        # branch in every TP worker for a condition that cannot change after
        # startup -- both flags are read at import -- and it only fired on a
        # sampled unfiltered batch, so an all-greedy batch silently ran relaxed
        # instead of being refused.
        assert RELAXED_TOP_N <= 1, (
            "ATOM_SPEC_LOSSLESS_SAMPLING and ATOM_ENABLE_RELAXED_MTP are "
            "mutually exclusive; this should have been refused at drafter "
            "construction"
        )
    if target_token_ids is not None:
        if target_token_ids.shape != target_logits_indices.shape:
            raise ValueError("Sampled target IDs must cover every verification row")
        target_token_ids = target_token_ids.contiguous()

    # Every `topk_select` below passes `tie="low"`, which is the whole reason
    # its answer is the same pick `torch.argmax`/`torch.topk` made. The default
    # promises no direction among equal scores, so a near-tie would resolve
    # differently from one TP rank to the next -- and the accept/reject pattern
    # these feed has to agree across ranks or `num_bonus_tokens` desyncs. It is
    # free at k=1: the reduction that serves k=1 can keep the promise anyway.
    # Indices come back int32, which is what the kernels here load.

    # Create output buffer. Each kernel program writes positions
    # [0 .. num_draft_tokens] for its request and fills the unwritten tail
    # [num_draft_tokens+1 .. num_spec_steps] with the -1 truncation sentinel
    # itself (variable-length verification / DSpark Phase 2), so modify triton kernel
    output_token_ids = torch.empty(
        (batch_size, num_spec_steps + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    # Rows: accepted drafts, rejected drafts (against `num_spec_steps`), and the
    # anchor -- the flat row of the last token emitted. One buffer, so a caller
    # that must align them across ranks broadcasts once.
    verdict = torch.empty((3, batch_size), dtype=torch.int32, device=device)

    if synthetic_acceptance_rates is not None:
        # Synthetic path: force a target acceptance length independent of the
        # real draft/target agreement. Draft tokens are accepted with the
        # configured per-position probability; on the first rejection we emit the
        # target argmax as the correction token and stop (same output layout /
        # num_bonus_tokens semantics as the greedy path).
        cond_rates = _get_synthetic_cond_rates(synthetic_acceptance_rates, device)
        _, target_argmax = topk_select(target_probs, 1, tie="low")
        target_argmax = target_argmax.view(-1)
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
            verdict,
            cu_num_draft_tokens,
            input_ids,
            target_logits_indices,
            target_argmax,
            bonus_token_ids,
            bonus_logits_indices,
            uniform,
            cond_rates,
            num_spec_steps,
            num_warps=1,
        )
    elif draft_probs is not None:
        # Probability-ratio path. Accept with `coin * q < p` against the q the
        # draft actually drew from, instead of requiring exact equality with the
        # target's own draw: same output distribution, higher acceptance.
        # Inseparable from draft-side sampling, because for a greedy delta q
        # sum_x min(p(x), q(x)) = p(x*) and the rule degenerates to exactly the
        # exact-match kernel with a full-vocab softmax bolted on.
        #
        # `target_probs` is raw logits here (only the relaxed path below ever
        # softmaxed them), so build p as a temperature softmax in fp32.
        target_p = _ratio_target_probs(target_probs, cu_num_draft_tokens, temperatures)
        if draft_probs_row_base is None:
            # Flat ragged q: row == start_idx + pos, so the base is the request's
            # own start. cu is exclusive-end, so shift it by one.
            draft_probs_row_base = torch.cat(
                (
                    cu_num_draft_tokens.new_zeros(1),
                    cu_num_draft_tokens[:-1],
                )
            ).to(torch.int32)
        draft_probs_row_base = draft_probs_row_base.contiguous()
        # Rank-consistent coins, same discipline as the synthetic path's, drawn
        # on the device so nothing here blocks on an H2D.
        generator = _ratio_generator(device)
        generator.manual_seed(_RATIO_RNG_BASE_SEED + int(ratio_step))
        # One coin per draft position plus one per request for the residual
        # CDF, as a single draw so a batch costs one kernel.
        coins = torch.rand(
            num_tokens + batch_size,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        rejection_ratio_sample_kernel[(batch_size,)](
            output_token_ids,
            verdict,
            cu_num_draft_tokens,
            input_ids,
            target_logits_indices,
            target_p,
            draft_probs,
            draft_probs_row_base,
            bonus_token_ids,
            bonus_logits_indices,
            coins[:num_tokens],
            coins[num_tokens:],
            num_spec_steps,
            VOCAB_SIZE=vocab_size,
            BLOCK_V=_RATIO_BLOCK_V,
            num_warps=4,
        )
    elif target_token_ids is not None or RELAXED_TOP_N <= 1:
        # Match the actual target draw for stochastic requests; greedy requests
        # take the target argmax and must match it exactly. The prefix kernel
        # needs only token IDs. `topk_select`, not `.argmax`: the tie rule
        # above is what keeps the accept/reject pattern equal across TP ranks.
        if target_token_ids is None:
            _, target_argmax = topk_select(target_probs, 1, tie="low")
            target_argmax = target_argmax.view(-1)
        else:
            target_argmax = target_token_ids
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            verdict,
            cu_num_draft_tokens,
            input_ids,
            target_logits_indices,
            target_argmax,
            bonus_token_ids,
            bonus_logits_indices,
            num_spec_steps,
            num_warps=1,
        )
    else:
        # Relaxed acceptance path: accept if draft is among top-N
        # candidates with prob >= (top1_prob - delta)
        probs = target_probs.softmax(dim=-1, dtype=torch.float32)
        # `sorted` is what puts the top-1 in slot 0 for `top1_probs` below.
        #
        # Among EQUAL probabilities the slot order is unspecified, here as it
        # was under `torch.topk` -- neither promises one. Measured over
        # 64-256 rows of three vocabularies: the id set per row is identical
        # every time and the values are bitwise equal, so `valid_mask` and the
        # kernel's membership test are unaffected; slot 0, which the kernel
        # emits as the correction token, moved on 2 rows of 256 and in both the
        # two ids held the same probability. Rank-consistent either way, since
        # `tie="low"` leaves only backends whose answer is a function of the row.
        topn_probs, topn_ids = topk_select(
            probs, RELAXED_TOP_N, sorted=True, tie="low", return_value=True
        )

        top1_probs = topn_probs[:, 0:1]
        valid_mask = topn_probs >= (top1_probs - RELAXED_DELTA)
        topn_ids[~valid_mask] = -1
        topn_ids = topn_ids.contiguous()

        rejection_relaxed_sample_kernel[(batch_size,)](
            output_token_ids,
            verdict,
            cu_num_draft_tokens,
            input_ids,
            target_logits_indices,
            topn_ids,
            bonus_token_ids,
            bonus_logits_indices,
            num_spec_steps,
            RELAXED_TOP_N,
            num_warps=1,
        )

    return output_token_ids, verdict


@triton.jit
def _draft_token(input_ids_ptr, target_logits_indices_ptr, r):
    """Draft r: the input token after the logits row that verifies it."""
    return tl.load(input_ids_ptr + tl.load(target_logits_indices_ptr + r) + 1)


@triton.jit
def _finish_request(
    output_token_ids_ptr,
    verdict_ptr,
    bonus_token_ids_ptr,
    bonus_logits_indices_ptr,
    req_idx,
    num_draft_tokens,
    last_checked,
    rejected,
    num_spec_steps,
):
    """The bonus token, the -1 tail, and the request's verdict.

    `last_checked` is the last draft position compared (-1 before any). The
    anchor -- the last emitted token's row -- sits one row per accepted draft
    into the segment that ends at the bonus row.
    """
    INVALID_TOKEN: tl.constexpr = -1
    row = output_token_ids_ptr + req_idx * (num_spec_steps + 1)
    num_accepted = last_checked
    if rejected:
        bonus_token_id = INVALID_TOKEN
    else:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        num_accepted += 1
    tl.store(row + num_draft_tokens, bonus_token_id)
    # The -1 tail is what downstream first-`-1` truncation reads under
    # variable-length verification (the output buffer is torch.empty).
    for pos in range(num_draft_tokens + 1, num_spec_steps + 1):
        tl.store(row + pos, INVALID_TOKEN)
    batch_size = tl.num_programs(0)
    anchor = tl.load(bonus_logits_indices_ptr + req_idx) - num_draft_tokens
    tl.store(verdict_ptr + req_idx, num_accepted)
    tl.store(verdict_ptr + batch_size + req_idx, num_spec_steps - num_accepted)
    tl.store(verdict_ptr + 2 * batch_size + req_idx, anchor + num_accepted)


@triton.jit(do_not_specialize=["num_spec_steps"])
# TODO use the same sampler as main model
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    verdict_ptr,  # [3, batch_size]: accepted drafts, rejected, anchor row
    cu_num_draft_tokens_ptr,  # [batch_size]
    input_ids_ptr,  # the step's input ids
    target_logits_indices_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    bonus_logits_indices_ptr,  # [batch_size]
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
            draft_token_id = _draft_token(
                input_ids_ptr, target_logits_indices_ptr, start_idx + pos
            )
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

    _finish_request(
        output_token_ids_ptr,
        verdict_ptr,
        bonus_token_ids_ptr,
        bonus_logits_indices_ptr,
        req_idx,
        num_draft_tokens,
        num_bonus_token,
        rejected,
        num_spec_steps,
    )


@triton.jit(do_not_specialize=["num_spec_steps"])
def rejection_synthetic_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    verdict_ptr,  # [3, batch_size]: accepted drafts, rejected, anchor row
    cu_num_draft_tokens_ptr,  # [batch_size]
    input_ids_ptr,  # the step's input ids
    target_logits_indices_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    bonus_logits_indices_ptr,  # [batch_size]
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
                output_id = _draft_token(
                    input_ids_ptr, target_logits_indices_ptr, start_idx + pos
                )
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

    _finish_request(
        output_token_ids_ptr,
        verdict_ptr,
        bonus_token_ids_ptr,
        bonus_logits_indices_ptr,
        req_idx,
        num_draft_tokens,
        num_bonus_token,
        rejected,
        num_spec_steps,
    )


@triton.jit(do_not_specialize=["num_spec_steps"])
def rejection_ratio_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    verdict_ptr,  # [3, batch_size]: accepted drafts, rejected, anchor row
    cu_num_draft_tokens_ptr,  # [batch_size]
    input_ids_ptr,  # the step's input ids
    target_logits_indices_ptr,  # [num_tokens]
    target_probs_ptr,  # [num_tokens, VOCAB_SIZE] fp32, already softmaxed
    draft_probs_ptr,  # [num_draft_rows, VOCAB_SIZE] fp32, the q the draft drew from
    draft_row_base_ptr,  # [batch_size] int32 -- this request's first q row
    bonus_token_ids_ptr,  # [batch_size]
    bonus_logits_indices_ptr,  # [batch_size]
    uniform_ptr,  # [num_tokens] accept coin, one per verification row
    uniform_final_ptr,  # [batch_size] residual-CDF coin, one per request
    num_spec_steps,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Probability-ratio speculative acceptance: accept iff ``coin * q < p``.

    The classic rejection-sampling rule, in ATOM's ragged
    ``cu_num_draft_tokens`` layout and its ``output_token_ids`` / ``verdict``
    output convention. A DSpark/EAGLE block is a chain rather than a tree, so
    the position is the node id and ATOM's dense ``[batch, num_spec_steps + 1]``
    row replaces the usual gather.

    The usual "final sampling" step covers both outcomes -- the residual on
    rejection, and a plain draw from ``target_probs[num_steps]`` when every
    draft was accepted. ATOM already samples that all-accepted token upstream
    (``bonus_token_ids`` comes out of the main sampler over ``bonus_logits``),
    so only the residual half is done here. Same distribution, and it is what
    lets the full-vocab pass be skipped on every accepted row: the chain stops
    at the first rejection, so at most one position per request ever reads [V]
    of q.
    """
    req_idx = tl.program_id(0)

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx
    # Where this request's q rows start. The draft wrote them at its batch slot
    # a step ago and the scheduler may have reordered the batch since, so the
    # row is reached by indirection rather than by `start_idx`.
    q_base = tl.load(draft_row_base_ptr + req_idx)

    # A negative base means this request has no live q: its persistent slot was
    # expired and handed to another request (`DSparkProposer._claim_q_slot`).
    # The sentinel is negative and not 0 because row 0 is in range and carries
    # q > 0 -- it would pass `q_is_prob`, and `coin * q < p` would then accept
    # unconditionally against a stranger's distribution. In-range was never the
    # safety property; being the right row is.
    #
    # `has_q` gates every q load; `q_safe_base` keeps the address in range so a
    # negative base cannot form a wild pointer. With q forced to 0 the token is
    # rejected and the residual below resamples from target `p` alone --
    # distributionally exact, costing one step's speculation.
    has_q = q_base >= 0
    q_safe_base = tl.where(has_q, q_base, 0)

    INVALID_TOKEN: tl.constexpr = -1
    rejected = False
    reject_pos = -1
    last_checked = -1

    for pos in range(num_draft_tokens):
        if rejected:
            tl.store(
                output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
                INVALID_TOKEN,
            )
        else:
            draft_token_id = _draft_token(
                input_ids_ptr, target_logits_indices_ptr, start_idx + pos
            )
            p_row = (start_idx + pos).to(tl.int64) * VOCAB_SIZE
            q_row = (q_safe_base + pos).to(tl.int64) * VOCAB_SIZE
            p = tl.load(target_probs_ptr + p_row + draft_token_id)
            q = tl.load(draft_probs_ptr + q_row + draft_token_id, mask=has_q, other=0.0)
            coin = tl.load(uniform_ptr + start_idx + pos)

            # X was sampled from q, so q(X) has to be a positive probability.
            # Anything else means this row is not the distribution X came from,
            # and `coin * q < p` would accept unconditionally on it.
            q_is_prob = (q > 0.0) & (q <= 1.0)

            # `coin * q < p` rather than `coin < p / q`: the product form has
            # no division, so a denormal q cannot become an inf ratio.
            if q_is_prob & (coin * q < p):
                tl.store(
                    output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
                    tl.cast(draft_token_id, tl.int32),
                )
            else:
                rejected = True
                reject_pos = pos
            last_checked += 1

    if rejected:
        # Residual resample from normalized relu(p - q) at the rejecting row.
        tp_base_ptr = (
            target_probs_ptr + (start_idx + reject_pos).to(tl.int64) * VOCAB_SIZE
        )
        dp_base_ptr = (
            draft_probs_ptr + (q_safe_base + reject_pos).to(tl.int64) * VOCAB_SIZE
        )
        coin_final = tl.load(uniform_final_ptr + req_idx)

        # Pass 1: the normalizer.
        norm_sum = 0.0
        for v_start in range(0, VOCAB_SIZE, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE
            p_val = tl.load(tp_base_ptr + v_offsets, mask=mask, other=0.0)
            q_val = tl.load(dp_base_ptr + v_offsets, mask=mask, other=0.0)
            # Any non-probability q (NaN, +-inf, negative) counts as 0, so the
            # residual falls back to p; a NaN comparison is false, so the range
            # test rejects NaN along with the infinities.
            q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
            # No live q: the residual is relu(p - 0) == p, a plain draw from
            # the target. Masking the load alone is not enough, since
            # `q_safe_base` aliases row 0.
            q_val = tl.where(has_q, q_val, 0.0)
            diff = p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)
            # Reduce with the SAME tree pass 2 walks with. `tl.sum` and
            # `tl.cumsum` lower differently and disagree by an ulp, which would
            # let `target_u = coin * norm_sum` exceed the CDF pass 2 actually
            # accumulates and miss in the final tile. val >= 0, so the scan is
            # monotone and its max is its last element.
            norm_sum += tl.max(tl.cumsum(val, axis=0), axis=0)

        # Pass 2: walk the CDF. A degenerate residual (norm_sum == 0, i.e.
        # p == q everywhere on rejection) leaves the cumsum at 0 <= target_u and
        # final_token falls back to VOCAB_SIZE - 1.
        target_u = coin_final * norm_sum
        cum_sum = 0.0
        final_token = VOCAB_SIZE - 1
        found = 0
        for v_start in range(0, VOCAB_SIZE, BLOCK_V):
            if found == 0:
                v_offsets = v_start + tl.arange(0, BLOCK_V)
                mask = v_offsets < VOCAB_SIZE
                p_val = tl.load(tp_base_ptr + v_offsets, mask=mask, other=0.0)
                q_val = tl.load(dp_base_ptr + v_offsets, mask=mask, other=0.0)
                q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
                # Same no-live-q handling as pass 1. The two passes must apply
                # identical q, or the CDF walked here disagrees with the
                # normalizer computed there and the residual draw is biased.
                q_val = tl.where(has_q, q_val, 0.0)
                diff = p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

                block_cumsum = tl.cumsum(val, axis=0)
                total_cumsum = cum_sum + block_cumsum

                candidates_mask = total_cumsum > target_u
                has_match = tl.max(candidates_mask, axis=0)
                if has_match:
                    match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                    final_token = v_start + match_idx
                    found = 1
                # Advance with the same reduction the hit test used above.
                # Mixing `tl.cumsum` (test) with `tl.sum` (advance) lets the two
                # disagree by an ulp; `target_u` landing in that gap makes this
                # tile miss, leaves `cum_sum` already above `target_u`, and the
                # next tile's element 0 then satisfies the test before its own
                # mass is consulted -- returning a multiple of BLOCK_V chosen by
                # position, with possibly zero residual mass.
                cum_sum += tl.max(block_cumsum, axis=0)

        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + reject_pos,
            tl.cast(final_token, tl.int32),
        )

    _finish_request(
        output_token_ids_ptr,
        verdict_ptr,
        bonus_token_ids_ptr,
        bonus_logits_indices_ptr,
        req_idx,
        num_draft_tokens,
        last_checked,
        rejected,
        num_spec_steps,
    )


@triton.jit(do_not_specialize=["num_spec_steps", "top_n"])
def rejection_relaxed_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    verdict_ptr,  # [3, batch_size]: accepted drafts, rejected, anchor row
    cu_num_draft_tokens_ptr,  # [batch_size]
    input_ids_ptr,  # the step's input ids
    target_logits_indices_ptr,  # [num_tokens]
    topn_ids_ptr,  # [num_tokens, top_n] — candidate token ids, -1 = invalid
    bonus_token_ids_ptr,  # [batch_size]
    bonus_logits_indices_ptr,  # [batch_size]
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
            draft_token_id = _draft_token(
                input_ids_ptr, target_logits_indices_ptr, start_idx + pos
            )

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

    _finish_request(
        output_token_ids_ptr,
        verdict_ptr,
        bonus_token_ids_ptr,
        bonus_logits_indices_ptr,
        req_idx,
        num_draft_tokens,
        num_bonus_token,
        rejected,
        num_spec_steps,
    )
