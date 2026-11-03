# SPDX-License-Identifier: MIT
"""Lossless speculative sampling for `DSparkProposer` (ATOM_SPEC_LOSSLESS_SAMPLING).

`DSparkLossless.arm` returns None, and the greedy pair stays, unless the flag
is set, the flavor has a sampled entry point and the engine can run it
(`lossless_unsupported_reason`). Armed, it owns the draft's q store, scratch
and staged inputs, and the proposer's lossless hooks delegate to it.
"""

import logging
from typing import NamedTuple

import torch

from atom.distributed.pcp_utils import get_pcp_world_size
from atom.kv_transfer.disaggregation import kv_config_has_producer
from atom.model_ops.dspark_markov_draw import (
    dspark_markov_sample_scratch,
    dspark_markov_sample_unsupported,
)
from atom.model_ops.ratio_verify import (
    RatioVerifyInputs,
    ratio_verify_scratch,
    warmup_ratio_verify,
)
from atom.model_ops.sampler import AITER_TOPK_TOPP_AVAILABLE, SAMPLER_EPS
from atom.model_ops.sparse_q_store import Q_WIDTH, RatioDraftQ, SparseQStore
from atom.models.dspark_draft import DraftSampling
from atom.spec_decode.draft_q_slots import DraftQSlots
from atom.spec_decode.draft_q_staging import DraftQStaging
from atom.utils import CpuGpuBuffer, envs

logger = logging.getLogger("atom")

#: Persistent q slots, as a multiple of `max_num_seqs`; one slot holds one
#: request's block of T rows. Slots are reclaimed by expiry (`DraftQSlots`), and
#: the headroom above the live set is what makes the expired entry usually a
#: finished request rather than a running one.
_Q_SLOT_OVERSUBSCRIBE = 2


class _LosslessInput(NamedTuple):
    """One staged input of the sampled draft or the ratio verify."""

    #: The `DraftQStaging` keyword its numpy view is passed as, the key
    #: `DraftQStaging.stage` reports its row count under, and, if `drafted`,
    #: the `DraftSampling` field its device tensor binds to.
    name: str
    #: One entry per batch row (`max_num_seqs`), or a single scalar.
    per_row: bool
    dtype: torch.dtype
    #: What rows past the staged batch hold; None leaves the zeros.
    idle: float | int | None
    #: Whether the draft reads it; False when only the verify does.
    drafted: bool = True

    @property
    def key(self) -> str:
        """The `metadata_buffers` key, a member of the `sampling` publication;
        prefixed because it shares `forward_vars` with the runner's."""
        return "draft_" + self.name


#: Every per-step input of lossless speculative sampling, in publication order.
#: Pad rows idle at temperature 1.0 rather than a 0 the kernel would divide by,
#: no filter, and row base -1, so a pad row writes no q.
_LOSSLESS_INPUTS = (
    _LosslessInput("temperatures", True, torch.float32, 1.0),
    _LosslessInput("greedy_mask", True, torch.int32, 1),
    _LosslessInput("seed", False, torch.int32, None),
    # The request's own top-k / top-p, for the draft to sample within.
    _LosslessInput("top_k", True, torch.int32, 0),
    _LosslessInput("top_p", True, torch.float32, 1.0),
    # First q row each batch row drafts into; -1 writes nothing.
    _LosslessInput("q_row_base", True, torch.int32, -1),
    # First q row each request's drafts were drawn into; -1 means none.
    _LosslessInput("verify_q_row_base", True, torch.int32, -1, drafted=False),
)


def lossless_unsupported_reason(drafter) -> str | None:
    """Why this engine cannot arm the feature it was asked for, or None.

    The text completes "ATOM_SPEC_LOSSLESS_SAMPLING ..."; the caller warns
    with it and keeps the greedy pair.
    """
    config = drafter.config
    if drafter.draft_tokens_per_seq != drafter.mtp_k:
        # The store holds `draft_tokens_per_seq` rows per slot but the verify
        # walks `mtp_k`: a narrower draft would read into the next slot.
        return (
            "needs the draft block to be as wide as the verify "
            f"(draft_tokens_per_seq={drafter.draft_tokens_per_seq}, "
            f"mtp_k={drafter.mtp_k}): the verify would read q rows past each "
            "request's slot."
        )
    if not AITER_TOPK_TOPP_AVAILABLE:
        # The verify filters as aiter does; `Sampler`'s native fallback would
        # draw the bonus from a different p.
        return (
            "needs aiter's top-k/top-p sampling ops, which this build "
            "lacks: the ratio verify filters as they do, and Sampler's "
            "native fallback filters differently."
        )
    head = drafter.model.markov_head
    unsupported = dspark_markov_sample_unsupported(
        head.markov_w1.weight, head.markov_w2.weight
    )
    if unsupported is not None:
        # Checked once here so the sampled op never has to fail mid-serve.
        return f"cannot run the fused sampled draft ({unsupported})."
    if kv_config_has_producer(config.kv_transfer_config) or (
        config.enable_rapidserve and not config.disagg_is_decode
    ):
        # A P/D producer drafts only for handoff and never verifies; the
        # consumer has no q for a handed-off request.
        return (
            "is not used on a P/D producer (prefill) engine: it drafts for "
            "handoff and verifies nothing. The decode engine still arms."
        )
    if config.pipeline_parallel_size > 1:
        # PP rings the `sampling` publication while the block head binds one
        # address, and a PP engine (`spec_decode_local = pp_size == 1`) never
        # reaches the ratio accept rule anyway.
        return (
            "is not implemented under pipeline parallelism "
            f"(pp={config.pipeline_parallel_size}): such an engine "
            "drafts for handoff and verifies nothing, so the ratio accept "
            "rule it pairs with is never reached."
        )
    if get_pcp_world_size() > 1:
        # PCP broadcasts the draft ids across ranks but not their q, so
        # rank > 0 would test q(x) for an x it did not draw.
        return (
            f"is not implemented under PCP (pcp={get_pcp_world_size()}): "
            "the draft ids are broadcast across PCP ranks but the "
            "distribution they were drawn from is not."
        )
    return None


class DSparkLossless:
    """An armed DSpark drafter's lossless state: the draft's `DraftSampling`,
    its host `DraftQStaging`, and the verify's scratch."""

    @classmethod
    def arm(cls, drafter) -> "DSparkLossless | None":
        """Arm lossless sampling for ``drafter`` and bind its draft model, or
        return None and keep the greedy pair. See `ATOM_SPEC_LOSSLESS_SAMPLING`
        in `docs/model_ops_guide.md`."""
        if not envs.ATOM_SPEC_LOSSLESS_SAMPLING:
            return None
        if not drafter.model.supports_lossless_draft:
            # Not ported to this flavor yet, which is not an operator error.
            return None
        reason = lossless_unsupported_reason(drafter)
        if reason is not None:
            logger.warning(
                "ATOM_SPEC_LOSSLESS_SAMPLING %s Falling back to greedy proposal "
                "with exact-match acceptance.",
                reason,
            )
            return None
        if envs.ATOM_ENABLE_RELAXED_MTP:
            # Relaxed top-N accepts tokens the target would not have drawn.
            raise ValueError(
                "ATOM_SPEC_LOSSLESS_SAMPLING and ATOM_ENABLE_RELAXED_MTP are "
                "mutually exclusive: relaxed top-N acceptance is approximate "
                "by construction and the probability-ratio rule is exact, so "
                "there is no defensible way to apply both. Turn one off."
            )
        lossless = cls(drafter)
        drafter.model.bind_draft_sampling(lossless.sampling)
        if drafter.runner.rank == 0:
            store = lossless.sampling.q_store
            q_rows, q_width = store.ids.shape
            t = drafter.draft_tokens_per_seq
            logger.info(
                "Lossless speculative sampling armed: draft q store %d slots x "
                "[%d, top %d of %d] (%.1f MiB) + %.1f MiB draft scores + %.1f "
                "MiB draft scratch + %.1f MiB verify scratch",
                q_rows // t,
                t,
                q_width,
                int(drafter.model.vocab_size),
                store.nbytes() / 2**20,
                lossless.sampling.s_scratch.numel() * 4 / 2**20,
                lossless.sampling.scratch.nbytes() / 2**20,
                lossless.verify_scratch.nbytes() / 2**20,
            )
        return lossless

    def __init__(self, drafter):
        """Allocate the draft's q store, scratch and staged inputs, and the
        verify's scratch, once, before the KV cache is sized, so the budget
        and the profiling run see them and no step allocates."""
        max_bs = drafter.config.max_num_seqs
        t = drafter.draft_tokens_per_seq
        vocab = int(drafter.model.vocab_size)
        device = drafter.device
        self.mtp_k = drafter.mtp_k
        self.verify_scratch = ratio_verify_scratch(max_bs, t, vocab, device)
        # Set by `warmup_verify`; checked against every verify's logits.
        self._verify_logits_dtype = None
        q_slots = _Q_SLOT_OVERSUBSCRIBE * max_bs
        # One row block per live request, not per batch slot: the verify reads
        # it a step later, after the batch may have been reordered.
        q_store = SparseQStore.allocate(q_slots * t, min(Q_WIDTH, vocab), device)
        # The verify borrows the draft's score scratch to densify a rejecting
        # request's q; the draft never runs while a verify does
        # (`ModelRunner.postprocess`).
        s_scratch = torch.empty((max_bs, vocab), dtype=torch.float32, device=device)
        draft_scratch = dspark_markov_sample_scratch(max_bs, vocab, device)
        # Members of the runner's `sampling` publication, uploaded by the H2D
        # `prepare_sample` issues; its `begin()` waits out the previous upload.
        pin = torch.device(device).type != "cpu"
        buffers = {}
        for spec in _LOSSLESS_INPUTS:
            buf = CpuGpuBuffer(
                max_bs if spec.per_row else 1,
                dtype=spec.dtype,
                device=device,
                pin_memory=pin,
                publication_group="sampling",
            )
            # Rows past the staged batch are never published, so they must
            # already hold a harmless value.
            if spec.idle is not None:
                buf.np[:] = spec.idle
                buf.gpu.fill_(spec.idle)
            buffers[spec.name] = buf
        drafter.metadata_buffers.update(
            {s.key: buffers[s.name] for s in _LOSSLESS_INPUTS}
        )
        self._verify_q_row_base = buffers["verify_q_row_base"].gpu
        self.staging = DraftQStaging(
            DraftQSlots(q_slots, t, log=drafter.runner.rank == 0),
            greedy_threshold=SAMPLER_EPS,
            **{name: buf.np for name, buf in buffers.items()},
        )
        self.sampling = DraftSampling(
            q_store=q_store,
            s_scratch=s_scratch,
            scratch=draft_scratch,
            **{s.name: buffers[s.name].gpu for s in _LOSSLESS_INPUTS if s.drafted},
        )

    def stage(
        self,
        batch,
        bs: int,
        group,
        *,
        drafts_verifiable: bool,
        verifies: bool,
        step: int,
    ) -> None:
        """Stage this step's draft inputs and both q row bases into the host
        buffers and claim their counts in ``group``, the `sampling`
        publication. The policy is `DraftQStaging.stage`'s."""
        rows = self.staging.stage(
            batch,
            bs,
            drafts_verifiable=drafts_verifiable,
            verifies=verifies,
            step=step,
        )
        for spec in _LOSSLESS_INPUTS:
            group.counts[group.indices[spec.key]] = rows[spec.name]

    def verify_inputs(
        self,
        bs: int,
        logits_dtype: torch.dtype,
        temperatures: torch.Tensor,
        top_ks: int | torch.Tensor | None,
        top_ps: float | torch.Tensor | None,
    ) -> RatioVerifyInputs:
        """The verify step's `RatioVerifyInputs`, around the batch's own
        temperatures and `Sampler` filters."""
        warmed = self._verify_logits_dtype
        if warmed is not None and logits_dtype != warmed:
            # The kernels were compiled for another logits dtype; this verify
            # would JIT them mid-serving while the TP/DP peers wait.
            raise RuntimeError(
                f"ratio verify was warmed up for {warmed} "
                f"logits but the LM head emits {logits_dtype}"
            )
        return RatioVerifyInputs(
            RatioDraftQ(self.sampling.q_store, self.sampling.s_scratch),
            self._verify_q_row_base[:bs],
            temperatures,
            top_ks,
            top_ps,
            self.verify_scratch,
        )

    def warmup_verify(self, logits_dtype: torch.dtype) -> None:
        """Compile the ratio verify at the widths serving gives it: the store's,
        the vocab's and ``mtp_k`` drafts (`warmup_ratio_verify`)."""
        self._verify_logits_dtype = logits_dtype
        warmup_ratio_verify(
            self.mtp_k,
            self.sampling.s_scratch.shape[1],
            self.sampling.q_store.ids.shape[1],
            logits_dtype,
            self.verify_scratch,
        )

    def clear_destinations(self) -> None:
        """Point every draft row at no q row, on the host and the device (on
        the current stream, the one the warmup that follows runs on)."""
        self.staging.clear_destinations()
        self.sampling.q_row_base.fill_(-1)
