# SPDX-License-Identifier: MIT
"""Lossless speculative sampling for `DSparkProposer` (ATOM_SPEC_LOSSLESS_SAMPLING).

`DSparkLossless.arm` returns None, and the greedy pair stays, unless the flag
is set, the flavor has a sampled entry point and the engine can run it
(`lossless_unsupported_reason`). Armed, it owns the draft's q store, scratch
and staged inputs, and is the drafter's ``lossless``: the runner binds, stages
and reads it directly.
"""

import logging

import torch

from atom.distributed.pcp_utils import get_pcp_world_size
from atom.kv_transfer.disaggregation import kv_config_has_producer
from atom.model_ops.dspark_markov_argmax import markov_weights_unsupported
from atom.model_ops.sampler import AITER_TOPK_TOPP_AVAILABLE, SAMPLER_EPS
from atom.model_ops.spec_sampling.dspark_markov_sample import (
    dspark_markov_sample_scratch,
)
from atom.model_ops.spec_sampling.ratio_verify import (
    RatioVerifyInputs,
    ratio_verify_scratch,
    warmup_ratio_verify,
)
from atom.model_ops.spec_sampling.sparse_q_store import Q_WIDTH, SparseQStore
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
    unsupported = markov_weights_unsupported(
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


def _sampling_buffer(rows: int, device, idle: int | None = None) -> CpuGpuBuffer:
    """An int32 member of the runner's `sampling` publication, its unpublished
    rows at ``idle``."""
    buf = CpuGpuBuffer(
        rows,
        dtype=torch.int32,
        device=device,
        pin_memory=torch.device(device).type != "cpu",
        publication_group="sampling",
    )
    if idle is not None:
        buf.np[:] = idle
        buf.gpu.fill_(idle)
    return buf


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
        if drafter.runner.rank == 0:
            store = lossless.q_store
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
                lossless.s_scratch.numel() * 4 / 2**20,
                lossless.draft_scratch.nbytes() / 2**20,
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
        self.q_store = SparseQStore.allocate(q_slots * t, min(Q_WIDTH, vocab), device)
        # The draft's own: the verify reads q from the store, not from here.
        self.s_scratch = torch.empty(
            (max_bs, vocab), dtype=torch.float32, device=device
        )
        self.draft_scratch = dspark_markov_sample_scratch(max_bs, vocab, device)
        # Members of the runner's `sampling` publication, uploaded by the H2D
        # `prepare_sample` issues; its `begin()` waits out the previous upload.
        # The Philox key, the first q row each batch row drafts into, and the
        # first q row each request's drafts were drawn into (-1: none). Rows
        # past the staged batch are never published, so they hold -1 already.
        self.seed = _sampling_buffer(1, device)
        self.q_row_base = _sampling_buffer(max_bs, device, idle=-1)
        self.verify_q_row_base = _sampling_buffer(max_bs, device, idle=-1)
        drafter.metadata_buffers.update(
            {
                "draft_seed": self.seed,
                "draft_q_row_base": self.q_row_base,
                "draft_verify_q_row_base": self.verify_q_row_base,
            }
        )
        self.staging = DraftQStaging(
            DraftQSlots(q_slots, t),
            seed=self.seed.np,
            q_row_base=self.q_row_base.np,
            verify_q_row_base=self.verify_q_row_base.np,
            greedy_threshold=SAMPLER_EPS,
        )
        # The draft model's `DraftSampling`, built by `bind_target`.
        self.sampling = None

    def bind_target(self, forward_vars, model) -> None:
        """Build the draft's `DraftSampling` around these buffers and the
        target's own temperatures, top-k and top-p (`ModelRunner.forward_vars`,
        which `prepare_sample` publishes every row of while this is armed), and
        bind ``model`` to it. Once, before any warmup or capture, so the
        recorded block reads them."""
        self.sampling = DraftSampling(
            q_store=self.q_store,
            s_scratch=self.s_scratch,
            q_row_base=self.q_row_base.gpu,
            seed=self.seed.gpu,
            scratch=self.draft_scratch,
            temperatures=forward_vars["temperatures"].gpu,
            top_k=forward_vars["top_ks"].gpu,
            top_p=forward_vars["top_ps"].gpu,
        )
        model.bind_draft_sampling(self.sampling)

    def stage(
        self,
        batch,
        bs: int,
        group,
        *,
        drafts_verifiable: bool,
        step: int,
    ) -> None:
        """Stage this step's draft inputs and both q row bases into the host
        buffers and claim their counts in ``group``, the `sampling`
        publication. The policy is `DraftQStaging.stage`'s."""
        rows = self.staging.stage(
            batch,
            bs,
            drafts_verifiable=drafts_verifiable,
            step=step,
        )
        counts, at = group.counts, group.indices
        counts[at["draft_seed"]] = rows.seed
        counts[at["draft_q_row_base"]] = rows.q_row_base
        counts[at["draft_verify_q_row_base"]] = rows.verify_q_row_base

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
            self.q_store,
            self.verify_q_row_base.gpu[:bs],
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
            self.s_scratch.shape[1],
            self.q_store.ids.shape[1],
            logits_dtype,
            self.verify_scratch,
        )

    def clear_destinations(self) -> None:
        """Point every draft row at no q row, on the host and the device (on
        the current stream, the one the warmup that follows runs on)."""
        self.staging.clear_destinations()
        self.q_row_base.gpu.fill_(-1)
