# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which Kimi-K3 decode steps the model takes over from the engine.

``install_mono_decode`` is Kimi-K3's installer (``models.registry``). With
``ATOM_KIMI_MONO_STEP_TAIL`` off it is the mono installer, unchanged. On, a
deployment the tail covers gets ``StepTailDecodeModel``: the mono routing
wrapper that is also a ``StepOverrideModel``. ``ModelRunner.run_model`` offers
it each graph decode step; on one it serves it replays the engine's recorded
forward (the model's mono launches, as today) and launches the head + verify
kernel on the hidden rows. The engine then computes no logits, takes the ids
and the verdict from ``verify_outputs``, skips the TP id broadcast (the ids are
identical on every rank) and waits for the output stream only. Everything
else is the production path: any other step, the draft's context KV and its
block backbone.

The draft block's epilogue is taken over separately (``_wrap_draft_head``):
where the tail runner has an agreed draft launch for the block's shape, the
draft LM head and the greedy Markov sampling run in it (no logits all-gather),
on any step that drafts, served or not: the ids are production's either way,
and the choice is the same on every rank and at every replay of a recording.

Coupling: the hook hands a model no handle on the engine's recorded forward,
so ``step_override`` reaches it through the drafter's ``runner``
(``req_graph.has`` / ``replay`` keyed by ``request_bucket``, and
``forward_vars["outputs"]``, the rows the recording leaves). A change to how
``ModelRunner`` keys or stores its decode graphs has to be mirrored in
``step_refusal`` and ``step_override``. (The alternative, graphs of the
wrapper's own, would double the recordings' private pools.)

A cross-rank wait that gives up fails the step (``StepTailRunner.check_waits``,
a RuntimeError): the launch went on with a stale peer line. Nothing is synced
for it. With deferred output the engine reads step N's ids on the host in step
N + 1's postprocess, after that step's ``verify_outputs``: the failure is
raised at step N + 1's offer if the count has landed and in its
``verify_outputs`` at the latest, so a stale step's ids never reach the host
(on the device they have by then fed the draft block and the next forward).
A step the tail leaves to the engine, and any forward of the engine's own,
settles the served steps' count first (``check_waits``). Without deferred
output the step's own ``verify_outputs`` raises. Not covered: an engine path
that reads a served step's ids without calling the model or offering it the
step (a decode step replayed from a token graph); ``check_waits`` is there
for the engine to call. Which mode the engine is in is read off the
drafter's ``runner.tokenID_processor.is_deferred_out``.

Every input a step is routed on is TP-uniform (the batch, the step's
metadata): the kernel waits on every peer, so ranks that disagreed would hang.
"""

import logging

import numpy as np
import torch

from atom.models.kimi_k3.mono import dispatch as mono_dispatch
from atom.models.kimi_k3.mono.dispatch import MonoDecodeModel
from atom.models.kimi_k3.step_tail import layout as L
from atom.mono.runtime.route import LazyRunner
from atom.mono.runtime.step_override import StepOverrideModel
from atom.utils import envs
from atom.utils.forward_context import get_forward_context, request_bucket

logger = logging.getLogger("atom")


def _has_constraints(batch) -> bool:
    from atom.model_engine.token_constraints import has_constraints

    return has_constraints(batch)


def _gfx() -> str:
    try:
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx()
    except ImportError:
        return ""


def _relaxed_acceptance() -> bool:
    return bool(envs.ATOM_ENABLE_RELAXED_MTP)


def _outputs_deferred(engine) -> bool:
    """Whether the engine reads a step's ids on the host a step late. An
    engine that does not say reads them at once, as far as the tail knows."""
    processor = getattr(engine, "tokenID_processor", None)
    return bool(getattr(processor, "is_deferred_out", False))


def tail_refusal(atom_config, drafter) -> str | None:
    """Why this deployment keeps the production step tail, or None: the tail
    is written for TP8 DSpark with greedy verification on gfx1250."""
    spec = getattr(atom_config, "speculative_config", None)
    checks = (
        (_gfx() == "gfx1250", "not gfx1250"),
        (
            atom_config.tensor_parallel_size == L.TP,
            f"TP {atom_config.tensor_parallel_size} (the tail is TP{L.TP})",
        ),
        (
            drafter is not None and spec is not None and spec.method == "dspark",
            "not DSpark",
        ),
        (
            spec is not None
            and 2 <= int(spec.num_speculative_tokens or 0) + 1 <= L.MAX_QLEN,
            f"draft width past {L.MAX_DRAFTS}",
        ),
        (
            getattr(drafter, "lossless", None) is None,
            "lossless speculative sampling (the tail verifies greedy steps)",
        ),
        (
            getattr(drafter, "verify_scheduler", None) is None,
            "confidence-scheduled verify",
        ),
        (
            getattr(spec, "synthetic_acceptance_rates", None) is None,
            "synthetic acceptance",
        ),
        (not _relaxed_acceptance(), "relaxed acceptance"),
        (
            getattr(atom_config, "prefill_context_parallel_size", 1) in (None, 1),
            "prefill context parallelism",
        ),
    )
    for ok, why in checks:
        if not ok:
            return why
    return None


class StepTailDecodeModel(MonoDecodeModel, StepOverrideModel):
    """Kimi-K3's mono routing wrapper, serving greedy DSpark verify steps
    whole (``step_override``)."""

    def __init__(self, model, drafter=None, metadata_builder=None) -> None:
        super().__init__(model, drafter, metadata_builder)

        def create():
            # imported on first use: the runner needs aiter and FlyDSL
            from atom.models.kimi_k3.step_tail.runner import StepTailRunner

            return StepTailRunner(model, drafter)

        self._tail = LazyRunner(create, "Kimi-K3 step tail")
        self._tail_drafter = drafter
        self._tail_served = None  # this step's runner when the tail served it
        self._tail_hidden = None
        self._tail_declines: set[str] = set()
        if drafter is not None:
            self._wrap_drafter(drafter)
            self._wrap_draft_head(drafter)

    def _wrap_drafter(self, drafter) -> None:
        """A served step returns no hidden states to the engine; the draft's
        context KV (production's) still reads the rows the forward left."""
        compute_draft_kv = drafter.compute_draft_kv

        def served_compute_draft_kv(positions, hidden_states, next_token_ids):
            if hidden_states is None and self._tail_served is not None:
                hidden_states = self._tail_hidden
            return compute_draft_kv(positions, hidden_states, next_token_ids)

        drafter.compute_draft_kv = served_compute_draft_kv

    def _wrap_draft_head(self, drafter) -> None:
        """The draft model's ``head_and_sample`` (the block pass's epilogue,
        recorded with it): the draft launch where ``draft_refusal`` allows,
        else the model's own. A recording bakes the side taken at its warmup,
        which is the same on every rank (every input of the choice is)."""
        model = getattr(drafter, "model", None)
        head_and_sample = getattr(model, "head_and_sample", None)
        if head_and_sample is None:
            return

        def tail_head_and_sample(out, anchor_ids, num_draft, *, greedy=False):
            why = self.draft_refusal(model, out, anchor_ids, num_draft, greedy)
            if why is not None:
                self._tail_declined(f"draft block: {why}")
                return head_and_sample(out, anchor_ids, num_draft, greedy=greedy)
            ids = self._tail.runner.launch_draft(
                anchor_ids.shape[0], num_draft, out, anchor_ids
            )
            return ids, None

        model.head_and_sample = tail_head_and_sample

    def draft_refusal(self, model, out, anchor_ids, num_draft, greedy) -> str | None:
        """Why this draft block keeps the model's own head and sampler, or
        None. Nothing is launched; a first sight of a shape builds its launch
        (collective, outside a capture)."""
        runner = self._tail.runner
        if runner is None:
            return "no step-tail runner yet"
        if not greedy and getattr(model, "draft_sampling", None) is not None:
            return "sampled drafting"
        if getattr(model, "confidence_head", None) is not None:
            return "a confidence head reads the Markov rows"
        requests = anchor_ids.shape[0]
        if (
            not isinstance(out, torch.Tensor)
            or tuple(out.shape) != (requests * num_draft, L.HIDDEN)
            or out.dtype != torch.bfloat16
            or not out.is_contiguous()
        ):
            return "draft rows the kernel cannot read"
        if (
            anchor_ids.dim() != 1
            or anchor_ids.dtype != torch.int32
            or not anchor_ids.is_contiguous()
        ):
            return f"anchor ids {anchor_ids.dtype}"
        if not runner.prepare_draft(requests, num_draft):
            return f"no draft launch for {requests} requests of {num_draft} drafts"
        return None

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        # a forward the engine ran itself (a warmup, a capture, a prefill, an
        # eager decode): not a step the tail served
        self._tail_served = self._tail_hidden = None
        self.check_waits()
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def _mono_forward(self, input_ids, positions, qlen):
        hidden = super()._mono_forward(input_ids, positions, qlen)
        if qlen >= 2:
            # build the width's tail beside its mono kernels, on the warmup
            # forward before its graph is recorded (nothing is built inside a
            # capture); collective, like the mono build
            self._tail.ready(input_ids.numel(), qlen)
        return hidden

    def _tail_declined(self, why: str) -> None:
        if why not in self._tail_declines:
            self._tail_declines.add(why)
            logger.info("Kimi-K3 step tail declined a step: %s", why)

    def check_waits(self) -> None:
        """Fail if a cross-rank wait of a served step or a draft block gave
        up, waiting for a count still queued: before a step the tail does
        not serve (the engine is about to read the served steps' ids)."""
        runner = self._tail.runner
        if runner is None:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        runner.check_waits(wait=True)

    def _engine(self):
        return getattr(self._tail_drafter, "runner", None)

    def step_refusal(self, input_ids, running_bs: int, batch) -> str | None:
        """Why this decode step keeps the production tail, or None. Host
        checks only: nothing is launched or staged."""
        if batch is None or batch.any_logprobs or _has_constraints(batch):
            return "logprobs or structured output"
        fc = get_forward_context()
        spec = getattr(fc, "spec_decode_metadata", None)
        if spec is None:
            return "no drafts to verify"
        qlen = int(spec.num_spec_steps) + 1
        rows = input_ids.numel()
        if rows != running_bs * qlen or L.step_refusal(rows, qlen):
            return f"{rows} rows for {running_bs} requests of {qlen}"
        if (np.asarray(spec.num_draft_tokens_np) != qlen - 1).any():
            return "ragged verify"
        bs = batch.total_seqs_num
        if np.asarray(batch.temperatures[:bs]).any():
            return "sampled requests (the tail verifies greedy steps)"
        ids = spec.input_ids
        if ids.dtype != torch.int32 or ids.numel() < rows or not ids.is_contiguous():
            return f"input ids {ids.dtype} [{ids.numel()}]"
        engine = self._engine()
        graph = getattr(engine, "req_graph", None)
        key = request_bucket(running_bs, fc.attn_metadata.max_seqlen_q)
        if graph is None or not graph.has(key):
            return "no recorded forward"
        hidden = engine.forward_vars["outputs"]
        if (
            hidden.dim() != 2
            or hidden.shape[0] < rows
            or hidden.shape[1] != L.HIDDEN
            or hidden.dtype != torch.bfloat16
            or not hidden.is_contiguous()
        ):
            return f"hidden rows {tuple(hidden.shape)} {hidden.dtype}"
        if not self._tail.ready(rows, qlen):
            return f"no step-tail build for {rows} rows"
        return None

    def step_override(self, input_ids, running_bs: int, batch):
        """``ModelRunner.run_model``, each graph decode step: serve it (the
        recorded forward, then the head + verify launch) and return the
        runner, or None for the production path."""
        self._tail_served = self._tail_hidden = None
        why = self.step_refusal(input_ids, running_bs, batch)
        if why is not None:
            self._tail_declined(why)
            self.check_waits()
            return None
        fc = get_forward_context()
        engine = self._engine()
        rows = input_ids.numel()
        runner = self._tail.runner
        # a stale step already known: nothing more is replayed or launched
        runner.check_waits(wait=False)
        runner.outputs_deferred = _outputs_deferred(engine)
        engine.req_graph.replay(
            key=request_bucket(running_bs, fc.attn_metadata.max_seqlen_q)
        )
        hidden = engine.forward_vars["outputs"][:rows]
        runner.launch_verify(
            rows, rows // running_bs, hidden, fc.spec_decode_metadata.input_ids
        )
        self._tail_served, self._tail_hidden = runner, hidden
        return runner


def install_mono_decode(model, atom_config, drafter, *, metadata_builder=None):
    """Kimi-K3's mono decode installer, with the step tail on top where
    ``ATOM_KIMI_MONO_STEP_TAIL`` asks for it and the deployment is covered."""
    if envs.ATOM_KIMI_MONO_STEP_LEAN or envs.ATOM_KIMI_MONO_EMBED_LOCAL:
        # (the step-gap hooks on the drafter and the embedding: whatever mono
        # and the tail do)
        from atom.models.kimi_k3.step_gap.install import install as install_step_gap

        install_step_gap(model, atom_config, drafter)
    installed = mono_dispatch.install_mono_decode(
        model, atom_config, drafter, metadata_builder=metadata_builder
    )
    if not envs.ATOM_KIMI_MONO_STEP_TAIL:
        return installed
    if not isinstance(installed, MonoDecodeModel):
        why = "Kimi-K3 mono decode is off"
    else:
        why = tail_refusal(atom_config, drafter)
    if why is not None:
        logger.info("Kimi-K3 step tail off: %s", why)
        return installed
    logger.info("Kimi-K3 step tail installed")
    return StepTailDecodeModel(model, drafter, metadata_builder)
