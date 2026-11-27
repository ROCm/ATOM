# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Which V4.1 forwards take the mono path.

``install_mono_decode`` wraps the loaded model, the way the TBO
``UBatchWrapper`` does, and every other attribute passes through to it. The
runtime model's own forward dispatches to its compiled graphs (``runtime``);
mono routing sits outside them, before any compiled code runs.

The configuration is checked once (``config.config_refusal``). A target step
is then routed only when it is a decode (a DSpark verify of any length a
request, or a token a request) of up to ``MAX_ROWS`` rows (a graph's padding
rows included), however many requests they are and however they split them.
The DSpark draft model is wrapped the same way (``MonoDraftModel``): its block backbone of
up to ``MAX_REQUESTS`` requests runs on ``draft_runner``, after a prefill as
after a decode. Every per-step input read is TP-uniform: the kernels wait on
every peer rank, so ranks that disagreed would wait on each other for ever (a
per-rank flag such as ``is_dummy_run`` must never gate this).
"""

import logging

import torch
from torch import nn

from atom.model_ops.verify_head import VerifyHead
from atom.models.deepseek_v41.mono import hi_gate
from atom.models.deepseek_v41.mono.config import (
    MAX_MOE_ROWS,
    MAX_REQUESTS,
    MAX_ROWS,
    config_refusal,
    indexer_enabled,
)
from atom.models.deepseek_v41.mono.draft_runner import DraftMonoRunner
from atom.models.deepseek_v41.mono.runner import V41MonoDecodeRunner
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.deployment import dp_attention_ranks
from atom.mono.runtime.route import LazyRunner
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def step_supported(rows: int, embeds: bool = False, dp_ranks: int = 0) -> bool:
    """Is this forward a decode step of ``rows`` rows, up to ``MAX_ROWS``
    (``embeds``: its input embeddings given, not its ids: no)? Its width,
    not the scheduled request count: a captured graph's width is the one
    its replays run, while the scheduled count differs from replay to replay.

    Under DP attention over ``dp_ranks`` ranks every rank runs its own requests
    (an idle one a dummy step), and the MoE spans them all: a step is mono on
    every rank or on none, of up to ``MAX_MOE_ROWS / dp_ranks`` rows a rank
    (the MoE's widest step their sum) and ``MAX_ROWS`` at most. It is only
    when no rank prefills
    (``running_tokens_are_unified``, agreed across DP); then every rank's width
    is the agreed ``running_tokens``, and an idle rank's step, requests or
    none, is a decode as well."""
    max_rows = min(MAX_ROWS, MAX_MOE_ROWS // dp_ranks) if dp_ranks else MAX_ROWS
    if embeds or not 0 < rows <= max_rows:
        return False
    fwd = get_forward_context()
    if fwd.context is None or fwd.context.is_prefill or fwd.ubatch_slices is not None:
        return False
    if dp_ranks and not fwd.context.running_tokens_are_unified:
        return False
    md = fwd.attn_metadata
    step = getattr(md, "step", None)
    if step is None or getattr(md, "image_mask", None) is not None:
        return False
    return step.decode and step.running_tokens == rows


class MonoVerifyHead(VerifyHead):
    """The mono LM head launch as the verify head (``atom.mono.lm_head``)
    on a step the mono kernels serve (``serves_step(rows)``), by
    ``runner()``'s launch: temperature and top-p rows drawn in it, every row's id
    picked over every rank's vocab shard alike on every rank."""

    samples = ids_agree = True

    def __init__(self, serves_step, runner) -> None:
        self.serves_step, self.runner = serves_step, runner

    def draws_top_p(self, rows: int) -> bool:
        return self.serves_step(rows)

    def sample_token_ids(self, hidden: torch.Tensor):
        if not self.serves_step(hidden.shape[0]):
            return None
        return self.runner().sample_token_ids(hidden)


class MonoDecodeModel(nn.Module):
    """The loaded V4.1 runtime model, with its supported steps routed to mono,
    and its verify head the mono LM head launch (``MonoVerifyHead``)."""

    def __init__(
        self,
        model: nn.Module,
        drafter,
        max_model_len: int,
        index_fp4: bool,
        dp_ranks: int,
        outputs,
        sampling,
        draft_context: bool = True,
        *,
        tp_size: int = 1,
        standalone_indexer: bool = True,
        indexer_workspace=None,
    ) -> None:
        super().__init__()
        self.model = model
        self._drafter = drafter
        # DP attention's ranks, 0 without it (``step_supported``)
        self._dp_ranks = dp_ranks
        self._tp_size = tp_size
        self._index_fp4 = index_fp4
        self._standalone_indexer = standalone_indexer
        # created on its first eligible step (``LazyRunner``); ``sampling``
        # None and ``draft_context`` False: an owner that neither verifies
        # through the mono LM head nor writes the draft's context through
        # ``write_context`` (``V41MonoDecodeRunner``)
        self._mono = LazyRunner(
            lambda: V41MonoDecodeRunner(
                model,
                drafter,
                max_model_len,
                index_fp4,
                outputs,
                sampling,
                draft_context,
                standalone_indexer=standalone_indexer,
                indexer_workspace=indexer_workspace,
            ),
            "V4.1 mono decode",
        )
        self.verify_head = MonoVerifyHead(self.serves_step, lambda: self._mono.runner)

    def serves_step(self, rows: int, embeds: bool = False) -> bool:
        """Whether the mono kernels serve this forward's ``rows``-row step:
        the decision ``forward`` takes, which the step's head and the draft's
        context write take again after it (a replayed graph does not call
        ``forward``)."""
        return (
            step_supported(rows, embeds, self._dp_ranks)
            and self.layer_rows_supported(rows)
            and self._mono.ready(rows)
        )

    def forward(self, input_ids, positions, inputs_embeds=None):
        if self.serves_step(input_ids.numel(), inputs_embeds is not None):
            return self._mono.runner.forward(input_ids, positions)
        if inputs_embeds is None:
            return self.model(input_ids, positions)
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def write_context(self, taps, positions) -> bool:
        """The draft's context KV of the target step that just ran, in one
        launch (``V41MonoDecodeRunner.write_context``), True if it ran: only
        when the mono kernels served that step (``serves_step``), whose step
        begin zeroed the launch's mailboxes."""
        fwd = get_forward_context()
        if fwd.context is None or fwd.context.is_dummy_run:
            return False
        metadata = fwd.attn_metadata
        rows = metadata.step.running_tokens
        if not self.serves_step(rows):
            return False
        return self._mono.runner.write_context(
            [tap[:rows] for tap in taps], positions[:rows], metadata.cache,
            metadata.step,
        )  # fmt: skip

    def uses_layer_mono(self, input_ids, inputs_embeds=None):
        return self.serves_step(input_ids.numel(), inputs_embeds is not None)

    def layer_rows_supported(self, rows):
        # At TP2, large FP4 decode batches are faster with native layers.
        # The outer wrapper still runs their standalone Mono indexer.
        return not (
            self._tp_size == 2
            and not self._dp_ranks
            and rows >= 40
            and self._standalone_indexer
            and indexer_enabled(self._index_fp4, self._tp_size)
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class MonoDraftModel(nn.Module):
    """The loaded DSpark draft model, with the block backbone of up to
    ``MAX_REQUESTS`` requests at the full block width routed to mono (every
    batch size up to it is captured, so a replay's requests are all real). The
    call is inside ``DraftGraph``'s recording, so the runner binds on the first
    such call outside one (the graph's warmup).

    Under DP attention over ``dp_ranks`` ranks the requests are the agreed
    padded batch (``DSparkProposer`` stages the anchors at ``running_bs``,
    an idle rank's as padding), the same on every rank, of up to
    ``MAX_REQUESTS`` a rank while the MoE's rows over every rank stay within
    ``MAX_MOE_ROWS``."""

    def __init__(self, draft: nn.Module, dp_ranks: int = 0) -> None:
        super().__init__()
        self.model = draft
        self._max_requests = (
            min(MAX_REQUESTS, MAX_MOE_ROWS // (dp_ranks * draft.block_size))
            if dp_ranks
            else MAX_REQUESTS
        )
        self._mono = LazyRunner(lambda: DraftMonoRunner(draft), "V4.1 mono draft")
        # the last block's backbone ran on the mono kernels: its head may too
        # (a pass's head follows its backbone, recorded or not)
        self._served = False
        # the target's mono context write (``MonoDecodeModel.write_context``),
        # set at install; None: the draft's own
        self.context_writer = None

    def write_context_kv(self, taps, positions, slot_mapping=None, **kw):
        """The target step's context KV into the draft's windows: on the
        target's mono launch where it serves, else the draft's own."""
        writer = self.context_writer
        if writer is not None and not kw and writer(taps, positions):
            return
        self.model.write_context_kv(taps, positions, slot_mapping, **kw)

    def block_backbone(self, input_ids, positions, num_draft):
        width = self.model.block_size if num_draft is None else num_draft
        requests = input_ids.numel()
        self._served = (
            self.model.draft_kv_dtype != "fp8"
            and 0 < requests <= self._max_requests
            and width == self.model.block_size
            and self._mono.ready(requests * width)
        )
        if self._served:
            return self._mono.runner.backbone(input_ids, positions)
        return self.model.block_backbone(input_ids, positions, num_draft)

    def head_and_sample(self, out, anchor_ids, num_draft, *, greedy=False):
        """The block's ids and confidences: after a mono backbone (``out`` its
        last stage's state) the tail, the LM head, the greedy Markov sampler
        and the confidence head in one launch (``DraftMonoRunner.head``; a
        confidence head it does not compute, on its collapsed and ``W1``
        rows); else, and where the block samples (``draft_sampling`` without
        ``greedy``), the model's own."""
        if self._served:
            runner = self._mono.runner
            samples = self.model.draft_sampling is not None and not greedy
            got = None if samples else runner.head(anchor_ids)
            if got is not None:
                ids, embeds, hidden, conf = got
                if conf is None:
                    conf = self.model.mtp[-1].confidence_head(hidden, embeds)
                return ids, conf
            out = runner.normed_rows(out, anchor_ids.numel())
        return self.model.head_and_sample(out, anchor_ids, num_draft, greedy=greedy)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _install_layer_decode(
    model: nn.Module, atom_config, drafter, *, metadata_builder
) -> nn.Module:
    """``model`` wrapped for mono decode (and ``drafter``'s DSpark draft model
    for its backbone), or ``model`` itself when this deployment is one mono
    does not serve."""
    from atom.models.deepseek_v41.mono import gfx1250

    if gfx1250.enabled():
        from atom.models.deepseek_v41.mono.gfx1250.dispatch import (
            install_single_step,
        )

        return install_single_step(model, atom_config, drafter)
    why = config_refusal(atom_config)
    if why is not None:
        if hi_gate.mode() == "1":
            raise RuntimeError(
                f"ATOM_DSV41_MONO_HI=1, but V4.1 mono decode is off: {why}"
            )
        logger.info("V4.1 mono decode off: %s", why)
        return model
    dp_ranks = dp_attention_ranks(atom_config)
    # The decoder layers on the high-interactivity monokernels where they serve
    # this deployment (``hi_gate``: ATOM_DSV41_MONO_HI, "auto" by default); any
    # other takes ATOM's mono decode below, unchanged.
    # A mismatch only the loaded model shows (``MonoUnsupported``, agreed over
    # the TP group before anything is built: ``install_hi``) falls back as the
    # gate's refusals do, on every rank.
    if hi_gate.decide(atom_config, drafter, metadata_builder, dp_ranks):
        from atom.models.deepseek_v41.mono.hi_dispatch import install_hi

        try:
            return install_hi(model, atom_config, drafter, metadata_builder, dp_ranks)
        except MonoUnsupported as why:
            if hi_gate.mode() == "1":
                raise
            hi_gate.log_once(logging.INFO, "V4.1 mono hi off: %s", why)
    draft = None
    if drafter is not None and hasattr(drafter.model, "block_backbone"):
        draft = drafter.model = MonoDraftModel(drafter.model, dp_ranks)
    standalone = indexer_enabled(
        atom_config.index_cache_dtype == "fp4",
        atom_config.tensor_parallel_size,
        dp_ranks=dp_ranks,
    )
    workspace = None
    if standalone:
        from atom.models.deepseek_v41.mono.indexer_workspace import IndexerWorkspace

        workspace = IndexerWorkspace(
            model, atom_config.max_model_len, atom_config.tensor_parallel_size
        )
    # the forward's output buffer and the step's staged sampling (one slot:
    # mono refuses pipeline parallel, whose forward_vars ring would swap them)
    staged = metadata_builder.model_runner.forward_vars
    decoder = MonoDecodeModel(
        model,
        drafter,
        atom_config.max_model_len,
        atom_config.index_cache_dtype == "fp4",
        dp_ranks,
        staged["outputs"],
        (staged["temperatures"].gpu, staged["top_ps"].gpu),
        tp_size=atom_config.tensor_parallel_size,
        indexer_workspace=workspace,
    )
    if draft is not None:
        draft.context_writer = decoder.write_context
    if standalone:
        from atom.models.deepseek_v41.mono.indexer_dispatch import MonoIndexerModel

        return MonoIndexerModel(decoder, model, atom_config, workspace=workspace)
    return decoder


def install_mono_decode(
    model: nn.Module, atom_config, drafter, *, metadata_builder=None
) -> nn.Module:
    dp_ranks = dp_attention_ranks(atom_config) if atom_config.enable_dp_attention else 0
    standalone = indexer_enabled(
        atom_config.index_cache_dtype == "fp4",
        atom_config.tensor_parallel_size,
        dp_ranks=dp_ranks,
    )
    # Native indexer math does not contain the layer runner's MoE. DP4 can
    # use a local hook without expanding ordinary layer Mono's DP2 admission.
    if (
        dp_ranks
        and standalone
        and hi_gate.mode() != "1"
        and config_refusal(atom_config) is not None
        and config_refusal(atom_config, supported_dpa=(2, 4)) is None
    ):
        from atom.models.deepseek_v41.mono.indexer_dispatch import (
            MonoIndexerModel,
            NativeDecodeModel,
        )
        from atom.models.deepseek_v41.mono.indexer_workspace import IndexerWorkspace

        workspace = IndexerWorkspace(model, atom_config.max_model_len, 1)
        return MonoIndexerModel(
            NativeDecodeModel(model), model, atom_config, workspace=workspace
        )
    return _install_layer_decode(
        model, atom_config, drafter, metadata_builder=metadata_builder
    )
