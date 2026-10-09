# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 decoder SWA bounded replay: prefill's late layers on a tail.

V4.1's global KV comes from its KV-source layers (2, 8, 14, 20). Every layer
after the last one -- ``late_layer_start`` (21) onward -- reuses layer 20's
compressed KV and selections and owns nothing but a sliding-window ring. What
a prefill leaves behind in those layers is therefore each request's last ring
of rows, plus the logits and DSpark aux rows of its last token. So an eager
prefill runs the early layers on every row and the late layers on each
request's tail only.

This is SGLang's ``--enable-decoder-swa-bounded-replay`` (``LateLayerTail``,
``late_layer_tail_layout``, ``enter_late_layer_tail``) and vLLM's
``--swa-bounded-replay`` (``DecoderReplayLayers``); the names follow SGLang.

The late layers see the tail as a chunked prefill that starts at the tail.
Every global row before it is in the cache already (the early layers wrote it
this forward); only the late layers' window rows below the tail were never
computed, so the tail step carries ``swa_replay_start``, which keeps the index
build from reading them. That truncated window is the approximation SGLang and
vLLM make too: the tail's first rows see less local context in layer 21, and
each later layer carries the difference at most one window further, so the
last token's logits and the window rows decode reads are close to, not equal
to, a full prefill's.

The tail is the window ring (``window + speculative tokens``, 133 on
V4.1-Flash with five DSpark tokens), so the prefill rewrites every ring slot.
SGLang keeps 128; the five extra rows cost nothing measurable.
"""

import contextlib
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from atom.model_ops.attentions.deepseek_v41.indices import fill_step_indptrs
from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep, RequestSpan
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.models.deepseek_v41.config import AttentionMode
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


# ---- the tail: which rows, and the step the late layers run on it ----------


def late_layer_tail_layout(step: BatchStep, tail_len: int):
    """Each request's last ``min(tail_len, length)`` rows of ``step``.

    Returns ``(spans, token_indices, swa_replay_start)`` on the host: the tail's
    request spans, the rows of ``step`` they keep (ascending), and per request
    the lowest position whose late-layer window row exists -- the tail's start
    for a trimmed request, 0 for one kept whole.
    """
    spans, token_indices, replay_start, offset = [], [], [], 0
    for span in step.requests:
        length = min(span.length, tail_len)
        position = span.end - length
        spans.append(RequestSpan(span.request_id, position, offset, length, span.slot))
        end = span.offset + span.length
        token_indices.append(np.arange(end - length, end, dtype=np.int64))
        replay_start.append(position if length < span.length else 0)
        offset += length
    return (
        tuple(spans),
        np.concatenate(token_indices),
        np.asarray(replay_start, dtype=np.int32),
    )


@dataclass(frozen=True)
class LateLayerTail:
    """The late layers' rows of one prefill: ``token_indices`` into the
    forward's rows, and ``step``, the ``BatchStep`` they run as."""

    token_indices: torch.Tensor
    step: BatchStep

    def rows(self, t: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return t.index_select(dim, self.token_indices)


def build_late_layer_tail(
    step: BatchStep, tail_len: int, late_specs, cache
) -> LateLayerTail:
    """``step`` cut down to each request's tail, with what the late layers read
    off the early ones carried over by row."""
    spans, indices, replay_start = late_layer_tail_layout(step, tail_len)
    device = step.positions.device
    token_indices = torch.from_numpy(indices).to(device, non_blocking=True)
    lengths = np.asarray([span.length for span in spans], dtype=np.int32)
    cu_seqlens_q = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    batch_ids = np.repeat(np.arange(len(spans), dtype=np.int32), lengths)

    def to_device(array):
        return torch.from_numpy(array).to(device, non_blocking=True)

    def rows(t, dim=0):
        return t.index_select(dim, token_indices)

    tail = BatchStep(
        spans,
        rows(step.positions),
        to_device(cu_seqlens_q),
        step.slots[: len(spans)],
        to_device(batch_ids),
        step.block_tables[: len(spans)],
        scheduled=int(lengths.sum()),
        max_q_len=int(lengths.max()),
        visible={ratio: rows(v) for ratio, v in step.visible.items()},
        plans=step.plans,
        request_positions=np.asarray([s.position for s in spans], dtype=np.int32),
        swa_replay_start=to_device(replay_start),
    )
    # A REUSE layer reads its owner's selection and every layer from the
    # candidate source on reads its candidates. The owners and sources among
    # the early layers already ran on every row; a late one (a REINDEX layer)
    # selects again on the tail itself.
    late = {s.layer_id for s in late_specs}
    owners = {s.topk_owner for s in late_specs if s.mode == AttentionMode.REUSE}
    for owner in owners - late:
        tail.selected[owner] = rows(step.selected[owner], dim=1)
    for source in {s.candidate_source for s in late_specs} - {None} - late:
        tail.candidates[source] = rows(step.candidates[source])
    tail.indptrs = fill_step_indptrs(tail, cache.geometry, cache.indptr_buffers)
    return LateLayerTail(token_indices, tail)


@contextlib.contextmanager
def late_layer_tail(metadata, context, tail: LateLayerTail):
    """Run the late layers on ``tail``: its step stands in for the forward's,
    and DSpark's aux capture learns which rows it is writing
    (``Context.late_layer_tail_rows``)."""
    full = metadata.step
    metadata.step = tail.step
    context.late_layer_tail_rows = tail.token_indices
    try:
        yield
    finally:
        metadata.step = full
        context.late_layer_tail_rows = None
        # The tail refilled the shared indptr buffers; give them back to the
        # forward's step for anything that reads them after the model.
        full.indptrs = fill_step_indptrs(
            full, metadata.cache.geometry, metadata.cache.indptr_buffers
        )


# ---- the model --------------------------------------------------------------


class DecoderReplayModel(nn.Module):
    """``model`` (the serving model, mono-wrapped or not) with every eligible
    prefill run eagerly on ``runtime``'s layers, the late ones on the tail."""

    def __init__(
        self, model: nn.Module, runtime: nn.Module, late_layer_start: int, tail_len: int
    ):
        super().__init__()
        self.model = model
        # Not a submodule: `model` owns it already.
        self.__dict__["runtime"] = runtime
        self.late_layer_start = late_layer_start
        self.tail_len = tail_len
        self.late_specs = runtime.topology[
            late_layer_start : runtime.config.num_hidden_layers
        ]
        self._logged = False

    def forward(self, input_ids, positions, inputs_embeds=None):
        if self._replays(input_ids, inputs_embeds):
            return self._forward_with_replay(input_ids)
        if inputs_embeds is None:
            return self.model(input_ids, positions)
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _replays(self, input_ids, inputs_embeds) -> bool:
        """An eager text prefill with a request longer than the tail: decode,
        warmup, draft, TBO microbatches and image rows take the model as is."""
        if inputs_embeds is not None:
            return False
        forward = get_forward_context()
        context, metadata = forward.context, forward.attn_metadata
        if (
            context is None
            or not context.is_prefill
            or context.is_dummy_run
            or context.is_draft
            or forward.ubatch_slices is not None
        ):
            return False
        step = getattr(metadata, "step", None)
        if (
            step is None
            or step.decode
            or getattr(metadata, "image_mask", None) is not None
            # no padding rows: the tail is laid out over the scheduled ones
            or input_ids.numel() != step.width
            or step.width != step.scheduled
        ):
            return False
        return any(span.length > self.tail_len for span in step.requests)

    def _run_layers(self, state, layers):
        m = self.runtime
        for spec, layer in layers:
            rope = m.global_rope if spec.ratio else m.window_rope
            state = layer(state, None, None, rope, None, image_mask=None)
        return state

    @torch.inference_mode()
    def _forward_with_replay(self, input_ids):
        m = self.runtime
        forward = get_forward_context()
        metadata = forward.attn_metadata
        num_tokens, hidden_size = input_ids.numel(), m.config.hidden_size
        layers = list(zip(m.topology, m.layers))

        hidden = m.embed(input_ids.flatten()).view(1, num_tokens, hidden_size)
        m.begin_forward(hidden, None)
        state = SinglePassHCState.from_embeddings(hidden, m.config.hc_mult)
        state = self._run_layers(state, layers[: self.late_layer_start])

        tail = build_late_layer_tail(
            metadata.step, self.tail_len, self.late_specs, metadata.cache
        )
        with late_layer_tail(metadata, forward.context, tail):
            state = state.take_rows(tail.token_indices)
            state = self._run_layers(state, layers[self.late_layer_start :])

        hidden = state.collapse()
        m.end_forward(hidden, None)
        # The trimmed rows' outputs stay zero; nothing reads them.
        out = hidden.new_zeros(num_tokens, hidden_size)
        out.index_copy_(0, tail.token_indices, hidden[0])
        if not self._logged:
            self._logged = True
            logger.info(
                "Decoder SWA bounded replay: layers %d.. ran %d of %d rows",
                self.late_layer_start,
                tail.token_indices.numel(),
                num_tokens,
            )
        return out


# ---- installation -----------------------------------------------------------


def decoder_replay_unsupported(hf_config, topology, speculative_config) -> str | None:
    """Why this model cannot replay its late layers on a tail, or None."""
    sources = list(getattr(hf_config, "kv_source_layer_ids", ()) or ())
    if not sources:
        return "the model declares no kv_source_layer_ids"
    start = max(sources) + 1
    late = topology[start : hf_config.num_hidden_layers]
    if not late:
        return "no layer follows the last KV source layer"
    if any(s.mode == AttentionMode.FULL or s.ratio not in (0, 1) for s in late):
        return "a layer after the last KV source layer compresses its own KV"
    if any(i >= start for i in getattr(hf_config, "engram_layer_ids", ())):
        return "an Engram layer sits after the last KV source layer"
    targets = getattr(hf_config, "dspark_target_layer_ids", ()) or ()
    if speculative_config is not None and any(i < start for i in targets):
        return "the drafter reads every row of a layer before the late ones"
    return None


def install_decoder_swa_bounded_replay(
    model: nn.Module, runtime: nn.Module, atom_config
) -> nn.Module:
    """``model`` wrapped for bounded replay under
    ``--enable-decoder-swa-bounded-replay``, else ``model`` itself."""
    if not getattr(atom_config, "enable_decoder_swa_bounded_replay", False):
        return model
    hf = runtime.config
    spec = atom_config.speculative_config
    reason = decoder_replay_unsupported(hf, runtime.topology, spec)
    if reason is not None:
        logger.warning("Decoder SWA bounded replay is off: %s.", reason)
        return model
    late_layer_start = max(hf.kv_source_layer_ids) + 1
    tail_len = hf.sliding_window + (spec.num_speculative_tokens if spec else 0)
    logger.info(
        "Decoder SWA bounded replay: eager prefill runs layers %d..%d on each "
        "request's last %d rows",
        late_layer_start,
        hf.num_hidden_layers - 1,
        tail_len,
    )
    return DecoderReplayModel(model, runtime, late_layer_start, tail_len)
