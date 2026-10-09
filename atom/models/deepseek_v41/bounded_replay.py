# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 decoder SWA bounded replay: prefill's late layers on a tail.

V4.1's global KV comes from its KV-source layers (2, 8, 14, 20). Every layer
after the last one -- ``late_layer_start`` (21) onward -- reuses layer 20's
compressed KV and selections and owns nothing but a sliding-window ring. What
a prefill leaves behind in those layers is each request's last ring of rows,
plus the logits and DSpark aux rows of its last token. So a prefill runs the
early layers on every row and the late layers on each request's tail only.

This is SGLang's ``--enable-decoder-swa-bounded-replay`` (``LateLayerTail``,
``late_layer_tail_layout``, ``enter_late_layer_tail``) and vLLM's
``--swa-bounded-replay`` (``DecoderReplayLayers``); the names follow SGLang.
The runtime model compiles the early and late layers as two graphs
(``runtime._EarlyLayers`` / ``_LateLayers``) and, on a replay, hands the late
graph the tail's rows with the tail's step in the forward context.

The late layers see the tail as a chunked prefill that starts at the tail.
Every global row before it is in the cache already (the early layers wrote it
this forward); only the late layers' window rows below the tail were never
computed, so the tail step carries ``swa_replay_start``, which keeps the index
build from reading them. That truncated window is the approximation SGLang and
vLLM make too: the tail's first rows see less local context in layer 21, and
each later layer carries the difference at most one window further, so the
last token's logits and the window rows decode reads are close to, not equal
to, a full prefill's.

The tail is the window ring (``geometry.ring_slots``: window + speculative
tokens, 133 on V4.1-Flash with five DSpark tokens), so the prefill rewrites
every ring slot and every row the draft's context write reads. SGLang keeps
128; the five extra rows cost nothing measurable.
"""

import contextlib
from dataclasses import dataclass

import numpy as np
import torch

from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep, RequestSpan
from atom.model_ops.attentions.token_layout.batch_ids import build_batch_ids
from atom.utils import upload_numpy


def late_layer_start(hf_config) -> int:
    """The first layer after the last KV source."""
    return max(hf_config.kv_source_layer_ids) + 1


def decoder_replay_unsupported(hf_config) -> str | None:
    """Why this model cannot replay its late layers on a tail, or None.

    Every layer after the last KV source must own no global KV: it compresses
    nothing (a KV source is the only layer with a compressor, so that holds by
    construction) and carries no Engram, whose n-gram rows read every token.
    """
    sources = list(getattr(hf_config, "kv_source_layer_ids", ()) or ())
    if not sources:
        return "the model declares no kv_source_layer_ids"
    start = max(sources) + 1
    if start >= hf_config.num_hidden_layers:
        return "no layer follows the last KV source layer"
    if any(i >= start for i in getattr(hf_config, "engram_layer_ids", ())):
        return "an Engram layer sits after the last KV source layer"
    return None


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


def build_late_layer_tail(step: BatchStep, tail_len: int, late_specs, cache):
    """``step`` cut down to each request's tail, with what the late layers read
    off the early ones carried over by row."""
    # Triton-backed; imported here so the host-side helpers above import
    # without it.
    from atom.model_ops.attentions.deepseek_v41.indices import fill_step_indptrs

    spans, indices, replay_start = late_layer_tail_layout(step, tail_len)
    device = step.positions.device
    token_indices = upload_numpy(indices, device)
    lengths = np.asarray([span.length for span in spans], dtype=np.int32)
    cu_seqlens_q = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)

    def rows(t, dim=0):
        return t.index_select(dim, token_indices)

    tail = BatchStep(
        spans,
        rows(step.positions),
        upload_numpy(cu_seqlens_q, device),
        step.slots[: len(spans)],
        upload_numpy(build_batch_ids(lengths), device),
        step.block_tables[: len(spans)],
        scheduled=int(lengths.sum()),
        max_q_len=int(lengths.max()),
        visible={ratio: rows(v) for ratio, v in step.visible.items()},
        plans=step.plans,
        request_positions=np.asarray([s.position for s in spans], dtype=np.int32),
        swa_replay_start=upload_numpy(replay_start, device),
    )
    # A REUSE layer reads its owner's selection and every layer from the
    # candidate source on reads its candidates. The owners and sources among
    # the early layers already ran on every row; a late one (a REINDEX layer)
    # selects again on the tail itself.
    late = {s.layer_id for s in late_specs}
    owners = {s.topk_owner for s in late_specs if s.topk_owner is not None}
    for owner in owners - late:
        tail.selected[owner] = rows(step.selected[owner], dim=1)
    for source in {s.candidate_source for s in late_specs} - {None} - late:
        tail.candidates[source] = rows(step.candidates[source])
    tail.indptrs = fill_step_indptrs(tail, cache.geometry, cache.indptr_buffers)
    return LateLayerTail(token_indices, tail)


@contextlib.contextmanager
def late_layer_tail(metadata, tail: LateLayerTail):
    """The late layers' step is the tail's while they run.

    The forward's own indptrs are not restored afterwards: nothing after the
    model reads them, and the next forward rebuilds its step.
    """
    full = metadata.step
    metadata.step = tail.step
    try:
        yield
    finally:
        metadata.step = full
