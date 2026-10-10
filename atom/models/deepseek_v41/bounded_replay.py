# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 decoder SWA bounded replay: prefill's late layers on a tail.

V4.1's global KV comes from its KV-source layers (2, 8, 14, 20). Every layer
after the last one -- ``late_layer_start`` (21) onward -- reuses layer 20's
compressed KV and selections and owns nothing but a sliding-window ring. What
a prefill leaves behind in those layers is each request's last ring of rows,
plus the logits and DSpark aux rows of its last token. So a prefill runs the
early layers on every row and the late layers on each request's tail only.

On by default (``--no-decoder-swa-bounded-replay`` turns it off). This is
SGLang's ``--enable-decoder-swa-bounded-replay`` (``LateLayerTail``,
``late_layer_tail_layout``, ``enter_late_layer_tail``) and vLLM's
``--swa-bounded-replay`` (``DecoderReplayLayers``, on by default there too);
the names follow SGLang. With it the runtime model compiles the early and late layers as two
graphs (``runtime._EarlyLayers`` / ``_LateLayers``) and, on a replay, hands
the late graph the tail's rows with the tail's step in the forward context.

The late layers see the tail as a chunked prefill that starts at the tail.
Every global row before it is in the cache already (the early layers wrote it
this forward); only the late layers' window rows below the tail were never
computed, so the tail step carries ``swa_replay_start``, which keeps the index
build from reading them.

Accuracy. The result is not a full prefill's. In layer 21 every tail row but
the last few sees a window truncated at the tail start. The tail is the ring,
``window + speculative tokens`` rows (133 on V4.1-Flash), only a few rows
longer than the window, so from layer 22 on even the last token's window
consists of rows that were themselves computed from truncated windows. The
last token's logits and every ring row decode reads therefore differ from a
full prefill's in all late layers. The global path -- the KV-source layers'
compressed KV and the top-k selections the late layers reuse -- is exact,
which is why the effect is small in practice (GSM8K and long-context
retrieval unchanged within run-to-run noise on V4.1-Flash). SGLang and vLLM
make the same approximation.
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
    if not list(getattr(hf_config, "kv_source_layer_ids", ()) or ()):
        return "the model declares no kv_source_layer_ids"
    start = late_layer_start(hf_config)
    if start >= hf_config.num_hidden_layers:
        return "no layer follows the last KV source layer"
    if any(i >= start for i in getattr(hf_config, "engram_layer_ids", ())):
        return "an Engram layer sits after the last KV source layer"
    return None


def replay_rows(forward, num_tokens: int) -> int | None:
    """The tail length when this forward replays, else None.

    A prefill with a request longer than the ring replays, image rows and
    input embeddings included (the late layers' only image-aware op, the MoE
    router, reads the tail's slice of the mask; see `late_layer_tail`).
    Decode, warmup, draft and TBO microbatches take every row through the late
    layers, and so does a padded step: prefill pads only under DP attention,
    where the MoE collectives are sized by every rank's token count, which a
    rank-local tail would change.
    """
    context, metadata = forward.context, forward.attn_metadata
    if (
        context is None
        or not context.is_prefill
        or context.is_dummy_run
        or context.is_draft
        or forward.ubatch_slices is not None
    ):
        return None
    step = getattr(metadata, "step", None)
    if (
        step is None
        or step.decode
        or num_tokens != step.width
        or step.width != step.scheduled
    ):
        return None
    ring = metadata.cache.geometry.ring_slots
    return ring if any(span.length > ring for span in step.requests) else None


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


def build_late_layer_tail(step: BatchStep, tail_len: int, late_specs) -> LateLayerTail:
    """``step`` cut down to each request's tail, with what the late layers read
    off the early ones carried over by row: one upload of the tail layout,
    then device-side gathers. No Triton kernel; the tail's indptrs are filled
    by ``late_layer_tail``."""
    spans, indices, replay_start = late_layer_tail_layout(step, tail_len)
    lengths = np.asarray([span.length for span in spans], dtype=np.int64)
    cu_seqlens_q = np.concatenate(([0], np.cumsum(lengths)))
    # One upload for the four per-tail arrays, sliced apart on the device.
    packed = upload_numpy(
        np.concatenate(
            (
                indices,
                cu_seqlens_q,
                build_batch_ids(lengths).astype(np.int64),
                replay_start.astype(np.int64),
            )
        ),
        step.positions.device,
    )
    rows = len(indices)
    token_indices = packed[:rows]
    cu, batch_ids, swa_replay_start = (
        part.to(torch.int32)
        for part in packed[rows:].split((len(spans) + 1, rows, len(spans)))
    )

    def take(t, dim=0):
        return t.index_select(dim, token_indices)

    tail = BatchStep(
        spans,
        take(step.positions),
        cu,
        step.slots[: len(spans)],
        batch_ids,
        step.block_tables[: len(spans)],
        scheduled=rows,
        max_q_len=int(lengths.max()),
        visible={ratio: take(v) for ratio, v in step.visible.items()},
        plans=step.plans,
        request_positions=np.asarray([s.position for s in spans], dtype=np.int32),
        swa_replay_start=swa_replay_start,
    )
    # A REUSE layer reads its owner's selection and every layer from the
    # candidate source on reads its candidates. The owners and sources among
    # the early layers already ran on every row; a late one (a REINDEX layer)
    # selects again on the tail itself.
    late = {s.layer_id for s in late_specs}
    owners = {s.topk_owner for s in late_specs if s.topk_owner is not None}
    for owner in owners - late:
        tail.selected[owner] = take(step.selected[owner], dim=1)
    for source in {s.candidate_source for s in late_specs} - {None} - late:
        tail.candidates[source] = take(step.candidates[source])
    return LateLayerTail(token_indices, tail)


@contextlib.contextmanager
def late_layer_tail(metadata, tail: LateLayerTail):
    """The late layers' step, and image mask, are the tail's while they run.

    Filling the tail's indptrs rewrites the cache's shared indptr buffers, and
    the late REINDEX layers rewrite its tile workspace. When the forward's step
    is restored its indptrs and memoized tiles are cleared, since those buffers
    now hold the tail's contents: a reader after the model fails loudly
    instead of reading them. The next forward's ``begin_step`` rebuilds both.
    """
    # Triton-backed; imported here so this module imports without it.
    from atom.model_ops.attentions.deepseek_v41.indices import fill_step_indptrs

    cache = metadata.cache
    tail.step.indptrs = fill_step_indptrs(
        tail.step, cache.geometry, cache.indptr_buffers
    )
    full, image_mask = metadata.step, getattr(metadata, "image_mask", None)
    metadata.step = tail.step
    if image_mask is not None:
        # [1, tokens]: the late layers' MoE router routes image rows by it
        metadata.image_mask = image_mask.index_select(1, tail.token_indices)
    try:
        yield
    finally:
        metadata.step = full
        if image_mask is not None:
            metadata.image_mask = image_mask
        full.indptrs = {}
        full.tiles.clear()
