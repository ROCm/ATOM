# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 decoder SWA bounded replay: a prefill computes only the rows
anything after it reads.

V4.1 is a causal encoder (layers 0-19) followed by a decoder (20-39). The
decoder's global KV comes from the encoder's output: its one KV source, layer
20, compresses it, and every layer after it -- ``late_layer_start`` (21)
onward -- reuses that KV and its selections and owns nothing but a
sliding-window ring. So, as the V4.1 report's SWA Bounded Replay prescribes,
the forward that finishes a prompt runs the decoder over its last ``W`` rows
alone (the tail; ``W`` is the window), and one that does not runs none of it:

- the encoder, and layer 20's compression (global KV, index keys): every row;
- layers 20-38: the tail;
- layer 39's KV: the tail; its query, attention and FFN: the last row.

Every decoder layer's window starts at the tail (``swa_replay_start``). Layer
39's queries are fewer than its KV rows (``QueryRows``): varlen attention whose
``cu_seqlens_q`` is shorter than its ``cu_seqlens_k``. A forward finishing a
prompt with fewer than ``W`` rows starts earlier, recomputing rows its cache
holds (``BlockManager.replay_start``, ``Config.min_final_prefill_rows``): their
global KV and the encoder's window rows stay as the cache has them, and the
rows from where the forward resumed read those rows, exact; only what the rows
feed the decoder is new.

On by default (``--no-decoder-swa-bounded-replay`` turns it off). The runtime
model compiles layers 0-19 and 21-38 as two graphs (``runtime._EarlyLayers`` /
``_LateLayers``) and runs layers 20 and 39 between and after them, which is
where a replay cuts their rows.

Accuracy. The result is not a full prefill's: in every decoder layer each tail
row but the last sees a window truncated at the tail start, and from layer 21
on even the last token's window holds rows computed from truncated windows. The
global path -- the encoder, the compressed KV and the top-k selections the
decoder reuses -- is exact. DeepSeek post-trains on this truncation.
"""

import contextlib
from dataclasses import dataclass, replace

import numpy as np
import torch

from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep, RequestSpan
from atom.model_ops.attentions.token_layout.batch_ids import build_batch_ids
from atom.utils import upload_numpy


def late_layer_start(hf_config) -> int:
    """The first layer after the last KV source."""
    return max(hf_config.kv_source_layer_ids) + 1


def decoder_replay_unsupported(hf_config) -> str | None:
    """Why this model cannot replay its decoder on a tail, or None.

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


def _gather_rows(rows, *tensors, dim=0):
    """`token_layout.gather_rows`: Triton-backed, imported here so this module
    imports without it."""
    from atom.model_ops.attentions.token_layout.gather import gather_rows

    return gather_rows(rows, *tensors, dim=dim)


def replayable(forward, num_tokens: int) -> bool:
    """Whether this forward may compute fewer than all of its rows.

    A prefill may, image rows and input embeddings included (the decoder's
    only image-aware op, the MoE router, reads its rows' slice of the mask;
    see `rows_of`). Decode, warmup, draft and TBO microbatches compute every
    row, and so do DP attention and padded steps: DP's MoE collectives are
    sized by every rank's token count, even when a rank needs no padding,
    and a rank-local tail would change those counts.
    """
    context, metadata = forward.context, forward.attn_metadata
    if (
        context is None
        or not context.is_prefill
        or context.is_dummy_run
        or context.is_draft
        or forward.ubatch_slices is not None
        or getattr(forward, "dp_metadata", None) is not None
    ):
        return False
    # Workers clear ubatch_slices to prevent recursive splitting. Their live
    # TBO context still identifies them, including the child at token offset 0.
    from atom.utils.tbo.ubatching import tbo_active

    if tbo_active():
        return False
    step = getattr(metadata, "step", None)
    return not (
        step is None
        or step.decode
        or num_tokens != step.width
        or step.width != step.scheduled
    )


@dataclass(frozen=True)
class QueryRows:
    """The rows of a step a layer is asked outputs for, each request's last
    rows of it, where the layer writes KV for all of them.

    ``index`` is the step with every other row's batch id -1, the padding mark
    the index build skips, so the queries' index lists come out adjacent and
    ``ptr_rows`` -- the queries, then the step's end -- picks their indptrs.
    """

    rows: torch.Tensor
    positions: torch.Tensor
    index: BatchStep
    ptr_rows: torch.Tensor

    def take(self, *tensors):
        """``tensors`` ([1, rows, ...]) at the queries, in one launch."""
        return _gather_rows(self.rows, *tensors, dim=1)

    def indices(self, cache, spec):
        """`cache.attention_indices` for the queries alone."""
        prefix, pptr, extend, eptr = cache.attention_indices(spec, self.index)
        return prefix, pptr[self.ptr_rows], extend, eptr[self.ptr_rows]


@dataclass(frozen=True)
class ReplayPlan:
    """A replayed prefill's parts (the module docstring); all None when no
    request finishes its prompt. ``tail``: the decoder's rows, the last
    layer's queries ``q39``; ``tail_rows`` and ``last_rows`` are their rows of
    the forward."""

    tail: BatchStep | None = None
    tail_rows: torch.Tensor | None = None
    q39: QueryRows | None = None
    last_rows: torch.Tensor | None = None


def build_replay_plan(step: BatchStep, window: int) -> ReplayPlan:
    """``step``'s tail: each finishing request's last ``window`` rows, a step
    of their own over its requests, PAGEs and slots, with no compress plan
    (their compression ran over ``step``). Laid out on the host and uploaded
    in one copy per dtype."""
    picks = [
        (span, min(span.length, window), i)
        for i, span in enumerate(step.requests)
        if step.request_finals[i]
    ]
    if not picks:
        return ReplayPlan()
    lengths = np.asarray([n for _, n, _ in picks])
    ends = np.cumsum(lengths)
    rows = np.concatenate(
        [s.offset + s.length - np.arange(n, 0, -1) for s, n, _ in picks]
    )
    positions = np.concatenate([s.end - np.arange(n, 0, -1) for s, n, _ in picks])
    starts = positions[ends - lengths]
    batch_ids = build_batch_ids(lengths)
    # layer 39's queries: each tail's last row
    last = ends - 1
    query_ids = np.full(len(rows), -1, np.int64)
    query_ids[last] = batch_ids[last]
    device = step.positions.device
    up = _upload(
        device,
        torch.int32,
        rows=rows,
        cu=np.concatenate(([0], ends)),
        batch_ids=batch_ids,
        # no window row below its first, in the layers it runs
        floors=starts,
        cached_until=np.zeros(len(picks)),
        slots=[s.slot for s, _, _ in picks],
        requests=[i for _, _, i in picks],
        last=last,
        query_ids=query_ids,
        query_ptrs=np.append(last, len(rows)),
        last_rows=rows[last],
    )
    up |= _upload(
        device,
        step.positions.dtype,
        positions=positions,
        last_positions=positions[last],
    )
    tail = BatchStep(
        tuple(
            RequestSpan(s.request_id, int(start), int(end - n), n, s.slot)
            for (s, n, _), start, end in zip(picks, starts, ends)
        ),
        up["positions"],
        up["cu"],
        up["slots"],
        up["batch_ids"],
        step.block_tables.index_select(0, up["requests"]),
        scheduled=int(ends[-1]),
        max_q_len=int(lengths.max()),
        visible=dict(
            zip(step.visible, _gather_rows(up["rows"], *step.visible.values()))
        ),
        request_positions=starts.astype(np.int32),
        swa_replay_start=up["floors"],
        swa_cached_until=up["cached_until"],
    )
    return ReplayPlan(
        tail=tail,
        tail_rows=up["rows"],
        q39=QueryRows(
            up["last"],
            up["last_positions"],
            replace(tail, batch_ids=up["query_ids"]),
            up["query_ptrs"],
        ),
        last_rows=up["last_rows"],
    )


def _upload(device, dtype, **arrays):
    """``arrays`` on ``device`` as ``dtype``, in one copy."""
    names = list(arrays)
    host = np.concatenate([np.asarray(arrays[k]) for k in names]).astype(
        torch.empty(0, dtype=dtype).numpy().dtype
    )
    return dict(
        zip(
            names,
            upload_numpy(host, device).split([len(arrays[k]) for k in names]),
        )
    )


def carry_selections(parent: BatchStep, plan: ReplayPlan, specs) -> None:
    """What the decoder reads off the encoder -- a REUSE layer's owner's
    selection, a candidate source's candidates -- at the tail's rows."""
    own = {s.layer_id for s in specs}
    owners = sorted(
        ({s.topk_owner for s in specs} - {None} - own) & parent.selected.keys()
    )
    sources = sorted(
        ({s.candidate_source for s in specs} - {None} - own) & parent.candidates.keys()
    )
    rows, tail = plan.tail_rows, plan.tail
    taken = _gather_rows(rows, *(parent.selected[o] for o in owners), dim=1)
    tail.selected.update(zip(owners, taken))
    taken = _gather_rows(rows, *(parent.candidates[s] for s in sources))
    tail.candidates.update(zip(sources, taken))


@contextlib.contextmanager
def rows_of(metadata, step, queries=None, rows=None):
    """``step`` and ``queries`` are the forward's while the layers in this
    block run, and so is the image mask at ``rows`` of the outer one's: the
    rows an MoE router routes here.

    Filling the step's indptrs rewrites the cache's shared indptr buffers, and
    a REINDEX layer rewrites its tile workspace. When the outer step is
    restored its indptrs and memoized tiles are cleared, since those buffers
    now hold the inner step's contents: a reader after this fails loudly
    instead of reading them; the next forward's ``begin_step`` rebuilds both.
    """
    # Triton-backed; imported here so this module imports without it.
    from atom.model_ops.attentions.deepseek_v41.indices import fill_step_indptrs

    indexed = step if queries is None else queries.index
    indexed.indptrs = fill_step_indptrs(
        indexed, metadata.cache.geometry, metadata.cache.indptr_buffers
    )
    outer = metadata.step, metadata.queries, metadata.image_mask
    metadata.step, metadata.queries = step, queries
    if rows is not None and outer[2] is not None:
        # [1, tokens]: an MoE router routes image rows by it
        metadata.image_mask = outer[2].index_select(1, rows)
    try:
        yield
    finally:
        metadata.step, metadata.queries, metadata.image_mask = outer
        outer[0].indptrs = {}
        outer[0].tiles.clear()
