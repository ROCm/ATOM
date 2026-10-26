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

The tail is a request bucket. Its rows are laid out for the step's
``running_bs`` requests, ``W`` each (``tail_width``): the finishing requests'
first, then zero-length padding requests and rows no request owns
(``lay_out_tail``). One shape per ``running_bs`` is what lets the decoder
record as a request graph (`decoder_graph.DecoderGraph`).

DP attention. The decoder's MoE collectives are sized by every rank's rows,
which the tail changes. ``running_bs`` is the group's, so every prefilling
rank runs the same tail width and request count, and a rank still decoding
its own rows; each derives that table from the step's DP sync
(``decoder_stages``). While any rank runs the decoder, a rank finishing no
prompt runs it on its tail of padding alone, to join the collectives. TBO-split
steps compute every row.

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

from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep
from atom.model_ops.attentions.token_layout.batch_ids import build_batch_ids
from atom.utils.forward_context import DPStage


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


def replayable(forward, running_tokens: int) -> bool:
    """Whether this forward may compute fewer than all of its rows: a prefill
    the backend published a decoder for (`BatchStep.decoder`) -- every
    serving prefill, an idle DP rank's dummy one included, never a warmup's.

    Image rows and input embeddings replay too (the decoder's only image-aware
    op, the MoE router, reads its rows' slice of the mask; see `rows_of`). A
    prefill padded to a token-graph bucket replays like any other: its padding
    rows belong to no request, so no tail picks them. Draft passes and TBO
    microbatches compute every row.
    """
    context, metadata = forward.context, forward.attn_metadata
    if (
        context is None
        or not context.is_prefill
        or context.is_draft
        or forward.ubatch_slices is not None
    ):
        return False
    # Workers clear ubatch_slices to prevent recursive splitting. Their live
    # TBO context still identifies them, including the child at token offset 0.
    from atom.utils.tbo.ubatching import tbo_active

    if tbo_active():
        return False
    step = getattr(metadata, "step", None)
    return (
        step is not None
        and step.decoder is not None
        and not step.decode
        and running_tokens == step.width
    )


def tail_width(requests: int, window: int, capacity: int) -> int:
    """The rows of a tail laid out for ``requests``: ``window`` each, within
    the ``capacity`` its buffers hold -- which every finishing request's rows
    fit in, since they are rows of the forward."""
    return min(requests * window, capacity)


@dataclass(frozen=True)
class TailLayout:
    """A tail's rows on the host (`lay_out_tail`). ``rows``: [width] each
    tail row's row of the forward, padding repeating a real one (or row 0).
    ``query_ids``: [width] layer 39's batch id at each finishing request's last
    tail row, -1 elsewhere. ``last``: [requests] layer 39's query rows, a
    padding request's the tail's last. ``query_ptrs``: [requests + 1] where
    each query's indptr is read, a padding request's past the end, so its list
    is empty and its predecessor's is whole. ``last_rows``: [finishing] the
    forward rows layer 39's outputs go to. ``filled``: the tail rows that are
    real, which lead."""

    rows: np.ndarray
    query_ids: np.ndarray
    last: np.ndarray
    query_ptrs: np.ndarray
    last_rows: np.ndarray
    filled: int


def lay_out_tail(picks, requests: int, width: int) -> TailLayout:
    """``picks`` (`tail_spans`) laid out as a tail of ``requests`` requests and
    ``width`` rows."""
    lengths = np.asarray([n for _, n, _ in picks], dtype=np.int64)
    ends = np.cumsum(lengths)
    filled = int(ends[-1]) if len(ends) else 0
    rows = np.zeros(width, dtype=np.int64)
    for (span, n, _), end in zip(picks, ends):
        rows[end - n : end] = span.offset + span.length - n + np.arange(n)
    if filled:
        rows[filled:] = rows[filled - 1]
    finals = ends - 1
    query_ids = np.full(width, -1, dtype=np.int64)
    query_ids[finals] = build_batch_ids(lengths)[finals]
    padding = requests - len(picks)
    return TailLayout(
        rows=rows,
        query_ids=query_ids,
        last=np.concatenate([finals, np.full(padding, width - 1)]),
        query_ptrs=np.concatenate([finals, np.full(padding + 1, width)]),
        last_rows=rows[finals],
        filled=filled,
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
        """`cache.attention_indices` for the queries alone: built afresh, never
        the index group its step's other layers already built over every row."""
        prefix, pptr, extend, eptr = cache.attention_indices(
            spec, replace(self.index, group_indices={})
        )
        return prefix, pptr[self.ptr_rows], extend, eptr[self.ptr_rows]


@dataclass(frozen=True)
class ReplayPlan:
    """The decoder's rows of a replayed prefill, on the device at the
    addresses every step of its request bucket publishes (`TailLayout`).
    ``tail``: their step."""

    tail: BatchStep
    tail_rows: torch.Tensor
    query_ids: torch.Tensor
    last: torch.Tensor
    query_ptrs: torch.Tensor
    last_rows: torch.Tensor
    filled: int

    def queries(self) -> QueryRows:
        """Layer 39's queries. Built where they are read, so a recording reads
        them through the tail's addresses."""
        tail = self.tail
        return QueryRows(
            self.last,
            tail.positions.index_select(0, self.last),
            replace(tail, batch_ids=self.query_ids),
            self.query_ptrs,
        )


def tail_spans(step: BatchStep, window: int):
    """``step``'s tail, as ``(span, rows, index)``: each finishing request's
    last ``window`` rows, ``index`` its request's in ``step``."""
    return [
        (span, min(span.length, window), i)
        for i, span in enumerate(step.requests)
        if step.request_finals[i]
    ]


def decoder_runs(forward) -> bool:
    """Whether this replaying forward runs the decoder: some rank of the
    group -- this one, or a peer whose collectives count on this rank --
    finishes a prompt, or still decodes every row. A capture context, which
    declares its shape rather than reading one, runs it."""
    mode = forward.context.forward_mode
    if mode is None:
        return True
    sync = mode.sync
    return sync.max_final_requests_across_dp > 0 or not all(sync.prefill_across_dp)


def decoder_stages(forward, parallel_config, width: int, requests: int, replaying):
    """The DP stages of layer 20's FFN through layer 38 and of layer 39, or
    ``(None, None)`` where every MoE layer runs the step's own rows.

    A prefilling rank replays and runs ``width`` and ``requests`` rows -- the
    same on every one, since the group's ``running_bs`` fixes both
    (`tail_width`) -- and a rank still decoding runs its own rows. Every rank
    of a step with no prefill or a TBO-split one runs its own rows, and so does
    a warmup's (``replaying`` False on a prefill: no rank replays). A capture
    context, which declares its shape, has every rank recording this one's
    step -- a prefill's or a decode's."""
    context = forward.context
    if forward.dp_metadata is None or (context.is_prefill and not replaying):
        return None, None
    mode = context.forward_mode
    across = context.running_tokens_across_dp
    if mode is None:
        prefill = (context.is_prefill,) * len(across)
    elif mode.sync.tbo_collective_active:
        return None, None
    else:
        prefill = mode.sync.prefill_across_dp
    if not any(prefill):
        return None, None

    def stage(rows):
        tokens = [rows if p else n for p, n in zip(prefill, across)]
        return DPStage.make(parallel_config, tokens)

    return stage(width), stage(requests)


def run_decoder_stages(forward, stages, first, middle, last):
    """Run the layers past the decoder's KV source, each under its DP stage
    (`decoder_stages`): ``first`` (layer 20 after its compression) and
    ``middle`` (21..38) under the decoder's, ``last`` (layer 39) under its own,
    so every MoE collective is sized the way the group runs it."""
    decoder, last_layer = stages
    with forward.use_dp_stage(decoder):
        state = middle(first())
        with forward.use_dp_stage(last_layer):
            return last(state)


def carry_selections(parent: BatchStep, plan: ReplayPlan, specs) -> None:
    """What the decoder reads off the encoder -- a REUSE layer's owner's
    selection, a candidate source's candidates -- at the tail's rows."""
    owners, sources = carried_layers(parent, specs)
    rows, tail = plan.tail_rows, plan.tail
    taken = _gather_rows(rows, *(parent.selected[o] for o in owners), dim=1)
    tail.selected.update(zip(owners, taken))
    taken = _gather_rows(rows, *(parent.candidates[s] for s in sources))
    tail.candidates.update(zip(sources, taken))


def carried_layers(parent: BatchStep, specs):
    """The encoder layers whose selections (owners) and candidates (sources)
    the decoder ``specs`` read, as `carry_selections` carries them."""
    own = {s.layer_id for s in specs}
    owners = sorted(
        ({s.topk_owner for s in specs} - {None} - own) & parent.selected.keys()
    )
    sources = sorted(
        ({s.candidate_source for s in specs} - {None} - own) & parent.candidates.keys()
    )
    return owners, sources


@contextlib.contextmanager
def rows_of(metadata, step, queries=None, rows=None):
    """``step`` and ``queries`` are the forward's while the layers in this
    block run, and so is the image mask at ``rows`` of the outer one's: the
    rows an MoE router routes here.

    The inner step's indptrs are filled by its first reader, into the
    cache's shared indptr buffers, and a REINDEX layer rewrites its tile
    workspace. When the outer step is restored its indptrs and memoized tiles
    are cleared, since those buffers now hold the inner step's contents: a
    reader after this fills them again.
    """
    indexed = step if queries is None else queries.index
    # Dropped on every entry, so each pass of a capture records the fill.
    indexed.indptrs = {}
    outer = metadata.step, metadata.queries, metadata.image_mask
    metadata.step, metadata.queries = step, queries
    if rows is not None and outer[2] is not None:
        # [1, tokens]: an MoE router routes image rows by it
        metadata.image_mask = outer[2].index_select(1, rows)
    try:
        yield
    finally:
        metadata.step, metadata.queries, metadata.image_mask = outer
        release_outer(outer[0])


def release_outer(step: BatchStep) -> None:
    """Drop ``step``'s indptrs and memoized tiles once an inner step's rows
    have overwritten their buffers (`rows_of`)."""
    step.indptrs = {}
    step.tiles.clear()
