# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's decoder on a replayed prefill's tail, as a request graph.

`bounded_replay` lays a tail out for the step's ``running_bs`` requests, at
addresses every step of that request bucket shares, so the decoder -- layer
20's attention and FFN, layers 21-38, layer 39 -- records once per bucket as a
`ReqGraph` and replays with no Python. A rank finishing no prompt replays its
bucket's idle recording, which joins its peers' collectives. Gathering the
tail's rows in and scattering layer 39's out stay outside the graph, and so
does any step a recording cannot serve (`DecoderGraph._replays`).
"""

import contextlib

import torch

from atom.model_ops.attentions.token_layout.gather import gather_rows
from atom.utils.region_graph import ReqGraph

from .bounded_replay import (
    decoder_stages,
    release_outer,
    rows_of,
    run_decoder_stages,
)


@contextlib.contextmanager
def _token_graphs_off(forward):
    """The late layers run as they are: a request graph records them inline,
    and no token bucket fits a tail (its rows are a request bucket's)."""
    outer, forward.tr_graph = forward.tr_graph, None
    try:
        yield
    finally:
        forward.tr_graph = outer


class DecoderGraph:
    """The decoder of `DeepseekV41RuntimeModel` on a tail (`run`)."""

    def __init__(self, model):
        self.model = model
        self.graph = ReqGraph()
        # A decoder that reads a selection the encoder carries off by address
        # (`carry_selections`) holds it only where a rank has requests, so
        # ranks could split on replaying; the topology says so once.
        specs = model.decoder_specs
        read = {s.topk_owner for s in specs} | {s.candidate_source for s in specs}
        self.records = not (read - {None} - {s.layer_id for s in specs})

    def run(self, forward, plan, normed, seam):
        """Layer 39's output rows, one per tail request (`ReplayPlan.last`),
        from the forward's ``normed`` rows and mHC ``seam`` at layer 20.
        Recorded under the runner's request-bucket pass ("record_req"),
        replayed where a recording serves the step, run directly otherwise."""
        requests = plan.last.numel()
        key = (requests, bool(plan.tail.requests))
        # A fused-quant `normed` (BF16 rows, their FP8 and e8m0 scales) travels
        # as three tensors, the scales as bytes: a gather moves bytes.
        fused = isinstance(normed, tuple)
        fields = (
            (normed[0], normed[1], normed[2].view(torch.uint8)) if fused else (normed,)
        )

        def tail_rows(out=None):
            gathered = gather_rows(
                plan.tail_rows,
                *fields,
                *seam,
                dim=1,
                out=None if out is None else tuple(t[None] for t in out),
            )
            return tuple(t.squeeze(0) for t in gathered)

        def decode(*rows):
            # Sized only where the stages run: a replay holds them recorded.
            stages = decoder_stages(
                forward,
                self.model.parallel_config,
                plan.tail.width,
                requests,
                replaying=True,
            )
            return self._decode(forward, plan, stages, rows, fused)

        if forward.tr_graph == "record_req" and self.records:
            return self.graph.record(decode, tail_rows(), key=key)
        if self._replays(forward, key):
            # Straight into the recording's inputs: the replay copies nothing.
            inputs = self.graph.inputs(key)
            tail_rows(out=inputs)
            output = self.graph.replay(inputs, key=key)
            # What `rows_of` does on its way out, which a replay skips.
            release_outer(forward.attn_metadata.step)
            return output
        return decode(*tail_rows())

    def _replays(self, forward, key) -> bool:
        """A recording serves the step: one exists for its bucket (the
        group's `running_bs`), no token-bucket pass is recording around it,
        and the DP group replays request graphs as one, as a decode's whole
        forward does (`ForwardMode.req_graphs`) -- an image prefill, whose rows
        the MoE router routes on their mask, being a step the group refuses a
        graph for."""
        mode = forward.context.forward_mode
        return (
            self.graph.has(key)
            and forward.tr_graph != "record"
            and mode is not None
            and mode.req_graphs
        )

    def _decode(self, forward, plan, stages, tail_rows, fused):
        model = self.model
        metadata = forward.attn_metadata
        start = model.decoder_start
        layer, rope = model.layers[start], model._rope(start)
        rows = [t.unsqueeze(0) for t in tail_rows]
        if fused:
            hidden, quant, scale, *seam = rows
            normed = (hidden, quant, scale.view(torch.float8_e8m0fnu))
        else:
            normed, *seam = rows
        queries = plan.queries()

        def last_layer(state):
            with rows_of(metadata, plan.tail, queries, queries.rows):
                return model._layer(model.last_layer, state, queries)

        with (
            _token_graphs_off(forward),
            rows_of(metadata, plan.tail, rows=plan.tail_rows),
        ):
            last = run_decoder_stages(
                forward,
                stages,
                lambda: layer.ffn_forward(
                    layer.attention_forward(normed, None, None, rope), *seam
                ),
                model._late_layers,
                last_layer,
            )
        return last.collapse().squeeze(0)
