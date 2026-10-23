# SPDX-License-Identifier: MIT
"""When the decoder on a tail records, replays or runs (`decoder_graph.py`)."""

from types import SimpleNamespace

import pytest
import torch

# the tail's row gather is a Triton kernel
pytest.importorskip("triton")

from atom.models.deepseek_v41 import decoder_graph
from atom.models.deepseek_v41.decoder_graph import DecoderGraph


class _Graph:
    """A `ReqGraph` that logs instead of capturing."""

    def __init__(self, recorded=()):
        self.recorded, self.log = set(recorded), []
        self.recorded_inputs = (torch.zeros(256, 4), torch.zeros(256, 4))

    def has(self, key, phase=None):
        return key in self.recorded

    def inputs(self, key, phase=None):
        return self.recorded_inputs

    def record(self, forward, args, kwargs=None, key=None, phase=None):
        self.log.append(("record", key))
        return forward(*args)

    def replay(self, args, kwargs=None, key=None, phase=None):
        self.log.append(("replay", key))
        # the recording's own inputs, gathered into in place: nothing to copy
        assert args is self.recorded_inputs
        return "replayed"


_FINISHING = (object(),)


def _gather(rows, *tensors, dim, out=None):
    return tensors if out is None else out


def _run(
    monkeypatch,
    *,
    recorded=(),
    mode=None,
    requests=_FINISHING,
    group_replays=True,
    specs=(),
):
    monkeypatch.setattr(decoder_graph, "gather_rows", _gather)
    graph = DecoderGraph(SimpleNamespace(decoder_specs=specs, parallel_config=None))
    graph.graph = _Graph(recorded)
    graph._decode = lambda forward, plan, stages, rows, fused: "decoded"
    step = SimpleNamespace(indptrs={1: 0}, tiles={1: 0}, selected={}, candidates={})
    forward = SimpleNamespace(
        tr_graph=mode,
        context=SimpleNamespace(
            is_prefill=True,
            forward_mode=SimpleNamespace(req_graphs=group_replays),
        ),
        dp_metadata=None,
        attn_metadata=SimpleNamespace(step=step),
    )
    plan = SimpleNamespace(
        tail=SimpleNamespace(width=256, requests=requests),
        tail_rows=None,
        last=torch.zeros(2),
    )
    rows = torch.zeros(1, 256, 4)
    out = graph.run(forward, plan, rows, [rows])
    return out, graph.graph.log, step


def test_the_request_bucket_pass_records_each_bucket_under_its_key(monkeypatch):
    out, log, _ = _run(monkeypatch, mode="record_req")
    assert out == "decoded" and log == [("record", (2, True))]
    # a tail of padding records apart: its attention sees no request
    _, log, _ = _run(monkeypatch, mode="record_req", requests=())
    assert log == [("record", (2, False))]


def test_a_recorded_bucket_replays_and_releases_the_outer_step(monkeypatch):
    out, log, step = _run(monkeypatch, recorded={(2, True)})
    assert out == "replayed" and log == [("replay", (2, True))]
    # what `rows_of` does on its way out, which a replay skips
    assert step.indptrs == {} and step.tiles == {}


@pytest.mark.parametrize(
    "kw",
    [
        {},  # no recording for this bucket
        {"recorded": {(2, False)}},  # only the idle one
        # the DP group refuses graphs this step (an image prefill on any rank,
        # a decoding peer): every rank runs it, not this one alone
        {"recorded": {(2, True)}, "group_replays": False},
        {"recorded": {(2, True)}, "mode": "record"},  # a token-bucket pass
    ],
)
def test_a_step_no_recording_serves_runs_directly(monkeypatch, kw):
    out, log, _ = _run(monkeypatch, **kw)
    assert out == "decoded" and log == []


def test_a_fused_quant_tail_gathers_its_three_planes_together(monkeypatch):
    """`prepare_attention` under fused quantization hands BF16 rows with their
    FP8 rows and e8m0 scales: all three ride the tail's gather, the scales as
    bytes, and the decoder gets the triple back with its own dtypes."""
    gathered, seen = [], {}

    def gather(rows, *tensors, dim, out=None):
        gathered.append(tuple(t.dtype for t in tensors))
        return tensors

    monkeypatch.setattr(decoder_graph, "gather_rows", gather)
    graph = DecoderGraph(SimpleNamespace(decoder_specs=(), parallel_config=None))
    graph.graph = _Graph()

    def decode(forward, plan, stages, rows, fused):
        seen["rows"], seen["fused"] = rows, fused
        return "decoded"

    graph._decode = decode
    forward = SimpleNamespace(
        tr_graph=None,
        context=SimpleNamespace(forward_mode=SimpleNamespace(req_graphs=True)),
        dp_metadata=None,
        attn_metadata=SimpleNamespace(step=None),
    )
    plan = SimpleNamespace(
        tail=SimpleNamespace(width=4, requests=_FINISHING),
        tail_rows=None,
        last=torch.zeros(1),
    )
    hidden = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    quant = torch.zeros(1, 4, 8, dtype=torch.float8_e4m3fn)
    scale = torch.full((1, 4, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    seam = torch.zeros(1, 4, 2)

    graph.run(forward, plan, (hidden, quant, scale), [seam])

    assert gathered == [(torch.bfloat16, torch.float8_e4m3fn, torch.uint8, seam.dtype)]
    assert seen["fused"] is True and len(seen["rows"]) == 4


def test_a_decoder_reading_the_encoders_selection_never_records(monkeypatch):
    """That selection is held only where a rank has requests, so a replay
    would split the group: a topology that carries one runs directly."""
    carried = (SimpleNamespace(layer_id=20, topk_owner=19, candidate_source=None),)
    out, log, _ = _run(monkeypatch, mode="record_req", specs=carried)
    assert out == "decoded" and log == []
