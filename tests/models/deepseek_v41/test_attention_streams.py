# SPDX-License-Identifier: MIT
"""The two side streams the attention forks, and where each one rejoins.

One stream per branch of the layer's dependency graph. The compressor reads
the hidden row and its own arena state, so it is issued before the
projections; what it writes is an index row whose first reader is the scorer,
so it rejoins there. The indexer is the critical path and moving it buys
nothing by itself -- what it buys is the Q/KV chain no longer sitting in front
of it on the main stream, which is why it gets a stream of its own and rejoins
before `_indices`.

Two positions and two join points are the whole feature: issue either later
and it overlaps nothing, join either later and something waits for no reason.
A fork that never happens and a join that never happens both leave a model
that still answers -- the first is the feature silently absent, the second a
read racing a write the captured graph then replays forever. So the edges are
what is pinned here, not the result.

`torch.cuda` is forced available: CI has no GPU, and a test comparing `None`
to `None` would pass just as well with the streams unplumbed.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from atom.models.deepseek_v41.attention import Attention
from atom.utils import forward_context


class FakeStream:
    """Stands in for `torch.cuda.Stream`; identity is all the plumbing reads."""


class RecordingStream:
    """A stream that writes down the ordering edges asked of it."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def wait_stream(self, other):
        self.log.append(f"{self.name} waits {other.name}")


@pytest.fixture
def created_streams(monkeypatch):
    made = []

    def make():
        made.append(FakeStream())
        return made[-1]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", make)
    return made


@pytest.fixture
def recorded(monkeypatch):
    """Main, compress and index streams, plus the log of every edge issued."""
    log = []
    streams = SimpleNamespace(
        main=RecordingStream(log, "main"),
        compress=RecordingStream(log, "compress"),
        index=RecordingStream(log, "index"),
    )

    @contextlib.contextmanager
    def enter(stream):
        log.append(f"enter {stream.name}")
        yield
        log.append(f"leave {stream.name}")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: streams.main)
    monkeypatch.setattr(torch.cuda, "stream", enter)
    return log, streams


def context(monkeypatch, main, *, in_hipgraph):
    """Feed the real `side_stream` a forward context, rather than stubbing it.

    The edges under test are that helper's, so replacing it would leave the
    thing being described untested.
    """
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(in_hipgraph=in_hipgraph, main_stream=main),
    )


def stub(log, *, compress=None, index=None, indexer=True):
    """An attention whose work only says that it ran, and in what order."""
    return SimpleNamespace(
        indexer=(
            SimpleNamespace(
                project=lambda *args: (log.append("project"), ("q", "w"))[1],
                score=lambda *args: log.append("score"),
            )
            if indexer
            else None
        ),
        compress_stream=compress,
        index_stream=index,
        _compress_batch=lambda *args: log.append("compress"),
    )


@pytest.mark.parametrize("entrypoint", ["runtime", "offline"])
def test_the_backbone_makes_three_distinct_streams_and_hands_two_down(
    monkeypatch, single_rank, unallocated_moe, build_v41, created_streams, entrypoint
):
    """Distinct, because a stream serializes what it carries.

    The MoE's shared expert has the third; sharing any of them would make one
    branch queue behind another it has no dependency on.
    """

    class RecordingAttention(torch.nn.Module):
        def __init__(self, config, spec, *, compress_stream=None, index_stream=None):
            super().__init__()
            self.given = (compress_stream, index_stream)

    from atom.models.deepseek_v41 import model as model_module

    monkeypatch.setattr(model_module.Block, "attention_cls", RecordingAttention)
    instance = build_v41(entrypoint)

    # Three for the model, one per concurrent pair, not three per layer.
    assert len(created_streams) == 3
    assert len({id(s) for s in created_streams}) == 3
    assert instance.layers
    for block in instance.layers:
        assert block.attn.given == (instance.compress_stream, instance.index_stream)


def test_compressor_forks_before_the_projections(monkeypatch, recorded):
    log, streams = recorded
    context(monkeypatch, streams.main, in_hipgraph=True)

    forked = Attention._fork_compress(stub(log, compress=streams.compress), *[None] * 4)

    assert log == [
        "compress waits main",
        "enter compress",
        "compress",
        "leave compress",
    ]
    # A bool, not a stream: the join belongs to whoever runs the scorer.
    assert forked is True


def test_indexer_forks_and_carries_the_compressor_join_into_its_own_stream(
    monkeypatch, recorded
):
    log, streams = recorded
    context(monkeypatch, streams.main, in_hipgraph=True)
    instance = stub(log, compress=streams.compress, index=streams.index)

    joined = Attention._fork_select(instance, *[None] * 6, compressed=True)

    # The compressor is waited for between the two halves: after the
    # projections that do not read what it wrote, before the scorer that
    # does -- and on the indexer's own stream, not the main one.
    assert log == [
        "index waits main",
        "enter index",
        "project",
        "index waits compress",
        "score",
        "leave index",
    ]
    assert joined is streams.main


def test_a_layer_without_an_indexer_forks_nothing_to_select(monkeypatch, recorded):
    """And so needs no join: only the mode that gives a layer a compressor
    gives it an indexer, so there is never one outstanding here."""
    log, streams = recorded
    context(monkeypatch, streams.main, in_hipgraph=True)
    instance = stub(log, compress=streams.compress, index=streams.index, indexer=None)

    assert Attention._fork_select(instance, *[None] * 6, compressed=False) is None
    assert log == []


@pytest.mark.parametrize(
    "reason,in_hipgraph,streamed",
    [("eager", False, True), ("no stream", True, False)],
)
def test_neither_forks_when_it_may_not(
    monkeypatch, recorded, reason, in_hipgraph, streamed
):
    """Outside the capture loop the launches have nothing to drain them.

    Eager mode would pile side-stream work up across layers; the recorded
    graph instead carries the edges and replays them. A machine with no CUDA
    has no stream to fork onto at all. Either way the work still runs, in the
    order it ran in before there were streams.
    """
    log, streams = recorded
    context(monkeypatch, streams.main, in_hipgraph=in_hipgraph)
    instance = stub(
        log,
        compress=streams.compress if streamed else None,
        index=streams.index if streamed else None,
    )

    forked = Attention._fork_compress(instance, *[None] * 4)
    joined = Attention._fork_select(instance, *[None] * 6, compressed=forked)

    assert log == ["compress", "project", "score"], reason
    assert forked is False, reason
    assert joined is None, reason
