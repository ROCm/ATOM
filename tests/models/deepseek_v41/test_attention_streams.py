# SPDX-License-Identifier: MIT
"""Which branches of a layer leave the main stream, and where each rejoins.

`ATOM_DSV41_SIDE_STREAMS` picks between three arrangements, and every one of
them answers correctly, so nothing about the output says which one ran. What
distinguishes them is edges: the compressor is issued before the projections
and waited at the scorer, its first reader; the indexer, where it is forked at
all, is waited before the attention reads what it selected. A fork that never
happens and a join that never happens both leave a model that still answers --
the first is the feature silently absent, the second a read racing a write
that a captured graph then replays forever. So the edges are what is pinned
here, and the streams' identities with them, since a borrowed handle and a
handle of its own are the same object graph to everything downstream.

`torch.cuda` is forced available: CI has no GPU, and a test comparing `None`
to `None` would pass just as well with the streams unplumbed.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="the attention module is built on AITER kernels")

from atom.models.deepseek_v41.attention import Attention
from atom.utils import envs, forward_context


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


class RecordingAttention(torch.nn.Module):
    """Takes the two handles and keeps them, so the plumbing is readable."""

    def __init__(self, config, spec, *, compress_stream=None, index_stream=None):
        super().__init__()
        self.given = (compress_stream, index_stream)


@pytest.mark.parametrize("entrypoint", ["runtime", "offline"])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_each_level_hands_every_attention_the_streams_it_declares(
    monkeypatch,
    single_rank,
    unallocated_moe,
    build_v41,
    created_streams,
    entrypoint,
    level,
):
    """What the model makes, and which of them reach the layer.

    0 forks nothing, and that has to reach the layer too: `side_stream`
    declines when it is handed nothing, so an attention that kept a handle
    anyway would fork on a stream the backbone never waits for. 1 and 2 both
    lend the MoE's stream to the compressor, whose live range ends a sublayer
    before the shared expert is issued -- identity is the whole claim there,
    since a handle of its own would be a hardware queue bought for an overlap
    that cannot happen. Only the indexer is a third concurrent branch, so only
    2 makes a stream, and it must not be either of the other two.
    """
    from atom.models.deepseek_v41 import model as model_module

    monkeypatch.setattr(envs, "ATOM_DSV41_SIDE_STREAMS", level, raising=False)
    monkeypatch.setattr(model_module.Block, "attention_cls", RecordingAttention)
    instance = build_v41(entrypoint)

    # One per branch that needs a queue of its own, for the model and not per
    # layer. The MoE's shared expert owns the one that is there at every level.
    assert len(created_streams) == (2 if level == 2 else 1)
    assert len({id(s) for s in created_streams}) == len(created_streams)
    expected = {
        0: (None, None),
        1: (instance.alt_stream, None),
        2: (instance.alt_stream, instance.index_stream),
    }[level]
    assert (instance.compress_stream, instance.index_stream) == expected
    if level == 2:
        assert instance.index_stream is not None
        assert instance.index_stream is not instance.alt_stream
    assert instance.layers
    for block in instance.layers:
        assert block.attn.given == expected


def test_an_unknown_level_is_refused_rather_than_rounded(
    monkeypatch, single_rank, unallocated_moe, build_v41, created_streams
):
    """Every arrangement answers, so a typo would only show up as throughput."""
    monkeypatch.setattr(envs, "ATOM_DSV41_SIDE_STREAMS", 3, raising=False)

    with pytest.raises(ValueError, match="0, 1 or 2"):
        build_v41("runtime")


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


def test_the_shipped_level_is_the_one_that_measured_fastest(monkeypatch):
    """0 is a measurement, not a preference (periods in the environment doc).

    Throughput is too coarse to see the effect that chose it, so a default
    flipped back on the strength of a benchmark would look justified and still
    be a regression.
    """
    monkeypatch.delenv("ATOM_DSV41_SIDE_STREAMS", raising=False)

    assert envs.ATOM_DSV41_SIDE_STREAMS == 0
