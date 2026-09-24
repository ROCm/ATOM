# SPDX-License-Identifier: MIT
"""The fork/join edges `side_stream` is responsible for emitting.

Every caller of it forks one branch of a layer beside another, and both ways
of getting it wrong leave a model that still answers: a fork that never
happens is the overlap silently absent, a join that never happens is a read
racing a write that a captured graph then replays forever. Neither shows up in
an output, so the edges are what is pinned.

Nothing here touches CUDA. The streams are recorders, because what is under
test is which wait is asked of which stream and in what order.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.utils import forward_context
from atom.utils.forward_context import side_stream


class RecordingStream:
    """Stands in for `torch.cuda.Stream`, writing down the edges asked of it."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def wait_stream(self, other):
        self.log.append(f"{self.name} waits {other.name}")


@pytest.fixture
def recorded(monkeypatch):
    """A main and a side stream, and the log of every edge and region."""
    log = []
    main, side = RecordingStream(log, "main"), RecordingStream(log, "side")

    class Region:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            log.append(f"enter {self.stream.name}")

        def __exit__(self, *_):
            log.append(f"leave {self.stream.name}")

    monkeypatch.setattr(torch.cuda, "stream", Region)
    return log, main, side


def use(monkeypatch, main, *, in_hipgraph):
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(in_hipgraph=in_hipgraph, main_stream=main),
    )


def test_forking_waits_on_the_main_stream_and_reports_how_to_rejoin(
    monkeypatch, recorded
):
    log, main, side = recorded
    use(monkeypatch, main, in_hipgraph=True)

    with side_stream(side) as (issuing, joining):
        log.append("work")
        assert issuing is side
        assert joining is main

    # The side stream waits before anything is issued to it; the caller is
    # handed the main stream to wait on it afterwards.
    assert log == ["side waits main", "enter side", "work", "leave side"]


@pytest.mark.parametrize(
    "reason,in_hipgraph,stream",
    [("outside the capture loop", False, True), ("no stream to fork", True, False)],
)
def test_declining_leaves_the_work_where_it_was(
    monkeypatch, recorded, reason, in_hipgraph, stream
):
    """Eager launches would pile up across layers with nothing to drain them,
    and a machine with no CUDA has no stream at all. Either way the work still
    runs, in the order it ran in before there was a stream, and there is
    nothing for the caller to rejoin."""
    log, main, side = recorded
    use(monkeypatch, main, in_hipgraph=in_hipgraph)

    with side_stream(side if stream else None) as (issuing, joining):
        log.append("work")
        assert issuing is main, reason
        assert joining is None, reason

    assert log == ["work"], reason
