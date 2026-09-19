# SPDX-License-Identifier: MIT
"""Which MoE layers get a side stream for the shared expert, and which do not.

The shared expert cannot be folded into a routed slot on any V4/V4.1
checkpoint -- it is FP8 where the routed experts are FP4 -- so it is a module
of its own reading the same rows the routed pass reads. A stream is how the
two come to run at once, and a stream that never reaches the layer is the
whole feature missing with nothing else to show for it.

`torch.cuda` is forced available here rather than read off the machine: CI has
no GPU, and a test that compared `None` to `None` would pass just as well with
the stream unplumbed.
"""

import pytest
import torch


class FakeStream:
    """Stands in for `torch.cuda.Stream`; identity is all these tests read."""


@pytest.fixture
def created_streams(monkeypatch):
    made = []

    def make():
        made.append(FakeStream())
        return made[-1]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", make)
    return made


@pytest.mark.parametrize("entrypoint", ["runtime", "offline"])
def test_backbone_gives_every_moe_the_one_stream_it_made(
    single_rank, unallocated_moe, build_v41, created_streams, entrypoint
):
    instance = build_v41(entrypoint)

    # One per model, not one per layer: a layer's attention is done before its
    # MoE starts and layers do not overlap, so the layers cannot contend.
    assert len(created_streams) == 1
    assert instance.alt_stream is created_streams[0]
    assert instance.layers
    for block in instance.layers:
        assert block.ffn.alt_stream is instance.alt_stream


@pytest.mark.parametrize("entrypoint", ["draft", "draft_offline"])
def test_draft_keeps_its_shared_expert_on_the_one_stream(
    single_rank, unallocated_moe, build_v41, created_streams, entrypoint
):
    """The draft does not fork, as V4's own DSpark layers do not.

    Not an oversight to be tidied up: forking here puts a second stream inside
    the propose graph's capture, which is a change with its own measurement to
    do. Pinned so that doing it is a decision rather than a side effect of
    touching the backbone.
    """
    instance = build_v41(entrypoint)

    assert created_streams == []
    assert instance.mtp
    for block in instance.mtp:
        assert block.ffn.alt_stream is None
