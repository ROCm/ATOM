# SPDX-License-Identifier: MIT
"""Which side stream each model's MoE layers get for the shared expert.

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
def test_draft_forks_on_the_stream_it_is_handed(
    single_rank, unallocated_moe, build_v41, created_streams, entrypoint
):
    """The draft forks too, on the backbone's stream rather than one of its own.

    The two models never run at once, so a second handle would buy nothing;
    serving hands this one down from `DSparkProposer`, which is where the
    backbone is. Building none of its own is half the contract -- a draft that
    quietly made one would fork on a stream the backbone never waits for.

    This used to pin the opposite, on the grounds that forking would put a
    second stream inside the propose graph's capture; the backbone's own decode
    graph already does exactly that, so the capture was never the obstacle.
    """
    handed = FakeStream()
    instance = build_v41(entrypoint, alt_stream=handed)

    assert created_streams == []
    assert instance.alt_stream is handed
    assert instance.mtp
    for block in instance.mtp:
        assert block.ffn.alt_stream is handed
