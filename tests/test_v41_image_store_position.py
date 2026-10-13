"""An image is stored only where the state it holds actually belongs.

`MambaPrefillCheckpointMetadata.checkpoint_offsets` says where INSIDE the
query the state must be captured -- vLLM calls it a mid-prefill checkpoint.
The store runs after the forward, so what it holds is the state at the END
of the query. Those coincide only when the offset is the query length.

Writing it anywhere else labels the state at 1362 tokens as the state at
1280, and nothing downstream can tell: the cursor contract compares the
position a request WANTS against the one the slot reports, and a restored
image sets both. This is the one defect in this mechanism that produces a
wrong answer rather than a dead engine, and it is what the single image
this branch ever stored contained.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.plugin.vllm import v41_state_image as mod


class _Meta:
    def __init__(self, offsets, blocks):
        self.checkpoint_offsets = torch.tensor(offsets)
        self.state_indices = torch.tensor(blocks)


def _publish(monkeypatch, offsets, blocks, query_lens):
    monkeypatch.setattr(mod, "image_block_table", lambda: torch.zeros(4, 32))
    monkeypatch.setattr(mod, "image_spec", lambda cfg: object())
    monkeypatch.setattr(
        mod, "_checkpoint_builder",
        lambda cfg, spec: SimpleNamespace(build=lambda m, rows: _Meta(offsets, blocks)),
    )
    starts = torch.tensor([0] + list(torch.tensor(query_lens).cumsum(0)))
    m = SimpleNamespace(query_start_loc_cpu=starts, query_start_loc=starts)
    mod.publish_image_halves(SimpleNamespace(), m, len(query_lens))
    return mod.take_store_marks()


def test_an_end_of_query_checkpoint_is_stored(monkeypatch):
    marks = _publish(monkeypatch, [1362], [77], [1362])
    assert marks == [77]


def test_an_interior_checkpoint_is_declined(monkeypatch):
    """1280 inside a 1362-token query: the post-forward state is not it."""
    before = mod.restore_declines()["image_write_col_interior"]
    marks = _publish(monkeypatch, [1280], [77], [1362])
    assert marks == [-1]
    assert mod.restore_declines()["image_write_col_interior"] == before + 1


def test_rows_are_judged_one_at_a_time(monkeypatch):
    marks = _publish(monkeypatch, [512, 1280], [11, 22], [512, 1362])
    assert marks == [11, -1]
