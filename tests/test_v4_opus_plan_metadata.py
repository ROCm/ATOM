import pytest
import torch

rows_module = pytest.importorskip(
    "atom.model_ops.indexer.rows",
    reason="the indexer's rows import aiter at load",
    exc_type=ImportError,
)


def test_natural_rows_plan_each_band_once_off_its_own_sequences(monkeypatch):
    """gfx1250's OPUS scorer: `fp4_rows` plans each band once, at the step,
    off the band's own sequences, its rows' bounds and each row's sequence --
    never per layer."""
    calls = []

    def fake_plan(query_start_loc, row_ends, *, buffers, total_q, row_to_batch):
        assert buffers is None, "an unrecorded step plans into fresh tensors"
        calls.append((query_start_loc.tolist(), row_ends.tolist(), total_q))
        return len(calls)

    monkeypatch.setattr(rows_module, "pa_mqa_logits_mxfp4_plan", fake_plan)
    sequences = rows_module.Fp4MqaRaggedMetadata(
        torch.tensor([0, 2, 5], dtype=torch.int32),
        3,
        torch.zeros(2, 4, dtype=torch.int32),
    )
    row_ends = torch.tensor([7, 8, 3, 4, 5], dtype=torch.int32)
    rows = rows_module.fp4_rows(
        row_ends,
        64,
        None,
        sequences,
        heads=64,
        page_size=64,
        layout=rows_module.NATURAL,
        row_to_batch=torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32),
        bands=((0, 3), (3, 5)),
    )

    # Each band's sequences count from its own first row.
    assert calls == [([0, 2, 3], [7, 8, 3], 3), ([0, 0, 2], [4, 5], 2)]
    assert [band["plan"] for band in rows.fp4_bands] == [1, 2]
