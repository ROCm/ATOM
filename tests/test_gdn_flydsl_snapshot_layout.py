"""CPU contracts for the layout of the GDN prefill's chunk-state snapshot."""

import pytest
import torch


def test_checkpoint_writer_rejects_an_unknown_snapshot_layout():
    pytest.importorskip("triton", reason="GDN checkpoints require Triton")
    from atom.model_ops.fla_ops.state_checkpoint import write_state_checkpoints

    pool = torch.zeros(2, 1, 4, 3)
    # Neither [K, V] nor [V, K] contiguous per state: every other column.
    strided = torch.zeros(1, 2, 1, 4, 6)[..., ::2]
    one = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="neither a KV nor a VK layout"):
        write_state_checkpoints(
            strided,
            pool,
            torch.zeros(64, 2),
            torch.zeros(2, 2, 3),
            one,
            one,
            one,
            one,
            one,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 64], dtype=torch.int32),
            64,
            3,
        )
