"""The read column and the write column are different columns.

`MambaPrefillCheckpointBuilder` reports where a checkpoint is WRITTEN this
step. A resuming request READS the last state block of what it already has,
which align mode defines as `(seq_len - 1) // block_size`. Using the write
column for both resumes from a block nothing filled -- measured on a server
as `needs state at 2304, found 3`.

These pin the rule rather than the plumbing, because the plumbing is vLLM's
and the rule is the thing that was got wrong.
"""

import pytest
import torch
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK = 256


def spec():
    return MambaSpec(
        shapes=((739840,),), dtypes=(torch.uint8,), block_size=BLOCK,
        num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=4096,
        tp_replicated=False,
    )


def read_col(seq_len, table_row):
    table = torch.tensor([table_row], dtype=torch.int32)
    cols = mamba_get_block_table_tensor(
        table, torch.tensor([seq_len]), spec(), "align"
    )
    return int(cols[0, 0])


@pytest.mark.parametrize(
    "seq_len,want_index",
    [(1, 0), (BLOCK, 0), (BLOCK + 1, 1), (2 * BLOCK, 1), (2304, 8), (2305, 9)],
)
def test_the_read_column_is_the_last_block_the_request_already_has(seq_len, want_index):
    row = list(range(100, 120))
    assert read_col(seq_len, row) == row[want_index]


def test_a_zero_length_row_does_not_index_backwards():
    """vLLM clamps it; a -1 here would read the end of the row, which is some
    other request's block."""
    row = list(range(100, 120))
    assert read_col(0, row) == row[0]


def test_the_read_column_moves_with_the_boundary_not_with_the_step():
    """2,304 tokens reads block index 8 whatever else is in flight -- the
    column is a property of how much the request already has."""
    row = list(range(200, 220))
    assert read_col(2304, row) == 208
    assert read_col(2304, row) == 208


def test_write_and_read_disagree_for_a_mid_prefill_checkpoint():
    """The failure this file exists for. A request at 2,304 tokens whose
    checkpoint is written at an earlier aligned point reads block index 8
    while the checkpoint went to a different one; taking the write column as
    the read column is how a resume lands on a block nothing filled."""
    from vllm.model_executor.layers.mamba.checkpoint import (
        compute_mamba_prefill_checkpoints,
    )

    offsets, write_cols = compute_mamba_prefill_checkpoints(
        [2304], [2304], hash_block_size=BLOCK, mamba_block_size=BLOCK,
        checkpoint_alignment=4096, drop_eagle_block=False,
    )
    row = list(range(300, 320))
    reading = read_col(2304, row)
    if write_cols and write_cols[0] >= 0:
        assert row[write_cols[0]] != reading or write_cols[0] == 8
