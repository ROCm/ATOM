# SPDX-License-Identifier: MIT
"""The ParallelConfig the SGLang plugin builds in each scheduler process.

Under dp-attention SGLang starts one scheduler process per GPU, and each one
builds its own ATOM ParallelConfig with its global DP rank. ParallelConfig's
constructor reads data_parallel_rank as the start of this node's slice and
rejects rank + data_parallel_size_local > data_parallel_size, so passing the
process rank there failed every rank but 0 at server start.
"""

import pytest

from atom.plugin.config import (
    _build_sglang_parallel_config,
    _normalize_sglang_parallel_config,
)


@pytest.fixture(autouse=True)
def _no_dp_env(monkeypatch):
    for name in (
        "ATOM_DP_SIZE",
        "ATOM_DP_RANK",
        "ATOM_DP_RANK_LOCAL",
        "ATOM_DP_SIZE_LOCAL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("tp_size", "tp_rank"),
    [(4, rank) for rank in range(4)] + [(8, rank) for rank in range(8)],
)
def test_dp_attention_every_rank_builds(tp_size, tp_rank):
    _, _, dp_size, dp_rank, _ = _normalize_sglang_parallel_config(
        tp_size=tp_size,
        dp_size=tp_size,
        tp_rank=tp_rank,
        enable_dp_attention=True,
    )

    parallel_config = _build_sglang_parallel_config(dp_size, dp_rank)

    assert parallel_config.data_parallel_size == tp_size
    assert parallel_config.data_parallel_size_local == tp_size
    assert parallel_config.data_parallel_rank == tp_rank
    assert parallel_config.data_parallel_rank_local == tp_rank


@pytest.mark.parametrize("dp_size", [1, 2])
def test_without_dp_attention_has_no_internal_dp_group(dp_size):
    _, _, atom_dp_size, atom_dp_rank, _ = _normalize_sglang_parallel_config(
        tp_size=8,
        dp_size=dp_size,
        tp_rank=5,
        enable_dp_attention=False,
    )

    parallel_config = _build_sglang_parallel_config(atom_dp_size, atom_dp_rank)

    assert parallel_config.data_parallel_size == 1
    assert parallel_config.data_parallel_size_local == 1
    assert parallel_config.data_parallel_rank == 0
    assert parallel_config.data_parallel_rank_local == 0
