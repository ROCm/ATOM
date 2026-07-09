# SPDX-License-Identifier: MIT
# CPP A·P0 — layer partition unit tests (pure, GPU-free).

import sys
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from atom.models.utils import get_pp_indices, make_layers, PPMissingLayer


class _RealLayer(nn.Module):
    def __init__(self, layer_num):
        super().__init__()
        self._layer_num = layer_num


def _partitions(num_layers, pp_size):
    """Reconstruct per-rank (start, end) intervals."""
    return [get_pp_indices(num_layers, r, pp_size) for r in range(pp_size)]


@pytest.mark.parametrize(
    "num_layers,pp_size",
    [(60, 8), (78, 8), (61, 4), (80, 3), (32, 1), (93, 8), (61, 2), (7, 7)],
)
def test_partition_covers_all_layers_contiguously(num_layers, pp_size):
    parts = _partitions(num_layers, pp_size)

    # first starts at 0, last ends at num_layers
    assert parts[0][0] == 0
    assert parts[-1][1] == num_layers

    # contiguous, non-overlapping, every partition non-empty
    for i in range(pp_size):
        start, end = parts[i]
        assert end > start, f"empty partition at rank {i}: {parts[i]}"
        if i > 0:
            assert start == parts[i - 1][1], f"gap/overlap at rank {i}: {parts}"

    # union == all layers
    covered = sum(end - start for start, end in parts)
    assert covered == num_layers


@pytest.mark.parametrize("num_layers,pp_size", [(61, 4), (78, 8), (80, 3), (93, 8)])
def test_remainder_never_lands_on_last_partition(num_layers, pp_size):
    """Last partition holds the norm/lm_head; it must keep the base size."""
    base = num_layers // pp_size
    parts = _partitions(num_layers, pp_size)
    last_size = parts[-1][1] - parts[-1][0]
    assert last_size == base


def test_env_override(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "15,15,15,16")
    assert get_pp_indices(61, 0, 4) == (0, 15)
    assert get_pp_indices(61, 1, 4) == (15, 30)
    assert get_pp_indices(61, 3, 4) == (45, 61)


def test_env_override_bad_count(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "15,15,31")
    with pytest.raises(ValueError):
        get_pp_indices(61, 0, 4)


def test_env_override_bad_sum(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "15,15,15,15")
    with pytest.raises(ValueError):
        get_pp_indices(61, 0, 4)


@pytest.mark.parametrize("pp_rank,pp_size", [(0, 4), (1, 4), (2, 4), (3, 4)])
def test_make_layers_places_real_and_missing(monkeypatch, pp_rank, pp_size):
    num_layers = 61
    grp = MagicMock()
    grp.rank_in_group = pp_rank
    grp.world_size = pp_size
    fake_ps = MagicMock()
    fake_ps.get_pp_group.return_value = grp
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", fake_ps)

    built = []

    def layer_fn(prefix, layer_num):
        built.append(layer_num)
        return _RealLayer(layer_num)

    start, end, modules = make_layers(num_layers, layer_fn, prefix="layers")

    assert (start, end) == get_pp_indices(num_layers, pp_rank, pp_size)
    assert len(modules) == num_layers
    # in-range are real layers, out-of-range are PPMissingLayer
    for idx in range(num_layers):
        if start <= idx < end:
            assert not isinstance(modules[idx], PPMissingLayer)
            assert modules[idx]._layer_num == idx
        else:
            assert isinstance(modules[idx], PPMissingLayer)
    # only in-range layers were actually constructed
    assert built == list(range(start, end))
