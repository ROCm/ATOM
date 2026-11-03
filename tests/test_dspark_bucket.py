import itertools

import pytest
import torch

from atom.spec_decode.dspark_bucket import (
    graph_row_indices,
    schedule_bucket_prefix_lengths_tensor,
)
from atom.spec_decode.dspark_scheduler import schedule_prefix_lengths_tensor


def decide(conf, sizes, qs, costs):
    table = torch.tensor([1000 / c for c in costs], dtype=torch.float32)
    rows = torch.tensor(graph_row_indices(sizes, qs, len(costs) - 1))
    return schedule_bucket_prefix_lengths_tensor(conf, table, rows, qs)


def objective(conf, ell, sizes, qs, costs):
    q = next(q for q in qs if q >= max(ell) + 1)
    n = len(ell) + sum(ell)
    graph = next(
        (g for g in sorted(sizes) if g >= n and g % q == 0 and g < len(costs)), None
    )
    if graph is None:
        return float("-inf")
    survival = conf.cumprod(1)
    expected = len(ell) + sum(float(survival[i, :e].sum()) for i, e in enumerate(ell))
    return expected / costs[graph]


def test_actual_q_divisibility_and_missing_coverage():
    rows = graph_row_indices([1, 2, 3, 4, 6, 8, 12], [1, 3, 6], 12)
    assert rows[1][4] == 4
    assert rows[2][4] == 6
    assert rows[4][7] == 12
    assert rows[3][10] == 12
    assert graph_row_indices([1, 3], [1, 3, 6], 12)[6][4] == -1


def test_global_search_crosses_plateau_without_discarding_same_graph_tokens():
    survival = torch.tensor([[0.9, 0.75, 0.5, 0.3, 0.04], [0.8, 0.6, 0.4, 0.2, 0.02]])
    conf = survival / torch.cat([torch.ones(2, 1), survival[:, :-1]], dim=1)
    # Large base cost makes all positive-survival tokens within graph 12 useful.
    costs = [10 + i for i in range(13)]
    sizes, qs = [1, 2, 3, 4, 6, 12], [1, 3, 6]
    chosen = decide(conf, sizes, qs, costs).tolist()
    best = max(
        objective(conf, e, sizes, qs, costs)
        for e in itertools.product(range(6), repeat=2)
    )
    assert objective(conf, chosen, sizes, qs, costs) == pytest.approx(best)


@pytest.mark.parametrize("seed", range(5))
def test_matches_exhaustive_allocation_oracle(seed):
    generator = torch.Generator().manual_seed(seed)
    conf = torch.rand((3, 3), generator=generator) * 0.98 + 0.01
    costs = (torch.rand(13, generator=generator).cumsum(0) + 1).tolist()
    sizes, qs = [1, 2, 3, 4, 6, 8, 12], [1, 2, 4]
    ell = decide(conf, sizes, qs, costs).tolist()
    actual = objective(conf, ell, sizes, qs, costs)
    optimum = max(
        objective(conf, e, sizes, qs, costs)
        for e in itertools.product(range(4), repeat=3)
    )
    assert actual == pytest.approx(optimum, rel=1e-6)


def test_equal_survival_keeps_prefix_order_and_is_deterministic():
    conf = torch.ones(3, 3)
    ell = decide(conf, [1, 3, 4, 6, 8, 12], [1, 2, 4], [1 + i for i in range(13)])
    assert (
        ell.tolist()
        == decide(
            conf, [1, 3, 4, 6, 8, 12], [1, 2, 4], [1 + i for i in range(13)]
        ).tolist()
    )
    assert torch.all((ell >= 0) & (ell <= 3))


def test_uncovered_batch_retains_full_lengths():
    ell = decide(torch.full((2, 3), 0.5), [], [1, 2, 4], [1.0] * 9)
    assert ell.tolist() == [3, 3]


def test_smooth_early_stop_policy_remains_available():
    conf = torch.tensor([[0.9, 0.8, 0.7]])
    table = torch.tensor([1000 / (1 + i) for i in range(5)])
    assert schedule_prefix_lengths_tensor(conf, table).tolist() == [2]
