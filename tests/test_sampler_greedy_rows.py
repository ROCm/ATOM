# SPDX-License-Identifier: MIT
"""Greedy correction preserves rows without a device-to-host mask decision."""

import pytest
import torch

sampler = pytest.importorskip(
    "atom.model_ops.sampler",
    reason="sampler requires the Triton/AITER runtime",
    exc_type=ImportError,
)


def _torch_topk_select(probs, k, *, tie):
    assert k == 1 and tie == "low"
    return None, probs.argmax(-1, keepdim=True).to(torch.int32)


@pytest.mark.parametrize("rows,vocab", [(1, 8), (4, 16), (17, 61), (64, 129)])
@pytest.mark.parametrize("frac", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("column", [False, True])
def test_greedy_correction_preserves_sampled_rows(
    monkeypatch, rows, vocab, frac, column
):
    monkeypatch.setattr(sampler, "topk_select", _torch_topk_select)
    probs = torch.rand(rows, vocab, generator=torch.Generator().manual_seed(31))
    temperatures = torch.ones(rows)
    # Scattered zero-temperature rows, not just a prefix.
    temperatures[torch.randperm(rows)[: int(rows * frac)]] = 0
    sampled = torch.full((rows, 1) if column else (rows,), -1, dtype=torch.long)
    got = sampler._apply_greedy_tokens(probs, temperatures, sampled)
    expected = torch.tensor(
        [
            int(probs[row].argmax()) if temperatures[row] == 0 else -1
            for row in range(rows)
        ],
        dtype=torch.int32,
    )
    assert got.shape == (rows,) and got.dtype == torch.int32
    assert torch.equal(got, expected)
    assert (sampled == -1).all()  # Caller-owned sampled results are not mutated.


def _noise_picks_the_high_tie(logits, temperatures, needs_independent_noise=False):
    """Stand-in for the Gumbel draw: on an exact tie it may land on either id."""
    return torch.full((logits.shape[0],), 5, dtype=torch.int)


@pytest.mark.parametrize("filters", [(None, None), (None, 0.9)])
def test_all_greedy_batch_takes_the_low_tie_whatever_its_filters(monkeypatch, filters):
    # Temperatures as the runner hands them over: clamped to SAMPLER_EPS.
    monkeypatch.setattr(sampler, "topk_select", _torch_topk_select)
    instance = sampler.Sampler()
    monkeypatch.setattr(instance, "_temperature_sample", _noise_picks_the_high_tie)
    logits = torch.zeros(3, 8)
    logits[:, 2] = logits[:, 5] = 1.0
    temperatures = torch.full((3,), sampler.SAMPLER_EPS)
    got = instance(logits, temperatures, *filters, all_greedy=True)
    assert got.tolist() == [2, 2, 2]


def test_mixed_batch_greedy_rows_take_the_low_tie(monkeypatch):
    monkeypatch.setattr(sampler, "topk_select", _torch_topk_select)
    instance = sampler.Sampler()
    monkeypatch.setattr(instance, "_temperature_sample", _noise_picks_the_high_tie)
    logits = torch.zeros(4, 8)
    logits[:, 2] = logits[:, 5] = 1.0
    temperatures = torch.tensor([sampler.SAMPLER_EPS, 0.7, sampler.SAMPLER_EPS, 1.0])
    got = instance(logits, temperatures, None, None, all_greedy=False)
    assert got.tolist() == [2, 5, 2, 5]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("path", ["aiter", "native"])
@pytest.mark.parametrize("greedy", ["none", "mixed", "all"])
def test_gpu_sampling_greedy_rows_and_no_mask_readback(path, greedy):
    # Real reducers and samplers. One-hot stochastic rows make their expected
    # samples exact without depending on RNG state; greedy rows have ties.
    rows, vocab = 8, 4096
    probs = torch.zeros(rows, vocab, device="cuda")
    probs[:, 7] = 1
    temperatures = torch.ones(rows, device="cuda")
    indices = {"none": [], "mixed": [1, 4, 6], "all": list(range(rows))}[greedy]
    for row in indices:
        temperatures[row] = 0
        probs[row, 3] = 1
    probs /= probs.sum(-1, keepdim=True)
    expected = torch.full((rows,), 7, dtype=torch.int32, device="cuda")
    for row in indices:
        expected[row] = 3
    top_ps = torch.ones(rows, device="cuda")  # Avoid independent scalar .item().
    instance = sampler.Sampler()

    def run():
        if path == "aiter":
            return instance._aiter_sample(
                probs, None, top_ps, False, True, temperatures
            )
        return instance._native_sample(probs, None, top_ps, temperatures)

    run()  # Compile before measuring.
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        result = run()
    assert result.shape == (rows,) and result.dtype == torch.int32
    assert torch.equal(result, expected)
    names = {event.key for event in profile.key_averages()}
    assert not names.intersection({"aten::is_nonzero", "aten::nonzero"})
    # RNG/sampling internals can extract their own scalar values. The greedy
    # correction itself must not extract any scalar, even for an empty mask.
    with torch.profiler.profile() as correction_profile:
        sampler._apply_greedy_tokens(probs, temperatures, result)
    correction_names = {event.key for event in correction_profile.key_averages()}
    assert not correction_names.intersection(
        {"aten::is_nonzero", "aten::nonzero", "aten::item", "aten::_local_scalar_dense"}
    )
