# SPDX-License-Identifier: MIT
"""ATOM_SAMPLER_SPLIT_V: which calls `Sampler` sends to the split-vocabulary
sampler, and with what noise (CPU; the kernels are in test_sampler_split.py)."""

import pytest
import torch
from aiter_stub import atom_under_stubs

V = 64


@pytest.fixture
def sampler_mod():
    with atom_under_stubs() as load:
        yield load("atom.model_ops.sampler")


@pytest.fixture
def calls(sampler_mod, monkeypatch):
    """Route every sampling kernel to a recorder; yields `[(kernel, noise)]`."""
    seen = []

    def split_sample(logits, exponential, temperatures, *, eps, greedy_eps):
        seen.append(("split", exponential.clone()))
        return torch.zeros(logits.shape[0], dtype=torch.int32)

    def mixed_sample_outer_exponential(out, logits, exponential, temps, eps=0.0):
        seen.append(("shared", exponential.clone()))
        out.zero_()

    def fused_mixed_sample(logits, exponential, temps, *, eps, greedy_eps):
        seen.append(("fused", exponential.clone()))
        return torch.zeros(logits.shape[0], dtype=torch.int32)

    def topk_select(logits, k, *, tie):
        return None, torch.zeros((logits.shape[0], 1), dtype=torch.int32)

    monkeypatch.setattr(sampler_mod, "split_sample", split_sample)
    monkeypatch.setattr(
        sampler_mod, "mixed_sample_outer_exponential", mixed_sample_outer_exponential
    )
    monkeypatch.setattr(sampler_mod, "fused_mixed_sample", fused_mixed_sample)
    monkeypatch.setattr(sampler_mod, "topk_select", topk_select)
    with torch.random.fork_rng(devices=[]):
        yield seen


def _sampler(sampler_mod, monkeypatch, enabled, max_reqs=8):
    monkeypatch.setenv("ATOM_SAMPLER_SPLIT_V", "1" if enabled else "0")
    monkeypatch.setenv("ATOM_SAMPLER_SPLIT_V_MAX_REQS", str(max_reqs))
    return sampler_mod.Sampler()


def _inputs(rows):
    return torch.randn(rows, V), torch.ones(rows)


@pytest.mark.parametrize("independent", [False, True])
def test_split_path_draws_the_noise_the_aiter_path_draws(
    sampler_mod, calls, monkeypatch, independent
):
    """Same RNG stream: the split sampler sees exactly the aiter path's noise."""
    logits, temps = _inputs(4)
    reference = _sampler(sampler_mod, monkeypatch, enabled=False)
    torch.manual_seed(0)
    reference(logits, temps, needs_independent_noise=independent)
    split = _sampler(sampler_mod, monkeypatch, enabled=True)
    torch.manual_seed(0)
    split(logits, temps, needs_independent_noise=independent)
    (ref_kernel, ref_noise), (kernel, noise) = calls
    assert ref_kernel == ("fused" if independent else "shared")
    assert kernel == "split"
    assert torch.equal(noise, ref_noise)


def test_flag_off_never_splits(sampler_mod, calls, monkeypatch):
    sampler = _sampler(sampler_mod, monkeypatch, enabled=False)
    sampler(*_inputs(2))
    sampler(*_inputs(2), needs_independent_noise=True)
    assert [k for k, _ in calls] == ["shared", "fused"]


def test_gate_on_requests(sampler_mod, calls, monkeypatch):
    sampler = _sampler(sampler_mod, monkeypatch, enabled=True, max_reqs=2)
    sampler(*_inputs(2))  # 2 rows, 2 requests: split
    sampler(*_inputs(3))  # 3 requests: aiter
    sampler(*_inputs(16), num_requests=2)  # 16 rows of 2 requests: split
    assert [k for k, _ in calls] == ["split", "shared", "split"]


def test_verification_counts_requests_not_rows(sampler_mod, calls, monkeypatch):
    """A verification call of 2 requests x 8 draft rows is 2 requests."""
    sampler = _sampler(sampler_mod, monkeypatch, enabled=True, max_reqs=2)
    logits, _ = _inputs(16)
    cu = torch.tensor([8, 16])
    sampler.sample_verification_tokens(logits, cu, torch.ones(2), None, None)
    cu3 = torch.tensor([6, 11, 16])
    sampler.sample_verification_tokens(logits, cu3, torch.ones(3), None, None)
    assert [k for k, _ in calls] == ["split", "fused"]


def test_filtered_and_greedy_calls_keep_their_paths(sampler_mod, calls, monkeypatch):
    sampler = _sampler(sampler_mod, monkeypatch, enabled=True)
    sampler(*_inputs(2), all_greedy=True)
    assert calls == []
