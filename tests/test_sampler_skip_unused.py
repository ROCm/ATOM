# SPDX-License-Identifier: MIT
"""The once-per-step sampling shortcuts leave every id and every later draw
where the full path puts them (dspark_options.sampler_skip_greedy_pick,
dspark_options.verify_skip_unused_target_draw). CPU only: AITER's two sampling ops are
stood in for by reference arithmetic, the generator is torch's own."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter

sampler_module = None
runner_module = None
recipe_module = None


@pytest.fixture(scope="module", autouse=True)
def _modules():
    global sampler_module, runner_module, recipe_module
    with atom_without_aiter() as load:
        runner_module = load("atom.model_engine.model_runner")
        sampler_module = load("atom.model_ops.sampler")
        recipe_module = load("atom.plugin_recipe")
        yield


CALLS = []


def _outer_exponential(out, logits, exponential, temperatures, eps=1e-10):
    """Reference for `mixed_sample_outer_exponential`: a pure function of the
    logits, the noise and the temperatures."""
    CALLS.append("score")
    scaled = logits.float() / temperatures.float().unsqueeze(-1)
    out.copy_((scaled - exponential.log()).argmax(-1))


def _fused_mixed_sample(logits, exponential, temperatures, eps=1e-10, greedy_eps=0.0):
    """Reference for `fused_mixed_sample`: the race on the noise and the
    lowest-id argmax for the rows at or below `greedy_eps`, in one call."""
    CALLS.append("fused")
    scaled = logits.float() / temperatures.float().unsqueeze(-1)
    drawn = (scaled - exponential.log()).argmax(-1)
    top = logits.float().max(-1, keepdim=True).values
    first = (logits.float() == top).int().argmax(-1)
    return torch.where(temperatures <= greedy_eps, first, drawn).to(torch.int32)


def _topk_select(values, k, tie="low", **kwargs):
    CALLS.append("argmax")
    assert k == 1 and tie == "low"
    top = values.float().max(-1, keepdim=True).values
    first = (values.float() == top).int().argmax(-1, keepdim=True)
    return None, first.to(torch.int32)


@pytest.fixture(autouse=True)
def reference_ops(monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(
        sampler_module, "mixed_sample_outer_exponential", _outer_exponential
    )
    monkeypatch.setattr(sampler_module, "topk_select", _topk_select)
    monkeypatch.setattr(sampler_module, "fused_mixed_sample", _fused_mixed_sample)
    monkeypatch.setattr(sampler_module, "_ANNOUNCED", set())
    _options(monkeypatch)


def _options(monkeypatch, **dspark_options):
    """The recipe's dspark_options as the runner reads them."""
    plan = SimpleNamespace(dspark_options=recipe_module.DSparkOptions(**dspark_options))
    monkeypatch.setattr(runner_module.plugin_host, "_plan", plan)


def _logits(rows, vocab=96, seed=11):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, vocab, generator=g).to(torch.bfloat16)


def _device_greedy_mask(temperatures: np.ndarray) -> torch.Tensor:
    """The mask `_apply_greedy_tokens` builds from what `prepare_sample` stages."""
    staged = np.empty(len(temperatures), dtype=np.float32)
    np.maximum(temperatures, sampler_module.SAMPLER_EPS, out=staged)
    return torch.from_numpy(staged) <= sampler_module.SAMPLER_EPS


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_host_no_greedy_rows_is_the_device_mask(dtype):
    eps = sampler_module.SAMPLER_EPS
    edge = [0.0, eps, np.nextafter(np.float32(eps), np.float32(1)), 1.0000000001e-10]
    edge += [5e-11, 1.5e-10, 1e-9, 1e-3, 0.6, 1.0, 2.0]
    rng = np.random.default_rng(3)
    batches = [np.array([t], dtype=dtype) for t in edge]
    batches += [np.array([1.0, t, 0.7], dtype=dtype) for t in edge]
    batches += [rng.uniform(0, 2, size=n).astype(dtype) for n in (1, 7, 64)]
    batches += [np.zeros(0, dtype=dtype)]
    seen = set()
    for temperatures in batches:
        expected = not bool(_device_greedy_mask(temperatures).any())
        assert sampler_module.host_no_greedy_rows(temperatures) is expected
        seen.add(expected)
    assert seen == {True, False}


@pytest.mark.parametrize("independent", [False, True])
def test_no_greedy_returns_the_same_ids_without_the_argmax(independent):
    logits = _logits(9)
    temperatures = torch.linspace(0.3, 1.7, 9)
    sampler = sampler_module.Sampler()
    torch.manual_seed(5)
    full = sampler(logits, temperatures, needs_independent_noise=independent)
    state_full = torch.get_rng_state()
    # Independent noise: the race and the greedy override are one call.
    assert CALLS == (["fused"] if independent else ["score", "argmax"])
    CALLS.clear()
    torch.manual_seed(5)
    short = sampler(
        logits, temperatures, needs_independent_noise=independent, no_greedy=True
    )
    assert CALLS == ["score"]
    assert short.dtype == full.dtype and short.shape == full.shape
    assert torch.equal(short, full)
    assert torch.equal(torch.get_rng_state(), state_full)


def test_a_greedy_row_changes_the_ids_so_the_host_must_not_claim_none():
    """The select is not a no-op when a row is greedy: the gate is what makes
    the shortcut exact, and the gate says no here."""
    logits = _logits(6)
    temperatures_np = np.array([1.0, 0.0, 1.0, 0.0, 0.8, 1.2])
    assert not sampler_module.host_no_greedy_rows(temperatures_np)
    temperatures = torch.from_numpy(
        np.maximum(temperatures_np, sampler_module.SAMPLER_EPS).astype(np.float32)
    )
    torch.manual_seed(5)
    full = sampler_module.Sampler()(logits, temperatures, needs_independent_noise=True)
    greedy = logits.float().argmax(-1).to(torch.int32)
    assert torch.equal(full[[1, 3]], greedy[[1, 3]])


def test_verification_noise_only_moves_the_generator_as_the_draw_does():
    logits = _logits(15)
    cu = torch.tensor([5, 10, 15], dtype=torch.int32)
    temperatures = torch.tensor([0.7, 1.0, 1.3])
    bonus_logits = _logits(3, seed=12)
    sampler = sampler_module.Sampler()

    torch.manual_seed(21)
    sampler.sample_verification_tokens(logits, cu, temperatures, None, None)
    state_full = torch.get_rng_state()
    bonus_full = sampler(bonus_logits, temperatures, needs_independent_noise=True)
    CALLS.clear()

    torch.manual_seed(21)
    assert sampler.verification_noise_only(logits, None, None) is True
    assert CALLS == []
    assert torch.equal(torch.get_rng_state(), state_full)
    bonus_short = sampler(bonus_logits, temperatures, needs_independent_noise=True)
    assert torch.equal(bonus_short, bonus_full)


def test_verification_noise_only_draws_nothing_for_no_rows_or_filters():
    sampler = sampler_module.Sampler()
    torch.manual_seed(2)
    before = torch.get_rng_state()
    assert sampler.verification_noise_only(_logits(0), None, None) is True
    assert torch.equal(torch.get_rng_state(), before)
    for top_ks, top_ps in ((5, None), (None, 0.9), (torch.tensor([5, 0]), 0.9)):
        assert sampler.verification_noise_only(_logits(4), top_ks, top_ps) is False
        assert torch.equal(torch.get_rng_state(), before)


class _Reached(Exception):
    pass


def _postprocess(monkeypatch, *, rates, top_ps=None, temperatures_np=None, bs=3):
    """`ModelRunner.postprocess` up to the rejection sampler, on the real
    `Sampler`: what the rejection sampler is handed, and the generator after."""
    width = 5
    rows = bs * (width + 1)
    temperatures_np = (
        np.array([0.7, 1.0, 1.3][:bs]) if temperatures_np is None else temperatures_np
    )
    bonus_rows = torch.arange(bs) * (width + 1) + width
    target_rows = torch.tensor(
        [r for r in range(rows) if r % (width + 1) != width], dtype=torch.int64
    )
    spec = SimpleNamespace(
        bonus_logits_indices=bonus_rows,
        target_logits_indices=target_rows,
        cu_num_draft_tokens=(torch.arange(bs, dtype=torch.int32) + 1) * width,
    )
    monkeypatch.setattr(
        runner_module,
        "get_forward_context",
        lambda: SimpleNamespace(spec_decode_metadata=spec),
    )
    monkeypatch.setattr(
        runner_module, "get_tp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(runner_module, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(runner_module, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(runner_module, "get_pcp_world_size", lambda: 1)
    seen = {}

    def forward(
        metadata, target_logits, bonus_token_ids, *, target_token_ids, ratio, **_
    ):
        seen.update(
            target_logits=target_logits.clone(),
            bonus=bonus_token_ids.clone(),
            target_ids=None if target_token_ids is None else target_token_ids.clone(),
            ratio=ratio,
            rng=torch.get_rng_state(),
            calls=list(CALLS),
        )
        raise _Reached

    runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(zero_spec_counts=torch.zeros(8)),
        sampler=sampler_module.Sampler(),
        rejection_sampler=SimpleNamespace(
            forward=forward, synthetic_acceptance_rates=rates
        ),
        _served=None,
        structured_output=None,
    )
    batch = SimpleNamespace(
        total_seqs_num=bs,
        return_logprobs=[False] * bs,
        any_logprobs=False,
        required_channel=None,
        structured_requests=None,
        num_spec_step=width,
        temperatures=temperatures_np,
    )
    temperatures = torch.from_numpy(
        np.maximum(temperatures_np, sampler_module.SAMPLER_EPS).astype(np.float32)
    )
    CALLS.clear()
    torch.manual_seed(77)
    with pytest.raises(_Reached):
        runner_module.ModelRunner.postprocess(
            runner,
            batch,
            _logits(rows, seed=31),
            temperatures,
            None,
            top_ps,
            bool((temperatures_np == 0).all()),
            hidden_states=None,
        )
    return seen


FORCED = (1.0, 1.0, 0.51, 0.0, 0.0)


@pytest.mark.parametrize("greedy_pick", [False, True])
def test_forced_acceptance_step_is_unchanged_without_the_target_draw(
    monkeypatch, greedy_pick
):
    """The benchmark's step: forced acceptance, unfiltered temperature rows.
    The rejection sampler gets the same logits and bonus ids, the generator
    ends where it did, and neither the target's scoring nor (with the second
    switch) any greedy reduction is launched."""
    base = _postprocess(monkeypatch, rates=FORCED)
    assert base["calls"] == ["fused", "fused"]
    assert base["target_ids"] is not None

    _options(
        monkeypatch,
        verify_skip_unused_target_draw=True,
        sampler_skip_greedy_pick=greedy_pick,
    )
    new = _postprocess(monkeypatch, rates=FORCED)
    assert new["calls"] == (["score"] if greedy_pick else ["fused"])
    # Forced acceptance reads no target id (`rejection_sample`'s synthetic
    # branch takes the argmax of `target_logits`), so None is the same step.
    assert new["target_ids"] is None and new["ratio"] is None
    assert torch.equal(new["target_logits"], base["target_logits"])
    assert torch.equal(new["bonus"], base["bonus"])
    assert new["bonus"].dtype == base["bonus"].dtype
    assert torch.equal(new["rng"], base["rng"])


def test_real_acceptance_keeps_the_target_draw(monkeypatch):
    """Exact match reads the draw: the switch changes nothing there, and the
    greedy-pick switch returns the same ids."""
    base = _postprocess(monkeypatch, rates=None)
    _options(monkeypatch, verify_skip_unused_target_draw=True)
    kept = _postprocess(monkeypatch, rates=None)
    assert kept["calls"] == base["calls"] == ["fused", "fused"]
    _options(
        monkeypatch,
        verify_skip_unused_target_draw=True,
        sampler_skip_greedy_pick=True,
    )
    short = _postprocess(monkeypatch, rates=None)
    assert short["calls"] == ["score", "score"]
    for seen in (kept, short):
        assert torch.equal(seen["target_ids"], base["target_ids"])
        assert torch.equal(seen["bonus"], base["bonus"])
        assert torch.equal(seen["rng"], base["rng"])


def test_a_greedy_row_keeps_the_reduce_and_select(monkeypatch):
    temperatures = np.array([0.7, 0.0, 1.3])
    base = _postprocess(monkeypatch, rates=None, temperatures_np=temperatures)
    _options(monkeypatch, sampler_skip_greedy_pick=True)
    new = _postprocess(monkeypatch, rates=None, temperatures_np=temperatures)
    assert new["calls"] == base["calls"] == ["fused", "fused"]
    assert torch.equal(new["target_ids"], base["target_ids"])
    assert torch.equal(new["bonus"], base["bonus"])


def test_a_filtered_batch_keeps_the_whole_target_draw(monkeypatch):
    """Top-p rows draw inside their sampling ops: nothing is skipped."""
    _options(monkeypatch, verify_skip_unused_target_draw=True)
    runner_sampler = sampler_module.Sampler
    with (
        mock.patch.object(
            runner_sampler,
            "sample_verification_tokens",
            return_value=torch.zeros(15, dtype=torch.int32),
        ) as full,
        mock.patch.object(
            runner_sampler, "forward", return_value=torch.zeros(3, dtype=torch.int32)
        ),
    ):
        _postprocess(monkeypatch, rates=FORCED, top_ps=0.9)
    full.assert_called_once()
