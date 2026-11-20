# SPDX-License-Identifier: MIT
"""Shared vs per-row exponential noise in the temperature sampler (CPU).

Without fan-out siblings the temperature path broadcasts one ``(1, vocab)``
noise row to every row of the batch, so rows with the same distribution always
pick the same token. ``--sampler-independent-noise`` (``Config.
sampler_independent_noise``) draws a fresh row per sequence instead. Both keep
each row's marginal distribution; only the joint distribution across rows
differs, which is what these tests pin down.

The AITER ops are replaced by their exponential race (argmax of p / noise),
the rule the real kernels implement, and the noise tensor each receives is
recorded: ``mixed_sample_outer_exponential`` for the shared row,
``fused_mixed_sample`` (race plus greedy override) for per-row noise.
The sampler itself is the production module, imported against the stub AITER
in ``tests/aiter_stub.py`` so the CPU gate runs it without a GPU build.
"""

import dataclasses
import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from aiter_stub import atom_under_stubs

PROBS = torch.tensor([0.4, 0.3, 0.2, 0.1])
ROWS = 8


@pytest.fixture(scope="module")
def sampler():
    with atom_under_stubs() as load:
        yield load("atom.model_ops.sampler")


def _fused_mixed_sample_reference(
    input, exponentials, temperatures, *, eps, greedy_eps
):
    """CPU stand-in with the documented semantics of aiter's fused_mixed_sample.

    A row at or below ``greedy_eps`` takes the lowest-id argmax of ``input``.
    Any other row takes the race
    ``argmax softmax(input / max(T, 1e-5)) / (E + eps)``.
    """
    scaled = input.float() / temperatures.clamp_min(1e-5).unsqueeze(-1)
    race = (torch.softmax(scaled, dim=-1) / (exponentials + eps)).argmax(-1)
    greedy = temperatures <= greedy_eps
    return torch.where(greedy, input.float().argmax(-1), race).to(torch.int32)


@pytest.fixture
def noise(sampler, monkeypatch):
    """Record the noise the sampler passes to the AITER kernels, and score it.

    Both kernels run the exponential race the real ones implement:
    ``argmax(p / (E + eps))``. ``fused_mixed_sample`` also takes the argmax of
    rows at or below ``greedy_eps``; ``topk_select`` keeps the greedy
    correction of the shared-row path on the CPU. Yields two lists, in call
    order: the noise each kernel call received, and the kernel it went to.
    """
    received, ops = [], []

    def race(logits, exponentials, temperatures, eps):
        probs = torch.softmax(logits.float() / temperatures.unsqueeze(-1), dim=-1)
        return (probs / (exponentials + eps)).argmax(-1)

    def mixed_sample_outer_exponential(
        out, logits, exponentials, temperatures, eps=1e-10
    ):
        received.append(exponentials)
        ops.append("shared")
        out.copy_(race(logits, exponentials, temperatures, eps).to(out.dtype))

    def fused_mixed_sample(input, exponentials, temperatures, *, eps, greedy_eps):
        received.append(exponentials)
        ops.append("fused")
        return _fused_mixed_sample_reference(
            input, exponentials, temperatures, eps=eps, greedy_eps=greedy_eps
        )

    def topk_select(probs, k, *, tie):
        assert k == 1 and tie == "low"
        return None, probs.argmax(-1, keepdim=True).to(torch.int32)

    monkeypatch.setattr(
        sampler, "mixed_sample_outer_exponential", mixed_sample_outer_exponential
    )
    monkeypatch.setattr(sampler, "fused_mixed_sample", fused_mixed_sample)
    monkeypatch.setattr(sampler, "topk_select", topk_select)
    # Each test seeds and draws its own stream; restore the global one after.
    with torch.random.fork_rng(devices=[]):
        yield received, ops


def _same_distribution_batch():
    return PROBS.log().expand(ROWS, -1).contiguous(), torch.ones(ROWS)


def _draw(instance, calls, **kwargs):
    logits, temperatures = _same_distribution_batch()
    return torch.stack(
        [instance(logits, temperatures, None, None, **kwargs) for _ in range(calls)]
    )


def _pair_coincidence(samples):
    pairs = list(itertools.combinations(range(samples.shape[1]), 2))
    hits = sum(int((samples[:, a] == samples[:, b]).sum()) for a, b in pairs)
    return hits / (samples.shape[0] * len(pairs))


def test_default_broadcasts_one_noise_row(sampler, noise):
    # Every way of asking for the default -- the bare constructor, the
    # explicit opt-out, and the engine's flag-off config -- keeps the shared
    # row.
    instances = (
        sampler.Sampler(),
        sampler.Sampler(always_independent_noise=False),
        sampler.sampler_from_config(SimpleNamespace(sampler_independent_noise=False)),
    )
    received, ops = noise
    torch.random.default_generator.manual_seed(0)
    for instance in instances:
        samples = _draw(instance, calls=200)
        last = received[-1]
        assert ops[-1] == "shared"
        assert last.shape == (ROWS, PROBS.numel()) and last.stride(0) == 0
        assert torch.equal(last, last[:1].expand_as(last))
        # Every row of a call picks the same token, and the call-to-call draw
        # is fresh, so the tokens still move between calls.
        assert (samples == samples[:, :1]).all()
        assert samples[:, 0].unique().numel() > 1
    # The flag-off sampler is the bare default path, RNG consumption
    # included: the same seed reproduces the same tokens.
    torch.random.default_generator.manual_seed(11)
    bare = _draw(instances[0], calls=20)
    torch.random.default_generator.manual_seed(11)
    configured = _draw(instances[2], calls=20)
    assert torch.equal(bare, configured)


def test_opt_in_draws_independent_rows(sampler, noise):
    torch.random.default_generator.manual_seed(0)
    samples = _draw(sampler.Sampler(always_independent_noise=True), calls=2000)
    received, ops = noise
    last = received[-1]
    assert set(ops) == {"fused"}
    assert last.shape == (ROWS, PROBS.numel()) and last.stride(0) > 0
    assert not torch.equal(last[0], last[1])
    # Independent rows with the same distribution coincide with probability
    # sum(p^2) = 0.30, not 1; each row's marginal is unchanged.
    assert _pair_coincidence(samples) == pytest.approx(
        float((PROBS**2).sum()), abs=0.02
    )
    counts = torch.bincount(samples.flatten().to(torch.int64), minlength=4)
    assert torch.allclose(counts / samples.numel(), PROBS, atol=0.02)
    for row in range(ROWS):
        counts = torch.bincount(samples[:, row].to(torch.int64), minlength=4)
        assert torch.allclose(counts / samples.shape[0], PROBS, atol=0.05)


def test_opt_in_is_seeded_reproducible_and_matches_the_fanout_path(sampler, noise):
    # The option reuses the noise allocation the n>1 path already draws, so
    # the same seed reproduces it and explicit per-call independence gives
    # identical draws.
    torch.random.default_generator.manual_seed(7)
    opt_in = _draw(sampler.Sampler(always_independent_noise=True), calls=20)
    torch.random.default_generator.manual_seed(7)
    fanout = _draw(sampler.Sampler(), calls=20, needs_independent_noise=True)
    torch.random.default_generator.manual_seed(7)
    again = _draw(sampler.Sampler(always_independent_noise=True), calls=20)
    assert torch.equal(opt_in, fanout) and torch.equal(opt_in, again)


def test_independent_noise_is_one_exponential_draw(sampler, noise):
    # The per-row path draws exactly what the unfused path did before
    # `fused_mixed_sample` took the race: one (rows, vocab) `exponential_`.
    # Same noise for a seed, and the RNG is left where that draw leaves it.
    logits, temperatures = _same_distribution_batch()
    torch.random.default_generator.manual_seed(5)
    sampler.Sampler()(logits, temperatures, None, None, needs_independent_noise=True)
    after = torch.random.get_rng_state()
    torch.random.default_generator.manual_seed(5)
    expected = torch.empty(ROWS, PROBS.numel()).exponential_(1)
    received, _ = noise
    assert torch.equal(received[-1], expected)
    assert torch.equal(after, torch.random.get_rng_state())


def test_independent_noise_calls_fused_mixed_sample(sampler, monkeypatch):
    # One aiter call does the race and the greedy override; the shared-row
    # kernel and the separate greedy select do not run.
    out = torch.full((ROWS,), 7, dtype=torch.int32)
    fused = MagicMock(return_value=out)
    race, select = MagicMock(), MagicMock()
    monkeypatch.setattr(sampler, "fused_mixed_sample", fused)
    monkeypatch.setattr(sampler, "mixed_sample_outer_exponential", race)
    monkeypatch.setattr(sampler, "topk_select", select)
    instance = sampler.Sampler()
    logits, temperatures = _same_distribution_batch()
    assert instance(logits, temperatures, needs_independent_noise=True) is out
    ((args, kwargs),) = fused.call_args_list
    assert args[0] is logits and args[2] is temperatures
    assert args[1].shape == logits.shape and args[1].dtype == torch.float32
    assert kwargs == {"eps": instance.eps, "greedy_eps": sampler.SAMPLER_EPS}
    race.assert_not_called()
    select.assert_not_called()


def test_independent_noise_batch_greedy_rows_take_the_low_tie(sampler, monkeypatch):
    # The per-row-noise path leaves the greedy override to fused_mixed_sample.
    # Pin what ATOM hands it (the logits, the temperatures and
    # greedy_eps=SAMPLER_EPS) and that the greedy rows come back as the
    # low-tie argmax, while the sampling rows keep the race.
    calls = []

    def fused_mixed_sample(input, exponentials, temperatures, *, eps, greedy_eps):
        calls.append((input, exponentials, temperatures, eps, greedy_eps))
        return _fused_mixed_sample_reference(
            input, exponentials, temperatures, eps=eps, greedy_eps=greedy_eps
        )

    monkeypatch.setattr(sampler, "fused_mixed_sample", fused_mixed_sample)
    instance = sampler.Sampler()
    rows = 16
    logits = torch.zeros(rows, 8)
    logits[:, 2] = logits[:, 5] = 1.0
    temperatures = torch.ones(rows)
    temperatures[::2] = sampler.SAMPLER_EPS
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        got = instance(logits, temperatures, None, None, needs_independent_noise=True)
    ((input, noise, temps, eps, greedy_eps),) = calls
    assert input is logits and temps is temperatures
    assert eps == instance.eps and greedy_eps == sampler.SAMPLER_EPS
    assert noise.shape == logits.shape and noise.stride(0) > 0
    assert got[::2].tolist() == [2] * (rows // 2)
    # On this seed the race alone puts some greedy rows on the high tie, so
    # the override is what makes them 2. Some sampling rows land there too, so
    # an override that reached them would show.
    race = _fused_mixed_sample_reference(
        logits, noise, temperatures, eps=eps, greedy_eps=-1.0
    )
    assert (race[::2] == 5).any() and (got[1::2] == 5).any()
    assert torch.equal(got[1::2], race[1::2])


@pytest.mark.parametrize(
    "always,requested,expected",
    [
        # The engine policy is ORed into the per-call request, so the
        # effective value covers both ways of asking.
        (False, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, True),
    ],
)
def test_forward_resolves_the_policy_once_for_every_consumer(
    sampler, monkeypatch, always, requested, expected
):
    # ``forward`` is where the per-call request arrives, and where the engine
    # option is ORed into it. Every consumer of that value must see the same
    # resolved boolean, filtered or not: an unfiltered batch takes the per-row
    # path exactly when it is set, and the filtered path receives it.
    seen = []

    def temperature_sample(self, logits, temperatures):
        seen.append("temperature")
        return torch.zeros(logits.shape[0], dtype=torch.int)

    def independent_noise_sample(self, logits, temperatures):
        seen.append("independent")
        return torch.zeros(logits.shape[0], dtype=torch.int)

    def topk_topp_sample(
        self, logits, temperatures, top_ks, top_ps, needs_independent_noise=False
    ):
        seen.append(("filtered", needs_independent_noise))
        return torch.zeros(logits.shape[0], dtype=torch.int)

    monkeypatch.setattr(sampler.Sampler, "_temperature_sample", temperature_sample)
    monkeypatch.setattr(
        sampler.Sampler, "_independent_noise_sample", independent_noise_sample
    )
    monkeypatch.setattr(sampler.Sampler, "_topk_topp_sample", topk_topp_sample)
    monkeypatch.setattr(
        sampler,
        "topk_select",
        lambda probs, k, *, tie: (
            None,
            probs.argmax(-1, keepdim=True).to(torch.int32),
        ),
    )

    instance = sampler.Sampler(always_independent_noise=always)
    logits = torch.zeros(3, 8)
    temperatures = torch.ones(3)
    instance(logits, temperatures, None, None, needs_independent_noise=requested)
    instance(logits, temperatures, top_ps=0.9, needs_independent_noise=requested)
    unfiltered = "independent" if expected else "temperature"
    assert seen == [unfiltered, ("filtered", expected)]


@pytest.mark.parametrize("always", [False, True])
def test_greedy_rows_take_the_argmax_whatever_the_option(sampler, noise, always):
    logits = torch.zeros(4, 8)
    logits[:, 2] = logits[:, 5] = 1.0
    temperatures = torch.tensor([sampler.SAMPLER_EPS, 1.0, sampler.SAMPLER_EPS, 1.0])
    got = sampler.Sampler(always_independent_noise=always)(
        logits, temperatures, None, None
    )
    assert got[0] == 2 and got[2] == 2


@pytest.mark.parametrize("always", [False, True])
def test_all_greedy_does_not_draw_noise(sampler, noise, always):
    logits = torch.zeros(4, 8)
    logits[:, 2] = logits[:, 5] = 1.0
    got = sampler.Sampler(always_independent_noise=always)(
        logits, torch.full((4,), sampler.SAMPLER_EPS), all_greedy=True
    )
    assert got.tolist() == [2, 2, 2, 2]
    received, _ = noise
    assert not received


@pytest.mark.parametrize("always", [False, True])
def test_filtered_sampling_is_inert_whatever_the_option(
    sampler, noise, monkeypatch, always
):
    # The documented scope: top-k/top-p already draws per-row noise from
    # torch's RNG, so the option has nothing to change there by design and no
    # exponential noise is drawn.
    monkeypatch.setattr(sampler, "AITER_TOPK_TOPP_AVAILABLE", False)
    monkeypatch.setattr(sampler, "_NATIVE_SAMPLING_WARNING_ISSUED", True)
    logits = torch.zeros(4, 8)
    logits[:, 3] = 10.0
    got = sampler.Sampler(always_independent_noise=always)(
        logits, torch.ones(4), top_ps=0.5
    )
    assert got.tolist() == [3, 3, 3, 3]
    received, _ = noise
    assert not received


# ---------------------------------------------------------------------------
# CLI / Config plumbing
# ---------------------------------------------------------------------------


def test_engine_args_defaults_to_off():
    from atom.model_engine.arg_utils import EngineArgs

    engine_args = EngineArgs()
    assert engine_args.sampler_independent_noise is False
    assert engine_args._get_engine_kwargs()["sampler_independent_noise"] is False


def test_engine_args_passes_the_option_to_engine_kwargs():
    from atom.model_engine.arg_utils import EngineArgs

    kwargs = EngineArgs(sampler_independent_noise=True)._get_engine_kwargs()
    assert kwargs["sampler_independent_noise"] is True


def test_cli_flag_parses():
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    # The server's parser, so the kebab/snake aliases are covered too.
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    for argv, expected in (
        ([], False),
        (["--sampler-independent-noise"], True),
        (["--sampler_independent_noise"], True),
        (["--no-sampler-independent-noise"], False),
        (["--no-sampler_independent_noise"], False),
    ):
        parsed = EngineArgs.from_cli_args(parser.parse_args(argv))
        assert parsed.sampler_independent_noise is expected
        assert parsed._get_engine_kwargs()["sampler_independent_noise"] is expected


def test_config_carries_the_field_defaulting_to_off():
    from atom.config import Config

    fields = {field.name: field for field in dataclasses.fields(Config)}
    assert fields["sampler_independent_noise"].default is False


def test_sampler_from_config_reads_the_field(sampler):
    assert (
        sampler.sampler_from_config(
            SimpleNamespace(sampler_independent_noise=True)
        ).always_independent_noise
        is True
    )
    assert (
        sampler.sampler_from_config(
            SimpleNamespace(sampler_independent_noise=False)
        ).always_independent_noise
        is False
    )
