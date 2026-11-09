# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`ATOM_SPEC_BONUS_FUSED`: which `Sampler.forward` paths take the fused draw.

The sampler needs AITER at import, which a CPU runner does not have, so it is
loaded here against a fake `aiter` under a private module name and the fakes are
removed again. aiter's `fused_mixed_sample` is faked the same way, so these
tests need neither AITER nor Triton and run on any CPU runner. Nothing runs a
kernel: the aiter ops and the fused draw are recording stand-ins. The one
exception binds the call to the real op's signature where aiter has the op, so
a change in #7416's signature fails here and not only in production.
"""

import importlib.util
import pathlib
import sys
import types
import warnings
from unittest.mock import MagicMock

import pytest
import torch

SAMPLER_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "atom" / "model_ops" / "sampler.py"
)
FUSED_MODULE = "aiter.ops.triton.fusions.fused_mixed_sample"


@pytest.fixture(scope="module")
def real_fused_mixed_sample():
    """aiter's real op, imported before `load_sampler` fakes aiter; else skip."""
    try:
        from aiter.ops.triton.fusions.fused_mixed_sample import fused_mixed_sample
    except ImportError as e:  # no aiter, or one without the op (CPU runners)
        pytest.skip(f"aiter's fused_mixed_sample is not importable: {e!r}")
    return fused_mixed_sample


@pytest.fixture
def load_sampler(monkeypatch):
    for name in (
        "aiter",
        "aiter.ops",
        "aiter.ops.sampling",
        "aiter.ops.triton",
        "aiter.ops.triton.softmax",
        "aiter.ops.triton.topk",
        "aiter.ops.triton.fusions",
    ):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    stub = sys.modules["aiter"]
    stub.mixed_sample_outer_exponential = MagicMock(name="aiter_race")
    stub.topk_select = MagicMock(
        side_effect=lambda x, k, tie: (
            None,
            torch.zeros(x.shape[0], 1, dtype=torch.int32),
        )
    )
    fused_module = types.ModuleType(FUSED_MODULE)
    fused_module.fused_mixed_sample = MagicMock(name="fused_draw")
    monkeypatch.setitem(sys.modules, FUSED_MODULE, fused_module)

    def load(fused: bool, aiter_has_op: bool = True, op=None):
        monkeypatch.setenv("ATOM_SPEC_BONUS_FUSED", "1" if fused else "0")
        if op is not None:
            fused_module.fused_mixed_sample = op
        if not aiter_has_op:
            # an aiter without the op: the import raises ImportError
            monkeypatch.setitem(sys.modules, FUSED_MODULE, None)
        spec = importlib.util.spec_from_file_location(
            "_sampler_under_test", SAMPLER_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, stub

    load.fused_draw = fused_module.fused_mixed_sample
    return load


def _batch(rows=4, vocab=64):
    return torch.randn(rows, vocab), torch.ones(rows)


def test_flag_on_wires_aiters_fused_mixed_sample(load_sampler):
    module, _ = load_sampler(fused=True)
    assert module.Sampler()._fused_temperature_sample is load_sampler.fused_draw


def test_flag_on_without_the_op_keeps_the_aiter_path(load_sampler):
    module, aiter = load_sampler(fused=True, aiter_has_op=False)
    with pytest.warns(UserWarning, match="fused_mixed_sample"):
        sampler = module.Sampler()
    assert sampler._fused_temperature_sample is None
    logits, temps = _batch()
    sampler(logits, temps, needs_independent_noise=True)
    aiter.mixed_sample_outer_exponential.assert_called_once()


def _renamed_eps(input, exponentials, temperatures, sample_eps=1e-10, greedy_eps=0.0):
    pass


def _no_greedy_eps(input, exponentials, temperatures, eps=1e-10):
    pass


def _new_required(input, exponentials, temperatures, seed, eps=1e-10, greedy_eps=0.0):
    pass


@pytest.mark.parametrize("op", [_renamed_eps, _no_greedy_eps, _new_required])
def test_flag_on_with_a_changed_signature_keeps_the_aiter_path(load_sampler, op):
    module, aiter = load_sampler(fused=True, op=op)
    with pytest.warns(UserWarning, match="signature"):
        sampler = module.Sampler()
    assert sampler._fused_temperature_sample is None
    logits, temps = _batch()
    sampler(logits, temps, needs_independent_noise=True)
    aiter.mixed_sample_outer_exponential.assert_called_once()


def test_real_fused_mixed_sample_takes_the_call(real_fused_mixed_sample, load_sampler):
    module, _ = load_sampler(fused=True, op=real_fused_mixed_sample)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler = module.Sampler()
    assert sampler._fused_temperature_sample is real_fused_mixed_sample


def test_flag_off_keeps_the_aiter_path(load_sampler):
    module, aiter = load_sampler(fused=False)
    sampler = module.Sampler()
    assert sampler._fused_temperature_sample is None
    logits, temps = _batch()
    sampler(logits, temps, needs_independent_noise=True)
    aiter.mixed_sample_outer_exponential.assert_called_once()


def test_flag_on_routes_the_per_row_noise_draw(load_sampler):
    module, aiter = load_sampler(fused=True)
    sampler = module.Sampler()
    fused = load_sampler.fused_draw
    fused.side_effect = lambda logits, *a, **k: torch.full(
        (logits.shape[0],), 7, dtype=torch.int32
    )
    logits, temps = _batch()
    out = sampler(logits, temps, needs_independent_noise=True)

    assert out.tolist() == [7] * 4
    aiter.mixed_sample_outer_exponential.assert_not_called()
    ((args, kwargs),) = fused.call_args_list
    got_logits, noise, got_temps = args
    assert got_logits is logits and got_temps is temps
    assert noise.shape == logits.shape and noise.dtype == torch.float32
    assert kwargs == {"eps": sampler.eps, "greedy_eps": module.SAMPLER_EPS}


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("no config"),  # no tuned config for this arch
        ValueError("the last dim of input and exponentials must be contiguous"),
    ],
)
def test_no_config_for_the_gpu_falls_back_on_the_same_noise(load_sampler, error):
    module, aiter = load_sampler(fused=True)
    sampler = module.Sampler()
    load_sampler.fused_draw.side_effect = error
    logits, temps = _batch()
    with pytest.warns(UserWarning, match=type(error).__name__):
        sampler(logits, temps, needs_independent_noise=True)

    ((fused_args, _),) = load_sampler.fused_draw.call_args_list
    ((race_args, race_kwargs),) = aiter.mixed_sample_outer_exponential.call_args_list
    assert race_args[2] is fused_args[1], "the fallback must reuse the drawn noise"
    assert race_kwargs == {"eps": sampler.eps}
    assert sampler._fused_temperature_sample is None

    sampler(logits, temps, needs_independent_noise=True)
    assert load_sampler.fused_draw.call_count == 1
    assert aiter.mixed_sample_outer_exponential.call_count == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"needs_independent_noise": False},
        {"needs_independent_noise": True, "top_ks": 5},
        {"needs_independent_noise": True, "top_ps": 0.9},
    ],
)
def test_flag_on_leaves_other_batches_to_aiter(load_sampler, kwargs):
    module, _ = load_sampler(fused=True)
    sampler = module.Sampler()
    sampler._fused_temperature_sample = MagicMock(name="fused")
    sampler._topk_topp_sample = MagicMock(
        return_value=torch.zeros(4, dtype=torch.int32)
    )
    logits, temps = _batch()
    sampler(logits, temps, **kwargs)
    sampler._fused_temperature_sample.assert_not_called()


def test_all_greedy_batches_never_reach_the_fused_draw(load_sampler):
    module, _ = load_sampler(fused=True)
    sampler = module.Sampler()
    sampler._fused_temperature_sample = MagicMock(name="fused")
    logits, temps = _batch()
    sampler(logits, temps, all_greedy=True, needs_independent_noise=True)
    sampler._fused_temperature_sample.assert_not_called()
