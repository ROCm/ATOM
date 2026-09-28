# SPDX-License-Identifier: MIT
# Shared fixtures for ATOM unit tests.
#
# Nothing here fakes a module. Tests import the same classes the engine
# imports, so a test cannot pass against an API the engine no longer has --
# which is what happened while this file hand-built stand-ins for `atom` and
# `atom.config`: the copy lost `CompilationLevel`, and four test modules
# silently stopped running on every machine.
#
# `atom.config` no longer needs the AITER build to import (`atom.quant_spec`
# resolves its two AITER handles on first use), and every other third-party
# import here is a declared dependency, so a plain CPU runner has them.

import dataclasses
import sys
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import pytest

# ── 1. Resolve ATOM root and ensure it's on sys.path ──────────────────────

ATOM_ROOT = str(Path(__file__).resolve().parent.parent)
if ATOM_ROOT not in sys.path:
    sys.path.insert(0, ATOM_ROOT)

# ── 2. Import atom submodules ──────────────────────────────────────────────

from atom.config import Config
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

# ── 3. MockConfig ──────────────────────────────────────────────────────────


class _MockHFConfig:
    """Minimal hf_config stub. Default is non-V4 so Scheduler's V4 SWA-warmup
    detection stays inert; pass architectures=[...] to exercise the V4 path."""

    def __init__(self, architectures=None, sliding_window=128):
        self.architectures = architectures or ["LlamaForCausalLM"]
        self.sliding_window = sliding_window


class MockConfig:
    """Lightweight stand-in for atom.config.Config.

    Provides exactly the attributes that BlockManager and Scheduler read,
    without triggering HuggingFace downloads or GPU init.
    """

    def __init__(self, **overrides):
        defaults = {
            "kv_cache_block_size": 4,
            "num_kvcache_blocks": 10,
            "enable_prefix_caching": False,
            "enable_log_stats": True,
            "throughput_log_interval": 10.0,
            "cache_hit_rate_window": 1000,
            "enable_chunked_prefill": True,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 64,
            "long_prefill_token_threshold": 0,
            "decode_context_parallel_size": 1,
            "max_model_len": 64,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "stop_token_ids": [],
            "scheduler_delay_factor": 0.0,
            "speculative_config": None,
            # Scheduler.__init__ reads config.hf_config.architectures for V4
            # SWA-warmup detection; a non-V4 stub keeps that path inert.
            "hf_config": _MockHFConfig(),
        }
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(self, k, v)


def atom_config_double(**overrides):
    """A stand-in for `atom.config.Config`, with the real Config's fields.

    Derived from `dataclasses.fields(Config)` rather than hand-listed, so a
    field production adds arrives here with its real default instead of
    raising `AttributeError` the first time a code path reads it. That is not
    hypothetical: `topK.is_rocm_aiter_fusion_shared_expert_enabled_for_quant_
    config` grew a read of `enable_dp_attention`, and the hand-built namespace
    in `test_shared_expert_dispatch` had no such attribute -- four tests red on
    every machine that can run them, which is only a machine with aiter,
    because the module `importorskip`s it. CI has no aiter, so CI never saw
    them and nobody was told.

    `MockConfig` below is the older, narrower answer to the same question --
    "exactly the attributes that BlockManager and Scheduler read" -- and it
    can drift the same way. It is left alone because its callers assert on the
    small surface it declares; new doubles should start here.

    An override naming something that is not a Config field is refused. That
    is the other direction of the same drift: a field renamed in production
    leaves a test setting an attribute nothing reads, which passes and means
    nothing.
    """
    values = {}
    for f in dataclasses.fields(Config):
        if f.default is not dataclasses.MISSING:
            values[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            values[f.name] = f.default_factory()
        else:
            # `model` and the `init=False` fields a real Config fills in from
            # the checkpoint. A test that needs one overrides it.
            values[f.name] = None
    unknown = sorted(set(overrides) - set(values))
    assert not unknown, (
        f"not Config fields: {unknown}. Either the name is wrong or "
        f"production renamed it and this override now sets nothing."
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# ── 4. Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def mock_config():
    return MockConfig()


@pytest.fixture
def mock_config_with_prefix_caching():
    return MockConfig(enable_prefix_caching=True)


@pytest.fixture
def block_manager(mock_config):
    return BlockManager(mock_config)


@pytest.fixture
def block_manager_prefix(mock_config_with_prefix_caching):
    return BlockManager(mock_config_with_prefix_caching)


@pytest.fixture
def scheduler(mock_config):
    return Scheduler(mock_config)


@pytest.fixture(autouse=True)
def reset_sequence_counter():
    """Reset Sequence.counter before each test for predictable IDs."""
    Sequence.counter = count()
    yield
    Sequence.counter = count()


def _duplicated_atom_classes():
    """Atom classes held by a module other than the one `sys.modules` publishes.

    Checking `sys.modules` identity alone is not enough: a fixture that pops
    atom modules, re-imports under a stub and then restores its snapshot leaves
    `sys.modules` looking untouched, while whatever imported during the window
    still refers to the SECOND copy. What is observable afterwards is a class
    whose own module no longer publishes it -- two `QuantType.No` objects that
    print identically and compare unequal.
    """
    out = []
    for name, module in list(sys.modules.items()):
        if not (name == "atom" or name.startswith("atom.")):
            continue
        for attr, value in list(vars(module).items()):
            origin = getattr(value, "__module__", None)
            if not isinstance(value, type) or not isinstance(origin, str):
                continue
            if not (origin == "atom" or origin.startswith("atom.")):
                continue
            home = sys.modules.get(origin)
            if home is not None and getattr(home, value.__name__, value) is not value:
                out.append(
                    f"{name}.{attr} is a stale copy of {origin}.{value.__name__}"
                )
    return sorted(set(out))


_atom_duplicates_seen: set[str] = set()


@pytest.fixture(autouse=True)
def atom_modules_are_imported_once():
    """Fail the test that leaves a second copy of an `atom.*` class alive.

    Test modules used to delete every `atom.*` entry from `sys.modules` and let
    the imports run again. Restoring the snapshot afterwards does not undo it:
    whatever imported during the window keeps the SECOND copy, so
    `atom.quant_spec` ends up existing twice and two `QuantType.No` objects
    compare unequal while printing identically. The victim was
    `test_qwen4_exp_quantization`, three files away -- it read
    `quant_type != QuantType.No` as true, built a quantized layer, and died on
    a `weight_scale` that branch never creates.

    The check is the duplicate itself, not a grep for `del sys.modules` and not
    `sys.modules` identity: a fixture that restores its snapshot leaves
    `sys.modules` pristine and the duplicate held elsewhere, which is exactly
    the case that got past the first version of this guard.

    Already-reported duplicates are remembered rather than re-reported, so the
    test that introduces one is named once instead of every test after it.
    """
    yield
    fresh = [d for d in _duplicated_atom_classes() if d not in _atom_duplicates_seen]
    _atom_duplicates_seen.update(fresh)
    assert not fresh, (
        "this test left a second copy of these atom classes alive, which makes "
        f"`is` and `==` disagree for every later test: {fresh}"
    )


@pytest.fixture(autouse=True)
def keep_envs_lazy():
    """Undo the permanent damage `monkeypatch.setattr(envs, ...)` leaves behind.

    `atom.utils.envs` reads each variable through a module-level `__getattr__`,
    which Python consults only for names NOT in the module dict. `monkeypatch`
    restores by `setattr`, so the value it read at patch time lands in that
    dict -- and from then on every `monkeypatch.setenv` for that name is
    ignored, in every later test, for the life of the process. The victim is
    whichever test asserts on that variable next, which is why this surfaced as
    three unrelated tests that pass alone and fail in the suite.

    Forty call sites across nine files use that idiom, so this is the one place
    to undo it rather than the forty. None of the lazy names is in the module
    dict after import, so finding one there is unambiguous.
    """
    yield
    from atom.utils import envs

    for name in list(vars(envs)):
        if name in envs.environment_variables:
            delattr(envs, name)


@pytest.fixture
def seq_factory():
    """Factory for creating Sequence objects with sensible defaults."""

    def make_sequence(token_ids, block_size=4, sampling_params=None, **kwargs):
        sp = sampling_params or SamplingParams()
        return Sequence(token_ids, block_size, sampling_params=sp, **kwargs)

    return make_sequence
