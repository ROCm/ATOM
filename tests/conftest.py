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


@pytest.fixture
def seq_factory():
    """Factory for creating Sequence objects with sensible defaults."""

    def make_sequence(token_ids, block_size=4, sampling_params=None, **kwargs):
        sp = sampling_params or SamplingParams()
        return Sequence(token_ids, block_size, sampling_params=sp, **kwargs)

    return make_sequence
