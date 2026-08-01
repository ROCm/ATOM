# SPDX-License-Identifier: MIT
# Shared fixtures for ATOM unit tests, and stand-ins for the third-party
# packages a plain CPU runner does not have.
#
# Only *external* packages are stubbed, and only when genuinely missing. No
# `atom.*` module is faked: a unit test imports the same class the engine
# imports, so it cannot pass against an API the engine no longer has.

import hashlib
import importlib
import importlib.util
import sys
import types
from itertools import count
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── 1. Resolve ATOM root and ensure it's on sys.path ──────────────────────

ATOM_ROOT = str(Path(__file__).resolve().parent.parent)
if ATOM_ROOT not in sys.path:
    sys.path.insert(0, ATOM_ROOT)

# ── 2. Stub AITER when it is absent ───────────────────────────────────────
# The unit gate runs on a plain CPU runner with no AITER build. Only the
# *external* boundary is stubbed: `atom.config` and friends are imported for
# real, so a test exercises the same class the engine does.
#
# Stubbing internal modules instead is what this replaced, and it rotted --
# the hand-written `atom.config` stand-in silently lost `CompilationLevel`,
# and because the modules that import it are guarded by a broad try/except
# that skips the whole file, four GPU kernel test modules stopped running on
# every machine, blaming a circular import that never existed.
#
# `atom.config` reaches exactly two AITER attributes (`QuantType` and
# `utility.dtypes.d_dtypes`); MagicMock covers them and anything added later.

if importlib.util.find_spec("aiter") is None:
    for _mod_name in ("aiter", "aiter.utility", "aiter.utility.dtypes"):
        sys.modules[_mod_name] = MagicMock()

# ── 3. Stub zmq / zmq.asyncio if not installed ────────────────────────────

if importlib.util.find_spec("zmq") is None:
    for _mod_name in ("zmq", "zmq.asyncio"):
        sys.modules[_mod_name] = MagicMock()

# ── 4. Stub xxhash with a hashlib-based fallback ──────────────────────────

if importlib.util.find_spec("xxhash") is None:
    _xxhash_mod = types.ModuleType("xxhash")

    class _XXH64:
        def __init__(self):
            self._h = hashlib.sha256()

        def update(self, data):
            if isinstance(data, (bytes, bytearray, memoryview)):
                self._h.update(data)
            else:
                raise TypeError(
                    f"expected bytes-like object, got {type(data).__name__}"
                )

        def intdigest(self):
            return int.from_bytes(self._h.digest()[:8], "little")

    _xxhash_mod.xxh64 = _XXH64
    sys.modules["xxhash"] = _xxhash_mod

# ── 5. Import atom submodules ──────────────────────────────────

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

# ── 6. MockConfig ──────────────────────────────────────────────────────────


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


# ── 7. Fixtures ────────────────────────────────────────────────────────────


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
