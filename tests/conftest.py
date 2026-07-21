# SPDX-License-Identifier: MIT
# Shared fixtures and module stubs for ATOM unit tests.
# Must be imported before any atom.* module to avoid triggering heavy imports.

import importlib
import importlib.util
import importlib.machinery
import sys
import os
import types
import enum
import hashlib
from itertools import count
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── 1. Resolve ATOM root and ensure it's on sys.path ──────────────────────

ATOM_ROOT = str(Path(__file__).resolve().parent.parent)
if ATOM_ROOT not in sys.path:
    sys.path.insert(0, ATOM_ROOT)

# ── 2. Stub the top-level `atom` package so __init__.py never runs ─────────
# atom/__init__.py imports LLMEngine which pulls in zmq, GPU init, etc.

_atom_pkg = types.ModuleType("atom")
_atom_pkg.__path__ = [os.path.join(ATOM_ROOT, "atom")]
_atom_pkg.__package__ = "atom"
# arg_utils does `from atom import LLMEngine`; stub it so the import resolves
# without running atom/__init__.py (which pulls in zmq / GPU init).
_atom_pkg.LLMEngine = MagicMock()
sys.modules["atom"] = _atom_pkg

# ── 3. Stub `atom.config` to avoid HuggingFace / torch heavy imports ──────

_atom_config = types.ModuleType("atom.config")
_atom_config.__package__ = "atom.config"


class _StubConfig:
    """Placeholder so `from atom.config import Config` doesn't fail."""

    pass


class _StubKVCacheTensor:
    """Placeholder for KVCacheTensor."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _StubParallelConfig:
    """Placeholder for ParallelConfig."""

    pass


class _StubEPLBConfig:
    """Placeholder for EPLBConfig with the defaults EPLB code reads."""

    load_window_size = 1000
    rebalance_interval = 3000
    p2p_batch_chunk_size = 32
    rebalance_layers_per_chunk = 64
    num_redundant_experts = 0
    rebalance_min_balancedness = 2.0
    rebalance_balancedness_agg = "min"
    placement_policy = "naive"

    def __init__(self, **kwargs):
        # AtomArgs builds EPLBConfig(**eplb_kwargs); mirror the real dataclass
        # constructor by accepting and storing those fields (falls back to the
        # class-level defaults above when constructed with no args).
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, cfg):
        # Mirror the real EPLBConfig.from_dict: arg_utils builds the config via
        # EPLBConfig.from_dict(--eplb-config JSON dict).
        return cls(**(cfg or {}))


class _StubAtomConfig:
    """Placeholder returned by get_current_atom_config in unit tests."""

    eplb_enable = False
    eplb_config = _StubEPLBConfig()


class _StubCUDAGraphMode(enum.Enum):
    """Mirror the real enum members so arg_utils' `CUDAGraphMode[name]` works."""

    NONE = 0
    PIECEWISE = 1
    FULL = 2


_atom_config.Config = _StubConfig
_atom_config.KVCacheTensor = _StubKVCacheTensor
_atom_config.ParallelConfig = _StubParallelConfig
_atom_config.EPLBConfig = _StubEPLBConfig
_atom_config.CUDAGraphMode = _StubCUDAGraphMode
# Config dataclasses arg_utils imports; tests only need them constructible.
_atom_config.CompilationConfig = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
_atom_config.SpeculativeConfig = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
_atom_config.DSparkConfig = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
# Present so tests can monkeypatch it (raising=True) and so EPLB code paths that
# call it without a patch get usable defaults.
_atom_config.get_current_atom_config = lambda: _StubAtomConfig()
sys.modules["atom.config"] = _atom_config

# ── 3b. Stub forward_context; Scheduler only needs get_kvconnector in tests ──

_forward_context = types.ModuleType("atom.utils.forward_context")
_forward_context.__package__ = "atom.utils"
_forward_context.__spec__ = importlib.machinery.ModuleSpec(
    "atom.utils.forward_context", loader=None
)
_forward_context.get_kvconnector = lambda *args, **kwargs: None
sys.modules["atom.utils.forward_context"] = _forward_context

# ── 4. Stub zmq / zmq.asyncio if not installed ────────────────────────────

if importlib.util.find_spec("zmq") is None:
    for _mod_name in ("zmq", "zmq.asyncio"):
        sys.modules[_mod_name] = MagicMock()

# ── 4b. Stub atom.utils.custom_register to avoid torch.library side effects

_cr = types.ModuleType("atom.utils.custom_register")
_cr.direct_register_custom_op = lambda **kwargs: None
sys.modules["atom.utils.custom_register"] = _cr

# ── 5. Stub xxhash with a hashlib-based fallback ──────────────────────────

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

# ── 6. Now safe to import atom submodules ──────────────────────────────────

from atom.sampling_params import SamplingParams  # noqa: E402
from atom.model_engine.sequence import Sequence  # noqa: E402
from atom.model_engine.block_manager import BlockManager  # noqa: E402
from atom.model_engine.scheduler import Scheduler  # noqa: E402

# ── 7. MockConfig ──────────────────────────────────────────────────────────


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
        defaults = dict(
            kv_cache_block_size=4,
            num_kvcache_blocks=10,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            max_num_seqs=4,
            max_num_batched_tokens=64,
            long_prefill_token_threshold=0,
            max_model_len=64,
            bos_token_id=1,
            eos_token_id=2,
            stop_token_ids=[],
            scheduler_delay_factor=0.0,
            speculative_config=None,
            # Scheduler.__init__ reads config.hf_config.architectures for V4
            # SWA-warmup detection; a non-V4 stub keeps that path inert.
            hf_config=_MockHFConfig(),
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(self, k, v)


# ── 8. Fixtures ────────────────────────────────────────────────────────────


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
