# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for MXFP4 activation quantization rounding alignment.

Covers:
  atom/model_ops/linear.py  — _resolve_mxfp4_act_round_mode()
  atom/quantization/quark/utils.py — quant_mxfp4_online_even() 3D reshape

Both changes ensure ATOM's runtime activation quant rounding matches the
Even rounding used by Quark during offline weight calibration, eliminating
the round-mode mismatch that degraded e2e accuracy.

These modules import ``aiter``/``triton`` at module load, which need a GPU
build absent on a CPU test runner. Rather than hand-mocking the whole
``atom.*`` import graph and loading the files by path, we install a
``sys.meta_path`` finder that fabricates a stub for *only* those GPU
packages, then import the modules under test the same way the engine does.
The real ``atom`` package (config, quant_spec, utils.envs, ...) is imported
unstubbed, so a test breaks if one of those APIs drifts.
"""

import importlib
import importlib.abc
import importlib.machinery
import math
import sys
import types
from unittest.mock import MagicMock

import pytest

# ── round-mode constants (mirror aiter.utility.mx_types.MxScaleRoundModeInt) ──

ROUND_DOWN = 0
ROUND_UP = 1
EVEN = 2
CEIL = 3


class _FakeMxScaleRoundModeInt:
    """Stand-in for aiter.utility.mx_types.MxScaleRoundModeInt."""

    RoundDown = ROUND_DOWN
    RoundUp = ROUND_UP
    Even = EVEN
    Ceil = CEIL

    def __init__(self, v=0):
        self.value = int(v)

    def __int__(self):
        return self.value


# ── GPU-dependency stub (aiter / triton only) ──────────────────────────────────
#
# aiter and triton are the only imports that require a GPU build; everything
# else linear.py / quark/utils.py pull in (torch, regex, and the real atom.*
# modules) is present on a CPU runner. This finder answers imports for those
# two roots with auto-populating stub modules and nothing else, so real
# packages keep their real behavior.

_STUB_ROOTS = ("aiter", "triton")


def _make_stub_module(name):
    if name == "aiter.utility.mx_types":
        mod = types.ModuleType(name)
        mod.MxScaleRoundModeInt = _FakeMxScaleRoundModeInt
        mod.MX_DEFAULT_ROUND_MODE = ROUND_UP
        return mod
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda _attr: MagicMock()
    mod.__path__ = []
    return mod


class _GpuDepFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        root = name.split(".")[0]
        if root in _STUB_ROOTS:
            return importlib.machinery.ModuleSpec(name, self)
        return None

    def create_module(self, spec):
        return _make_stub_module(spec.name)

    def exec_module(self, module):
        pass


@pytest.fixture(scope="module", autouse=True)
def _modules_under_test():
    """Import the modules under test against stubbed GPU deps.

    Installs the aiter/triton finder, imports linear.py and quark/utils.py
    normally, exposes the functions under test, then removes the finder and
    the stub modules so nothing leaks into other test files.
    """
    finder = _GpuDepFinder()
    sys.meta_path.insert(0, finder)

    # A partial *real* aiter/triton build may already be in sys.modules (CI has
    # one; a CPU dev box does not). Evict those entries plus the two modules
    # under test so the finder owns every aiter/triton import during load, then
    # restore the originals on teardown so nothing leaks into other test files.
    evict = [
        "atom.model_ops.linear",
        "atom.quantization.quark.utils",
    ]
    evict += [n for n in sys.modules if n.split(".")[0] in _STUB_ROOTS]
    saved = {n: sys.modules.pop(n) for n in evict if n in sys.modules}
    try:
        linear = importlib.import_module("atom.model_ops.linear")
        utils = importlib.import_module("atom.quantization.quark.utils")
        yield types.SimpleNamespace(
            resolve=linear._resolve_mxfp4_act_round_mode,
            quant=utils.quant_mxfp4_online_even,
            utils_mod=utils,
        )
    finally:
        sys.meta_path.remove(finder)
        # Drop everything imported under the stubs, then restore what was there.
        for name in list(sys.modules):
            if name.split(".")[0] in _STUB_ROOTS or name in (
                "atom.model_ops.linear",
                "atom.quantization.quark.utils",
            ):
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture
def resolve(_modules_under_test):
    return _modules_under_test.resolve


@pytest.fixture
def quant(_modules_under_test):
    return _modules_under_test.quant


# ── helpers ───────────────────────────────────────────────────────────────────


class _FakeTensor:
    """CPU-only fake tensor with .shape, .dim(), .reshape(), .contiguous(), .to()."""

    def __init__(self, shape, dtype="bfloat16"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._numel = math.prod(shape)

    def dim(self):
        return len(self.shape)

    def contiguous(self):
        return self

    def to(self, dtype):
        return _FakeTensor(self.shape, dtype)

    def view(self, dtype):
        return _FakeTensor(self.shape, dtype)

    def reshape(self, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            new_shape = list(args[0])
        else:
            new_shape = list(args)
        neg = [i for i, s in enumerate(new_shape) if s == -1]
        if neg:
            known = math.prod(s for s in new_shape if s != -1)
            new_shape[neg[0]] = self._numel // known
        assert (
            math.prod(new_shape) == self._numel
        ), f"reshape {self.shape} → {new_shape} invalid"
        return _FakeTensor(new_shape, self.dtype)


def _make_fake_hip():
    """Return (fake_quant_mxfp4_hip, call_log) for injection into quark/utils."""
    call_log = []

    def _hip(w, round_mode):
        call_log.append({"shape": w.shape, "round_mode": int(round_mode)})
        rows, k = w.shape
        return _FakeTensor((rows, k // 2)), _FakeTensor((rows, k // 32))

    return _hip, call_log


# ── Tests: _resolve_mxfp4_act_round_mode ──────────────────────────────────────


class TestResolveMxfp4ActRoundMode:

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv("ATOM_ACT_QUANT_HIP_ROUNDUP", raising=False)

    def test_returns_even_by_default(self, resolve):
        assert resolve() == EVEN

    def test_env_var_forces_roundup(self, resolve, monkeypatch):
        """ATOM_ACT_QUANT_HIP_ROUNDUP=1 forces the legacy RoundUp path."""
        monkeypatch.setenv("ATOM_ACT_QUANT_HIP_ROUNDUP", "1")
        assert resolve() == ROUND_UP

    def test_env_var_zero_stays_even(self, resolve, monkeypatch):
        """ATOM_ACT_QUANT_HIP_ROUNDUP=0 keeps the default Even path."""
        monkeypatch.setenv("ATOM_ACT_QUANT_HIP_ROUNDUP", "0")
        assert resolve() == EVEN


# ── Tests: quant_mxfp4_online_even 3D reshape ─────────────────────────────────


class TestQuantMxfp4OnlineEven3DReshape:

    @pytest.fixture(autouse=True)
    def _inject_fake_hip(self, _modules_under_test):
        """Inject fake quant_mxfp4_hip via sys.modules so lazy imports resolve it."""
        fake_hip, call_log = _make_fake_hip()
        self._calls = call_log
        # utils.py does `from aiter.ops.quant import quant_mxfp4_hip` at call
        # time, so we patch the sys.modules entry that the lazy import will hit.
        sys.modules["aiter.ops.quant"].quant_mxfp4_hip = fake_hip
        _modules_under_test.utils_mod.quant_mxfp4_hip = fake_hip
        yield
        sys.modules["aiter.ops.quant"].quant_mxfp4_hip = MagicMock()
        _modules_under_test.utils_mod.quant_mxfp4_hip = MagicMock()

    def test_2d_weight_passes_unchanged(self, quant):
        """Standard [N, K] weight: no reshape, kernel called once with original shape."""
        q, s = quant(_FakeTensor((512, 128)))
        assert len(self._calls) == 1
        assert self._calls[0]["shape"] == (512, 128)
        assert q.shape == (512, 64)
        assert s.shape == (512, 4)

    def test_3d_moe_weight_reshaped_before_kernel(self, quant):
        """[num_experts, N, K] → kernel sees [num_experts*N, K]."""
        quant(_FakeTensor((8, 512, 128)))
        assert len(self._calls) == 1
        assert self._calls[0]["shape"] == (8 * 512, 128)

    def test_3d_output_leading_dim_restored(self, quant):
        """Packed weight and scale recover the [E, N, ...] leading dims."""
        q, s = quant(_FakeTensor((4, 256, 64)))
        assert q.shape == (4, 256, 32), f"q.shape={q.shape}"
        assert s.shape == (4, 256, 2), f"s.shape={s.shape}"

    def test_4d_weight_also_handled(self, quant):
        """[batch, experts, N, K] tensors reshape and restore correctly."""
        q, s = quant(_FakeTensor((2, 4, 128, 64)))
        assert self._calls[0]["shape"] == (2 * 4 * 128, 64)
        assert q.shape == (2, 4, 128, 32)
        assert s.shape == (2, 4, 128, 2)

    def test_even_round_mode_passed_to_kernel(self, quant):
        """quant_mxfp4_hip must receive Even (2) as round_mode for 2D input."""
        quant(_FakeTensor((64, 32)))
        assert self._calls[0]["round_mode"] == EVEN

    def test_3d_even_round_mode_preserved_through_reshape(self, quant):
        """Even round mode is not lost when the 3D reshape path is taken."""
        quant(_FakeTensor((8, 128, 64)))
        assert self._calls[0]["round_mode"] == EVEN

    @pytest.mark.parametrize(
        "num_experts,n,k",
        [
            (8, 512, 128),  # DeepSeek-R1 expert shape
            (256, 128, 64),  # Qwen3.5-35B-A3B expert shape
            (64, 256, 32),  # smaller expert
        ],
    )
    def test_various_moe_shapes(self, quant, num_experts, n, k):
        """3D MoE expert shapes all reshape, quantize, and restore correctly."""
        q, s = quant(_FakeTensor((num_experts, n, k)))
        assert self._calls[0]["shape"] == (num_experts * n, k)
        assert q.shape == (num_experts, n, k // 2)
        assert s.shape == (num_experts, n, k // 32)
