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

These tests run without a GPU: all aiter / torch GPU calls are mocked so
the real function implementations are imported and exercised directly.
"""

import atexit
import contextlib
import importlib.util
import math
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ATOM_ROOT = str(Path(__file__).resolve().parent.parent)


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


# ── sys.modules mock context ───────────────────────────────────────────────────


@contextlib.contextmanager
def _temporary_mocks(extra_patches=None):
    """Patch sys.modules with lightweight stubs covering all aiter / torch imports."""
    mx_types_mod = types.ModuleType("aiter.utility.mx_types")
    mx_types_mod.MxScaleRoundModeInt = _FakeMxScaleRoundModeInt
    mx_types_mod.MX_DEFAULT_ROUND_MODE = ROUND_UP

    dtypes_mod = MagicMock()
    dtypes_mod.fp4x2 = "fp4x2"
    dtypes_mod.fp8_e8m0 = "fp8_e8m0"

    aiter_mod = types.ModuleType("aiter")
    aiter_mod.QuantType = MagicMock()
    aiter_mod.dtypes = dtypes_mod
    aiter_mod.get_hip_quant = MagicMock()
    # Expose all the symbols linear.py imports from aiter at the top level
    for sym in [
        "gemm_a4w4",
        "gemm_a8w8",
        "gemm_a8w8_blockscale_bpreshuffle",
        "gemm_a8w8_bpreshuffle",
        "gemm_a8w8_blockscale",
        "get_hip_quant",
        "dtypes",
    ]:
        setattr(aiter_mod, sym, MagicMock())
    aiter_mod.__path__ = []

    aiter_ops_quant = types.ModuleType("aiter.ops.quant")
    # quant_mxfp4_hip will be replaced per-test; provide a default stub
    aiter_ops_quant.quant_mxfp4_hip = MagicMock()

    mock_torch = MagicMock()

    # Only stub torch when it is not already installed.  In the ATOM CI
    # environment torch is always present; replacing it would break other
    # test modules that compare real torch types (e.g. torch.uint8).
    _torch_patches = (
        {}
        if "torch" in sys.modules
        else {"torch": mock_torch, "torch.nn": mock_torch.nn}
    )

    patches = {
        **_torch_patches,
        "torch.distributed": MagicMock(),
        # aiter top-level and sub-modules
        "aiter": aiter_mod,
        "aiter.dtypes": dtypes_mod,
        "aiter.ops": types.ModuleType("aiter.ops"),
        "aiter.ops.quant": aiter_ops_quant,
        "aiter.ops.triton": MagicMock(),
        "aiter.ops.triton.gemm_afp4wfp4": MagicMock(),
        "aiter.ops.triton.gemm": MagicMock(),
        "aiter.ops.triton.gemm.basic": MagicMock(),
        "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale": MagicMock(),
        "aiter.ops.triton.gemm.basic.gemm_a8w8": MagicMock(),
        "aiter.dist": types.ModuleType("aiter.dist"),
        "aiter.dist.parallel_state": MagicMock(),
        "aiter.jit": types.ModuleType("aiter.jit"),
        "aiter.jit.utils": types.ModuleType("aiter.jit.utils"),
        "aiter.jit.utils.torch_guard": MagicMock(),
        "aiter.tuned_gemm": MagicMock(),
        "aiter.utility": MagicMock(),
        "aiter.utility.mx_types": mx_types_mod,
        "aiter.utility.dtypes": MagicMock(),
        "aiter.utility.fp4_utils": MagicMock(),
        # atom dependencies (not under test)
        "atom.config": MagicMock(),
        "atom.quant_spec": MagicMock(),
        "atom.model_ops.utils": MagicMock(),
        "atom.utils": MagicMock(),
        "atom.utils.envs": MagicMock(),
        "atom.utils.decorators": MagicMock(),
        "atom.quantization": types.ModuleType("atom.quantization"),
        "atom.quantization.quark": types.ModuleType("atom.quantization.quark"),
        "atom.quantization.quark.utils": MagicMock(),
        # other heavy deps
        "regex": MagicMock(),
        "triton": MagicMock(),
        "triton.language": MagicMock(),
    }

    if extra_patches:
        patches.update(extra_patches)

    saved = {}
    for name, mock in patches.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock
    try:
        yield patches
    finally:
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


def _load_module(rel_path: str, module_name: str):
    """Load a source file by path with sys.modules already patched."""
    path = os.path.join(ATOM_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── install mocks and load real modules ───────────────────────────────────────
# Lazy imports inside function bodies (e.g. `from aiter.utility.mx_types ...`)
# execute at call time, so the stubs must stay in sys.modules for the full
# test session.  We install the patches now and register a pytest session-end
# finalizer to restore the original sys.modules entries, preventing leakage
# into other test modules that run after ours.

_mocks_ctx = _temporary_mocks()
_installed_patches = _mocks_ctx.__enter__()
# Restore sys.modules when the process exits so we don't pollute other
# test modules that run in the same pytest session (e.g. test_lmcache_*).
atexit.register(_mocks_ctx.__exit__, None, None, None)

_linear = _load_module("atom/model_ops/linear.py", "atom.model_ops.linear")
_utils = _load_module(
    "atom/quantization/quark/utils.py", "atom.quantization.quark.utils"
)

_resolve_mxfp4_act_round_mode = _linear._resolve_mxfp4_act_round_mode
quant_mxfp4_online_even = _utils.quant_mxfp4_online_even


# ── helpers ───────────────────────────────────────────────────────────────────


class _FakeQuantConfig:
    """Minimal stand-in for QuantizationConfig."""

    def __init__(self, raw=None):
        self._raw = raw

    @property
    def global_quant_config(self):
        return self._raw


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
    """Return (fake_quant_mxfp4_hip, call_log) for injection into _utils."""
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

    def test_returns_even_by_default_no_config(self):
        assert _resolve_mxfp4_act_round_mode(None) == EVEN

    def test_reads_even_from_config(self):
        cfg = _FakeQuantConfig({"input_tensors": {"scale_calculation_mode": "even"}})
        assert _resolve_mxfp4_act_round_mode(cfg) == EVEN

    def test_reads_floor_from_config(self):
        cfg = _FakeQuantConfig({"input_tensors": {"scale_calculation_mode": "floor"}})
        assert _resolve_mxfp4_act_round_mode(cfg) == ROUND_DOWN

    def test_reads_ceil_from_config(self):
        cfg = _FakeQuantConfig({"input_tensors": {"scale_calculation_mode": "ceil"}})
        assert _resolve_mxfp4_act_round_mode(cfg) == CEIL

    def test_reads_round_up_from_config(self):
        cfg = _FakeQuantConfig(
            {"input_tensors": {"scale_calculation_mode": "round_up"}}
        )
        assert _resolve_mxfp4_act_round_mode(cfg) == ROUND_UP

    def test_case_insensitive(self):
        cfg = _FakeQuantConfig({"input_tensors": {"scale_calculation_mode": "EVEN"}})
        assert _resolve_mxfp4_act_round_mode(cfg) == EVEN

    def test_unknown_scale_mode_falls_back_to_even(self):
        cfg = _FakeQuantConfig({"input_tensors": {"scale_calculation_mode": "weird"}})
        assert _resolve_mxfp4_act_round_mode(cfg) == EVEN

    def test_missing_input_tensors_falls_back_to_even(self):
        cfg = _FakeQuantConfig({})
        assert _resolve_mxfp4_act_round_mode(cfg) == EVEN

    def test_non_dict_global_quant_config_falls_back_to_even(self):
        """global_quant_config='ptpc_fp8' (online quant string) → Even fallback."""
        cfg = _FakeQuantConfig("ptpc_fp8")
        assert _resolve_mxfp4_act_round_mode(cfg) == EVEN

    def test_env_var_forces_roundup(self, monkeypatch):
        """ATOM_ACT_QUANT_HIP_ROUNDUP=1 overrides Even config."""
        monkeypatch.setenv("ATOM_ACT_QUANT_HIP_ROUNDUP", "1")
        cfg = _FakeQuantConfig({"input_tensors": {"scale_calculation_mode": "even"}})
        assert _resolve_mxfp4_act_round_mode(cfg) == ROUND_UP

    def test_env_var_zero_does_not_force_roundup(self, monkeypatch):
        """ATOM_ACT_QUANT_HIP_ROUNDUP=0 does not override config."""
        monkeypatch.setenv("ATOM_ACT_QUANT_HIP_ROUNDUP", "0")
        cfg = _FakeQuantConfig({"input_tensors": {"scale_calculation_mode": "even"}})
        assert _resolve_mxfp4_act_round_mode(cfg) == EVEN

    def test_exception_in_config_access_falls_back_to_even(self):
        """Broken quant_config (property raises) → Even fallback, no crash."""

        class _Broken:
            @property
            def global_quant_config(self):
                raise RuntimeError("intentional")

        assert _resolve_mxfp4_act_round_mode(_Broken()) == EVEN

    @pytest.mark.parametrize(
        "model,expected_mode",
        [
            ("Kimi-K2.6-MXFP4", EVEN),
            ("Qwen3.5-35B-A3B-MXFP4", EVEN),
            ("GLM-5-MXFP4", EVEN),
        ],
    )
    def test_known_quark_checkpoint_modes(self, model, expected_mode):
        """All known AMD MXFP4 checkpoints use Even — verify dispatch is correct."""
        cfg = _FakeQuantConfig({"input_tensors": {"scale_calculation_mode": "even"}})
        assert (
            _resolve_mxfp4_act_round_mode(cfg) == expected_mode
        ), f"{model}: expected Even ({expected_mode})"


# ── Tests: quant_mxfp4_online_even 3D reshape ─────────────────────────────────


class TestQuantMxfp4OnlineEven3DReshape:

    @pytest.fixture(autouse=True)
    def _inject_fake_hip(self):
        """Inject fake quant_mxfp4_hip via sys.modules so lazy imports resolve it."""
        fake_hip, call_log = _make_fake_hip()
        self._calls = call_log
        # utils.py does `from aiter.ops.quant import quant_mxfp4_hip` at call
        # time, so we patch the sys.modules entry that the lazy import will hit.
        sys.modules["aiter.ops.quant"].quant_mxfp4_hip = fake_hip
        _utils.quant_mxfp4_hip = fake_hip  # also patch module attr for safety
        yield
        sys.modules["aiter.ops.quant"].quant_mxfp4_hip = MagicMock()
        _utils.quant_mxfp4_hip = MagicMock()

    def test_2d_weight_passes_unchanged(self):
        """Standard [N, K] weight: no reshape, kernel called once with original shape."""
        q, s = quant_mxfp4_online_even(_FakeTensor((512, 128)))
        assert len(self._calls) == 1
        assert self._calls[0]["shape"] == (512, 128)
        assert q.shape == (512, 64)
        assert s.shape == (512, 4)

    def test_3d_moe_weight_reshaped_before_kernel(self):
        """[num_experts, N, K] → kernel sees [num_experts*N, K]."""
        quant_mxfp4_online_even(_FakeTensor((8, 512, 128)))
        assert len(self._calls) == 1
        assert self._calls[0]["shape"] == (8 * 512, 128)

    def test_3d_output_leading_dim_restored(self):
        """Packed weight and scale recover the [E, N, ...] leading dims."""
        q, s = quant_mxfp4_online_even(_FakeTensor((4, 256, 64)))
        assert q.shape == (4, 256, 32), f"q.shape={q.shape}"
        assert s.shape == (4, 256, 2), f"s.shape={s.shape}"

    def test_4d_weight_also_handled(self):
        """[batch, experts, N, K] tensors reshape and restore correctly."""
        q, s = quant_mxfp4_online_even(_FakeTensor((2, 4, 128, 64)))
        assert self._calls[0]["shape"] == (2 * 4 * 128, 64)
        assert q.shape == (2, 4, 128, 32)
        assert s.shape == (2, 4, 128, 2)

    def test_even_round_mode_passed_to_kernel(self):
        """quant_mxfp4_hip must receive Even (2) as round_mode for 2D input."""
        quant_mxfp4_online_even(_FakeTensor((64, 32)))
        assert self._calls[0]["round_mode"] == EVEN

    def test_3d_even_round_mode_preserved_through_reshape(self):
        """Even round mode is not lost when the 3D reshape path is taken."""
        quant_mxfp4_online_even(_FakeTensor((8, 128, 64)))
        assert self._calls[0]["round_mode"] == EVEN

    @pytest.mark.parametrize(
        "num_experts,n,k",
        [
            (8, 512, 128),  # DeepSeek-R1 expert shape
            (256, 128, 64),  # Qwen3.5-35B-A3B expert shape
            (64, 256, 32),  # smaller expert
        ],
    )
    def test_various_moe_shapes(self, num_experts, n, k):
        """3D MoE expert shapes all reshape, quantize, and restore correctly."""
        q, s = quant_mxfp4_online_even(_FakeTensor((num_experts, n, k)))
        assert self._calls[0]["shape"] == (num_experts * n, k)
        assert q.shape == (num_experts, n, k // 2)
        assert s.shape == (num_experts, n, k // 32)
