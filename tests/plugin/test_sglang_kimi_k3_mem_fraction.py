"""Kimi-K3 mem_fraction restore predicates (CPU / mocked SGLang only)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Bridge imports torch at module scope; stub it so CPU hosts without CUDA libs
# can still exercise the restore predicates.
sys.modules.setdefault("torch", MagicMock())

from atom.plugin.sglang.kimi_k3_bridge import (
    _KIMI_K3_MEM_FRACTION_OVERRIDE,
    _kimi_k3_mem_fraction_already_restored,
    _restore_kimi_k3_mem_fraction,
)


def test_already_restored_calls_overrides_log_as_method():
    """0.5.20 exposes overrides_log as a method — must call it, not iterate it."""
    ctx = MagicMock()
    ctx.overrides_log = MagicMock(
        return_value=[
            (_KIMI_K3_MEM_FRACTION_OVERRIDE, {"mem_fraction_static": 0.9}),
        ]
    )
    assert _kimi_k3_mem_fraction_already_restored(ctx) is True
    ctx.overrides_log.assert_called_once_with()


def test_already_restored_empty_method_log():
    ctx = MagicMock()
    ctx.overrides_log = MagicMock(return_value=[])
    assert _kimi_k3_mem_fraction_already_restored(ctx) is False


def _install_sglang_stubs(*, ctx, exec_bag, schedule, honor_explicit: bool):
    rt = types.ModuleType("sglang.srt.runtime_context")
    rt.get_context = lambda: ctx
    rt.get_exec = lambda: exec_bag
    rt.get_schedule = lambda: schedule

    envs = MagicMock()
    envs.SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION.get.return_value = honor_explicit
    environ = types.ModuleType("sglang.srt.environ")
    environ.envs = envs

    return {
        "sglang": MagicMock(),
        "sglang.srt": MagicMock(),
        "sglang.srt.runtime_context": rt,
        "sglang.srt.environ": environ,
    }


def test_restore_overrides_once_then_skips():
    log: list[tuple] = []

    def overrides_log():
        return list(log)

    def override(source, **fields):
        log.append((source, dict(fields)))

    ctx = SimpleNamespace(
        server_args=SimpleNamespace(_raw_input={"mem_fraction_static": 0.85}),
        overrides_log=overrides_log,
        override=override,
    )
    exec_bag = SimpleNamespace(kernel=SimpleNamespace(attention_backend="aiter"))
    # Upstream scaled operator input 0.85 by *0.85 → schedule holds 0.7225.
    schedule = SimpleNamespace(mem_fraction_static=0.85 * 0.85)
    owner = SimpleNamespace(model_config=SimpleNamespace(context_len=16384))

    stubs = _install_sglang_stubs(
        ctx=ctx, exec_bag=exec_bag, schedule=schedule, honor_explicit=False
    )
    with patch.dict(sys.modules, stubs):
        _restore_kimi_k3_mem_fraction(owner)
        _restore_kimi_k3_mem_fraction(owner)

    assert len(log) == 1
    assert log[0][0] == _KIMI_K3_MEM_FRACTION_OVERRIDE
    assert log[0][1]["mem_fraction_static"] == 0.85


def test_restore_skips_when_honor_explicit_mem_fraction():
    """HONOR on + explicit flag: upstream did not *0.85, so schedule is 0.85.

    If the env lookup is wrong, restore would divide 0.85/0.85 and override 1.0.
    """
    override = MagicMock()
    ctx = SimpleNamespace(
        server_args=SimpleNamespace(_raw_input={"mem_fraction_static": 0.85}),
        overrides_log=lambda: [],  # noqa: PIE807
        override=override,
    )
    exec_bag = SimpleNamespace(kernel=SimpleNamespace(attention_backend="aiter"))
    schedule = SimpleNamespace(mem_fraction_static=0.85)
    owner = SimpleNamespace(model_config=SimpleNamespace(context_len=16384))

    stubs = _install_sglang_stubs(
        ctx=ctx, exec_bag=exec_bag, schedule=schedule, honor_explicit=True
    )
    with patch.dict(sys.modules, stubs):
        _restore_kimi_k3_mem_fraction(owner)

    override.assert_not_called()


@pytest.mark.parametrize("resolved_backend", ["", "auto"])
def test_restore_skips_unresolved_attention_backend(resolved_backend):
    """Restore must read resolved exec.kernel.attention_backend, not raw input."""
    override = MagicMock()
    ctx = SimpleNamespace(
        server_args=SimpleNamespace(
            attention_backend="aiter",
            _raw_input={"mem_fraction_static": 0.85},
        ),
        overrides_log=lambda: [],  # noqa: PIE807
        override=override,
    )
    exec_bag = SimpleNamespace(
        kernel=SimpleNamespace(attention_backend=resolved_backend)
    )
    # If restore wrongly used raw "aiter", 0.7225/0.85 would override 0.85.
    schedule = SimpleNamespace(mem_fraction_static=0.85 * 0.85)
    owner = SimpleNamespace(model_config=SimpleNamespace(context_len=16384))

    stubs = _install_sglang_stubs(
        ctx=ctx, exec_bag=exec_bag, schedule=schedule, honor_explicit=False
    )
    with patch.dict(sys.modules, stubs):
        _restore_kimi_k3_mem_fraction(owner)

    override.assert_not_called()
