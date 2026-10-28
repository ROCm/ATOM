# SPDX-License-Identifier: MIT
"""Prefill CP on HIP when ATOM's SGLang model package is active.

SGLang 0.5.20 rejects --enable-prefill-cp on HIP in its
`validate_prefill_cp_platform` resolution step. The ATOM plugin replaces that
step through SGLang's resolution-hook registry so ATOM models, which run
prefill CP themselves, can still use it on ROCm. Only that combination may
skip the built-in check.

SGLang is stubbed so CPU CI collects this with no SGLang present.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from atom.plugin.sglang import register

ATOM_MODEL_PACKAGE = "atom.plugin.sglang.models"


def _module(name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def sglang(monkeypatch):
    hooks = {}
    state = SimpleNamespace(hooks=hooks, is_hip=True, package=ATOM_MODEL_PACKAGE)

    def register_resolution_hook(name):
        def decorator(fn):
            hooks.setdefault(name, []).append(fn)
            return fn

        return decorator

    envs = SimpleNamespace(
        SGLANG_EXTERNAL_MODEL_PACKAGE=SimpleNamespace(get=lambda: state.package)
    )
    for name, module in {
        "sglang": _module("sglang"),
        "sglang.srt": _module("sglang.srt"),
        "sglang.srt.arg_groups": _module("sglang.srt.arg_groups"),
        "sglang.srt.arg_groups.resolution_hooks": _module(
            "sglang.srt.arg_groups.resolution_hooks",
            register_resolution_hook=register_resolution_hook,
        ),
        "sglang.srt.utils": _module("sglang.srt.utils", is_hip=lambda: state.is_hip),
        "sglang.srt.environ": _module("sglang.srt.environ", envs=envs),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(register, "_PREFILL_CP_PLATFORM_HOOK_REGISTERED", False)
    return state


def _run_step(state):
    calls = []
    (hook,) = state.hooks["validate_prefill_cp_platform"]
    hook(SimpleNamespace(enable_prefill_cp=True), calls.append)
    return calls


def test_hip_with_atom_models_skips_builtin_rejection(sglang):
    register._allow_prefill_cp_on_hip_for_atom()

    assert _run_step(sglang) == []


@pytest.mark.parametrize(
    ("is_hip", "package"),
    [(True, None), (True, "other.models"), (False, ATOM_MODEL_PACKAGE)],
)
def test_other_setups_keep_builtin_check(sglang, is_hip, package):
    sglang.is_hip = is_hip
    sglang.package = package
    register._allow_prefill_cp_on_hip_for_atom()

    assert len(_run_step(sglang)) == 1


def test_registers_once(sglang):
    register._allow_prefill_cp_on_hip_for_atom()
    register._allow_prefill_cp_on_hip_for_atom()

    assert len(sglang.hooks["validate_prefill_cp_platform"]) == 1


def test_sglang_without_hook_registry_is_a_no_op(sglang, monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang.srt.arg_groups.resolution_hooks", None)

    register._allow_prefill_cp_on_hip_for_atom()

    assert sglang.hooks == {}
    assert register._PREFILL_CP_PLATFORM_HOOK_REGISTERED is False
