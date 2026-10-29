# SPDX-License-Identifier: MIT
"""Prefill CP on HIP when ATOM's SGLang model package is active.

SGLang 0.5.20 rejects --enable-prefill-cp on HIP in its
`validate_prefill_cp_platform` resolution step. The ATOM plugin replaces that
step through SGLang's resolution-hook registry so ATOM models, which run
prefill CP themselves, can still use it on ROCm. Only that combination may
skip the built-in check.

At runtime 0.5.20 routes CP extend through `EagerRunner._execute_extend_cp`,
which shards inputs itself and bypasses the model wrapper forward. The plugin
sends ATOM wrappers back through their own forward with the full batch.

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


class _Model:
    def __init__(self, atom_owns_prefill_cp):
        if atom_owns_prefill_cp:
            self.atom_owns_prefill_cp = True
        self.forward_calls = []

    def forward(self, input_ids, positions, forward_batch, **kwargs):
        self.forward_calls.append((input_ids, positions, forward_batch, kwargs))
        return "atom_forward_output"


@pytest.fixture
def eager_runner(monkeypatch):
    class EagerRunner:
        def __init__(self, model):
            self.model_runner = SimpleNamespace(model=model)
            self.builtin_calls = []

        def _execute_extend_cp(self, forward_batch, kwargs):
            self.builtin_calls.append((forward_batch, kwargs))
            return "builtin_cp_output"

    for name, module in {
        "sglang": _module("sglang"),
        "sglang.srt": _module("sglang.srt"),
        "sglang.srt.model_executor": _module("sglang.srt.model_executor"),
        "sglang.srt.model_executor.runner": _module("sglang.srt.model_executor.runner"),
        "sglang.srt.model_executor.runner.eager_runner": _module(
            "sglang.srt.model_executor.runner.eager_runner", EagerRunner=EagerRunner
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return EagerRunner


def _forward_batch():
    return SimpleNamespace(input_ids="full_input_ids", positions="full_positions")


def test_cp_extend_runs_atom_wrapper_forward_on_full_batch(eager_runner):
    register._route_prefill_cp_to_atom_model_forward()
    model = _Model(atom_owns_prefill_cp=True)
    runner = eager_runner(model)
    forward_batch = _forward_batch()
    kwargs = {"pp_proxy_tensors": None}

    assert runner._execute_extend_cp(forward_batch, kwargs) == "atom_forward_output"
    assert model.forward_calls == [
        ("full_input_ids", "full_positions", forward_batch, kwargs)
    ]
    assert runner.builtin_calls == []


def test_cp_extend_keeps_builtin_path_for_other_models(eager_runner):
    register._route_prefill_cp_to_atom_model_forward()
    model = _Model(atom_owns_prefill_cp=False)
    runner = eager_runner(model)
    forward_batch = _forward_batch()

    assert runner._execute_extend_cp(forward_batch, {}) == "builtin_cp_output"
    assert runner.builtin_calls == [(forward_batch, {})]
    assert model.forward_calls == []


def test_cp_extend_patch_installs_once(eager_runner):
    register._route_prefill_cp_to_atom_model_forward()
    patched = eager_runner._execute_extend_cp
    register._route_prefill_cp_to_atom_model_forward()

    assert eager_runner._execute_extend_cp is patched


def test_sglang_without_cp_extend_runner_is_a_no_op(eager_runner):
    del eager_runner._execute_extend_cp

    register._route_prefill_cp_to_atom_model_forward()

    assert not hasattr(eager_runner, "_execute_extend_cp")
    assert not getattr(eager_runner, "_atom_prefill_cp_patched", False)
