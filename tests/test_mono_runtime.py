# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mono framework's host mechanisms that need no GPU: TP agreement, kernel
argument tables, compile-only builds, and the framework's dependency rules."""

import ast
import os
import pathlib
import tempfile

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.compile import (
    _COMPILE_ONLY,
    compile_only,
    kernel_resources,
    scratch_bytes,
)
from atom.mono.runtime.consensus import tp_agree

FRAMEWORK = pathlib.Path(__file__).resolve().parents[1] / "atom" / "mono"


class _Kernel:
    """What ``@flyc.kernel`` leaves: the Python function as ``_func``."""

    def __init__(self, fn):
        self._func = fn


class _Launcher:
    """What ``@flyc.jit`` leaves: the Python function as ``func``."""

    def __init__(self, fn):
        self.func = fn


def test_pack_orders_by_name():
    abi = KernelAbi(("a", "b", "c"))
    assert abi.pack({"c": 3, "a": 1, "b": 2}) == [1, 2, 3]
    assert abi.zeros() == [0, 0, 0]


@pytest.mark.parametrize(
    "values",
    [{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3, "d": 4}, {"a": 1, "b": 2, "d": 4}],
)
def test_pack_rejects_a_missing_or_unexpected_name(values):
    with pytest.raises(TypeError, match="ABI mismatch"):
        KernelAbi(("a", "b", "c")).pack(values)


def test_check_accepts_matching_signatures():
    def kernel(a, b):
        pass

    def launcher(a, b, stream=None):
        pass

    KernelAbi(("a", "b")).check(_Kernel(kernel), _Launcher(launcher))


def test_check_rejects_a_reordered_kernel():
    def kernel(b, a):
        pass

    def launcher(a, b, stream=None):
        pass

    with pytest.raises(TypeError, match="kernel parameters"):
        KernelAbi(("a", "b")).check(_Kernel(kernel), _Launcher(launcher))


def test_check_rejects_a_launcher_without_stream():
    def kernel(a, b):
        pass

    def launcher(a, b):
        pass

    with pytest.raises(TypeError, match="launcher parameters"):
        KernelAbi(("a", "b")).check(_Kernel(kernel), _Launcher(launcher))


@pytest.mark.parametrize("before", [None, "0"])
def test_compile_only_is_scoped(monkeypatch, before):
    if before is None:
        monkeypatch.delenv(_COMPILE_ONLY, raising=False)
    else:
        monkeypatch.setenv(_COMPILE_ONLY, before)
    with pytest.raises(RuntimeError), compile_only():
        assert os.environ[_COMPILE_ONLY] == "1"
        raise RuntimeError
    assert os.environ.get(_COMPILE_ONLY) == before


def test_agree_alone():
    assert tp_agree(True, None) and not tp_agree(False, None)


def _agree_worker(rank, world, init, refuser, out):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    out[rank] = tp_agree(rank != refuser, dist.group.WORLD)
    dist.destroy_process_group()


@pytest.mark.parametrize("refuser", [None, 0, 2])
def test_one_refusing_rank_turns_every_rank_down(refuser):
    world = 3
    with tempfile.TemporaryDirectory() as d:
        out = mp.Manager().dict()
        mp.spawn(
            _agree_worker,
            args=(world, f"file://{d}/store", refuser, out),
            nprocs=world,
            join=True,
        )
    assert dict(out) == {r: refuser is None for r in range(world)}


def _module_files():
    return sorted(FRAMEWORK.rglob("*.py"))


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_framework_imports_no_model(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            assert not name.startswith("atom.models"), f"{path.name} imports {name}"


def test_the_framework_package_is_found():
    # the parametrized rule above passes vacuously on an empty glob
    assert _module_files(), FRAMEWORK


# --------------------------------------------------------------- shared runner parts
from atom.mono.runtime import route
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.widths import WidthBuilds


def test_bind_agreed_reraises_this_ranks_refusal():
    def refuse():
        raise MonoUnsupported("no drafter")

    with pytest.raises(MonoUnsupported, match="no drafter"):
        bind_agreed(refuse, None)
    bind_agreed(lambda: None, None)


def test_bind_agreed_converts_other_failures_to_refusal():
    """A non-refusal failure must still leave every rank on the same path.

    Peers only ever see MonoUnsupported, so a rank whose bind raised something
    else -- an OOM, typically -- has to refuse the same way instead of dying
    alone with an exception LazyRunner does not catch."""

    import torch

    def oom():
        raise torch.OutOfMemoryError("out of memory")

    with pytest.raises(MonoUnsupported, match="OutOfMemoryError: out of memory"):
        bind_agreed(oom, None)


def _mlir_escaped(data: bytes) -> str:
    return "".join(
        chr(b) if 0x20 <= b < 0x7F and b not in b'\\"' else f"\\{b:02X}" for b in data
    )


class _CompiledLauncher:
    """A compiled ``@flyc.jit`` launcher as ``scratch_bytes`` reads it: one
    artifact whose module text holds a code object's scratch size (``value``:
    its msgpack encoding)."""

    def __init__(self, value: bytes):
        elf = b"\x7fELF\x02\x01" + b"\x9b.private_segment_fixed_size" + value + b"\x00"
        text = (
            f'gpu.binary @m [#gpu.object<#rocdl.target, bin = "{_mlir_escaped(elf)}">]'
        )
        self._mem_cache = {"key": type("Artifact", (), {"_ir_text": text})()}

    def __call__(self, *args):  # already compiled: the compile-only call is a no-op
        pass


class _NoArgs:
    @staticmethod
    def zeros():
        return ()


NO_SCRATCH = _CompiledLauncher(b"\x00")


@pytest.mark.parametrize(
    "value, size",
    [(b"\x00", 0), (b"\x24", 36), (b"\xcc\x90", 144), (b"\xcd\x14\x8c", 5260)],
)
def test_scratch_bytes_reads_the_code_objects_private_segment(value, size):
    assert scratch_bytes(_CompiledLauncher(value)) == size


def _note(key: bytes, value: bytes) -> bytes:
    return bytes([0xA0 | len(key)]) + key + value


@pytest.mark.parametrize(
    "symbol, name",
    [
        (b"_ZN4atom12v41_layer_s6E.kd", "atom::v41_layer_s6"),
        (b"plain_kernel.kd", "plain_kernel"),
    ],
)
def test_kernel_resources_reads_name_registers_scratch_lds_and_spills(symbol, name):
    elf = b"\x7fELF" + b"".join(
        [
            _note(b".vgpr_count", b"\xcc\x9d"),
            _note(b".sgpr_count", b"\x6a"),
            _note(b".private_segment_fixed_size", b"\x00"),
            _note(b".group_segment_fixed_size", b"\xce\x00\x01\x84\xa0"),
            _note(b".sgpr_spill_count", b"\xcc\x8d"),
            b"\xa7.symbol" + b"\xd9" + bytes([len(symbol)]) + symbol,
        ]
    )
    assert kernel_resources(elf) == {
        "name": name, "vgpr": 157, "sgpr": 106, "scratch": 0, "lds": 99488,
        "vgpr_spill": None, "sgpr_spill": 141,
    }  # fmt: skip


def test_scratch_bytes_refuses_a_launcher_without_a_binary():
    launcher = type("Uncompiled", (), {"_mem_cache": {}})()
    with pytest.raises(RuntimeError, match="no compiled binary"):
        scratch_bytes(launcher)


def test_a_width_is_built_and_compiled_once():
    built, compiled = [], []

    def build(rows):
        built.append(rows)
        return f"k{rows}"

    def kernels(k):
        compiled.append(k)
        return [(NO_SCRATCH, _NoArgs)]

    widths = WidthBuilds(build, kernels, None, "test")
    assert widths.prepare(6) and widths.prepare(6)
    assert built == [6] and compiled == ["k6"] and widths[6] == "k6"


def test_a_width_that_fails_to_build_is_not_kept():
    def build(rows):
        raise RuntimeError("trace error")

    widths = WidthBuilds(build, lambda k: [(NO_SCRATCH, _NoArgs)], None, "test")
    assert not widths.prepare(12)
    with pytest.raises(KeyError):
        widths[12]


def test_a_refused_width_is_not_built_again():
    tries = []

    def build(rows):
        tries.append(rows)
        raise ValueError("LDS outgrows a CU's")

    widths = WidthBuilds(build, lambda k: [(NO_SCRATCH, _NoArgs)], None, "test")
    assert not widths.prepare(48) and not widths.prepare(48)
    assert tries == [48]


def test_a_kernel_with_scratch_refuses_its_width_with_a_warning(caplog):
    tries = []

    def build(rows):
        tries.append(rows)
        return f"k{rows}"

    widths = WidthBuilds(
        build,
        lambda k: [(NO_SCRATCH, _NoArgs), (_CompiledLauncher(b"\x34"), _NoArgs)],
        None,
        "test",
    )
    with caplog.at_level("WARNING", logger="atom"):
        assert not widths.prepare(48) and not widths.prepare(48)
    assert tries == [48]
    assert "52 B of scratch" in caplog.text
    with pytest.raises(KeyError):
        widths[48]


class _Runner:
    def __init__(self, ok=True):
        self.ok = ok

    def prepare(self, rows):
        return self.ok


def test_lazy_runner_waits_out_a_capture(monkeypatch):
    capturing = [True]
    monkeypatch.setattr(
        route.torch.cuda, "is_current_stream_capturing", lambda: capturing[0]
    )
    mono = route.LazyRunner(_Runner, "test")
    assert not mono.ready(6) and mono.runner is None and mono.enabled
    capturing[0] = False
    assert mono.ready(6) and mono.runner is not None


def test_lazy_runner_create_refusal_is_for_good(monkeypatch):
    """A `create()` refusal is a deployment verdict: mono cannot serve at all."""
    monkeypatch.setattr(route.torch.cuda, "is_current_stream_capturing", lambda: False)

    def create():
        raise MonoUnsupported("TP 8")

    mono = route.LazyRunner(create, "test")
    assert not mono.ready(6) and not mono.enabled and not mono.ready(6)


def test_lazy_runner_width_refusal_spares_other_widths(monkeypatch):
    """A `prepare(width)` refusal is about that width and nothing else.

    A step padded to an uncaptured width must not take mono away from the
    widths that do build -- the runner stays enabled and a later good width
    still runs.
    """
    monkeypatch.setattr(route.torch.cuda, "is_current_stream_capturing", lambda: False)

    class _ByWidth:
        def prepare(self, rows):
            return rows != 20

    mono = route.LazyRunner(_ByWidth, "test")
    assert not mono.ready(20)  # the padded width is declined
    assert mono.enabled  # ...but mono is not switched off
    assert mono.ready(6)  # ...and a width that builds still runs
    assert not mono.ready(20)  # the refused width stays refused


def test_aiter_stub_teardown_leaves_no_second_copy():
    """Dropping a module must clear its parent's attribute too.

    `sys.modules.pop` alone leaves the parent package bound to the same object,
    so a later import builds a fresh module while attribute access still finds
    the old one -- two copies of every class in it, and a dotted-path
    monkeypatch can reach the copy the test is not holding.
    """
    import sys
    import types

    from tests.aiter_stub import _drop

    pkg = types.ModuleType("fake_pkg")
    sub = types.ModuleType("fake_pkg.sub")
    pkg.sub = sub
    sys.modules["fake_pkg"] = pkg
    sys.modules["fake_pkg.sub"] = sub
    try:
        _drop("fake_pkg.sub")
        assert "fake_pkg.sub" not in sys.modules
        assert not hasattr(pkg, "sub"), "parent still points at the dropped module"
    finally:
        sys.modules.pop("fake_pkg.sub", None)
        sys.modules.pop("fake_pkg", None)


def test_aiter_stub_teardown_spares_modules_that_took_nothing():
    """The teardown must drop poisoned `atom` modules and only those.

    Dropping every `atom` module imported under the stub is the two-copies
    bug from the other end: a caller still holding a module that never
    touched AITER gets a second copy of its classes on the next import.
    `test_scheduler_metrics` hit exactly that -- `engine_core_protocol` has
    no AITER dependency, and pickling an enum member out of it raised
    `PicklingError` once the module had been dropped and re-imported.

    Synthetic modules, so the assertion does not depend on which real module
    some earlier test happened to import first.
    """
    import sys
    import types

    from tests.aiter_stub import stubbed_aiter

    # Force the stub path: with a real AITER present nothing is installed and
    # the sweep under test never runs.
    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "aiter" or name.startswith("aiter.")
    }
    clean = types.ModuleType("atom.took_nothing")
    poisoned = types.ModuleType("atom.took_a_mock")
    # Holds a class *from* the poisoned module and nothing from AITER itself,
    # which is how `engine_core` relates to `async_proc`. Keeping it would
    # leave the stale second copy `conftest.atom_modules_are_imported_once`
    # fails on, so the taint has to reach it transitively.
    importer = types.ModuleType("atom.imports_the_poisoned_one")

    class Borrowed:
        pass

    Borrowed.__module__ = "atom.took_a_mock"
    try:
        with stubbed_aiter():
            import aiter

            poisoned.borrowed = aiter.some_symbol
            poisoned.Borrowed = Borrowed
            importer.Borrowed = Borrowed
            sys.modules["atom.took_nothing"] = clean
            sys.modules["atom.took_a_mock"] = poisoned
            sys.modules["atom.imports_the_poisoned_one"] = importer

        assert (
            sys.modules.get("atom.took_nothing") is clean
        ), "a module that never touched AITER was dropped"
        assert "atom.took_a_mock" not in sys.modules, "a poisoned module survived"
        assert "atom.imports_the_poisoned_one" not in sys.modules, (
            "a module holding a class from a dropped module survived; that is "
            "the stale second copy the conftest guard fails on"
        )
    finally:
        for name in (
            "atom.took_nothing",
            "atom.took_a_mock",
            "atom.imports_the_poisoned_one",
        ):
            sys.modules.pop(name, None)
        sys.modules.update(saved)
