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


@pytest.mark.parametrize("refusal", ["create", "prepare"])
def test_lazy_runner_refusal_is_for_good(monkeypatch, refusal):
    monkeypatch.setattr(route.torch.cuda, "is_current_stream_capturing", lambda: False)

    def create():
        if refusal == "create":
            raise MonoUnsupported("TP 8")
        return _Runner(ok=False)

    mono = route.LazyRunner(create, "test")
    assert not mono.ready(6) and not mono.enabled and not mono.ready(6)
