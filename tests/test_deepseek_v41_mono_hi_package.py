# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono hi runtime (``atom/models/deepseek_v41/mono/hi``): of ATOM it imports only the shared mono pieces
(``atom.mono``), the hook and its entry modules import, and no file of the tree carries the name of the project the runtime grew out of.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "atom" / "models" / "deepseek_v41" / "mono" / "hi"
PKG = "atom.models.deepseek_v41.mono.hi"
# (spelled in pieces, so this file does not match itself)
OLD_NAME = re.compile("mono" + r"[\s_-]?" + "fly", re.IGNORECASE)
SCANNED = ("atom", "tests", "docs", "recipes", ".github", "docker")
TEXT = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".cfg",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
    ".ini",
    ".rst",
    "",
}


def test_no_file_carries_the_old_project_name():
    bad = []
    paths = [
        p for top in SCANNED if (ROOT / top).exists() for p in (ROOT / top).rglob("*")
    ]
    # (not ``.git``: in a worktree a file naming the checkout's path)
    paths += [p for p in ROOT.iterdir() if p.is_file() and p.name != ".git"]
    for path in paths:
        rel = path.relative_to(ROOT)
        if "__pycache__" in rel.parts:
            continue
        if OLD_NAME.search(str(rel)):
            bad.append(f"{rel} (path)")
        if not path.is_file() or path.suffix not in TEXT:
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if OLD_NAME.search(line):
                bad.append(f"{rel}:{i}")
    assert not bad, bad


def _imports(path: Path):
    """(line, absolute module or None, level) of every import statement of ``path``."""
    for n in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield n.lineno, a.name, 0
        elif isinstance(
            n, ast.ImportFrom
        ):  # (absolute: each name as a module path, ``atom.utils.envs``)
            for a in n.names if n.level == 0 and n.module else [None]:
                yield n.lineno, f"{n.module}.{a.name}" if a else n.module, n.level


# what the package may import of ATOM (the rest of its imports are relative or third-party: torch, flydsl, ...)
ATOM_IMPORTS = ("atom.mono",)


def test_the_package_is_self_contained():
    """Within the package every import is relative, of a third-party module, or of ``ATOM_IMPORTS``: no engine
    module (the hook is the package's only way into the engine)."""
    paths = sorted(PACKAGE.rglob("*.py"))
    assert paths
    bad = []
    for path in paths:
        for line, module, level in _imports(path):
            ok = (
                any(module == m or module.startswith(m + ".") for m in ATOM_IMPORTS)
                if module
                else False
            )
            if level == 0 and module and module.split(".")[0] == "atom" and not ok:
                bad.append(f"{path.relative_to(ROOT)}:{line}: {module}")
    assert not bad, bad


ENTRIES = (
    "v41.layer",
    "v41.head",
    "v41.draft",
    "v41.engram_io",
    "v41.config",
    "plan.program",
    "runtime.kernel",
    "runtime.peer",
    "runtime.dual",
    "blocks.gfx950.engram",
)


def test_the_hook_and_its_entry_modules_import():
    """The hook's modules and every package module it imports (FlyDSL needed: the blocks are FlyDSL code)."""
    pytest.importorskip("flydsl")
    pytest.importorskip(
        "aiter"
    )  # (the hi device code re-exports atom.mono.device's primitives)
    pytest.importorskip("triton")
    importlib.import_module("atom.models.deepseek_v41.mono.hi_runner")
    importlib.import_module("atom.models.deepseek_v41.mono.hi_bridge")
    for entry in ENTRIES:
        importlib.import_module(f"{PKG}.{entry}")


@pytest.mark.parametrize("tp", [2, 4])
def test_the_served_schedules_build_dual_copy_mailboxes(tp):
    """V4.1's schedules (MI355X, TP2 and TP4) put its tagged mailboxes in dual copies (``NUMA_DUAL``, "numa2"
    regions): the served programs need the run-time half of the dual-copy scratch (``runtime.dual``).
    """
    pytest.importorskip("flydsl")
    pytest.importorskip(
        "aiter"
    )  # (the hi device code re-exports atom.mono.device's primitives)
    from atom.models.deepseek_v41.mono.hi.v41 import schedules

    for kind in schedules.KINDS:
        for rows in (6, 12, 24):
            assert schedules.knobs(kind, rows, tp=tp)["NUMA_DUAL"] > 0, (
                kind,
                rows,
            )


def test_a_dual_copy_scratch_takes_its_calibrated_word(monkeypatch):
    """``MonoKernel.buffers`` of a dual-copy program: the run area at the start of a 2 MiB page, the copy word
    ``runtime.dual.scratch_word`` calibrates (stubbed here: its probe times the device) in word 0 of the program's
    private buffer; on a shared arena, the arena's word as given (no calibration).
    """
    pytest.importorskip("flydsl")
    pytest.importorskip(
        "aiter"
    )  # (the hi device code re-exports atom.mono.device's primitives)
    from types import SimpleNamespace

    import torch

    from atom.models.deepseek_v41.mono.hi.plan.program import DUAL_PAGE, DUAL_RUN
    from atom.models.deepseek_v41.mono.hi.runtime import dual, kernel
    from atom.models.deepseek_v41.mono.hi.target import MI355X

    calls = []

    def word(scratch, area, name=""):
        calls.append((scratch.data_ptr() % DUAL_PAGE, area, name))
        return 0xF00F

    monkeypatch.setattr(dual, "scratch_word", word)
    area = 4 * DUAL_RUN
    k = kernel.MonoKernel.__new__(kernel.MonoKernel)
    k.program = SimpleNamespace(name="dual_prog", dual_on=True, scratch_bytes=area + 4096, dual_bytes=area,
                                private={}, private_bytes=256, dyn_rows=False, target=MI355X)  # fmt: skip
    b = k.buffers(device="cpu", uncached=False)
    assert calls == [(0, area, "dual_prog")]
    assert b.scratch.data_ptr() % DUAL_PAGE == 0 and b.scratch.numel() == area + 4096
    assert int(b.private[:4].view(torch.int32)[0]) == 0xF00F
    raw = torch.zeros(3 * DUAL_PAGE, dtype=torch.uint8)
    arena = raw[-raw.data_ptr() % DUAL_PAGE :][: 2 * DUAL_PAGE]
    b2 = k.buffers(device="cpu", scratch=arena, word=0x1234)
    assert (
        calls == [(0, area, "dual_prog")] and b2.scratch.data_ptr() == arena.data_ptr()
    )
    assert int(b2.private[:4].view(torch.int32)[0]) == 0x1234
    with pytest.raises(ValueError, match="page"):
        k.buffers(device="cpu", scratch=arena[256:], word=0)


def test_a_launch_does_not_trace_the_program_again(monkeypatch):
    """After the first launch, ``MonoKernel.launch`` calls the compiled function: later launches trace nothing and
    still pass their own ``layer``. The program's one task stores its launch's tag
    (``step * LAYER_SLOTS + layer + 1``)."""
    pytest.importorskip("flydsl")
    pytest.importorskip(
        "aiter"
    )  # (the hi device code re-exports atom.mono.device's primitives)
    from atom.models.deepseek_v41.mono.hi.target import MI355X, device_arch

    if device_arch() != MI355X.arch:
        pytest.skip("an MI355X program")
    import torch
    from flydsl.compiler.kernel_function import KernelFunction

    from atom.models.deepseek_v41.mono.hi.device.gfx950 import buffer_ops as bo
    from atom.models.deepseek_v41.mono.hi.device.gfx950.ops import rsrc
    from atom.models.deepseek_v41.mono.hi.plan.program import (
        Program,
        Scenario,
        Stage,
    )
    from atom.models.deepseek_v41.mono.hi.runtime.kernel import MonoKernel

    def store_tag(k, t):
        bo.buffer_store(k.tag, rsrc(k.args["out"]), 0)

    stage = Stage("tag", 1, store_tag, args=("out",), base=0)
    kernel = MonoKernel(Program("launch_tag", Scenario(1, 1), [stage]))
    bufs, out = kernel.buffers(), torch.zeros(1, dtype=torch.int32, device="cuda")
    table = kernel.table({"out": out})
    kernel.launch(table, bufs)
    torch.cuda.synchronize()
    assert out.item() == 1
    emitted = []
    emit = KernelFunction._emit_kernel
    monkeypatch.setattr(
        KernelFunction,
        "_emit_kernel",
        lambda *a, **k: emitted.append(1) or emit(*a, **k),
    )
    for layer in (1, 2):
        kernel.launch(table, bufs, layer=layer)
        torch.cuda.synchronize()
        assert out.item() == layer + 1
    assert not emitted


def test_a_flydsl_change_keeps_flydsls_locations_and_warns_once(monkeypatch, caplog):
    """``atom.mono.runtime.srcloc.scoped``: on a fingerprint miss (a FlyDSL whose location capture is not the one
    reproduced) FlyDSL's own capture stays in place and one WARNING a process says what is skipped and its cost.
    """
    pytest.importorskip("flydsl")
    from atom.mono.runtime import srcloc

    original = srcloc.meta.capture_user_location
    monkeypatch.setattr(srcloc, "FINGERPRINT", "0" * 16)
    srcloc._warn_miss.cache_clear()
    with caplog.at_level("WARNING", logger="atom"):
        for _ in range(3):
            with srcloc.scoped():
                assert srcloc.meta.capture_user_location is original
    srcloc._warn_miss.cache_clear()
    warned = [r for r in caplog.records if "srcloc" in r.getMessage()]
    assert len(warned) == 1 and "twice the build time" in warned[0].getMessage()
