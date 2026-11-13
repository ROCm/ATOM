# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

ATOM_ROOT = Path(__file__).resolve().parents[1] / "atom"
# Every FP4 scorer takes its rows through `aiter.ops.flydsl`'s exports
# (`Fp4MqaRaggedMetadata`), prefill and decode alike, never a kernel module.
# Widen this only alongside a caller.
CURRENT_MODULES: set[str] = set()


def _fp4_aiter_imports() -> list[tuple[Path, int, str]]:
    imports = []
    for path in ATOM_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "pa_mqa_logits_fp4" in node.module
            ):
                imports.append((path, node.lineno, node.module))
    return imports


def test_dsv4_fp4_imports_use_current_aiter_package():
    imports = _fp4_aiter_imports()
    modules = {module for _, _, module in imports}

    unexpected_imports = [item for item in imports if item[2] not in CURRENT_MODULES]
    assert not unexpected_imports
    assert modules == CURRENT_MODULES


def test_fp8_and_flydsl_indexer_import_without_opus_api():
    # A fresh process catches eager imports even if another test already
    # loaded OPUS. The older non-OPUS AITER exports remain available.
    code = textwrap.dedent("""
        import importlib
        import sys
        from importlib.abc import MetaPathFinder
        import torch
        from tests.aiter_stub import stubbed_aiter

        class NoOpus(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith("aiter.ops.opus"):
                    raise ModuleNotFoundError("OPUS API unavailable", name=fullname)

        sys.meta_path.insert(0, NoOpus())
        with stubbed_aiter(
            "aiter.jit", "aiter.jit.utils", "aiter.jit.utils.chip_info",
            "aiter.ops", "aiter.ops.flydsl", "aiter.ops.topk",
            "aiter.ops.triton", "aiter.ops.triton.attention",
            "aiter.ops.triton.attention.pa_mqa_logits",
            "aiter.ops.triton.fp8_mqa_logits",
        ):
            from atom.model_ops.indexer import rows
            importlib.import_module("atom.model_ops.indexer.score")
            ends = torch.tensor([8], dtype=torch.int32)
            tables = torch.tensor([[0]], dtype=torch.int32)
            planned = rows.fp8_rows(ends, 64, None, tables)
            assert planned.row_ends is ends
            seqs = rows.Fp4MqaRaggedMetadata(
                torch.tensor([0, 1], dtype=torch.int32), 1, tables
            )
            planned = rows.fp4_rows(
                ends, 64, None, seqs, heads=64, page_size=64,
                layout=rows.PRESHUFFLE,
            )
            assert len(planned.fp4_bands) == 1
            assert not any(n.startswith("aiter.ops.opus") for n in sys.modules)
            # Selecting OPUS still requires the new API; never reinterpret
            # its cache as FlyDSL to hide an incompatible AITER installation.
            try:
                rows.fp4_rows(
                    ends, 64, None, seqs, heads=64, page_size=64,
                    layout=rows.MFMA,
                )
            except ModuleNotFoundError as exc:
                assert exc.name.startswith("aiter.ops.opus")
            else:
                raise AssertionError("OPUS unexpectedly accepted the older AITER")
        """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ATOM_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
