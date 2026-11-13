# SPDX-License-Identifier: MIT
"""Layout declarations must remain usable on CPU runners without AITER."""

import subprocess
import sys
import textwrap
from pathlib import Path


def test_layout_exports_and_v41_plane_import_without_aiter():
    # A fresh process avoids cached modules and other tests' AITER stubs.
    code = textwrap.dedent("""
        import sys
        from importlib.abc import MetaPathFinder

        class NoAiter(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "aiter" or fullname.startswith("aiter."):
                    raise AssertionError(f"layout declarations imported {fullname}")

        sys.meta_path.insert(0, NoAiter())

        from atom.model_ops.indexer import MFMA, NATURAL, PRESHUFFLE
        from atom.model_ops.deepseek_v41.index_plane import (
            IndexUnits, fp4_index_layout_for_arch,
        )

        assert fp4_index_layout_for_arch("gfx950", backend="flydsl") == PRESHUFFLE
        assert fp4_index_layout_for_arch("gfx950", backend="opus") == MFMA
        assert fp4_index_layout_for_arch("gfx1250", backend="opus") == NATURAL
        assert fp4_index_layout_for_arch("gfx1250", backend="flydsl") == NATURAL
        assert IndexUnits(None).layout == PRESHUFFLE
        assert len({MFMA, NATURAL, PRESHUFFLE}) == 3
        assert "atom.model_ops.indexer.rows" not in sys.modules
        """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
