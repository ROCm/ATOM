# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The blockwise-FP8 recipe requires a TE that carries the CDNA4 running-scale fix.

A version gate rather than a functional probe: a synthetic GEMM with a large
per-128-block magnitude spread does not reproduce the fault on the pre-fix build
(measured: 0 non-finite across K in {2048, 4096} and spread in {1e18, 1e25, 1e30}),
so there is no cheap startup check that discriminates. See
ROCm/TransformerEngine #737 and #744.
"""

from __future__ import annotations

import sys
import types

import pytest
from atomrl.engine.training.megatron_native_engine import MegatronNativeEngine as E


@pytest.fixture
def te_version(monkeypatch):
    try:
        import transformer_engine as te_root
    except ImportError:
        # The check reads only __version__, so a stand-in lets it run without TE.
        te_root = types.ModuleType("transformer_engine")
        monkeypatch.setitem(sys.modules, "transformer_engine", te_root)

    def _set(v):
        monkeypatch.setattr(te_root, "__version__", v, raising=False)

    return _set


@pytest.mark.parametrize("v", ["2.15.0", "2.18.0.dev0+73162dd8c", "1.9.0", "2.19.0"])
def test_a_te_without_the_fix_is_refused(te_version, monkeypatch, v):
    monkeypatch.delenv("ATOMRL_SKIP_TE_BLOCKWISE_CHECK", raising=False)
    te_version(v)
    with pytest.raises(RuntimeError, match="blockwise"):
        E._check_te_supports_blockwise()


@pytest.mark.parametrize("v", ["2.20.0.dev0+c468afa60", "2.20.0", "2.21.0", "3.0.0"])
def test_a_te_with_the_fix_passes(te_version, monkeypatch, v):
    monkeypatch.delenv("ATOMRL_SKIP_TE_BLOCKWISE_CHECK", raising=False)
    te_version(v)
    E._check_te_supports_blockwise()


def test_the_escape_hatch_is_honoured(te_version, monkeypatch):
    """Someone running a patched older build must be able to proceed."""
    te_version("2.15.0")
    monkeypatch.setenv("ATOMRL_SKIP_TE_BLOCKWISE_CHECK", "1")
    E._check_te_supports_blockwise()


def test_an_unparseable_version_does_not_block_the_run(te_version, monkeypatch):
    """Better to let TE raise for itself than to refuse on a string we cannot read."""
    monkeypatch.delenv("ATOMRL_SKIP_TE_BLOCKWISE_CHECK", raising=False)
    te_version("not-a-version")
    E._check_te_supports_blockwise()


def test_the_container_default_te_is_refused(te_version, monkeypatch):
    """The atomrl image ships TE 2.15.0.dev0, which lacks the fix.

    So a run that forgets PYTHONPATH=/data/rl_data/te-current must be refused rather
    than quietly training on the buggy GEMM. This is the gate's reason to exist.
    """
    monkeypatch.delenv("ATOMRL_SKIP_TE_BLOCKWISE_CHECK", raising=False)
    te_version("2.15.0.dev0+6e541a104")
    with pytest.raises(RuntimeError, match="2.15.0"):
        E._check_te_supports_blockwise()
