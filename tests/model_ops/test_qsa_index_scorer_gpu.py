# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU tests (gfx1250): golden replay through ATOM's QSA index scorer call site.

Skipped unless a gfx1250 device, ``$QSA_GOLDEN_DIR`` and
``aiter.ops.qsa_index_scorer`` are all available. Root runs these under the
node's gpu-lock (see docs/qsa_index_scorer.md, "GPU run plan").
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _gfx1250() -> bool:
    if not torch.cuda.is_available():
        return False
    return str(torch.cuda.get_device_properties(0).gcnArchName).startswith("gfx1250")


def _aiter_ok() -> bool:
    try:
        import aiter.ops.qsa_index_scorer  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_gfx1250() and os.environ.get("QSA_GOLDEN_DIR") and _aiter_ok()),
    reason="needs gfx1250 + QSA_GOLDEN_DIR + aiter.ops.qsa_index_scorer",
)


@pytest.fixture(scope="module")
def adapter_result(tmp_path_factory):
    from atom.model_ops.qwen4_exp.qsa_index_scorer import golden_adapter

    out = tmp_path_factory.mktemp("qsa") / "golden_adapter.json"
    rc = golden_adapter.main(["--json", str(out)])
    return rc, json.loads(out.read_text())


def _records(result, mode):
    return [r for r in result[1]["records"] if r.get("mode") == mode]


def test_golden_replay_site(adapter_result):
    recs = _records(adapter_result, "site")
    assert len(recs) == 22
    bad = [r["name"] for r in recs if not r["passed"]]
    assert not bad, bad
    assert all(r["path"] == "fused" for r in recs)


def test_golden_replay_engine_path(adapter_result):
    recs = _records(adapter_result, "engine")
    bad = [r["name"] for r in recs if not r["passed"]]
    assert not bad, bad
    assert all(r["routed_native_equals_stock"] for r in recs if not r.get("skipped"))


def test_engine_pad_m(adapter_result):
    recs = _records(adapter_result, "engine_pad")
    assert recs and all(r["passed"] for r in recs)


def test_every_reachable_fallback_reason(adapter_result):
    recs = _records(adapter_result, "fallbacks")
    bad = [
        (r["name"], r.get("reason"), r.get("error")) for r in recs if not r["passed"]
    ]
    assert not bad, bad


def test_kernel_env_mismatch_runs_native():
    """A QSA_* kernel switch off its profiled value => kernel_env_mismatch, native."""
    code = r"""
import torch, sys
from atom.model_ops.qwen4_exp.qsa_index_scorer.route import QSAScorerRoute
from atom.model_ops.qwen4_exp.qsa_index_scorer import golden_adapter as A
d = A.fixture(dict(regime="decode", rows=4, requests=1, fmt="bf16", table_width=16, cache_pages=64))
r = QSAScorerRoute("qsa_page4_gfx1250_v1", site="env")
a = A.scorer_args(d, row_starts=torch.empty(4, dtype=torch.int32, device="cuda"))
reason, _ = r.decision_for(*a)
out = r(*a)
ref = r.native(*a)
ok = reason == "kernel_env_mismatch" and torch.equal(out[0].view(torch.int32), ref[0].view(torch.int32))
print(reason); sys.exit(0 if ok else 1)
"""
    env = dict(os.environ, QSA_PAD_BUCKET="1")
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=1800,
    )
    assert res.returncode == 0, res.stdout + res.stderr
