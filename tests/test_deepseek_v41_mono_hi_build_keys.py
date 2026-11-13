# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hi programs' names and build keys (``Program.key()``: the compile cache's key and the kernel symbol's digest)
held to a snapshot, composed without compiling (CPU only): every program ``hi_compose.HiPrograms`` builds at the
served widths at TP4, on either FP4 index layout, its build fallbacks included (the split verify heads, the draft's
stage-2 tail and plain heads).

- bf16 KV (``--kv_cache_dtype bf16``): every name and key as in the snapshot (``FIXTURE``), so a change that should
  leave the served bf16 programs alone (a defaulted field, a new knob, the fp8 pool's builds) does;
- fp8 KV: the programs that read or write ATOM's KV pool (the compressed layers, the DSpark context write and draft
  stages) are their own builds, named "_kv8", each keyed apart from its bf16 twin; every other program is the bf16
  one, name and key.

Keys depend on the Linear prefetch fits only through the fit (``linear._fit``, which compiles probes): stubbed here to
the plan's own depth, so the keys are this source's alone (any FlyDSL). Regenerate the snapshot after an intended
change of the bf16 programs: ``python tests/test_deepseek_v41_mono_hi_build_keys.py --write``.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")
pytest.importorskip("triton")

FIXTURES = Path(__file__).parent / "models" / "deepseek_v41" / "fixtures"
CONFIG = FIXTURES / "config.json"
FIXTURE = FIXTURES / "hi_bf16_build_keys.json"
# the builds that fail in the second composition, so it takes their fallbacks: the wide verify heads (split heads of
# one request) and the draft's chained heads past one request (a stage-2 tail and plain heads)
FALLBACKS = (r"^atom_v41_head_chain_.*_b\d+$", r"^atom_v41_draft_head_r\d+_b\d+$")


def build_keys(kv_dtype: str = "bf16") -> dict:
    """{program name: its build key's digest} of every composition the module docstring names, on ATOM's KV pool of
    ``kv_dtype``."""
    import os

    os.environ.setdefault("ARCH", "gfx950")
    os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx950")
    from atom.models.deepseek_v41.config import build_attention_topology
    from atom.models.deepseek_v41.mono import hi_compose, hi_contract
    from atom.models.deepseek_v41.mono.hi.blocks.gfx950 import linear
    from atom.models.deepseek_v41.mono.hi.emit.compile import ResourceError
    from atom.models.deepseek_v41.mono.hi.plan.program import plan
    from atom.models.deepseek_v41.mono.hi.runtime import kernel
    from atom.models.deepseek_v41.mono.hi_gate import INDEX_LAYOUTS, MAX_CTX

    keys, failing = {}, []

    class Planned:
        """A ``MonoKernel`` planned, not emitted or compiled: its program's key recorded by name."""

        def __init__(self, program):
            if any(re.match(r, program.name) for r in failing):
                raise ResourceError(f"{program.name}: a fallback's build")
            self.program = plan(program)
            digest = hashlib.sha256(repr(self.program.key()).encode()).hexdigest()[:20]
            if keys.setdefault(program.name, digest) != digest:
                raise AssertionError(f"two programs named {program.name}")

    saved = kernel.MonoKernel, linear._fit
    kernel.MonoKernel, linear._fit = Planned, lambda lin, sc, p: p
    try:
        cfg = hi_contract.config(str(CONFIG))
        topology = build_attention_topology(cfg)[: cfg.num_hidden_layers]
        for fails in ((), FALLBACKS):
            failing[:] = fails
            for layout in INDEX_LAYOUTS:
                hi_compose.HiPrograms(cfg, topology, list(hi_compose.ROWS), MAX_CTX, 4, index_layout=layout,
                                      kv_dtype=kv_dtype)  # fmt: skip
    finally:
        kernel.MonoKernel, linear._fit = saved
    return dict(sorted(keys.items()))


@pytest.fixture(scope="module")
def bf16():
    return build_keys("bf16")


def test_the_bf16_programs_are_the_snapshots(bf16):
    want = json.loads(FIXTURE.read_text())
    assert sorted(bf16) == sorted(want), (
        f"programs added {sorted(set(bf16) - set(want))}, gone {sorted(set(want) - set(bf16))}"
    )
    changed = sorted(n for n in want if bf16[n] != want[n])
    assert not changed, f"bf16 build keys changed: {changed}"
    assert not [n for n in bf16 if "kv8" in n]


def test_the_fp8_pool_builds_only_the_kv_programs_apart(bf16):
    fp8 = build_keys("fp8")
    kv8 = {n: k for n, k in fp8.items() if "_kv8" in n}
    same = {n: k for n, k in fp8.items() if "_kv8" not in n}
    # the rest: the bf16 programs as they are (the window layers, the verify and draft heads, the Engram launch)
    assert same == {n: bf16[n] for n in same}
    assert not [n for n in same if re.match(r"atom_v41_(consumer|kv|index|ctx|draft\d)", n)], sorted(same)
    # each KV program: its bf16 twin's name with "_kv8", keyed apart
    twins = {n: n.replace("_kv8", "") for n in kv8}
    assert set(twins.values()) | set(same) == set(bf16)
    assert all(kv8[n] != bf16[t] for n, t in twins.items())


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        print(__doc__)
        sys.exit(2)
    FIXTURE.write_text(json.dumps(build_keys("bf16"), indent=1) + "\n")
    print(f"{FIXTURE}: written")
