# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.5 layer kernel's hand-off contract on real builds: traced compile-only
for gfx950 (no GPU launch), every mailbox access checked against
``layout.mailbox_regions``.

The served shapes: plain decode of 4 requests, an MTP-3 verify of one request (4 rows),
the verifies' GDN on the snapshot state and on ReplaySSM records; GDN and full attention,
FP8 and MXFP4; full attention also as ATOM_PA_FLYDSL's, which alone serves the MTP-4 / MTP-5
verify (5 / 6 rows a request), of one request and of several. The negative control checks
a trace against a wrong table instead of building again."""

import dataclasses

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.qwen3_5.mono import contract
from atom.models.qwen3_5.mono.layout import mailbox_regions
from atom.mono.plan.check import ContractError, check
from atom.mono.runtime.compile import trace

# (full, weights, tokens a request, ReplaySSM records, ATOM_PA_FLYDSL, rows)
BUILDS = (
    [
        (False, w, T, cap, False, 6 if T == 6 else 4)
        for w in ("fp8", "mxfp4")
        for T, cap in ((1, 0), (4, 0), (4, 16), (6, 0), (6, 16))
    ]
    + [(True, w, T, 0, False, 4) for w in ("fp8", "mxfp4") for T in (1, 4)]
    + [
        (True, w, T, 0, True, S)
        for w, T, S in (
            ("fp8", 1, 4),
            ("fp8", 4, 4),
            ("fp8", 4, 8),
            ("fp8", 5, 5),
            ("fp8", 6, 6),
            ("fp8", 6, 12),
            ("mxfp4", 1, 4),
            ("mxfp4", 6, 6),
        )
    ]
)


def rows(build):
    return build[5]


@pytest.fixture(scope="module")
def traces():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ARCH", "gfx950")
        patch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        return {
            b: trace(
                *contract.build_layer(
                    rows(b), b[0], b[1], T=b[2], cap=b[3], pa_flydsl=b[4]
                )
            )
            for b in BUILDS
        }


@pytest.mark.parametrize("build", BUILDS)
def test_the_layer_kernel_holds(traces, build):
    full, weights = build[:2]
    check(traces[build], mailbox_regions(rows(build), full, weights, contract.TP))


def test_a_wrong_declaration_is_caught(traces):
    """The positive control: the same build against a table that names the wrong
    writer for the MoE mid rows."""
    decls = mailbox_regions(4, False, "fp8", contract.TP)
    wrong = [
        dataclasses.replace(d, writer="down") if d.name == "mid" else d for d in decls
    ]
    with pytest.raises(ContractError, match="mid put by ug, its writer is down"):
        check(traces[(False, "fp8", 1, 0, False, 4)], wrong)
