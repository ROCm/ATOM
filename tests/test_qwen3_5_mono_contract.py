# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.5 layer kernel's hand-off contract on real builds: traced compile-only
for gfx950 (no GPU launch), every mailbox access checked against
``layout.mailbox_regions``.

The served shapes at 4 rows: plain decode of 4 requests and an MTP-3 verify of one
request (its GDN on the snapshot state and on ReplaySSM records), GDN and full
attention, FP8 and MXFP4; the negative control checks a trace against a wrong table
instead of building again."""

import dataclasses

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.qwen3_5.mono import contract
from atom.models.qwen3_5.mono.layout import mailbox_regions
from atom.mono.plan.check import ContractError, check
from atom.mono.runtime.compile import trace

ROWS = 4
# (full, weights, tokens a request, ReplaySSM records)
BUILDS = [
    (full, w, T, cap)
    for full in (False, True)
    for w in ("fp8", "mxfp4")
    for T, cap in ((1, 0), (4, 0), (4, 0 if full else 16))
]
BUILDS = list(dict.fromkeys(BUILDS))


@pytest.fixture(scope="module")
def traces():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ARCH", "gfx950")
        patch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        return {
            b: trace(*contract.build_layer(ROWS, b[0], b[1], T=b[2], cap=b[3]))
            for b in BUILDS
        }


@pytest.mark.parametrize("build", BUILDS)
def test_the_layer_kernel_holds(traces, build):
    full, weights = build[:2]
    check(traces[build], mailbox_regions(ROWS, full, weights, contract.TP))


def test_a_wrong_declaration_is_caught(traces):
    """The positive control: the same build against a table that names the wrong
    writer for the MoE mid rows."""
    decls = mailbox_regions(ROWS, False, "fp8", contract.TP)
    wrong = [
        dataclasses.replace(d, writer="down") if d.name == "mid" else d for d in decls
    ]
    with pytest.raises(ContractError, match="mid put by ug, its writer is down"):
        check(traces[(False, "fp8", 1, 0)], wrong)
