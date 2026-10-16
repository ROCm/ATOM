# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The fused layer kernels' hand-off contract on real builds: traced compile-only
for gfx950 (no GPU launch), every mailbox access checked against
``layout.mailbox_regions``.

One build of the served model (one MTP-3 request, indexer CP across the TP
ranks), each kernel traced once; the negative control checks the layer
kernel's trace against a wrong table instead of building again."""

import dataclasses

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.minimax_m3.mono import contract
from atom.models.minimax_m3.mono.config import TP
from atom.models.minimax_m3.mono.kernels.dense_post import DENSE_POST_ABI
from atom.models.minimax_m3.mono.kernels.post_attn import K4_ABI
from atom.models.minimax_m3.mono.layout import dense_mailbox_regions, mailbox_regions
from atom.mono.plan.check import ContractError, check
from atom.mono.runtime.compile import launcher_resources, trace

TOKENS = 4  # one request: the token and its 3 MTP drafts
M31_TOKENS = (1, 16, 32)
INDEX_HEADS = TP


@pytest.fixture(scope="module")
def traces():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ARCH", "gfx950")
        patch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        m31 = {
            tokens: contract.build_m31_layer(tokens, INDEX_HEADS)
            for tokens in M31_TOKENS
        }
        m31_single_s32 = contract.build_m31_layer(32, 1)
        return {
            "layer": trace(contract.build_layer(TOKENS, INDEX_HEADS), K4_ABI),
            "m31": {
                tokens: trace(launcher, K4_ABI)
                for tokens, launcher in m31.items()
            },
            "m31_single_s32": trace(m31_single_s32, K4_ABI),
            "m31_resources": {
                (16, INDEX_HEADS): launcher_resources(m31[16]),
                (32, INDEX_HEADS): launcher_resources(m31[32]),
                (32, 1): launcher_resources(m31_single_s32),
            },
            "dense": trace(contract.build_dense(TOKENS), DENSE_POST_ABI),
        }


def test_the_layer_kernel_holds(traces):
    check(traces["layer"], mailbox_regions(TOKENS, INDEX_HEADS, True, True))


def test_the_dense_layer_kernel_holds(traces):
    check(traces["dense"], dense_mailbox_regions(TOKENS))


def test_the_m31_layer_kernel_holds(traces):
    for tokens, traced in traces["m31"].items():
        check(
            traced,
            mailbox_regions(tokens, INDEX_HEADS, True, True, shared_fp8=True),
        )
    check(
        traces["m31_single_s32"],
        mailbox_regions(32, 1, True, True, shared_fp8=True),
    )


def test_the_m31_resource_boundaries_stay_fully_resident(traces):
    for build, [resources] in traces["m31_resources"].items():
        assert resources["vgpr_spill"] == 0, (build, resources)
        assert resources["scratch"] == 0, (build, resources)
        assert resources["lds"] <= 160 * 1024, (build, resources)


def test_a_wrong_declaration_is_caught(traces):
    """The positive control: the same build against a table that names the wrong
    writer for the MoE mid rows."""
    decls = mailbox_regions(TOKENS, INDEX_HEADS, True, True)
    assert any(d.name == "mid" for d in decls)
    wrong = [
        dataclasses.replace(d, writer="down") if d.name == "mid" else d for d in decls
    ]
    with pytest.raises(ContractError, match="mid put by ug, its writer is down"):
        check(traces["layer"], wrong)
