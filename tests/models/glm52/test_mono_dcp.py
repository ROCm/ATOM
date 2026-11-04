# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GLM-5.2 MonoKernel under decode context parallelism (DCP == TP4): which
deployments mono serves, which weights it binds and which sparse indptr it
hands the kernel. Whether the kernel's DCP decode is right needs a GPU."""

from types import SimpleNamespace

import pytest
import torch
from test_mono_config import _served_tp4_fp8_config

from atom.models.glm52.mono.config import config_refusal


def _dcp_config(dcp=4, tp=4, interleave=1, qrep=True):
    cfg = _served_tp4_fp8_config(
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=3)
    )
    cfg.tensor_parallel_size = tp
    cfg.decode_context_parallel_size = dcp
    cfg.dcp_config = SimpleNamespace(
        interleave_size=interleave, enable_query_replication=qrep
    )
    return cfg


@pytest.fixture
def dcp_on(monkeypatch):
    monkeypatch.setenv("ATOM_GLM52_MONO_DCP", "1")
    monkeypatch.setattr(
        "atom.models.glm52.mono.config._is_gfx950", lambda: True, raising=True
    )


def test_dcp_is_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("ATOM_GLM52_MONO_DCP", raising=False)
    monkeypatch.setattr(
        "atom.models.glm52.mono.config._is_gfx950", lambda: True, raising=True
    )
    refusal = config_refusal(_dcp_config())
    assert refusal is not None and "ATOM_GLM52_MONO_DCP" in refusal


def test_dcp_equal_to_tp4_is_served(dcp_on):
    assert config_refusal(_dcp_config()) is None


@pytest.mark.parametrize(
    "over,why",
    [
        ({"dcp": 2}, "DCP 2 with TP 4"),
        ({"interleave": 2}, "interleave"),
        ({"qrep": False}, "query replication"),
    ],
)
def test_any_other_dcp_deployment_is_refused_by_name(dcp_on, over, why):
    refusal = config_refusal(_dcp_config(**over))
    assert refusal is not None and why in refusal


def test_dcp_at_tp8_is_refused(dcp_on):
    """The kernel's geometry takes DCP 4 only (``glm5_attention_heads``)."""
    refusal = config_refusal(_dcp_config(dcp=8, tp=8))
    assert refusal is not None and "DCP 8 with TP 8" in refusal


def test_the_fused_indexer_refuses_dcp(monkeypatch):
    from test_fused_indexer_config import _servable_atom_config

    from atom.models.glm52.mono.config import fused_indexer_refusal

    monkeypatch.setenv("ATOM_GLM52_FUSED_INDEXER", "1")
    assert fused_indexer_refusal(_servable_atom_config()) is None
    refusal = fused_indexer_refusal(
        _servable_atom_config(decode_context_parallel_size=4)
    )
    assert refusal is not None and "DCP" in refusal


# ------------------------------------------------------------------ weights
def _linear(rows, cols, padded=None):
    lin = SimpleNamespace(
        weight=SimpleNamespace(
            ndim=2,
            shape=(rows, cols),
            dtype=torch.float8_e4m3fn,
            is_shuffled=True,
            data="w",
        ),
        weight_scale=SimpleNamespace(
            dtype=torch.float32, numel=lambda: rows, data=torch.zeros(rows)
        ),
        quant_type=SimpleNamespace(name="per_Token"),
        params_dtype=torch.float8_e4m3fn,
    )
    if padded is not None:
        lin.is_output_padded = True
        lin._output_size_before_padding = padded
    return lin


def test_a_wider_ptpc_weight_is_refused_not_cut():
    """A 64-head q_b checked as 16 heads used to pass, handing the kernel the
    first 16 heads of another layout without a word."""
    from atom.mono.runtime.consensus import MonoUnsupported
    from atom.mono.weights.layer import linear_ptpc_fp8

    with pytest.raises(MonoUnsupported, match="weight shape"):
        linear_ptpc_fp8(
            _linear(64 * 256, 2048),
            name="q_b",
            logical_rows=16 * 256,
            logical_cols=2048,
        )
    w, _ = linear_ptpc_fp8(
        _linear(16 * 256, 2048), name="q_b", logical_rows=16 * 256, logical_cols=2048
    )
    assert w == "w"
    # output padding stays allowed, its logical size checked
    linear_ptpc_fp8(
        _linear(16 * 256 + 64, 2048, padded=16 * 256),
        name="q_b",
        logical_rows=16 * 256,
        logical_cols=2048,
    )


# ------------------------------------------------------------- DCP indptr
def _layer_with(buf):
    impl = SimpleNamespace(dcp_sparse_kv_indptr_buffer=buf)
    return SimpleNamespace(
        self_attn=SimpleNamespace(mla_attn=SimpleNamespace(impl=impl))
    )


def test_the_dcp_indptr_is_the_layers_shared_buffer():
    from atom.models.glm52.mono.contract import shared_dcp_indptr

    buf = torch.zeros(65, dtype=torch.int32)
    assert shared_dcp_indptr([_layer_with(buf), _layer_with(buf)], 8) is buf


@pytest.mark.parametrize(
    "bufs",
    [
        [torch.zeros(65, dtype=torch.int64)],
        [torch.zeros(5, dtype=torch.int32)],
        [torch.zeros(65, dtype=torch.int32), torch.zeros(65, dtype=torch.int32)],
        [None],
    ],
)
def test_a_dcp_indptr_the_kernel_cannot_read_is_refused(bufs):
    from atom.models.glm52.mono.contract import shared_dcp_indptr
    from atom.mono.runtime.consensus import MonoUnsupported

    with pytest.raises(MonoUnsupported, match="DCP sparse indptr"):
        shared_dcp_indptr([_layer_with(b) for b in bufs], 8)


# ------------------------------------------------------ uvl / uv under DCP
@pytest.mark.parametrize("samples", [1, 4, 8, 12])
def test_dcp_rebuilds_each_head_once_and_runs_only_owned_uv_tiles(samples):
    from atom.models.glm52.mono.kernels.layout import stage_tasks

    def table(heads, dcp):
        return stage_tasks(
            samples, heads, 2048, npes=4, inter=512, split_head_merge=True, dcp_size=dcp
        )

    names = [n for n, _ in table(64, 4)]
    tasks = dict(table(64, 4))
    assert names.index("split") < names.index("uvl") == names.index("uv") - 1
    assert tasks["uvl"] == samples * 64
    # 16 owned heads x 4 tiles: the same uv table as the 16-head DCP1 build
    assert tasks["uv"] == dict(table(16, 1))["uv"] == samples * 64
    assert "uvl" not in dict(table(16, 1))


@pytest.mark.parametrize("samples", [1, 8])
def test_dcp_summaries_tile_the_peer_buffer(samples):
    from atom.models.glm52.mono.kernels.layout import (
        DCP_SUMMARY_PAIRS,
        dcp_summary_index,
        layout,
    )

    dcp, heads, owned = 4, 64, 16
    _, sym = layout(
        samples, heads, 4, 2048, inter=512, output_heads=owned, dcp_size=dcp
    )
    # every rank's partial of every (sample, head) lands in its owner's
    # buffer at a distinct pair, and nothing else does
    seen = {
        dcp_summary_index(src, s, head % owned, item, samples, owned)
        for src in range(dcp)
        for s in range(samples)
        for head in range(heads)
        if head // owned == 0
        for item in range(DCP_SUMMARY_PAIRS)
    }
    n = dcp * samples * owned * DCP_SUMMARY_PAIRS
    assert seen == set(range(n))
    assert sym["_dcp_part_stride"] == n * 8
    assert sym["_bytes"] - sym["dcp"] == 2 * n * 8
