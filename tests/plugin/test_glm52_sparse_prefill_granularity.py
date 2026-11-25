# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""GLM-5.2 SGLang sparse prefill plans BF16 rows whole, as the native builder does."""

import importlib
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

TOPK = 40
TOKENS = 50


def _package(name):
    mod = ModuleType(name)
    mod.__path__ = []
    return mod


@pytest.fixture
def bridge():
    # Import the real dependencies first: patch.dict drops every module first
    # imported inside it, and aiter's JIT registry does not survive a reimport.
    import atom.distributed.pcp_utils
    import atom.model_ops.attentions.sparse_mla_head_groups
    import atom.plugin
    import atom.utils.forward_context

    utils_mod = ModuleType("sglang.kernels.ops.attention.utils")
    utils_mod.create_flashinfer_kv_indices_triton = lambda *args, **kwargs: None
    resolver_mod = ModuleType("atom.plugin.sglang.runtime.attention_backend_resolver")
    resolver_mod.resolve_sglang_runtime = lambda forward_batch: None
    model_arch_mod = ModuleType("atom.plugin.sglang.runtime.model_arch")
    model_arch_mod.is_glm52_dsa_config = lambda config: True
    fake_modules = {
        "sglang": _package("sglang"),
        "sglang.kernels": _package("sglang.kernels"),
        "sglang.kernels.ops": _package("sglang.kernels.ops"),
        "sglang.kernels.ops.attention": _package("sglang.kernels.ops.attention"),
        "sglang.kernels.ops.attention.utils": utils_mod,
        "atom.plugin.sglang": _package("atom.plugin.sglang"),
        "atom.plugin.sglang.runtime": _package("atom.plugin.sglang.runtime"),
        "atom.plugin.sglang.runtime.attention_backend_resolver": resolver_mod,
        "atom.plugin.sglang.runtime.model_arch": model_arch_mod,
    }
    fake_modules["atom.plugin.sglang"].__path__ = [
        os.path.join(os.path.dirname(atom.plugin.__file__), "sglang")
    ]
    with patch.dict(sys.modules, fake_modules):
        sys.modules.pop("atom.plugin.sglang.glm52_dsa_bridge", None)
        yield importlib.import_module("atom.plugin.sglang.glm52_dsa_bridge")


@pytest.mark.parametrize(
    "builder", ["build_prefill_metadata", "build_mtp_draft_extend_prefill_metadata"]
)
@pytest.mark.parametrize("pcp", [False, True])
@pytest.mark.parametrize("kv_dtype,granularity", [("bf16", 64), ("fp8", 16)])
def test_sparse_prefill_plans_bf16_rows_whole(
    bridge, monkeypatch, builder, pcp, kv_dtype, granularity
):
    """Every sparse_prefill_* plan (and the PCP rank's rebuild) of a BF16 batch
    uses one work per row; FP8 keeps the page granularity."""
    from atom.distributed import pcp_utils

    planned = []
    monkeypatch.setattr(
        bridge,
        "get_mla_metadata_info_v1",
        lambda *args, **kwargs: [((1,), torch.int32)] * 6,
    )
    monkeypatch.setattr(
        bridge,
        "get_mla_metadata_v1",
        lambda *args, **kwargs: planned.append(kwargs["kv_granularity"]),
    )
    lens = np.array([TOKENS], dtype=np.int32)
    monkeypatch.setattr(bridge, "get_seq_lens_cpu", lambda batch, bs: lens)
    monkeypatch.setattr(bridge, "get_extend_lens_cpu", lambda batch, pos, bs: lens)
    monkeypatch.setattr(bridge, "real_prefill_seq_lens", lambda seq, ext, batch: seq)
    monkeypatch.setattr(
        bridge,
        "build_token_table",
        lambda *args, **kwargs: torch.zeros(1, TOKENS, dtype=torch.int32),
    )
    monkeypatch.setattr(
        bridge, "flatten_kv_indices", lambda table, lengths: torch.arange(TOKENS)
    )
    monkeypatch.setattr(bridge, "_publish_fp8_prefill_index_rows", lambda md, c: None)
    monkeypatch.setattr(pcp_utils, "pcp_is_enabled", lambda: pcp)
    monkeypatch.setattr(pcp_utils, "get_pcp_world_size", lambda: 2)
    monkeypatch.setattr(
        pcp_utils,
        "pcp_round_robin_query_indices",
        lambda n, size: torch.arange(0, n, size),
    )

    forward_batch = SimpleNamespace(
        batch_size=1,
        out_cache_loc=torch.arange(TOKENS),
        seq_lens=torch.tensor([TOKENS]),
    )
    atom_config = SimpleNamespace(
        hf_config=SimpleNamespace(index_topk=TOPK, num_attention_heads=16),
        tensor_parallel_size=1,
        kv_cache_dtype=kv_dtype,
    )
    md = getattr(bridge, builder)(
        forward_batch,
        torch.arange(TOKENS),
        token_to_kv_pool=SimpleNamespace(page_size=1),
        req_to_token_pool=None,
        atom_config=atom_config,
    )
    assert md.run_sparse
    assert planned == [granularity] * (2 if pcp else 1)
