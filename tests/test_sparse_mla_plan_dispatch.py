# SPDX-License-Identifier: MIT
"""CPU contracts for native and folded sparse-prefill work plans."""

import ast
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

WORK = (
    "work_meta_data",
    "work_info_set",
    "work_indptr",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)


@pytest.fixture
def production(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "atom/model_ops/attention_mla.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "mla_kernel_num_heads",
        "mla_dcp_sparse_prefill_num_heads",
        "_forward_prefill_mla",
        "_rebuild_sparse_dcp_persistent_metadata",
    }
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id.startswith("_MLA_")
            for target in node.targets
        )
    ]
    namespace = {
        "torch": torch,
        "AttentionMetaData": SimpleNamespace,
        "PERSISTENT_WORK_BUFFERS": WORK,
        "envs": SimpleNamespace(ATOM_MLA_PAGE_SIZE=1),
        "NONPS_MAX_Q_ROWS": 32768,
        "_dcp_sparse_prefill_width_warned": False,
        "logger": logging.getLogger(__name__),
        "get_forward_context": lambda: SimpleNamespace(
            context=SimpleNamespace(scheduled_bs=2)
        ),
        "mla_dcp_sparse_prefill_kv_splits": lambda *_: 2,
    }
    exec(  # noqa: S102 - checked-in production bodies; GPU boundaries below are stubs
        compile(
            ast.Module(body=constants + functions, type_ignores=[]), str(path), "exec"
        ),
        namespace,
    )
    dcp = ModuleType("atom.model_ops.dcp_ops")
    dcp.zero_nonfinite_rows = lambda *_: None
    monkeypatch.setitem(sys.modules, dcp.__name__, dcp)
    return namespace


@pytest.mark.parametrize(
    "native,supported", [(False, True), (True, None), (True, False), (True, True)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_each_owner_rebuilds_folded_or_unknown_plans(
    production, native, supported, dtype
):
    ns = production
    queries, rebuilds, patches, launches = [], [], [], []
    q = torch.empty(2, 64, 12, dtype=dtype)
    kv = torch.empty(16, 1, 12, dtype=dtype)
    ptr = torch.tensor([0, 4, 9], dtype=torch.int32)
    md = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 2]),
        max_seqlen_q=2,
        kv_indptr=None,
        kv_indices=None,
        kv_last_page_lens=None,
        sparse_cu_seqlens_q=torch.arange(3, dtype=torch.int32),
        sparse_kv_indptr=torch.tensor([0, 9, 18], dtype=torch.int32),
        sparse_kv_indices=torch.zeros(18, dtype=torch.int32),
        sparse_kv_last_page_lens=torch.ones(2, dtype=torch.int32),
        dcp_sparse_kv_indptr=ptr,
    )
    buffers = {name: torch.zeros(1) for name in WORK}
    for name, value in buffers.items():
        setattr(md, f"sparse_prefill_{name}", value)

    def support(width, max_q, q_dtype, kv_dtype, **kwargs):
        queries.append((width, max_q, q_dtype, kv_dtype))
        assert kwargs == {"persistent": True, "return_lse": True, "page_size": 1}
        return SimpleNamespace(native=native, supported=supported)

    def rebuild(*args, **kwargs):
        assert args[0] is md.sparse_cu_seqlens_q
        assert torch.equal(args[1], md.dcp_sparse_kv_indptr)
        assert args[2] is md.sparse_kv_last_page_lens
        assert args[3:6] == (64, 1, True)
        for index, (value, expected) in enumerate(zip(args[6:], buffers.values())):
            assert value is expected
            value.fill_(100 * (len(rebuilds) + 1) + index)
        assert kwargs["dtype_q"] == kwargs["dtype_kv"] == dtype
        assert (
            kwargs["page_size"]
            == kwargs["max_seqlen_qo"]
            == kwargs["uni_seqlen_qo"]
            == 1
        )
        assert kwargs["kv_granularity"] == (2048 if dtype == torch.bfloat16 else 16)
        rebuilds.append(args[1].clone())

    def patch(info, indptr, owned, rows):
        assert (
            native and supported is True
        ), "folded or unknown plans must rebuild, not patch row indices"
        assert info is buffers["work_info_set"] and indptr is buffers["work_indptr"]
        assert torch.equal(owned, md.dcp_sparse_kv_indptr) and rows == 2
        patches.append(owned.clone())

    def launch(query, cache, out, qptr, kvptr, *args, **kwargs):
        assert query is q and cache.dtype == dtype
        assert torch.equal(kvptr, md.dcp_sparse_kv_indptr)
        for name, value in buffers.items():
            assert kwargs[name] is value
        launches.append(kvptr.clone())
        out.zero_()
        return out, torch.zeros(query.shape[:2])

    ns.update(
        get_mla_decode_shape_support=support,
        get_mla_metadata_v1=rebuild,
        patch_work_kv_ranges=patch,
        mla_decode_fwd=launch,
    )
    methods = {
        name: ns[name]
        for name in ("_forward_prefill_mla", "_rebuild_sparse_dcp_persistent_metadata")
    }
    attn = type("Attention", (), methods)()
    for name, value in {
        "is_sparse_mla": True,
        "dcp_world_size": 8,
        "num_heads": 8,
        "use_seg_mla": False,
        "kv_lora_rank": 8,
        "dtype": torch.bfloat16,
        "kv_cache_dtype": "bf16" if dtype == torch.bfloat16 else "fp8",
        "_q_scale": torch.ones(1),
        "_k_scale": torch.ones(1),
        "scale": 1.0,
        "dcp_sparse_prefill_persistent": True,
        "sparse_dcp_metadata_rebuild": True,
        "dcp_persistent_supported": True,
        "dcp_sparse_prefill_nonps": False,
        "owns_sparse_indexer": True,
        "sparse_prefill_head_groups": None,
        "_sparse_prefill_kv_granularity": 2048,
    }.items():
        setattr(attn, name, value)
    attn._restore_sparse_prefill_query_heads = lambda value, _: value

    for owned in (ptr, torch.tensor([0, 2, 5], dtype=torch.int32)):
        md.dcp_sparse_kv_indptr = owned
        attn._forward_prefill_mla(q, kv, md, return_lse=True, q_prepadded=True)
    # IndexShare consumers keep the preceding owner's descriptors.
    attn.owns_sparse_indexer = False
    attn._forward_prefill_mla(q, kv, md, return_lse=True, q_prepadded=True)
    assert queries == [(64, 1, dtype, dtype)] * 2
    assert len(launches) == 3
    if native and supported is True:
        assert len(patches) == 2 and not rebuilds
    else:
        assert len(rebuilds) == 2 and not patches
        assert rebuilds[0].tolist() == [0, 4, 9]
        assert rebuilds[1].tolist() == [0, 2, 5]


@pytest.mark.parametrize(
    "dtype,expected", [(torch.bfloat16, 128), (torch.float8_e4m3fn, 96)]
)
def test_width_uses_actual_support_and_preserves_bf16_head_groups(
    production, dtype, expected
):
    calls = []

    def support(width, max_q, q_dtype, kv_dtype, **kwargs):
        assert (max_q, q_dtype, kv_dtype) == (1, dtype, dtype)
        assert kwargs == {"persistent": True, "return_lse": True, "page_size": 1}
        calls.append(width)
        return SimpleNamespace(native=width in (96, 128), supported=True)

    production["get_mla_decode_shape_support"] = support
    assert (
        production["mla_dcp_sparse_prefill_num_heads"](
            12,
            8,
            persistent=True,
            dtype_q=dtype,
            dtype_kv=dtype,
        )
        == expected
    )
    assert expected in calls


@pytest.mark.parametrize(
    "native,supported", [(False, True), (True, None), (True, False)]
)
def test_unqualified_width_keeps_conservative_padding(production, native, supported):
    production["get_mla_decode_shape_support"] = lambda *a, **k: SimpleNamespace(
        native=native, supported=supported
    )
    assert (
        production["mla_dcp_sparse_prefill_num_heads"](
            8,
            8,
            persistent=True,
            dtype_q=torch.bfloat16,
            dtype_kv=torch.bfloat16,
        )
        == 128
    )
