# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
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


def granularity_helpers():
    """The production sparse-prefill granularity helpers, without importing aiter."""
    path = (
        Path(__file__).resolve().parents[1]
        / "atom/model_ops/attentions/sparse_mla_head_groups.py"
    )
    names = {"sparse_prefill_rows_whole", "sparse_prefill_kv_granularity"}
    functions = [
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {f.name for f in functions} == names
    namespace = {"torch": torch}
    exec(  # noqa: S102 - checked-in production helpers
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return {name: namespace[name] for name in names}


@pytest.fixture
def production(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "atom/model_ops/attention_mla.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "mla_kernel_num_heads",
        "mla_dcp_sparse_prefill_num_heads",
        "_forward_prefill_mla",
        "_rebuild_sparse_dcp_persistent_metadata",
        "_configure_dcp_decode_head_padding",
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
        **granularity_helpers(),
    }
    attention = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLAAttention"
    )
    constructor = next(
        node
        for node in attention.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    dtype_assignments = [
        node
        for node in constructor.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr in ("kv_cache_dtype", "dtype")
            for target in node.targets
        )
    ]

    def initialize_dtypes(layer, cache_dtype, activation_dtype):
        exec(  # noqa: S102 - actual constructor normalization, before configuring DCP
            compile(
                ast.Module(body=dtype_assignments, type_ignores=[]), str(path), "exec"
            ),
            {"self": layer, "kv_cache_dtype": cache_dtype, "dtype": activation_dtype},
        )

    namespace["initialize_dtypes"] = initialize_dtypes
    dtype_path = path.parents[1] / "distributed/dcp_utils.py"
    dtype_fn = next(
        n
        for n in ast.parse(dtype_path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "mla_cache_dtype"
    )
    aiter = ModuleType("aiter")
    aiter.dtypes = SimpleNamespace(fp8=torch.float8_e4m3fn)
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    exec(  # noqa: S102 - shared production dtype resolver
        compile(ast.Module(body=[dtype_fn], type_ignores=[]), str(dtype_path), "exec"),
        namespace,
    )
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
    rebuilds, patches, launches, partials = [], [], [], []
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

    def support(*args, **kwargs):
        raise AssertionError("a prefill layer queried fixed AITER shape support")

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
        partials.append(kwargs["has_partials"])
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
        "dcp_sparse_prefill_plan_native": native and supported is True,
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
    assert len(launches) == 3
    # Only whole-row plans skip the reduce: any bf16 plan, or the native one.
    whole = dtype == torch.bfloat16 or (native and supported is True)
    assert partials == [not whole] * 3
    if native and supported is True:
        assert len(patches) == 2 and not rebuilds
    else:
        assert len(rebuilds) == 2 and not patches
        assert rebuilds[0].tolist() == [0, 4, 9]
        assert rebuilds[1].tolist() == [0, 2, 5]


@pytest.mark.parametrize(
    "dtype,dcp,expected",
    [
        (torch.bfloat16, 2, 32),
        (torch.bfloat16, 4, 64),
        (torch.bfloat16, 8, 128),
        (torch.float8_e4m3fn, 8, 96),
    ],
)
def test_width_uses_actual_support_and_preserves_bf16_head_groups(
    production, dtype, dcp, expected
):
    calls = []

    def support(width, max_q, q_dtype, kv_dtype, **kwargs):
        assert (max_q, q_dtype, kv_dtype) == (1, dtype, dtype)
        assert kwargs == {"persistent": True, "return_lse": True, "page_size": 1}
        calls.append(width)
        # gfx950 advertises intermediate BF16 widths as native too.
        return SimpleNamespace(
            native=dtype == torch.bfloat16 or width in (96, 128), supported=True
        )

    production["get_mla_decode_shape_support"] = support
    assert production["mla_dcp_sparse_prefill_num_heads"](
        12,
        dcp,
        persistent=True,
        dtype_q=dtype,
        dtype_kv=dtype,
    ) == (expected, True)
    assert expected in calls


@pytest.mark.parametrize(
    "native,supported", [(False, True), (True, None), (True, False)]
)
@pytest.mark.parametrize(
    "heads,dcp,expected",
    [(16, 2, 32), (16, 4, 64), (8, 4, 32), (8, 8, 64), (12, 2, 32), (12, 4, 64)],
)
def test_unqualified_width_keeps_conservative_padding(
    production, native, supported, heads, dcp, expected
):
    production["get_mla_decode_shape_support"] = lambda *a, **k: SimpleNamespace(
        native=native, supported=supported
    )
    assert production["mla_dcp_sparse_prefill_num_heads"](
        heads,
        dcp,
        persistent=True,
        dtype_q=torch.bfloat16,
        dtype_kv=torch.bfloat16,
    ) == (expected, False)


@pytest.mark.parametrize(
    "native,supported,nonps",
    [
        (True, True, False),
        (False, True, False),
        (True, None, False),
        (True, False, False),
        (True, True, True),
    ],
)
@pytest.mark.parametrize(
    "cache_dtype,activation_dtype,query_dtype",
    [
        ("bf16", torch.bfloat16, torch.bfloat16),
        ("bf16", torch.float16, torch.bfloat16),
        ("fp8", torch.bfloat16, torch.float8_e4m3fn),
    ],
)
def test_plan_support_is_bound_to_the_final_width_and_reset(
    production, native, supported, nonps, cache_dtype, activation_dtype, query_dtype
):
    queries = []

    def support(width, max_q, q_dtype, kv_dtype, **kwargs):
        assert q_dtype is query_dtype and kv_dtype is query_dtype
        queries.append((width, max_q, q_dtype, kv_dtype))
        return SimpleNamespace(native=native, supported=supported)

    production.update(
        dtypes=SimpleNamespace(
            fp8=torch.float8_e4m3fn,
            d_dtypes={"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn},
        ),
        get_mla_decode_shape_support=support,
        mla_dcp_persistent_width=lambda *a, **k: 64,
        mla_dcp_kernel_num_heads=lambda *a, **k: 64,
        mla_dcp_decode_is_persistent=lambda *a, **k: True,
        mla_dcp_sparse_prefill_is_persistent=lambda *a, **k: True,
        mla_dcp_sparse_prefill_uses_nonps=lambda *a: nonps,
        mla_dcp_sparse_prefill_mode=lambda n, p, h: (False, 64) if n else (p, h),
        get_current_atom_config=lambda: SimpleNamespace(max_num_batched_tokens=16384),
    )
    layer = SimpleNamespace(
        num_heads=8,
        min_query_heads=16,
        is_sparse_mla=True,
        sparse_dcp_metadata_rebuild=True,
    )
    production["initialize_dtypes"](layer, cache_dtype, activation_dtype)
    assert layer.kv_cache_dtype == ("fp8" if cache_dtype == "fp8" else "auto")
    assert layer.dtype is activation_dtype
    configure = production["_configure_dcp_decode_head_padding"]
    for dcp, width in ((8, 64), (4, 32)):
        first_query = len(queries)
        configure(layer, dcp)
        assert layer.dcp_sparse_prefill_num_heads == (64 if nonps else width)
        assert layer.dcp_sparse_prefill_plan_native == (
            native and supported is True and not nonps
        )
        queried = queries[first_query:]
        assert len({q[0] for q in queried}) == len(queried), "same width queried twice"
        assert queried[0] == (width, 1, query_dtype, query_dtype)
    before = list(queries)
    configure(layer, 1)
    assert not layer.dcp_sparse_prefill_plan_native
    assert queries == before


@pytest.mark.parametrize(
    "dtype,dcp,block,topk,q_granularity,local_granularity",
    [
        # BF16 rows are whole under either transport.
        (torch.bfloat16, 1, 128, 2048, 2048, 2048),
        (torch.bfloat16, 8, 1, 2048, 2048, 2048),
        (torch.bfloat16, 8, 1, 2176, 4096, 4096),
        (torch.float8_e4m3fn, 1, 128, 2048, 128, 128),
        (torch.float8_e4m3fn, 8, 1, 2048, 2048, 16),
        (torch.float8_e4m3fn, 8, 1, 2176, 4096, 16),
    ],
)
@pytest.mark.parametrize("local_heads", [False, True])
def test_work_plan_preserves_transport_granularity(
    dtype, dcp, block, topk, q_granularity, local_granularity, local_heads
):
    path = (
        Path(__file__).resolve().parents[1] / "atom/model_ops/attentions/aiter_mla.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    builder = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AiterMLAMetadataBuilder"
    )
    init = next(
        n
        for n in builder.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    init_granularity = [
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute)
            and t.attr
            in (
                "_sparse_prefill_local_kv_granularity",
                "_sparse_prefill_kv_granularity",
            )
            for t in n.targets
        )
    ]
    plan = next(
        n
        for n in builder.body
        if isinstance(n, ast.FunctionDef) and n.name == "_sparse_prefill_work_plan"
    )
    counts, plans = [], []

    def power(value):
        counts.append("power")
        return 1 << (value - 1).bit_length()

    helper = granularity_helpers()["sparse_prefill_kv_granularity"]

    def granularity(*args):
        counts.append("helper")
        return helper(*args)

    work = {name: object() for name in WORK}
    instance = SimpleNamespace(
        is_sparse=True,
        dcp_world_size=dcp,
        block_size=block,
        index_topk_out=topk,
        dcp_sparse_prefill_nonps=False,
        dtype_q=dtype,
        dtype_kv=dtype,
        model_runner=SimpleNamespace(
            forward_vars={f"sparse_prefill_{k}": v for k, v in work.items()}
        ),
        _sparse_prefill_work_views=lambda prefix, heads: work,
    )
    ns = {
        "self": instance,
        "triton": SimpleNamespace(next_power_of_2=power),
        "sparse_prefill_kv_granularity": granularity,
        "AttentionMetaData": SimpleNamespace,
        "torch": torch,
        "PERSISTENT_WORK_BUFFERS": WORK,
        "_MLA_SPLIT_BUDGET_AUTO": -1,
        "get_mla_metadata_v1": lambda *a, **k: plans.append(k["kv_granularity"]),
    }
    exec(  # noqa: S102 - execute the production initializer field and plan builder
        compile(
            ast.Module(body=init_granularity + [plan], type_ignores=[]),
            str(path),
            "exec",
        ),
        ns,
    )
    initialized = list(counts)
    metadata = SimpleNamespace(sparse_cu_seqlens_q=None, sparse_kv_indptr=None)
    for _ in range(2):
        ns["_sparse_prefill_work_plan"](
            instance, metadata, "", 64, None, local_heads=local_heads
        )
    want = local_granularity if local_heads else q_granularity
    assert plans == [want, want]
    # Both granularities are fixed at init; a step only reads them.
    assert counts == initialized == ["helper"] + (["power"] if dcp > 1 else [])
