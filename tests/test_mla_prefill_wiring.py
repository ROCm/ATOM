# SPDX-License-Identifier: MIT
"""CPU controls at production initialization and prefill dispatch boundaries."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _tree(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def _run(nodes, namespace):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    program = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(program, "<production wiring>", "exec"), namespace)  # noqa: S102


def _constructor(name):
    cls = next(
        n
        for n in _tree("atom/models/kimi_k3/model.py").body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )


def _sets(node, attr):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Attribute) and t.attr == attr for t in node.targets
    )


@pytest.mark.parametrize(
    "dsa,shared,expected",
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_model_passes_its_created_stream_to_decoder_layers(dsa, shared, expected):
    nodes = _constructor("KimiLinearModel").body
    first = next(i for i, n in enumerate(nodes) if _sets(n, "alt_stream"))
    last = next(
        i
        for i, n in enumerate(nodes)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Tuple) for t in n.targets)
    )
    stream, received = object(), []
    config = SimpleNamespace(
        num_shared_experts=int(shared),
        dsa_layer_indices=(0,) if dsa else (),
        num_hidden_layers=2,
    )

    def make_layers(count, factory, **kwargs):
        return 0, count, [factory(f"layer{i}", i) for i in range(count)]

    _run(
        nodes[first : last + 1],
        {
            "self": SimpleNamespace(),
            "config": config,
            "atom_config": None,
            "prefix": "model",
            "torch": SimpleNamespace(cuda=SimpleNamespace(Stream=lambda: stream)),
            "make_layers": make_layers,
            "KimiDecoderLayer": lambda *a, **k: received.append(k.get("alt_stream")),
        },
    )
    assert received == [stream if expected else None] * 2


def test_decoder_passes_the_model_stream_to_full_attention():
    node = next(
        n
        for n in _constructor("KimiDecoderLayer").body
        if isinstance(n, ast.If) and any(_sets(child, "self_attn") for child in n.body)
    )
    stream, received = object(), []
    _run(
        [node],
        {
            "self": SimpleNamespace(),
            "config": SimpleNamespace(kimi_kda_layers=()),
            "layer_num": 3,
            "atom_config": None,
            "quant_config": None,
            "prefix": "layer3",
            "alt_stream": stream,
            "KimiFullAttention": lambda *a, **k: received.append(k.get("alt_stream")),
        },
    )
    assert received == [stream]


@pytest.mark.parametrize("has_indexer", [False, True])
def test_full_attention_assigns_the_stream_to_its_indexer(has_indexer):
    nodes = _constructor("KimiFullAttention").body
    first = next(i for i, n in enumerate(nodes) if _sets(n, "indexer"))
    last = next(
        i
        for i, n in enumerate(nodes[first:], first)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "mla_modules" for t in n.targets)
    )
    stream = object()
    layer = SimpleNamespace(hidden_size=8, q_lora_rank=4)
    _run(
        nodes[first:last],
        {
            "self": layer,
            "Indexer": lambda *a, **k: SimpleNamespace(exchange_stream=None),
            "atom_config": SimpleNamespace(index_cache_dtype="fp8"),
            "config": None,
            "quant_config": None,
            "prefix": "layer",
            "alt_stream": stream,
            "has_indexer": has_indexer,
        },
    )
    assert (
        layer.indexer is None
        if not has_indexer
        else layer.indexer.exchange_stream is stream
    )


@pytest.mark.parametrize(
    "topk,width,expected", [(2048, 2048, 2048), (2048, 2176, 4096), (None, None, 16)]
)
def test_layer_initialization_publishes_the_actual_sparse_granularity(
    topk, width, expected
):
    cls = next(
        n
        for n in _tree("atom/model_ops/attention_mla.py").body
        if isinstance(n, ast.ClassDef) and n.name == "MLAAttention"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    nodes = [n for n in init.body if _sets(n, "_sparse_prefill_kv_granularity")]
    layer = SimpleNamespace(topk_tokens=topk)
    _run(
        nodes,
        {
            "self": layer,
            "mla_modules": SimpleNamespace(
                indexer=SimpleNamespace(topk_out_width=width)
            ),
            "triton": SimpleNamespace(
                next_power_of_2=lambda n: 1 << (n - 1).bit_length()
            ),
        },
    )
    assert getattr(layer, "_sparse_prefill_kv_granularity", None) == expected


@pytest.fixture
def absorb():
    names = {
        "_configure_dcp_prefill_routes",
        "_absorb_into_q_out",
        "process_weights_after_loading",
    }
    nodes = [
        n
        for n in ast.walk(_tree("atom/model_ops/attention_mla.py"))
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    for node in nodes:
        node.decorator_list = []
    namespace = {
        "torch": torch,
        "dtypes": SimpleNamespace(fp8=torch.float8_e4m3fn),
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
    }
    _run(nodes, namespace)
    layer = SimpleNamespace(
        qrep_enabled=True,
        W_K_qrep_scale=None,
        W_K_qrep=torch.empty(1, dtype=torch.bfloat16),
        dtype=torch.bfloat16,
        kv_shuffled=False,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        pbm_enabled=False,
        _dcp_project_merge_out=lambda: None,
        _dcp_project_merge_out_head_major=lambda: None,
    )
    for name in names:
        setattr(layer, name, MethodType(namespace[name], layer))
    namespace.update(
        _runs_on_gfx950=lambda: True, is_rocm_aiter_fp4bmm_enabled=lambda: False
    )
    return layer, namespace


@pytest.mark.parametrize(
    "change,expected",
    [
        ({}, True),
        ({"gfx": False}, False),
        ({"dtype": torch.bfloat16}, False),
        ({"empty": True}, False),
        ({"fp4": True}, False),
        ({"kv_shuffled": True}, False),
        ({"qk_nope_head_dim": 192}, False),
        ({"qk_nope_head_dim": 512}, False),
        ({"kv_lora_rank": 96}, False),
        ({"W_K_qrep_scale": object()}, False),
        ({"W_K_qrep": torch.empty(1, dtype=torch.float16)}, False),
        ({"model_dtype": torch.float16}, False),
        ({"qrep_enabled": False}, False),
    ],
)
def test_absorb_fixed_gate_is_cached_and_live_cache_dtype_still_gates(
    absorb, change, expected
):
    layer, namespace = absorb
    change = dict(change)
    dtype = change.pop("dtype", torch.float8_e4m3fn)
    layer.dtype = change.pop("model_dtype", torch.bfloat16)
    empty = change.pop("empty", False)
    namespace["_runs_on_gfx950"] = lambda: change.get("gfx", True)
    namespace["is_rocm_aiter_fp4bmm_enabled"] = lambda: change.get("fp4", False)
    for name, value in change.items():
        if name not in ("gfx", "fp4"):
            setattr(layer, name, value)
    layer._configure_dcp_prefill_routes()
    namespace["_runs_on_gfx950"] = lambda: pytest.fail(
        "architecture repeated in forward"
    )
    cache = torch.empty(0 if empty else 1)
    assert layer._absorb_into_q_out(cache, dtype) is expected
    assert layer._absorb_into_q_out(cache, dtype) is expected


def test_weight_processing_initializes_the_cached_absorb_route(absorb):
    layer, namespace = absorb
    layer.num_heads, layer.v_head_dim = 2, 4
    layer.kv_b_proj = object()
    layer.dcp_group = SimpleNamespace(all_gather=lambda x, dim: x)
    weight = torch.zeros(
        (
            layer.num_heads * (layer.qk_nope_head_dim + layer.v_head_dim),
            layer.kv_lora_rank,
        ),
        dtype=torch.bfloat16,
    )
    namespace.update(
        get_and_maybe_dequant_weights=lambda _: weight,
        _mla_absorb_bf16_enabled=lambda: True,
    )
    layer.process_weights_after_loading()
    assert getattr(layer, "_absorb_q_pack_enabled", False)
    assert layer.W_K_qrep_scale is None


@pytest.mark.parametrize(
    "qrep,prefill,called",
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_default_and_decode_steps_do_not_call_the_absorb_gate(qrep, prefill, called):
    assignments = [
        n
        for n in ast.walk(_tree("atom/model_ops/attention_mla.py"))
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "absorb_into_q_out" for t in n.targets
        )
    ]
    calls = []
    ns = {
        "self": SimpleNamespace(_absorb_into_q_out=lambda *a: calls.append(a) or True),
        "use_qrep": qrep,
        "context": SimpleNamespace(is_prefill=prefill),
        "kv_cache": object(),
        "attn_metadata": SimpleNamespace(dtype_q=torch.float8_e4m3fn),
    }
    _run(assignments, ns)
    assert bool(calls) is called and ns["absorb_into_q_out"] is called


def test_patch_work_ranges_uses_integer_launch_geometry():
    fn = next(
        n
        for n in _tree("atom/model_ops/sparse_mla_work_plan.py").body
        if isinstance(n, ast.FunctionDef) and n.name == "patch_work_kv_ranges"
    )
    grids = []

    class Kernel:
        def __getitem__(self, grid):
            grids.append(grid)
            return lambda *a, **k: None

    def repeated(*args):
        pytest.fail("runtime wrapper repeated a fixed support/ceil-div query")

    ns = {
        "torch": torch,
        "triton": SimpleNamespace(cdiv=repeated),
        "BLOCK": 256,
        "_patch_work_kv_ranges": Kernel(),
    }
    _run([fn], ns)
    ns["patch_work_kv_ranges"](
        torch.zeros(257, 8, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(258, dtype=torch.int32),
        257,
    )
    assert grids == [(2,)]


@pytest.mark.parametrize("tokens", [0, 33])
@pytest.mark.parametrize("storage_heads", [96, 128])
def test_prefill_binds_aiter_nope_writer_to_real_head_output(tokens, storage_heads):
    calls = [
        n
        for n in ast.walk(_tree("atom/model_ops/attention_mla.py"))
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "fused_bf16_bmm_cat_quant_pad_mla"
    ]
    assert len(calls) == 1
    query = torch.empty(tokens, 96, 192, dtype=torch.bfloat16)
    q_nope = query[..., :128]
    weight = torch.empty(96, 512, 128, dtype=torch.bfloat16)
    output = torch.empty(tokens, storage_heads, 576, dtype=torch.float8_e4m3fn)
    real_output = output[:, :96]
    scale = torch.tensor([0.37])
    observed = []

    def absorb(q, w, rope, q_scale, *, out):
        assert q.shape == (96, tokens, 128)
        assert q.stride() == q_nope.transpose(0, 1).stride()
        assert q.data_ptr() == q_nope.data_ptr()
        assert w.shape == (96, 128, 512)
        assert w.stride() == (65536, 1, 128)
        assert w.data_ptr() == weight.data_ptr()
        assert rope is None and q_scale is scale
        assert out is real_output and out.shape == (tokens, 96, 576)
        observed.append(out)

    _run(
        calls,
        {
            "fused_bf16_bmm_cat_quant_pad_mla": absorb,
            "q_nope": q_nope,
            "q_out_write": real_output,
            "self": SimpleNamespace(W_K_qrep=weight, _q_scale=scale),
        },
    )
    assert observed == [real_output]
