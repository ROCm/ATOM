# SPDX-License-Identifier: MIT
"""CPU validation of M3's PP-stage payloads and head-major PAGE geometry."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.mha_head_transfer import head_slices
from atom.kv_transfer.disaggregation.page_region import page_region
from atom.kv_transfer.disaggregation.types import KVTransferTensors
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField

ROOT = Path(__file__).resolve().parents[1]


def _method(path, cls, method, scope):
    """Run the production method without importing GPU-only model dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    body = (
        tree.body
        if cls is None
        else next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == cls
        )
    )
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == method)
    node.decorator_list = []
    scope = dict(scope)
    exec(  # noqa: S102 - execute only a method parsed from this checkout
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(path),
            "exec",
            flags=__import__("__future__").annotations.compiler_flag,
        ),
        scope,
    )
    return scope[method]


@pytest.mark.parametrize("groups", [1, 8])
@pytest.mark.parametrize("heads,tp", [(8, 4), (4, 4), (2, 4), (8, 1)])
@pytest.mark.parametrize("width", [64, 2048])
def test_head_slices_preserve_page_head_and_block_order(groups, heads, tp, width):
    src = (
        torch.arange(3 * groups * heads * width, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
    )
    src = src.reshape(3, groups, heads, width)
    dh = max(1, heads // tp)
    for rank in range(tp):
        dst = torch.full((5, groups, dh, width), 255, dtype=torch.uint8)
        for sb, db in [(2, 1), (0, 4)]:
            for off, target, size in head_slices(
                src[0].numel(), dst[0].numel(), (groups, heads), (groups, dh), rank, tp
            ):
                dst[db].flatten()[target : target + size] = src[sb].flatten()[
                    off : off + size
                ]
            first = rank * dh if heads >= tp else rank // (tp // heads)
            assert torch.equal(dst[db], src[sb, :, first : first + dh])
        assert torch.all(dst[[0, 2, 3]] == 255)


@pytest.mark.parametrize(
    "args",
    [
        (128, 128, None, (1, 1), 0, 4),
        (128, 64, None, None, 0, 4),
        (128, 32, (8, 8), (1, 2), 0, 4),
        (128, 64, (1, 8), (1, 2), 0, 4),
        (128, 32, (1, 8), (1, 2), 4, 4),
        (128, 32, (0, 8), (1, 2), 0, 4),
    ],
)
def test_head_slices_reject_incompatible_geometry(args):
    with pytest.raises(ValueError):
        head_slices(*args)


def test_index_page_is_replicated():
    for rank in range(4):
        assert head_slices(18432, 18432, None, None, rank, 4) == [(0, 0, 18432)]


def test_builder_labels_global_layers_and_sparse_physical_pages():
    pool = MhaKvPool(
        layers=2,
        block_size=128,
        num_kv_heads=8,
        head_dim=128,
        kv_dtype=torch.float8_e4m3fn,
        extra_fields=[EntryField("index", 1, (128 * 144,), torch.uint8)],
    )
    pool.allocate(2, "cpu")

    class Layer:
        def __init__(self, number, sparse):
            self.layer_num, self.sparse = number, sparse

    dense, sparse = Layer(2, False), Layer(3, True)
    builder = SimpleNamespace(
        kv_pools={"geometry": pool},
        pool_rows={"geometry": {dense: 0, sparse: 1}, "index": {sparse: 0}},
        model_runner=SimpleNamespace(
            block_size=128, config=SimpleNamespace(hf_config=object())
        ),
    )
    fn = _method(
        "atom/model_ops/attentions/aiter_attention.py",
        "AiterAttentionMetadataBuilder",
        "get_kv_transfer_tensors",
        {
            "_is_minimax_m3_config": lambda _: True,
            "_is_indexed_sparse_attention": lambda m: m.sparse,
            "INDEX_ROWS": "index",
            "page_region": page_region,
            "KVTransferTensors": KVTransferTensors,
        },
    )
    result = fn(builder)
    assert len(result.pages) == 9
    for page in result.pages:
        role = page.region.semantic_role
        if ".index." in role:
            assert role.startswith("m3.layer_3.index.")
            assert page.tp_head_groups is None
        elif "layer_2." in role:
            assert ".shuffle128." in role and page.tp_head_groups == (1, 8)
        else:
            assert ".shuffle16." in role and page.tp_head_groups == (8, 8)


class _Intermediate:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        return self.tensors[key]


def test_pp4_forward_preserves_eagle_aux_from_earlier_stages():
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)

    def layer_fn(i):
        def run(positions, hidden, residual, capture_aux=False):
            hidden = hidden * (1 + i / 100) + i
            residual = hidden if residual is None else residual + i
            return (
                (hidden, residual, hidden + residual)
                if capture_aux
                else (hidden, residual)
            )

        return run

    layers = [layer_fn(i) for i in range(60)]
    forward = _method(
        "atom/models/minimax_m3/model.py",
        "MiniMaxM3Model",
        "forward",
        {
            "get_pp_group": lambda: pp,
            "IntermediateTensors": _Intermediate,
            "fused_allreduce_gemma_rms_norm": lambda h, r, _: (h + r, None),
        },
    )
    state = SimpleNamespace(
        start_layer=0,
        end_layer=60,
        layers=layers,
        norm=None,
        aux_hidden_state_layers=(2, 30, 57),
    )
    inputs = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    baseline = forward(state, None, None, inputs_embeds=inputs)
    intermediate = None
    for stage in range(4):
        pp.is_first_rank, pp.is_last_rank = stage == 0, stage == 3
        state.start_layer, state.end_layer = stage * 15, (stage + 1) * 15
        result = forward(
            state, None, None, intermediate_tensors=intermediate, inputs_embeds=inputs
        )
        if stage < 3:
            assert isinstance(result, _Intermediate)
            intermediate = result
    assert torch.equal(result[0], baseline[0])
    assert len(result[1]) == 3
    assert all(torch.equal(a, b) for a, b in zip(result[1], baseline[1]))


@pytest.mark.parametrize(
    "method", ["send_intermediate_tensors", "async_send_intermediate_tensors"]
)
def test_pp_transport_keeps_aux_and_filters_unrelated_values(method):
    pp = SimpleNamespace(
        rank_in_group=0, world_size=4, ranks=[0, 1, 2, 3], cpu_group=None
    )
    sent = []
    pp.send_tensor_dict = lambda tensors, **kw: sent.append(tensors)
    fn = _method(
        "atom/distributed/pp_comm.py",
        None,
        method,
        {
            "get_pp_group": lambda: pp,
            "pp_send_allgather_group": lambda: None,
            "_PP_PROXY_KEYS": ("hidden_states", "residual", "sparse_kv_indices"),
            "_split_tensor_dict": lambda tensors: (sent.append(tensors), []),
            "_async_send_object": lambda *args: [],
        },
    )
    values = {
        k: object()
        for k in (
            "hidden_states",
            "residual",
            "eagle_aux_2",
            "eagle_aux_30",
            "unrelated",
        )
    }
    fn(_Intermediate(values))
    assert set(sent[0]) == set(values) - {"unrelated"}
