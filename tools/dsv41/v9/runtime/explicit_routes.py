# ruff: noqa: I001, S102
# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
"""Explicit raw-route producer/consumer qualification; no production dispatch installed."""

import ast
import hashlib
import importlib
import inspect
import textwrap
import torch
from aiter.dist.parallel_state import get_tp_group
from atom.utils.forward_context import get_forward_context
from atom.utils.custom_register import direct_register_custom_op
from adapter import PushJoinRS
from routes_interface import call_native_routes

fm = importlib.import_module("aiter.fused_moe")
mm = importlib.import_module("atom.model_ops.moe")


def route_frontend(*args, **kwargs):
    bound = inspect.signature(fm.fused_moe).bind(*args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments
    for name in (
        "shared_w1",
        "shared_w2",
        "shared_w1_scale",
        "shared_w2_scale",
        "stage2_scatter",
        "output",
    ):
        assert a.pop(name) is None, name
    assert a.pop("shared_expert_id") == -1 and a.pop("clamp_shared") is True
    assert a.pop("splitk") == 0
    for name in ("activation", "quant_type", "quant_type_a"):
        if a[name] is not None and hasattr(a[name], "value"):
            a[name] = a[name].value
    a["block_size_M"] = a["block_size_M"] or -1
    hidden = a.pop("hidden_states")
    w1 = a.pop("w1")
    w2 = a.pop("w2")
    weight = a.pop("topk_weight")
    ids = a.pop("topk_ids")
    return call_native_routes(hidden, w1, w2, weight, ids, **a)


def build_apply_shadow():
    source = textwrap.dedent(inspect.getsource(mm.Mxfp4MoEMethod.apply))
    original = ast.parse(source)
    tree = ast.parse(source)
    original.body[0].decorator_list = []
    tree.body[0].decorator_list = []
    edits = []

    class ReplaceFinal(ast.NodeTransformer):
        def visit_Return(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "fused_moe"
            ):
                edits.append(node.value.func)
                node.value.func = ast.Name(
                    id="_explicit_route_frontend", ctx=ast.Load()
                )
            return self.generic_visit(node)

    tree = ReplaceFinal().visit(tree)
    assert len(edits) == 1
    tree.body[0].name = "_raw_native_apply"
    restored = ast.parse(ast.unparse(tree))
    restored.body[0].name = original.body[0].name

    class Restore(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "_explicit_route_frontend":
                return ast.Name(id="fused_moe", ctx=ast.Load())
            return node

    restored = Restore().visit(restored)
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )
    ast.fix_missing_locations(tree)
    native = inspect.unwrap(mm.Mxfp4MoEMethod.apply)
    namespace = dict(native.__globals__)
    namespace["_explicit_route_frontend"] = route_frontend
    exec(compile(tree, "<dsv41_native_apply_raw_route>", "exec"), namespace)
    return namespace["_raw_native_apply"], {
        "native_apply_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "restored_AST_equal": True,
        "transform": "Remove tracing decorator from shadow only; replace exactly one native fused_moe return with parity frontend/raw G2 override. Native object never changed.",
    }


APPLY, APPLY_AUDIT = build_apply_shadow()


def forward_routes(experts, hidden, logits, scale):
    assert experts.supports_prequantized_input()
    assert (experts.tp_size, experts.ep_size, experts.dp_size) == (4, 1, 1)
    assert not experts.moe_parallel_config.pcp_merge and not experts.use_ep
    assert experts.expert_mask is None and experts.num_fused_shared_experts == 0
    assert hidden.shape[1] == 5120 and 4096 <= hidden.shape[0] <= 16384
    assert hidden.dtype == mm.dtypes.fp8 and scale.dtype == mm.dtypes.fp8_e8m0
    assert scale.shape == (hidden.shape[0], 160) and logits.shape == (
        hidden.shape[0],
        384,
    )
    method = experts.quant_method
    weights, ids = method.select_experts(experts, hidden, logits)
    weights, ids = method.place_experts(experts, weights, ids, False)
    assert weights.shape == ids.shape == (hidden.shape[0], 6)
    return APPLY(
        method,
        layer=experts,
        x=hidden,
        topk_weights=weights,
        topk_ids=ids,
        global_num_experts=experts.global_num_experts,
        expert_map=experts.expert_map,
        activation=experts.activation,
        apply_router_weight_on_input=experts.apply_router_weight_on_input,
        prefix=f"{experts.prefix}.fused_moe",
        x_scale=scale,
        scheduled_tokens_gpu=None,
    )


def route_producer(
    hidden: torch.Tensor, logits: torch.Tensor, scale: torch.Tensor, layer_name: str
) -> torch.Tensor:
    experts = get_forward_context().no_compile_layers[layer_name]
    return forward_routes(experts, hidden, logits, scale)


def route_producer_fake(
    hidden: torch.Tensor, logits: torch.Tensor, scale: torch.Tensor, layer_name: str
) -> torch.Tensor:
    return hidden.new_empty((hidden.shape[0], 6, 5120), dtype=torch.bfloat16)


candidates = {}


def route_consumer(routes: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    rows = shared.shape[0]
    if rows not in candidates:
        candidates[rows] = PushJoinRS(get_tp_group(), rows)
    return candidates[rows](routes, shared, 6)


def route_consumer_fake(routes: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    return shared.new_empty(((shared.shape[0] + 3) // 4, 5120))


for name, real, fake in (
    ("dsv41_qualify_prefill_routes", route_producer, route_producer_fake),
    ("dsv41_qualify_prefill_join_rs", route_consumer, route_consumer_fake),
):
    direct_register_custom_op(name, real, [], fake_impl=fake)
