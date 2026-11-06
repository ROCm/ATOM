"""Native owned prefill FFN with explicit route and join/reduce-scatter ops.

`build_owned` re-compiles `prefill_ffn.prequantized_owned_ffn` with exactly two
calls swapped: the expert `forward_impl` becomes the route producer op and the
final combine/scatter becomes the push-join reduce-scatter op. Everything else
(preparation, shared expert, streams) is the native body.
"""

import ast
import importlib
import inspect

import torch

from . import explicit_routes  # noqa: F401  (registers the two custom ops)

pf = importlib.import_module("atom.models.deepseek_v41.owner.prefill_ffn")


def build_owned():
    source = inspect.getsource(pf.prequantized_owned_ffn)
    tree = ast.parse(source)
    edits = []

    class Replace(ast.NodeTransformer):
        def visit_Return(self, node):
            call = node.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "forward_impl"
            ):
                assert ast.unparse(call.func.value) == "moe.experts"
                edits.append(("producer", node.value))
                node.value = ast.parse(
                    "_explicit_route(value, logits, scale, moe.experts.layer_name)",
                    mode="eval",
                ).body
            elif (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "v41_ffn_combine_scatter"
            ):
                edits.append(("consumer", node.value))
                call.func.id = "_explicit_join"
            return self.generic_visit(node)

    tree = Replace().visit(tree)
    assert [e[0] for e in edits] == ["producer", "consumer"]
    tree.body[0].name = "_explicit_owned_ffn"
    ast.fix_missing_locations(tree)
    namespace = dict(pf.prequantized_owned_ffn.__globals__)
    namespace["_explicit_route"] = torch.ops.aiter.dsv41_prefill_routes
    namespace["_explicit_join"] = torch.ops.aiter.dsv41_prefill_join_rs
    exec(  # noqa: S102
        compile(tree, "<dsv41_explicit_owned_prefill>", "exec"), namespace
    )
    return namespace["_explicit_owned_ffn"]


OWNED = build_owned()


def explicit_owned_ffn(moe, local, rows):
    assert 4096 <= rows <= 16384 and not getattr(moe, "prefill_route_packet", False)
    assert (
        moe.experts.supports_prequantized_input()
        and moe.experts.tp_size == 4
        and moe.experts.ep_size == 1
    )
    return OWNED(moe, local, rows)
