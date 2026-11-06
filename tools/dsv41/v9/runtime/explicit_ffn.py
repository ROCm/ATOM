# ruff: noqa: I001, S102
# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
"""Native owned prefill with explicit 3D route and 2D join/RS opaque boundaries."""

import ast, hashlib, importlib, inspect
import torch
from explicit_routes import APPLY_AUDIT  # noqa: F401

pf = importlib.import_module("atom.models.deepseek_v41.prefill_ffn")


def build_owned():
    source = inspect.getsource(pf.prequantized_owned_ffn)
    original = ast.parse(source)  # noqa: F841
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
    # Preserve exact native preparation/shared/stream body; only the two interface calls change.
    tree.body[0].name = "_explicit_owned_ffn"
    ast.fix_missing_locations(tree)
    namespace = dict(pf.prequantized_owned_ffn.__globals__)
    namespace["_explicit_route"] = torch.ops.aiter.dsv41_qualify_prefill_routes
    namespace["_explicit_join"] = torch.ops.aiter.dsv41_qualify_prefill_join_rs
    exec(compile(tree, "<dsv41_explicit_owned_prefill>", "exec"), namespace)
    return namespace["_explicit_owned_ffn"], {
        "native_prefill_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "edits": [
            "native forward_impl -> explicit route opaque op",
            "native combine/scatter -> explicit route consumer opaque op",
        ],
        "preparation_shared_stream_body_unchanged": True,
        "shadow_source": ast.unparse(tree),
    }


OWNED, OWNED_AUDIT = build_owned()


def explicit_owned_ffn(moe, local, rows):
    assert 4096 <= rows <= 16384 and not getattr(moe, "prefill_route_packet", False)
    assert (
        moe.experts.supports_prequantized_input()
        and moe.experts.tp_size == 4
        and moe.experts.ep_size == 1
    )
    return OWNED(moe, local, rows)
