# ruff: noqa: I001, S102
# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
"""Explicit route tensor prototype using the pinned native stage2 implementation.

No producer/consumer pending dictionary. Opaque registration and service dispatch
remain separate integration work; this module alone enables no model path.
"""

import ast
import hashlib
import inspect
import torch
import aiter.fused_moe as _import_alias  # noqa: F401
import importlib

fm = importlib.import_module("aiter.fused_moe")


def build_shadow():
    source = inspect.getsource(fm._flydsl_v2_stage2_wrapper)
    original = ast.parse(source)
    tree = ast.parse(source)
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef) and isinstance(fn.body[-1], ast.Return)
    assert isinstance(fn.body[-1].value, ast.Name) and fn.body[-1].value.id == "out"
    skipped = []

    class DropReduction(ast.NodeTransformer):
        def visit_Expr(self, node):
            c = node.value
            if (
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == "_run_moe_reduction"
            ):
                skipped.append(node)
                return ast.copy_location(
                    ast.Expr(value=ast.Constant(value="dsv41_drop_final_reduction")),
                    node,
                )
            return self.generic_visit(node)

    tree = DropReduction().visit(tree)
    assert len(skipped) == 1
    fn = tree.body[0]
    old_return = fn.body[-1]
    fn.body[-1] = ast.Return(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="target", ctx=ast.Load()), attr="view", ctx=ast.Load()
            ),
            args=[
                ast.Name(id=n, ctx=ast.Load())
                for n in ("token_num", "topk", "model_dim_runtime")
            ],
            keywords=[],
        )
    )
    fn.name = "_dsv41_stage2_routes_shadow"
    ast.fix_missing_locations(tree)
    # Restore the two intended edits to prove no unrelated source transform.
    restored = ast.parse(ast.unparse(tree))
    restored.body[0].name = original.body[0].name
    restored.body[0].body[-1] = old_return

    class RestoreReduction(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Constant)
                and node.value.value == "dsv41_drop_final_reduction"
            ):
                return skipped[0]
            return self.generic_visit(node)

    restored = RestoreReduction().visit(restored)
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )
    namespace = dict(fm._flydsl_v2_stage2_wrapper.__globals__)
    exec(compile(tree, "<dsv41_explicit_routes_shadow>", "exec"), namespace)
    return namespace[fn.name], {
        "original_function_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "transform": "Only drop one final _run_moe_reduction expression and replace final return out with target.view(M,topk,H). Early scatter returns untouched, runtime guards require reduce/BF16/noFP8 route.",
        "restored_AST_equal": True,
        "shadow_source": ast.unparse(tree),
    }


SHADOW, SOURCE_AUDIT = build_shadow()


def override_stage2(*, ordinary_stage2, stage2_args, stage2_kwargs):
    assert ordinary_stage2.func is fm._flydsl_v2_stage2_wrapper
    kwargs = dict(ordinary_stage2.keywords or {})
    kwargs.update(stage2_kwargs)
    bound = inspect.signature(fm._flydsl_v2_stage2_wrapper).bind(*stage2_args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments
    cfg = fm.parse_flydsl_v2_gemm2_kernel(a["kernelName"])
    assert cfg and cfg["epilog"] == "reduce"
    assert a["expert_mask"] is None and a["topk"] == 6
    assert not fm._flydsl_stage2_fp8_enabled()
    out = a["out"]
    assert out.dtype == torch.bfloat16 and out.ndim == 2 and out.shape[1] == 5120
    assert 4096 <= out.shape[0] <= 16384
    route = SHADOW(*stage2_args, **kwargs)
    assert (
        route.shape == (out.shape[0], 6, 5120)
        and route.dtype == out.dtype
        and route.is_contiguous()
    )
    return route


def call_native_routes(hidden_states, w1, w2, topk_weights, topk_ids, **kwargs):
    """Real CUDA helper; must be enclosed in a new opaque op with matching fake."""
    assert (
        hidden_states.ndim == 2
        and 4096 <= hidden_states.shape[0] <= 16384
        and hidden_states.shape[1] == 5120
    )
    assert topk_ids.shape == topk_weights.shape == (hidden_states.shape[0], 6)
    assert kwargs.get("expert_mask") is None
    kwargs = dict(kwargs)
    assert kwargs.pop("dtype", torch.bfloat16) == torch.bfloat16

    def guard_metadata(meta):
        assert not meta.run_1stage and meta.stage2.func is fm._flydsl_v2_stage2_wrapper
        cfg = fm.parse_flydsl_v2_gemm2_kernel(meta.stage2.keywords["kernelName"])
        assert cfg and cfg["epilog"] == "reduce"
        return meta

    assert "_stage2_override" in inspect.signature(fm._fused_moe_impl).parameters
    return fm._fused_moe_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        **kwargs,
        dtype=torch.bfloat16,
        _metadata_transform=guard_metadata,
        _stage2_override=override_stage2,
    )


def routes_fake(hidden_states, topk_ids):
    return torch.empty(
        (hidden_states.shape[0], topk_ids.shape[1], 5120),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )


def join_scatter_fake(shared):
    return torch.empty(
        ((shared.shape[0] + 3) // 4, 5120), device=shared.device, dtype=torch.bfloat16
    )
