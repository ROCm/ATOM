import ast
from pathlib import Path


def _tree():
    model_path = Path(__file__).parents[1] / "atom" / "models" / "kimi_k3.py"
    return ast.parse(model_path.read_text())


def _ops_tree():
    ops_path = (
        Path(__file__).parents[1] / "atom" / "model_ops" / "attention_residual.py"
    )
    return ast.parse(ops_path.read_text())


def _attn_res_calls(node):
    """Calls to an AttnRes submodule, e.g. ``self.mlp_attn_res(...)``."""
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr.endswith("attn_res")
    ]


def _block_residual_shape_tests(node):
    """``block_residual.shape[...]`` subscripts, i.e. "any candidates yet?"."""
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Subscript)
        and isinstance(n.value, ast.Attribute)
        and n.value.attr == "shape"
        and isinstance(n.value.value, ast.Name)
        and n.value.value.id == "block_residual"
    ]


def test_attn_res_call_sites_unpack_both_outputs():
    tree = _tree()
    calls = {id(c) for c in _attn_res_calls(tree)}
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and id(node.value) in calls
    ]

    assert assignments
    for assignment in assignments:
        assert len(assignment.targets) == 1
        assert isinstance(assignment.targets[0], ast.Tuple), (
            f"AttnRes.forward returns (mixed_output, prefix_out), but the call at "
            f"kimi_k3.py:{assignment.lineno} does not unpack both outputs"
        )


def test_post_mlp_add_is_deferred_to_next_attn_res():
    tree = _tree()
    decoder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KimiDecoderLayer"
    )
    forward = next(
        node
        for node in decoder.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )

    # The MLP output is returned as pending state rather than materialized by an
    # elementwise add at the end of every decoder layer. _ffn goes further: an
    # MoE's routed and shared outputs come back unsummed, so even the add
    # BETWEEN them is deferred into the next layer's kernel.
    ffn_line = next(
        node.lineno
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_ffn"
    )
    trailing_adds = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and node.lineno > ffn_line
    ]
    assert not trailing_adds
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and len(node.value.elts) == 4
        and [getattr(e, "id", None) for e in node.value.elts[1:3]]
        == ["hidden_states", "shared"]
        for node in ast.walk(forward)
    )

    # Both the next-layer entry and the final model attn-res consume each
    # pending, as an argument to the kernel rather than a separate add. Only
    # the no-attn-residual fallback in KimiDecoderLayer.forward and the
    # not-last-PP-rank branch may add them by hand.
    for name in ("pending_add", "pending_add2"):
        consumers = [
            call
            for call in _attn_res_calls(tree)
            if any(
                isinstance(a, ast.Name) and a.id == name
                for a in list(call.args) + [k.value for k in call.keywords]
            )
        ]
        assert len(consumers) == 2, f"{name} at {len(consumers)} attn_res sites"


def test_attn_res_owns_the_empty_block_branch():
    """The branch on "are there candidates yet" lives in AttnRes, not callers.

    Callers pass block_residual straight through; if a caller re-grew a
    ``block_residual.shape[1]`` test, the branching would be back where this
    refactor removed it from.
    """
    # AttnRes lives in model_ops now; the branch has to have moved with it
    # rather than been left behind or duplicated in the model file.
    attn_res = next(
        node
        for node in _ops_tree().body
        if isinstance(node, ast.ClassDef) and node.name == "AttnRes"
    )
    assert _block_residual_shape_tests(attn_res)

    shape_tests = _block_residual_shape_tests(_tree())
    assert not shape_tests, (
        "block_residual.shape is inspected outside AttnRes at kimi_k3.py lines "
        f"{[n.lineno for n in shape_tests]}"
    )
