import ast
from pathlib import Path


def test_inline_call_wrapper_forwards_new_torch_arguments():
    decorators_path = (
        Path(__file__).resolve().parents[1] / "atom" / "utils" / "decorators.py"
    )
    module = ast.parse(decorators_path.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "patched_inline_call"
    )

    assert wrapper.args.vararg is not None
    assert wrapper.args.vararg.arg == "inline_args"
    assert wrapper.args.kwarg is not None
    assert wrapper.args.kwarg.arg == "inline_kwargs"

    forwarded_call = next(
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "inline_call"
    )
    assert any(
        isinstance(arg, ast.Starred)
        and isinstance(arg.value, ast.Name)
        and arg.value.id == "inline_args"
        for arg in forwarded_call.args
    )
    assert any(
        keyword.arg is None
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "inline_kwargs"
        for keyword in forwarded_call.keywords
    )
