"""Bind the private adapter without duplicating or patching production math."""

import ast
import copy
import inspect
import textwrap


def build_ast(source, *, marked=False):
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected exactly the native forward_owner function")
    function = tree.body[0]
    original = copy.deepcopy(function)
    imports = [
        (i, node)
        for i, node in enumerate(function.body)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "owner_head_attention"
    ]
    if len(imports) != 1:
        raise ValueError("Native FP8 binding changed; inspect before using A0")
    import_index, old_import = imports[0]
    if [(a.name, a.asname) for a in old_import.names] != [
        ("attend_fp8_head_shards", None)
    ]:
        raise ValueError("Unexpected symbols in the native adapter import")
    del function.body[import_index]
    # Verify the entire remaining native function, including four wo_b
    # partials and all state updates, is bytecode-source structurally intact.
    restored = copy.deepcopy(function)
    restored.body.insert(import_index, old_import)
    if ast.dump(restored) != ast.dump(original):
        raise AssertionError("A0 binding changed native forward arithmetic")
    if marked:
        body = []
        inserted = []
        for node in function.body:
            target = (
                node.targets[0].id
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                else None
            )
            if target == "attend":
                body.append(ast.parse('_a0_boundary("attention_start")').body[0])
                inserted.append("attention_start")
            body.append(node)
            if target == "output":
                if (
                    not isinstance(node.value, ast.Call)
                    or not isinstance(node.value.func, ast.Name)
                    or node.value.func.id != "attend"
                ):
                    raise ValueError("Native attention output boundary changed")
                body.append(ast.parse('_a0_boundary("attention_end")').body[0])
                inserted.append("attention_end")
        if inserted != ["attention_start", "attention_end"]:
            raise ValueError("Native attention boundaries are not unique")
        function.body = body
    return ast.fix_missing_locations(tree)


def bind_forward(original, adapter, *, boundary=None):
    """Original module/package globals remain available to relative imports.

    `boundary` is only used in separate observer graphs. The complete native
    window and the unobserved A0 have no marker calls.
    """
    tree = build_ast(inspect.getsource(original), marked=boundary is not None)
    namespace = original.__globals__.copy()
    namespace["attend_fp8_head_shards"] = adapter
    namespace["_a0_boundary"] = boundary
    # Execute only the AST checked above, taken from the fixed local runtime.
    code = compile(tree, __file__ + ":native_forward_binding", "exec")
    exec(code, namespace)  # noqa: S102
    return namespace[original.__name__]
