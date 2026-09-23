# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3's sparse attention must break the breakable cudagraph.

vLLM auto-enables ``VLLM_USE_BREAKABLE_CUDAGRAPH`` for this architecture, so
plugin mode compiles nothing and splits nothing: one stream capture drives the
whole forward, and only ops carrying ``@eager_break_during_capture`` end a
segment. Drop the decorator and 57 of M3's 60 attention layers get captured
wholesale, freezing this batch's token counts, ``block_table``, ``seq_lens``
and topk indices into every replay.

Nothing downstream notices. A cold prompt prefills past the largest captured
size and runs eagerly, so it answers correctly; reuse a prefix and the short
remainder lands inside a captured size and answers fluently from the wrong KV.
Catching that costs a two-run accuracy sweep on four GPUs, which is why this
guard is here instead.

The scan reads the source rather than importing it: the module imports ``aiter``
at module scope, so an importing test would skip on every CI runner -- exactly
where the guard is supposed to fire.
"""

import ast
from pathlib import Path

M3_ATTENTION = (
    Path(__file__).resolve().parents[2]
    / "atom"
    / "plugin"
    / "vllm"
    / "attention"
    / "minimax_m3_attnetion.py"
)

BREAK_DECORATOR = "eager_break_during_capture"
SPARSE_OP = "minimax_m3_sparse_attention"


def _module() -> ast.Module:
    return ast.parse(M3_ATTENTION.read_text())


def _decorator_names(node: ast.FunctionDef) -> set[str]:
    names = set()
    for dec in node.decorator_list:
        target = getattr(dec, "func", dec)
        name = getattr(target, "id", None) or getattr(target, "attr", None)
        if name:
            names.add(name)
    return names


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_sparse_attention_run_is_an_eager_break():
    tree = _module()
    run = _find_function(tree, "_sparse_attn_run")
    assert run is not None, (
        "MiniMaxM3SparseAttentionForVllm._sparse_attn_run is gone -- whatever "
        "replaced it still has to carry the eager break, or M3's 57 sparse "
        "layers go back inside the captured graph"
    )
    assert BREAK_DECORATOR in _decorator_names(run), (
        f"_sparse_attn_run lost @{BREAK_DECORATOR}; sparse attention would be "
        "captured with this batch's metadata frozen into every replay"
    )


def test_the_break_writes_into_a_caller_owned_buffer():
    """The decorator replays the Python kernel, so its output must be passed in.

    A tensor allocated inside lands at a new address on every replay while the
    captured segments that consume it still read the address recorded at
    capture -- the failure ``eager_break_during_capture`` documents by name.
    """
    tree = _module()

    op = _find_function(tree, SPARSE_OP)
    assert op is not None, f"{SPARSE_OP} is gone"
    assert "output" in {a.arg for a in op.args.args}, (
        f"{SPARSE_OP} no longer takes `output` from its caller; allocating it "
        "inside breaks every replay after the first"
    )

    mark = next(
        (
            dec
            for dec in op.decorator_list
            if getattr(getattr(dec, "func", dec), "id", None) == "mark_spliting_op"
        ),
        None,
    )
    assert mark is not None, f"{SPARSE_OP} is no longer registered as a custom op"
    mutates = next((kw.value for kw in mark.keywords if kw.arg == "mutates_args"), None)
    assert isinstance(mutates, ast.List), "mutates_args must be a literal list"
    assert "output" in {
        el.value for el in mutates.elts if isinstance(el, ast.Constant)
    }, (
        f"{SPARSE_OP} must declare `output` in mutates_args, or torch records "
        "it as a pure return and the schema stops marking the write"
    )

    run = _find_function(tree, "_sparse_attn_run")
    assert run is not None and "output" in {
        a.arg for a in run.args.args
    }, "the eager break must receive the buffer rather than allocate one"


# ---------------------------------------------------------------------------
# The other half of the contract: the decorator above only does anything while
# breakable cudagraph is on, and for M3 nobody turns it on explicitly -- vLLM
# does it from a hardcoded tuple of architecture names in its own config. So
# M3's prefix-reuse correctness rests on a literal in a package ATOM tracks
# daily and does not own. Drop the name there (a rename, a refactor of the
# gate, an upstream decision to enable breakable some other way) and the
# decorator becomes dead code, in silence: the server still boots, cold
# prompts still answer correctly, and only reused prefixes read the wrong KV.
#
# ATOM's own `splitting_ops` (atom/config.py) cannot cover for it. That list
# belongs to ATOM's native engine, and in plugin mode with breakable enabled
# vLLM sets CompilationMode.NONE -- there is no FX split to fall back to. This
# test is the whole fallback.

M3_ARCHITECTURES = (
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
)

BREAKABLE_ENV = "VLLM_USE_BREAKABLE_CUDAGRAPH"


def _vllm_config_source() -> Path | None:
    """Path to vllm/config/vllm.py without importing vllm."""
    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(next(iter(spec.submodule_search_locations))) / "config" / "vllm.py"
    return path if path.exists() else None


def _auto_enable_gate(tree: ast.Module) -> ast.If | None:
    """The ``if`` whose body switches the breakable env var on.

    Matched by what it does, not by where it sits or what the tuple is called,
    so upstream reformatting does not turn this guard into a false alarm.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for stmt in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == BREAKABLE_ENV
                ):
                    return node
    return None


def test_vllm_still_auto_enables_breakable_cudagraph_for_m3():
    source = _vllm_config_source()
    assert source is not None, (
        "vllm is not importable, so the architecture gate M3 depends on cannot "
        "be checked -- this guard must not pass quietly in that state"
    )

    gate = _auto_enable_gate(ast.parse(source.read_text()))
    assert gate is not None, (
        f"no `if` in {source} assigns os.environ[{BREAKABLE_ENV!r}] any more. "
        "vLLM changed how breakable cudagraph gets enabled; re-derive how M3 "
        "turns it on before deleting this test, because M3's eager break is "
        "inert whenever it is off"
    )

    named = {
        n.value
        for n in ast.walk(gate.test)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    missing = [arch for arch in M3_ARCHITECTURES if arch not in named]
    assert not missing, (
        f"{missing} no longer appear in vLLM's breakable auto-enable gate "
        f"({source}). M3's @{BREAK_DECORATOR} only fires while breakable "
        "cudagraph is enabled, so with this gate closed M3's sparse layers are "
        "captured with one batch's block_table and topk indices frozen into "
        "every replay. Cold prompts still look right; reused prefixes answer "
        f"fluently from the wrong KV. Either get the names restored upstream, "
        f"or set {BREAKABLE_ENV}=1 for M3 from the ATOM plugin"
    )
