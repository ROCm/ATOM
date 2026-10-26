# SPDX-License-Identifier: MIT
"""The plugin's pool geometry and core's must be the same geometry.

The CSA2 pool is declared twice: once by the scheduler that sizes the proxy
KV allocation, and once by the runtime that addresses it. The plugin derives
the first from its own copy of the arithmetic (`_build_v41_pool_geometry`),
core derives the second inside `DeepseekV41MetadataBuilder.__init__`. If the
two ever compute different shapes, vLLM allocates N bytes while the kernels
read M -- pool corruption, not an exception, and not visible until answers go
wrong somewhere unrelated.

Every other duplication this PR introduced is a copy whose drift shows up as
a crash or a missing feature. This one does not, which is why it gets a test
of its own rather than a source-fingerprint canary: the fields are compared,
so a change to either side has to be made on both or this goes red.

Core's copy is inline in a constructor that needs a model runner, so it is
read out by building the same `V41PoolGeometry` call from the same source --
the AST of that constructor -- rather than by instantiating a builder, which
would need a GPU.

It compares source text and imports nothing, so it lives outside
`tests/plugin` and CPU CI collects it. It used to sit inside that directory
behind an `importorskip("vllm")` it never needed, which meant the one check
standing between a silent geometry drift and pool corruption ran only in the
approval-gated GPU job.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "atom/model_ops/attentions/deepseek_v41/backend.py"
PLUGIN = ROOT / "atom/plugin/vllm/deepseek_v41_bridge.py"


def _geometry_call(source: str, func: str) -> ast.Call:
    """The single `V41PoolGeometry(...)` construction inside `func`."""
    fn = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == func
    )
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "V41PoolGeometry"
    ]
    assert len(calls) == 1, f"{func} builds {len(calls)} geometries, expected 1"
    return calls[0]


def _shape(call: ast.Call) -> tuple[list[str], dict[str, str]]:
    """Positional args and keywords, normalized to source text.

    Normalized because the two differ in exactly one way by construction:
    core reads `self.config.x` where the plugin reads `hf_config.x`, and core
    reads `self.block_size` where the plugin takes `block_size`. Anything
    beyond that renaming is a real divergence.
    """

    def norm(node):
        text = ast.unparse(node)
        for before, after in (
            ("self.config.", "hf_config."),
            ("self.block_size", "block_size"),
            ("num_drafts", "speculative_tokens"),
            ("model_runner.config.kv_cache_dtype == 'fp4'", "packed"),
            ("model_runner.config.index_cache_dtype == 'fp4'", "index_fp4"),
        ):
            text = text.replace(before, after)
        return text

    return (
        [norm(a) for a in call.args],
        {k.arg: norm(k.value) for k in call.keywords},
    )


def test_both_sides_build_the_same_geometry():
    core_args, core_kwargs = _shape(_geometry_call(CORE.read_text(), "__init__"))
    plugin_args, plugin_kwargs = _shape(
        _geometry_call(PLUGIN.read_text(), "_build_v41_pool_geometry")
    )

    assert core_args == plugin_args, (
        "the positional geometry arguments diverged; core and the plugin would "
        "size the same pool differently"
    )
    assert set(core_kwargs) == set(plugin_kwargs), (
        f"keyword sets diverged: core only {sorted(set(core_kwargs) - set(plugin_kwargs))}, "
        f"plugin only {sorted(set(plugin_kwargs) - set(core_kwargs))}"
    )
    for name in sorted(core_kwargs):
        assert core_kwargs[name] == plugin_kwargs[name], (
            f"`{name}` differs: core {core_kwargs[name]!r}, "
            f"plugin {plugin_kwargs[name]!r}"
        )


def test_the_comparison_actually_found_both_sides():
    """A query that matched nothing would pass every assertion above while
    comparing two empty shapes."""
    core_args, core_kwargs = _shape(_geometry_call(CORE.read_text(), "__init__"))
    assert len(core_args) >= 7, core_args
    assert "speculative_tokens" in core_kwargs
    assert "index_fp4" in core_kwargs


def _plugin_geometry_call_site() -> dict[str, str]:
    """The keywords `v41_proxy_geometry` hands `_build_v41_pool_geometry`."""
    tree = ast.parse(PLUGIN.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "_build_v41_pool_geometry"
    ]
    assert len(calls) == 1, f"expected one call site, found {len(calls)}"
    return {k.arg: ast.unparse(k.value) for k in calls[0].keywords}


def test_the_hardcoded_index_plane_still_matches_what_the_config_resolves():
    """The one geometry field the parity test above cannot see.

    That test compares the two `V41PoolGeometry(...)` constructions, and both
    spell this field as a name -- core reads `index_cache_dtype == 'fp4'`, the
    plugin passes its own `index_fp4` parameter through, and the normalization
    makes those equal. What differs is the *value supplied at the call site*,
    which the plugin writes as a literal `False`.

    That literal is correct only while V4.1's `index_cache_dtype` resolves to
    fp8, which `atom/config.py` does unconditionally for this model type and
    which the plugin config translation never overrides. Both halves are
    asserted here, so whoever wires an index-cache dtype through to plugin
    mode -- the way `kv_transfer_config` was wired through -- gets a red test
    pointing at the literal instead of a pool sized for the wrong plane.
    """
    assert _plugin_geometry_call_site()["index_fp4"] == "False"

    config = ast.parse((ROOT / "atom/config.py").read_text())
    # Only this branch's own body. Walking the whole `If` would sweep in its
    # `else`, which is where every other model takes the KV dtype -- a sibling,
    # not something V4.1 can reach.
    resolved = [
        ast.unparse(node.value)
        for branch in ast.walk(config)
        if isinstance(branch, ast.If)
        and "deepseek_v41_text" in ast.unparse(branch.test)
        for statement in branch.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Assign)
        and [ast.unparse(t) for t in node.targets] == ["self.index_cache_dtype"]
    ]
    assert resolved == ["'fp8'"], (
        "V4.1's index cache no longer resolves to fp8 unconditionally, so the "
        "plugin's literal `index_fp4=False` may now size the pool for the "
        f"wrong index plane; found {resolved}"
    )
