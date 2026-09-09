# SPDX-License-Identifier: MIT
"""How a KV pool learns the count it is built at.

Sizing runs in every runner subprocess and they disagree by a few blocks --
each measures its own free memory -- so EngineCore takes one answer and
broadcasts it. A builder that reads a count off the runner instead can build
its pool at one number while something sized off another is built at a second;
that shipped once, as an indexer cache sized against this rank's own estimate
while the pool was built at the broadcast one, and only a real server surfaced
it. The count arrives as an argument so there is nothing else to read.

Read statically, by path, for the reason `test_layout_packages` gives: these
modules import aiter, so a test that imported them to inspect them would not
run on CI, which is where the regression would land.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ATTENTIONS = (
    pathlib.Path(__file__).resolve().parent.parent / "atom/model_ops/attentions"
)
# The draft's builder is not an attention backend -- it holds a pool without
# knowing what flavor fills it -- but it answers the same runner hook, so the
# same contract binds it.
SOURCES = sorted(ATTENTIONS.glob("*.py")) + [
    pathlib.Path(__file__).resolve().parent.parent / "atom/spec_decode/draft_kv.py"
]

HOOK = "allocate_kv_cache_tensors"
RUNNER = (
    pathlib.Path(__file__).resolve().parent.parent / "atom/model_engine/model_runner.py"
)
# The one place a pool is backed, and the two that reach it: one allocates the
# buffer, the other receives it over IPC. Named with their class because
# `RapidServeModelRunner` holds the second.
BACKER = "_back_paged_pools"
BACKING_PATHS = [
    ("ModelRunner", "allocate_kv_cache"),
    ("RapidServeModelRunner", "_bind_kv_cache_to_modules"),
]


def _hooks(path: pathlib.Path) -> list[tuple[str, ast.FunctionDef]]:
    """Every `(class, def)` pair defining the allocate hook in one file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        (cls.name, fn)
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef)
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == HOOK
    ]


ALL_HOOKS = [
    pytest.param(path, cls, fn, id=f"{path.stem}.{cls}")
    for path in SOURCES
    for cls, fn in _hooks(path)
]


def test_every_flavor_that_owns_a_pool_is_covered():
    """A count of the implementations, so the per-hook checks below cannot
    quietly stop covering one. The base declaration plus MHA, MLA, Kimi's
    MLA+GDN, V4, and the draft's -- GDN-hybrid has none of its own: counting
    the modules it owns leaves the linear-attention layers out by itself."""
    assert len(ALL_HOOKS) == 6


@pytest.mark.parametrize("path, cls, fn", ALL_HOOKS)
@pytest.mark.parametrize("arg", ["blocks", "buf"])
def test_the_pool_is_told_where_to_build_itself(path, cls, fn, arg):
    """Keyword-only, so a caller cannot pass one positionally into the slot
    another implementation gave a different meaning, and so no implementation
    can quietly stop taking it and go back to reading the runner.

    `buf` is the region of the runner's one paged allocation this builder's
    pool lives in. Same argument as `blocks` in kind: a builder that allocated
    its own buffer instead would leave the runner holding one pool while the
    kernels read another.
    """
    del path, cls
    assert arg in {a.arg for a in fn.args.kwonlyargs}


def _publishes_the_hooks_result(fn: ast.FunctionDef) -> bool:
    """Whether `fn` loops over `<...>.allocate_kv_cache_tensors(...).items()`
    and `setattr`s each pair."""
    for loop in ast.walk(fn):
        if not isinstance(loop, ast.For):
            continue
        items = loop.iter
        if not (
            isinstance(items, ast.Call)
            and isinstance(items.func, ast.Attribute)
            and items.func.attr == "items"
            and isinstance(items.func.value, ast.Call)
            and isinstance(items.func.value.func, ast.Attribute)
            and items.func.value.func.attr == HOOK
        ):
            continue
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            for node in ast.walk(loop)
        ):
            return True
    return False


def _method(owner: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(RUNNER.read_text(), filename=str(RUNNER))
    found = [
        fn
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == owner
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == name
    ]
    assert len(found) == 1, f"{owner}.{name} is defined {len(found)} times"
    return found[0]


def test_the_one_backer_publishes_what_the_hook_returns():
    """What the hook returns is not decoration: the aligned indexer dimension,
    the compact index-cache layer maps and GLM-5.3's k-pool tail reach their
    readers as runner attributes, and `build_kv_cache_tensor` dereferences them
    right after. The decode side of a P/D pair used to back its pool through a
    hook of its own that returned nothing, and bound against attributes nobody
    set -- on a decode rank only, which no test here reaches."""
    assert _publishes_the_hooks_result(_method("ModelRunner", BACKER))


@pytest.mark.parametrize("owner, method", BACKING_PATHS)
def test_both_backing_paths_go_through_it(owner, method):
    """And neither reaches the hook on its own, which is what stops the two
    from drifting again."""
    fn = _method(owner, method)
    called = {
        node.func.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert BACKER in called
    assert HOOK not in called


@pytest.mark.parametrize("path, cls, fn", ALL_HOOKS)
@pytest.mark.parametrize("arg", ["blocks", "buf"])
def test_neither_has_a_default(path, cls, fn, arg):
    """A default would let a caller omit it and get a pool built at someone
    else's number, or in a buffer nobody else can find -- which is the failure
    these arguments exist to remove."""
    del path, cls
    defaults = dict(zip((a.arg for a in fn.args.kwonlyargs), fn.args.kw_defaults))
    assert defaults.get(arg, "absent") is None


# ── Who builds a draft a pool of its own ───────────────────────────────────
#
# Two members that have to agree: the flag `draft_kv_builder` reads, and the
# factory it calls once the flag says yes. Getting the flag wrong is the silent
# half -- a backend that builds pools but forgets to say so leaves its drafts
# sharing the target's rows, which is a real layout for MLA and a wrong one
# here. The other half is loud: the base factory raises.

POOL_FACTORY = "make_kv_pool"
DRAFT_FLAG = "DRAFT_OWNS_KV_POOL"
# The declaration, which defines the factory in order to refuse it.
FACTORY_DECLARATION = "AttentionBackend"


def _classes_defining_the_factory() -> list[tuple[str, ast.ClassDef]]:
    return [
        (path.stem, cls)
        for path in SOURCES
        for cls in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(cls, ast.ClassDef)
        and any(
            isinstance(fn, ast.FunctionDef) and fn.name == POOL_FACTORY
            for fn in cls.body
        )
    ]


def _sets_the_flag_true(cls: ast.ClassDef) -> bool:
    """Whether this class body assigns `DRAFT_OWNS_KV_POOL = True`, annotated
    or not. Its own body: an inheriting backend gets the answer with the
    factory, and the two cannot part."""
    for stmt in cls.body:
        targets = (
            [stmt.target]
            if isinstance(stmt, ast.AnnAssign)
            else getattr(stmt, "targets", [])
        )
        if any(isinstance(t, ast.Name) and t.id == DRAFT_FLAG for t in targets):
            return stmt.value is not None and getattr(stmt.value, "value", None) is True
    return False


def test_the_factory_is_declared_where_the_flag_is():
    """Both on `AttentionBackend`, so a backend inherits a coherent pair
    (no pool, and a factory that says so) rather than half of one."""
    declarations = _classes_defining_the_factory()

    assert FACTORY_DECLARATION in [cls.name for _, cls in declarations]


@pytest.mark.parametrize(
    "cls",
    [
        pytest.param(cls, id=f"{stem}.{cls.name}")
        for stem, cls in _classes_defining_the_factory()
        if cls.name != FACTORY_DECLARATION
    ],
)
def test_a_backend_that_can_build_a_draft_pool_says_so(cls):
    """The silent direction. Without the flag `draft_kv_builder` returns None
    and the draft binds into the target's rows -- no error, no log, and a
    draft reading K and V that are not its own."""
    assert _sets_the_flag_true(cls)


def test_a_geometry_row_space_names_itself_readably():
    """`KvGeometry` is the one row space that is not a plain string, so it is
    the one that needs a `__str__`: the default repr put
    `KvGeometry(num_kv_heads=2, head_dim=256)` in a startup log beside
    `linear_state`, and into the `semantic_role` a transfer region carries into
    its error messages. Pinned by value because a shape is what a reader
    recognizes."""
    from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry

    assert str(KvGeometry(2, 256)) == "h2d256"
    assert f"mha.{KvGeometry(8, 128)}.k.layer_0" == "mha.h8d128.k.layer_0"
