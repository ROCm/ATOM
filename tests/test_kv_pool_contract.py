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


@pytest.mark.parametrize("path, cls, fn", ALL_HOOKS)
@pytest.mark.parametrize("arg", ["blocks", "buf"])
def test_neither_has_a_default(path, cls, fn, arg):
    """A default would let a caller omit it and get a pool built at someone
    else's number, or in a buffer nobody else can find -- which is the failure
    these arguments exist to remove."""
    del path, cls
    defaults = dict(zip((a.arg for a in fn.args.kwonlyargs), fn.args.kw_defaults))
    assert defaults.get(arg, "absent") is None
