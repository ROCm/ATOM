# SPDX-License-Identifier: MIT
"""Every linear-attention builder must wire BOTH halves of the state cache.

The state cache is not a builder-local optimization: the engine (BlockManager
/ StateCachePool) clamps hits, publishes checkpoints and sets
`seq.has_recurrent_state` on its own, with no callback into the attention
builder. So a builder that omits the two calls does NOT get "no state cache" —
it gets an engine that believes the cache is working while the kernel resumes
the recurrence from a runtime slot nothing ever wrote. Slots are recycled
without scrubbing, so that is another sequence's state: silently wrong output
on a prefix hit, at full cache speed.

`KimiAiterMLAGDNMetadataBuilder` shipped in exactly that state. It composes
`GDNStateMixin` with the MLA builder rather than inheriting
`GDNAttentionMetadataBuilder`, so it inherited the two methods but not the
`prepare_prefill` that calls them. A test asserting the methods *resolve*
passes on the broken code — only the call site distinguishes them.

Source-parsing rather than importing, because `tests/conftest.py` stubs
`atom.config` (no `SSM_STATE_KERNEL_CHUNK`), so importing the attention
modules under pytest fails. Same style as `test_ssm_state_cache_sizing.py`:
the shipped source is the thing under test.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Both halves must be present. Loading without saving resumes from an unwritten
# slot; saving without loading publishes checkpoints nothing can use.
REQUIRED_CALLS = ("apply_state_cache_loads", "_checkpoint_targets")

# (module path, builder class). Every concrete builder for a model whose layers
# carry per-request recurrent state. Add a linear-attention builder here when
# you add one — that is the point of the file.
BUILDERS = [
    ("atom/model_ops/attentions/gdn_attn.py", "GDNAttentionMetadataBuilder"),
    (
        "atom/model_ops/attentions/kimi_mla_gdn_attn.py",
        "KimiAiterMLAGDNMetadataBuilder",
    ),
    (
        "atom/model_ops/attentions/kimi_mla_gdn_attn.py",
        "KimiTritonMLAGDNMetadataBuilder",
    ),
]


def _module(rel: str) -> ast.Module:
    return ast.parse((ROOT / rel).read_text())


def _classdef(rel: str, name: str) -> ast.ClassDef:
    for node in ast.walk(_module(rel)):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in {rel}")


def _bases(rel: str, name: str) -> list[str]:
    return [ast.unparse(b) for b in _classdef(rel, name).bases]


def _prepare_prefill_source(rel: str, name: str) -> str:
    """Source of the `prepare_prefill` this builder actually runs.

    Walks the base chain within the same file, so a builder that inherits the
    method (rather than overriding it) is checked against the definition it
    really uses.
    """
    seen: set[tuple[str, str]] = set()
    queue = [(rel, name)]
    while queue:
        cur_rel, cur_name = queue.pop(0)
        if (cur_rel, cur_name) in seen:
            continue
        seen.add((cur_rel, cur_name))
        try:
            cls = _classdef(cur_rel, cur_name)
        except AssertionError:
            continue  # base lives in another module; not a state-cache owner
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "prepare_prefill":
                return ast.unparse(item)
        for base in cls.bases:
            queue.append((cur_rel, ast.unparse(base).split("[")[0].strip()))
        # Kimi's builders are thin `pass` bodies over a shared mixin in the
        # same file; follow that too.
        if cur_rel != "atom/model_ops/attentions/gdn_attn.py":
            queue.append(("atom/model_ops/attentions/gdn_attn.py", "GDNStateMixin"))
    raise AssertionError(f"no prepare_prefill reachable from {name} in {rel}")


@pytest.mark.parametrize("rel,name", BUILDERS)
@pytest.mark.parametrize("call", REQUIRED_CALLS)
def test_prepare_prefill_wires_the_state_cache(rel, name, call):
    """The call must appear in the `prepare_prefill` this builder runs."""
    src = _prepare_prefill_source(rel, name)
    assert f"{call}(" in src, (
        f"{name}.prepare_prefill never calls {call}(). The engine will still "
        "clamp hits and set has_recurrent_state, so the recurrence resumes "
        "from an unwritten slot — silently wrong output on a prefix hit, not "
        "a missed optimization."
    )


@pytest.mark.parametrize("rel,name", BUILDERS)
def test_checkpoint_targets_feed_chunk_offsets(rel, name):
    """`ssm_checkpoints` is useless without the matching `ssm_chunk_offsets`.

    `write_state_checkpoints` maps a target's offset to a chunk index through
    `chunk_offsets[row]`; passing None there indexes from the batch start and
    captures one sequence's state into another's checkpoint.
    """
    src = _prepare_prefill_source(rel, name)
    assert "ssm_chunk_offsets" in src, (
        f"{name}.prepare_prefill sets ssm_checkpoints but not "
        "ssm_chunk_offsets; the write kernel would index chunks from the "
        "batch start rather than the sequence start."
    )


@pytest.mark.parametrize("rel,name", BUILDERS)
def test_builder_uses_the_state_mixin(rel, name):
    """A builder in this list must actually own state-cache machinery.

    Guards the case where someone lists a builder here that does not use
    `GDNStateMixin`, which would make the assertions above vacuous.
    """
    chain, queue = set(), [(rel, name)]
    while queue:
        cur_rel, cur_name = queue.pop(0)
        if cur_name in chain:
            continue
        chain.add(cur_name)
        try:
            for base in _bases(cur_rel, cur_name):
                queue.append((cur_rel, base.split("[")[0].strip()))
        except AssertionError:
            continue
    assert "GDNStateMixin" in chain, (
        f"{name} does not inherit GDNStateMixin, so the state-cache "
        "assertions in this file do not apply to it."
    )
