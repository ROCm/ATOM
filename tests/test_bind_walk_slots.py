# SPDX-License-Identifier: MIT
"""Which row of a pool a layer binds to, counted rather than derived.

A module's row used to be recovered from `layer_id` -- the bind walk's running
count of every layer it registered -- by knowing the model's shape: which
layers of a Qwen3-Next hybrid are full attention (`layer_id // interval`), that
a linear-attention layer takes the complement of that (`(layer_id // interval)
* (interval - 1) + layer_id % interval`), and where a draft's stack starts
(`num_full_attn + (layer_id - mtp_start)`). Three formulas over one counter,
each carrying an assumption about a model it does not name.

They are `take_slot(kind)` now: a module's row is its position among the
modules of its kind, which is the order the walk visits them in. What is pinned
here is that the two agree on the shapes the formulas were written for, and
that a re-bind starts over -- the formulas were stateless and a counter is not,
which is the one thing the change could break.

`take_slot` is `AttentionMetadataBuilder`'s, and importing that pulls in aiter,
so the tests drive the method through a bare subclass of nothing: the mixin
under test reads no state but its own.
"""

from __future__ import annotations

import ast
import pathlib

BACKENDS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "atom/model_ops/attentions/backends.py"
)


def _slot_holder():
    """`take_slot` / `reset_slots` lifted off the class, with no aiter import.

    Compiling the two methods out of the source rather than importing the
    module they live in: the file imports the attention stack, which CI has no
    build of, and a module-level skip would mean these never run there.
    """
    tree = ast.parse(BACKENDS.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AttentionMetadataBuilder"
    )
    wanted = [
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("take_slot", "reset_slots")
    ]
    assert len(wanted) == 2, "the slot methods moved; this test has to follow"
    namespace: dict = {}
    exec(  # noqa: S102 - the source is this repository's own, read above
        compile(ast.Module(body=wanted, type_ignores=[]), str(BACKENDS), "exec"),
        namespace,
    )
    holder = type("Builder", (), namespace)
    return holder()


# Qwen3-Next's shape: full attention closes every group of `INTERVAL` layers,
# and the MTP draft's layers are all full attention, appended after them.
INTERVAL = 4
FULL_LAYERS = 3
MTP_LAYERS = 2
MTP_START = INTERVAL * FULL_LAYERS


def _layer_kinds() -> list[str]:
    """The bind walk's view of that model: one entry per registered layer."""
    kinds = [
        "kv" if (i + 1) % INTERVAL == 0 else "gdn"
        for i in range(INTERVAL * FULL_LAYERS)
    ]
    return kinds + ["kv"] * MTP_LAYERS


def _row_by_formula(kind: str, layer_id: int) -> int:
    """What the deleted arithmetic answered, kept here as the oracle."""
    if kind == "gdn":
        return (layer_id // INTERVAL) * (INTERVAL - 1) + (layer_id % INTERVAL)
    if layer_id < MTP_START:
        return layer_id // INTERVAL
    return FULL_LAYERS + (layer_id - MTP_START)


def _walk(builder) -> list[tuple[str, int]]:
    return [(kind, builder.take_slot(kind)) for kind in _layer_kinds()]


def test_the_counter_gives_every_layer_the_row_the_formula_did():
    builder = _slot_holder()
    builder.reset_slots()

    rows = _walk(builder)

    assert rows == [
        (kind, _row_by_formula(kind, layer_id))
        for layer_id, kind in enumerate(_layer_kinds())
    ]


def test_the_rows_of_one_kind_are_the_whole_range():
    """Dense from zero and each row handed out once -- what a pool sized for
    exactly this many layers needs, and what an off-by-one walk would break."""
    builder = _slot_holder()
    builder.reset_slots()

    rows = _walk(builder)

    for kind, count in (("kv", FULL_LAYERS + MTP_LAYERS), ("gdn", FULL_LAYERS * 3)):
        assert sorted(r for k, r in rows if k == kind) == list(range(count))


def test_a_second_walk_starts_over():
    """The P/D import binds a pool again, and so does a rollout wake. A counter
    that survived either would hand the next walk's first layer a row past the
    end of the pool -- the one failure a stateless formula could not have."""
    builder = _slot_holder()
    builder.reset_slots()
    first = _walk(builder)

    builder.reset_slots()

    assert _walk(builder) == first


def test_kinds_are_counted_apart():
    """One counter per kind: an MHA row, a linear-attention slot and an
    indexer's compact row are three ranges over the same walk."""
    builder = _slot_holder()
    builder.reset_slots()

    assert [builder.take_slot("kv"), builder.take_slot("index")] == [0, 0]
    assert [builder.take_slot("kv"), builder.take_slot("index")] == [1, 1]
