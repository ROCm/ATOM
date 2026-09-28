# SPDX-License-Identifier: MIT
"""Which row of a pool a layer binds to: a property of the module, not of a walk.

A module's row was once recovered from `layer_id` -- the bind walk's running
count of every layer it registered -- by knowing the model's shape: which
layers of a Qwen3-Next hybrid are full attention (`layer_id // interval`), that
a linear-attention layer takes the complement of that, and where a draft's
stack starts (`num_full_attn + (layer_id - mtp_start)`). Three formulas over
one counter, each carrying an assumption about a model it does not name.

It is `pool_rows[kind][module]` now, assigned by the builder's own walk over
`_pooled_modules`. Pinned here: that the assignment agrees with those formulas
on the shapes they were written for, that it is the same walk sizing counts, and
that a module the walk never saw gets an error rather than the next free row --
which is what a counter following someone else's walk would have given it.

`pool_layout/pool_rows.py` imports nothing at all, so this could import it --
but reading it statically is what survives the module ever gaining an import,
which is the breakage the arrangement exists to prevent. It reads nothing but
`self._pooled_models()`, which the fixtures below supply.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

MIXIN_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "atom/model_ops/attentions/pool_layout/pool_rows.py"
)


def _mixin():
    """`PoolRowsMixin` lifted off the module, reading no imports of its own."""
    tree = ast.parse(MIXIN_SOURCE.read_text())
    wanted = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PoolRowsMixin"
    )
    namespace: dict = {}
    exec(  # noqa: S102 - the source is this repository's own, read above
        compile(ast.Module(body=[wanted], type_ignores=[]), str(MIXIN_SOURCE), "exec"),
        namespace,
    )
    return namespace["PoolRowsMixin"]


# Qwen3-Next's shape: full attention closes every group of `INTERVAL` layers,
# and the MTP draft's layers are all full attention, appended after them.
INTERVAL = 4
FULL_LAYERS = 3
MTP_LAYERS = 2
MTP_START = INTERVAL * FULL_LAYERS


class _Layer:
    """A module the walk can tell apart, and use as a dict key."""

    def __init__(self, kind: str):
        self.kind = kind


class _Model:
    """The tree `model.modules()` walks, in bind order."""

    def __init__(self, layers: list[_Layer]):
        self._layers = layers

    def modules(self):
        return iter(self._layers)


def _target_layers() -> list[_Layer]:
    return [
        _Layer("kv" if (i + 1) % INTERVAL == 0 else "gdn")
        for i in range(INTERVAL * FULL_LAYERS)
    ]


def _draft_layers() -> list[_Layer]:
    return [_Layer("kv") for _ in range(MTP_LAYERS)]


def _builder(target: list[_Layer], draft: list[_Layer] | None = None):
    """The mixin over a fake model tree, plus the one hook it needs."""
    models = [_Model(target)] + ([_Model(draft)] if draft is not None else [])

    class Builder(_mixin()):
        def _pooled_models(self):
            return models

        def _module_kinds(self, module):
            return (module.kind,)

    builder = Builder()
    builder.invalidate_pool_rows()
    return builder


def _row_by_formula(kind: str, layer_id: int) -> int:
    """What the deleted arithmetic answered, kept here as the oracle."""
    if kind == "gdn":
        return (layer_id // INTERVAL) * (INTERVAL - 1) + (layer_id % INTERVAL)
    if layer_id < MTP_START:
        return layer_id // INTERVAL
    return FULL_LAYERS + (layer_id - MTP_START)


def test_a_module_gets_the_row_the_formula_gave_its_layer():
    target, draft = _target_layers(), _draft_layers()
    builder = _builder(target, draft)

    rows = builder.pool_rows

    for layer_id, layer in enumerate(target + draft):
        assert rows[layer.kind][layer] == _row_by_formula(layer.kind, layer_id)


def test_the_rows_of_one_kind_are_the_whole_range():
    """Dense from zero and each row handed out once -- what a pool sized for
    exactly this many layers needs, and what an off-by-one walk would break."""
    builder = _builder(_target_layers(), _draft_layers())

    for kind, count in (("kv", FULL_LAYERS + MTP_LAYERS), ("gdn", FULL_LAYERS * 3)):
        assert sorted(builder.pool_rows[kind].values()) == list(range(count))


def test_sizing_counts_what_binding_hands_out():
    """`row_counts` is what `sub_pool_specs` charges for. Reading it off the
    same map is the whole point: a pool cannot be sized for one number of rows
    and bound at another."""
    builder = _builder(_target_layers(), _draft_layers())

    assert builder.row_counts() == {
        "kv": FULL_LAYERS + MTP_LAYERS,
        "gdn": FULL_LAYERS * 3,
    }


def test_a_module_the_walk_never_saw_has_no_row():
    """The failure a counter could not report.

    The runner's bind walk offers a draft's modules to the target builder
    whenever a sibling builder declines them. A counter would have handed one
    the next row -- past the end of a pool sized without it -- and said nothing.
    """
    builder = _builder(_target_layers())
    stranger = _Layer("kv")

    with pytest.raises(KeyError):
        builder.pool_rows["kv"][stranger]


def test_kinds_are_numbered_apart():
    """An MHA row, a linear-attention slot and an indexer's compact row are
    three ranges over the same walk."""
    builder = _builder(_target_layers())

    kv = builder.pool_rows["kv"]
    gdn = builder.pool_rows["gdn"]

    assert min(kv.values()) == min(gdn.values()) == 0


def test_a_re_bind_re_reads_the_tree():
    """The P/D import binds a pool again, and so does a rollout wake. Neither
    promises the same modules, so the assignment is dropped rather than kept."""
    target = _target_layers()
    builder = _builder(target)
    first = dict(builder.pool_rows["kv"])

    target.append(_Layer("kv"))
    assert dict(builder.pool_rows["kv"]) == first  # held until told otherwise

    builder.invalidate_pool_rows()

    assert len(builder.pool_rows["kv"]) == len(first) + 1
