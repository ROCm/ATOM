# SPDX-License-Identifier: MIT

"""Pure control-plane tests for backend state-transfer capabilities."""

import pickle

import pytest

from atom.model_engine.state_pool import StateMaintenanceOps, StateTransfer


def test_wire_round_trip_keeps_the_complete_capability():
    transfers = (
        StateTransfer.none(),
        StateTransfer.fork(7),
        StateTransfer.copy("layout-v2"),
    )

    for transfer in transfers:
        wire = pickle.loads(pickle.dumps(transfer.to_wire()))
        assert StateTransfer.from_wire(wire) == transfer
    assert transfers[-1].to_wire()["paged_layout_id"] == "layout-v2"


def test_invalid_kind_token_and_layout_combinations_are_rejected():
    invalid = (
        ("copy", 0, None),
        ("copy", 1, "layout-v1"),
        ("fork", 0, None),
        ("fork", 1, "layout-v1"),
        ("none", 1, None),
        ("none", 0, "layout-v1"),
        ("unknown", 0, None),
    )

    for args in invalid:
        with pytest.raises(ValueError):
            StateTransfer(*args)


def test_copy_factory_requires_a_non_empty_layout():
    with pytest.raises(ValueError, match="layout"):
        StateTransfer.copy("")


def test_wire_shape_is_exact():
    with pytest.raises(ValueError, match="fields"):
        StateTransfer.from_wire(
            {
                "kind": "copy",
                "fork_tokens": 0,
                "paged_layout_id": "layout-v1",
                "other": 1,
            }
        )


def test_state_maintenance_bundle_is_typed_and_immutable():
    empty = StateMaintenanceOps()
    populated = StateMaintenanceOps(relocations=((1, 2), (3, 4)))

    assert empty.empty
    assert not populated.empty
    assert populated.relocations == ((1, 2), (3, 4))
    with pytest.raises(AttributeError):
        populated.relocations = ()
