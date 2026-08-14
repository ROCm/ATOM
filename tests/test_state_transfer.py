# SPDX-License-Identifier: MIT

"""Pure control-plane tests for backend state-transfer capabilities."""

import pickle

import pytest

from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_engine.state_pool import (
    StateMaintenanceOps,
    StateRuntime,
    StateTransfer,
)

COPY_SPEC = PagedStateCheckpointSpec(10, 25, "layout-v2")


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


def test_state_runtime_accepts_copy_fork_and_none_contracts():
    none_runtime = StateRuntime()
    fork_runtime = StateRuntime(transfer=StateTransfer.fork(7))
    copy_runtime = StateRuntime(
        transfer=StateTransfer.copy(COPY_SPEC.layout_id),
        checkpoint_spec=COPY_SPEC,
    )

    assert none_runtime.transfer == StateTransfer.none()
    assert none_runtime.checkpoint_spec is None
    assert fork_runtime.transfer.forks and fork_runtime.checkpoint_spec is None
    assert copy_runtime.transfer.copies
    assert copy_runtime.checkpoint_spec is COPY_SPEC


def test_state_runtime_rejects_every_transfer_spec_mismatch():
    invalid = (
        (StateTransfer.copy(COPY_SPEC.layout_id), None),
        (StateTransfer.copy("different-layout"), COPY_SPEC),
        (StateTransfer.fork(1), COPY_SPEC),
        (StateTransfer.none(), COPY_SPEC),
    )

    for transfer, checkpoint_spec in invalid:
        with pytest.raises(ValueError):
            StateRuntime(transfer=transfer, checkpoint_spec=checkpoint_spec)


def test_state_runtime_wire_round_trip_revalidates_the_complete_contract():
    runtimes = (
        StateRuntime(),
        StateRuntime(transfer=StateTransfer.fork(7)),
        StateRuntime(
            transfer=StateTransfer.copy(COPY_SPEC.layout_id),
            checkpoint_spec=COPY_SPEC,
        ),
    )

    for runtime in runtimes:
        wire = pickle.loads(pickle.dumps(runtime.to_wire()))
        assert StateRuntime.from_wire(wire) == runtime

    invalid_wire = runtimes[-1].to_wire()
    invalid_wire["checkpoint_spec"] = None
    with pytest.raises(ValueError, match="requires a PAGE checkpoint spec"):
        StateRuntime.from_wire(invalid_wire)

    with pytest.raises(ValueError, match="fields"):
        StateRuntime.from_wire({"transfer": StateTransfer.none().to_wire()})


def test_state_maintenance_bundle_is_typed_and_immutable():
    empty = StateMaintenanceOps()
    populated = StateMaintenanceOps(relocations=((1, 2), (3, 4)))

    assert empty.empty
    assert not populated.empty
    assert populated.relocations == ((1, 2), (3, 4))
    with pytest.raises(AttributeError):
        populated.relocations = ()
