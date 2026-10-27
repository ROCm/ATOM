# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every scheduler backend must explicitly implement block retention."""

import pytest

from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase


class _LegacyScheduler(KVConnectorSchedulerBase):
    is_producer = True

    def get_num_new_matched_tokens(self, seq):
        return 0, False

    def build_connector_meta(self):
        return None

    def update_state_after_alloc(self, seq):
        pass

    def request_finished(self, seq):
        pass


def _no_retention(self, seq):
    return False


def _no_op(self, value):
    pass


_RETENTION_METHODS = {
    "should_defer_free": _no_retention,
    "send_finished": _no_op,
    "source_blocks_released": _no_op,
}


@pytest.mark.parametrize("missing", _RETENTION_METHODS)
def test_missing_retention_hook_prevents_instantiation(missing):
    incomplete = type(
        "IncompleteScheduler",
        (_LegacyScheduler,),
        {
            name: method
            for name, method in _RETENTION_METHODS.items()
            if name != missing
        },
    )

    with pytest.raises(TypeError, match=missing):
        incomplete()


def test_explicit_no_ops_satisfy_the_retention_contract():
    complete = type("CompleteScheduler", (_LegacyScheduler,), _RETENTION_METHODS)
    scheduler = complete()

    assert scheduler.should_defer_free(None) is False
    assert scheduler.send_finished(7) is None
    assert scheduler.source_blocks_released(None) is None
