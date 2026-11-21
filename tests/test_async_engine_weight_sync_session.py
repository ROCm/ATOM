from contextlib import contextmanager

import pytest

from atom.model_engine.collective_rpc import DISCARD_WEIGHT_SYNC_CMD
from atom.rollout.async_engine import AsyncLLMEngine


class _CoreManager:
    def __init__(self):
        self.commands = []

    @contextmanager
    def weight_sync_session(self):
        yield

    def broadcast_utility_command_sync(self, command, **kwargs):
        self.commands.append((command, kwargs))
        return []


def test_async_weight_sync_session_discards_on_sender_exception():
    engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
    engine.core_mgr = _CoreManager()

    with (
        pytest.raises(RuntimeError, match="sender failed"),
        engine.weight_sync_session(),
    ):
        raise RuntimeError("sender failed")

    assert engine.core_mgr.commands == [
        (
            DISCARD_WEIGHT_SYNC_CMD,
            {"failed_cmd": "weight_sync_session: RuntimeError"},
        )
    ]
