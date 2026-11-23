from contextlib import contextmanager

import pytest

from atom.model_engine.collective_rpc import DISCARD_WEIGHT_SYNC_CMD
from atom.rollout.async_engine import AsyncLLMEngine
from atom.rollout.weight_sync import _abort_weight_sync


class _CoreManager:
    def __init__(self, discard_error=None):
        self.commands = []
        self.discard_error = discard_error
        self.abort_claimed = False

    @contextmanager
    def weight_sync_session(self):
        yield

    def broadcast_utility_command_sync(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if self.discard_error is not None:
            raise self.discard_error
        return []

    def claim_weight_sync_abort(self):
        if self.abort_claimed:
            return False
        self.abort_claimed = True
        return True


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
            {
                "timeout": 30.0,
                "failed_cmd": "weight_sync_session: RuntimeError",
                "error": (
                    "weight_sync_session: RuntimeError sender abandoned the sync "
                    "before its final bucket"
                ),
            },
        )
    ]


def test_async_weight_sync_session_preserves_sender_error_when_discard_fails():
    engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
    engine.core_mgr = _CoreManager(discard_error=TimeoutError("discard timed out"))

    with (
        pytest.raises(RuntimeError, match="sender failed"),
        engine.weight_sync_session(),
    ):
        raise RuntimeError("sender failed")


def test_nested_weight_sync_abort_is_sent_only_once():
    engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
    engine.core_mgr = _CoreManager()

    with (
        pytest.raises(RuntimeError, match="sender failed"),
        engine.weight_sync_session(),
    ):
        _abort_weight_sync(engine.core_mgr, "update_weights_shm")
        raise RuntimeError("sender failed")

    assert len(engine.core_mgr.commands) == 1
