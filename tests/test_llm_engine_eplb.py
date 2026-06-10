# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock

from atom.model_engine.llm_engine import LLMEngine


def test_offline_eplb_rebalance_sends_utility_command():
    engine = LLMEngine.__new__(LLMEngine)
    engine.core_mgr = MagicMock()

    engine.offline_eplb_rebalance()

    engine.core_mgr.send_utility_command.assert_called_once_with(
        "offline_eplb_rebalance"
    )

