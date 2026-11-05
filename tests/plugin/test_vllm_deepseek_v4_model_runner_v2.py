# SPDX-License-Identifier: MIT
"""What ATOM's DeepSeek-V4 refuses under vLLM's Model Runner V2.

Prefix caching and the MTP draft still hook V1's runner alone, so V2 has to
refuse them at model construction rather than serve them wrong.
"""

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("vllm", reason="the V4 bridge imports vLLM")

from atom.plugin.vllm import deepseek_v4_bridge as bridge
from atom.plugin.vllm import deepseek_v4_state_checkpoint as checkpoints


@pytest.fixture
def prefix_caching(monkeypatch):
    state = {"on": False}
    monkeypatch.setattr(
        checkpoints, "deepseek_v4_state_checkpoints_on", lambda cfg: state["on"]
    )
    return state


def _config(*, v2, spec=None):
    return NS(use_v2_model_runner=v2, speculative_config=spec)


def test_v1_is_left_alone(prefix_caching):
    prefix_caching["on"] = True
    bridge.check_deepseek_v4_model_runner(_config(v2=False, spec=object()), is_mtp=True)


def test_v2_serves_the_plain_target(prefix_caching):
    bridge.check_deepseek_v4_model_runner(_config(v2=True), is_mtp=False)


def test_v2_refuses_prefix_caching(prefix_caching):
    prefix_caching["on"] = True
    with pytest.raises(ValueError, match="prefix caching"):
        bridge.check_deepseek_v4_model_runner(_config(v2=True), is_mtp=False)


@pytest.mark.parametrize("is_mtp", [False, True])
def test_v2_refuses_mtp_on_the_target_and_the_draft(prefix_caching, is_mtp):
    config = _config(v2=True, spec=None if is_mtp else object())
    with pytest.raises(ValueError, match="MTP"):
        bridge.check_deepseek_v4_model_runner(config, is_mtp=is_mtp)
