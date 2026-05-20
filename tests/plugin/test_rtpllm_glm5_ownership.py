"""Ownership contract tests for GLM5 rtp-llm M0."""

from atom.plugin.rtpllm.attention_backend.rtp_mla_metadata import (
    GLM5_RTP_BRIDGE_MODE,
    GLM5_RTP_BRIDGE_MODE_M0_DENSE,
    GLM5_RTP_OWNERSHIP,
)


def test_glm5_bridge_mode_starts_in_m0_dense():
    assert GLM5_RTP_BRIDGE_MODE == GLM5_RTP_BRIDGE_MODE_M0_DENSE


def test_glm5_ownership_unique_and_separates_rope_paths():
    required = {
        "main_q_norm",
        "main_kv_norm",
        "main_rope",
        "main_kv_cache",
        "indexer_k_norm",
        "indexer_rope",
        "indexer_cache",
        "topk_selector",
    }

    assert required <= set(GLM5_RTP_OWNERSHIP)
    for key in required:
        owner = GLM5_RTP_OWNERSHIP[key]
        assert isinstance(owner, str)
        assert owner

    assert GLM5_RTP_OWNERSHIP["main_rope"] != GLM5_RTP_OWNERSHIP["indexer_rope"]


def test_glm5_ownership_forbids_qwen_and_mha_components():
    forbidden = ("GatedDeltaNet", "RTPFullAttention", "Qwen3Next")
    for owner in GLM5_RTP_OWNERSHIP.values():
        assert all(name not in owner for name in forbidden)

