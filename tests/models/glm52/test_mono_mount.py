# SPDX-License-Identifier: MIT
"""GLM-5.2 mono is mounted on the GLM class, not the shared DeepSeek one."""

from pathlib import Path

SOURCE = (Path(__file__).parents[3] / "atom" / "models" / "deepseek_v2.py").read_text()


def test_the_shared_deepseek_class_carries_no_glm_mono_code():
    shared = SOURCE[
        SOURCE.index("class DeepseekV2ForCausalLM") : SOURCE.index(
            "class GlmMoeDsaForCausalLM"
        )
    ]
    assert "glm52" not in shared.lower()
    assert "glm_moe_dsa" not in shared


def test_the_glm_subclass_routes_to_mono():
    glm = SOURCE[SOURCE.index("class GlmMoeDsaForCausalLM") :]
    assert "from atom.models.glm52.mono.dispatch import MonoDecode" in glm
    assert ".supports(" in glm
