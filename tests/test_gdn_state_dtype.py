# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Tests which dtype the GDN temporal state pool stores.
# A checkpoint that declares `mamba_ssm_dtype` (Qwen3.5 / Qwen3.8: "float32")
# gets what it asks for, as vLLM does; one that does not keeps the state at the
# model dtype, which is what every GDN model got before the field was honoured.
# The KDA families keep their own switch: `ATOM_GDN_SSM_DTYPE`.


from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_ops.attentions import gdn_attn
from atom.model_ops.attentions.gdn_attn import GDNStateMixin


def dtypes_for(model_dtype=torch.bfloat16, **hf_fields):
    config = SimpleNamespace(
        torch_dtype=model_dtype, hf_config=SimpleNamespace(**hf_fields)
    )
    stub = SimpleNamespace(model_runner=SimpleNamespace(config=config))
    return GDNStateMixin._state_dtypes(stub)


class TestACheckpointThatDeclaresItsStateDtype:
    def test_qwen3_5_float32_is_honoured(self):
        """Qwen3.5 / Qwen3.8 configs say "float32"; the conv state stays at
        the model dtype, only the temporal state widens."""
        assert dtypes_for(model_type="qwen3_5", mamba_ssm_dtype="float32") == (
            torch.bfloat16,
            torch.float32,
        )

    def test_bfloat16_is_honoured_too(self):
        assert dtypes_for(model_type="qwen3_5", mamba_ssm_dtype="bfloat16") == (
            torch.bfloat16,
            torch.bfloat16,
        )

    def test_short_spellings_are_accepted(self):
        """So the field and `ATOM_GDN_SSM_DTYPE` agree on names."""
        assert dtypes_for(model_type="qwen3_5", mamba_ssm_dtype="fp32")[1] is (
            torch.float32
        )
        assert dtypes_for(model_type="qwen3_5", mamba_ssm_dtype="FLOAT32")[1] is (
            torch.float32
        )

    def test_an_unknown_value_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="mamba_ssm_dtype='float8'"):
            dtypes_for(model_type="qwen3_5", mamba_ssm_dtype="float8")


class TestACheckpointThatDoesNot:
    def test_keeps_the_model_dtype(self):
        """Qwen3-Next configs carry no `mamba_ssm_dtype`; nothing changes for
        them."""
        assert dtypes_for(model_type="qwen3_next") == (
            torch.bfloat16,
            torch.bfloat16,
        )

    def test_follows_the_model_dtype(self):
        assert dtypes_for(model_dtype=torch.float16, model_type="qwen3_next") == (
            torch.float16,
            torch.float16,
        )


class TestTheKdaFamiliesKeepTheirOwnSwitch:
    def test_the_env_wins_even_when_the_config_has_the_field(self, monkeypatch):
        monkeypatch.setattr(gdn_attn.envs, "ATOM_GDN_SSM_DTYPE", "fp16")
        assert dtypes_for(model_type="kimi_linear", mamba_ssm_dtype="float32") == (
            torch.bfloat16,
            torch.float16,
        )

    def test_the_env_default_is_fp32(self, monkeypatch):
        monkeypatch.setattr(gdn_attn.envs, "ATOM_GDN_SSM_DTYPE", "fp32")
        assert dtypes_for(model_type="glm5_next_text")[1] is torch.float32
