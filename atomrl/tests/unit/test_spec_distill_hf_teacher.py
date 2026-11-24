# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The HF teacher must provide every teacher tensor the Eagle3 train step reads.

examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/configs/smoke_test_hf.yaml failed its
first step with AttributeError: the HF path set only the lm_head weight, while
the vLLM and ATOM paths also set the embedding table.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from atomrl.trainer.spec_distill_trainer import SpecDistillTrainer


def test_hf_teacher_sets_embedding_table(tmp_path) -> None:
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    model = transformers.LlamaForCausalLM(cfg)
    model.save_pretrained(tmp_path)

    trainer = SpecDistillTrainer.__new__(SpecDistillTrainer)
    trainer._rank = 0
    trainer._device = torch.device("cpu")
    trainer._setup_teacher_hf(str(tmp_path))

    expected = model.get_input_embeddings().weight.detach().to(torch.bfloat16)
    torch.testing.assert_close(trainer._embed_weight, expected)
    assert trainer._lm_head_weight.shape == (64, 16)
    assert trainer._norm_weight is None


def test_cleanup_continues_to_mooncake_after_teacher_backend_error() -> None:
    events = []

    class FailingTeacher:
        def shutdown(self) -> None:
            events.append("teacher")
            raise LookupError("teacher actor already stopped")

    class MooncakeMaster:
        def shutdown(self) -> None:
            events.append("mooncake")

    trainer = SpecDistillTrainer.__new__(SpecDistillTrainer)
    trainer._rank = 0
    trainer._teacher_engine = FailingTeacher()
    trainer._mooncake_master = MooncakeMaster()
    trainer._teacher_model = object()
    trainer._draft_model = object()
    trainer._lm_head_weight = object()
    trainer._optimizer = object()

    trainer.cleanup()

    assert events == ["teacher", "mooncake"]
    assert trainer._teacher_engine is None
    assert trainer._mooncake_master is None
