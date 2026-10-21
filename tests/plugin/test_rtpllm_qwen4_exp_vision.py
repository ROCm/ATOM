"""RTP Qwen4Exp image-position and multimodal registration contracts."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("rtp_llm")

from rtp_llm.multimodal.multimodal_mixin_register import (
    get_multimodal_mixin_cls,
)

from atom.plugin.rtpllm.models.qwen4_exp import _ATOMQwen4ExpRuntime
from atom.plugin.rtpllm.models.qwen4_exp_vision import Qwen4ExpVisionMixin


def test_qwen4_exp_vision_mixin_registered():
    assert get_multimodal_mixin_cls("atom_qwen4_exp") is Qwen4ExpVisionMixin


def test_vision_weight_budget_reads_checkpoint_metadata(tmp_path):
    save_file(
        {"model.visual.weight": torch.zeros(2, 3)}, tmp_path / "visual.safetensors"
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.visual.weight": "visual.safetensors"}})
    )
    params = SimpleNamespace(config={"ckpt_path": str(tmp_path)})
    assert Qwen4ExpVisionMixin.eval_mm_model_param_count(params, "", "") == 6
    assert Qwen4ExpVisionMixin.eval_mm_model_size(params, "", "") == 12


@pytest.mark.parametrize(
    "positions",
    [
        torch.tensor([0, 0, 0, 1, 2, 3], dtype=torch.int32),
        torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.int32),
    ],
)
def test_rtp_interleaved_mrope_uses_token_major_order(positions):
    inputs = SimpleNamespace(
        attention_inputs=SimpleNamespace(combo_position_ids=positions),
        combo_position_ids=None,
    )
    actual = _ATOMQwen4ExpRuntime._rope_positions(inputs, torch.device("cpu"), 2)
    assert torch.equal(actual, torch.tensor([[0, 1], [0, 2], [0, 3]]))


def test_logical_cache_position_is_not_image_rope_position():
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    attn_inputs = SimpleNamespace(
        is_prefill=True,
        input_lengths=torch.tensor([4], dtype=torch.int32),
        prefix_lengths=torch.tensor([0], dtype=torch.int32),
    )
    actual = runtime._logical_positions(attn_inputs, torch.device("cpu"))
    assert torch.equal(actual, torch.arange(4, dtype=torch.int32))


def test_decode_cache_position_uses_sequence_length_not_mrope():
    runtime = object.__new__(_ATOMQwen4ExpRuntime)
    attn_inputs = SimpleNamespace(
        is_prefill=False,
        sequence_lengths_plus_1_device=torch.tensor([129, 257], dtype=torch.int32),
    )
    actual = runtime._logical_positions(attn_inputs, torch.device("cpu"))
    assert torch.equal(actual, torch.tensor([128, 256], dtype=torch.int32))
