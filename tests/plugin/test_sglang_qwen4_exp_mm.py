"""Qwen3.8-Flash-Next SGLang plugin multimodal wiring."""

from types import SimpleNamespace

import torch

# Patch AutoConfig.register before any SGLang config import (qwen3_asr clash).
from atom.plugin.sglang.patches.qwen4_exp_recognition_patch import (
    Qwen4ExpConfig,
    Qwen4ExpMultimodalProcessor,
    apply_qwen4_exp_recognition_patch,
    qwen4_exp_mm_token_id,
    qwen4_exp_uses_mrope,
    register_qwen4_exp_processor,
)
from atom.plugin.sglang.patches.qwen4_exp_recognition_patch import (
    Qwen4ExpForConditionalGeneration as ProcessorArch,
)

apply_qwen4_exp_recognition_patch()

from atom.plugin.sglang.models.qwen4_exp import (
    _cat_mm_field,
    _lm_positions_for_runtime,
    _sequence_positions,
    _should_embed_mm,
    _skip_visual_prefixes,
)
from atom.plugin.sglang.qwen4_exp_bridge import (
    _num_tokens_from_positions,
    _sequence_index_positions,
)


class _DecodeMode:
    @staticmethod
    def is_decode():
        return True

    @staticmethod
    def is_target_verify():
        return False


class _PrefillMode:
    @staticmethod
    def is_decode():
        return False

    @staticmethod
    def is_target_verify():
        return False


def test_qwen4_exp_uses_mrope_from_rope_parameters():
    hf = SimpleNamespace(
        rope_parameters={"mrope_section": [8, 12, 12]},
        rope_scaling=None,
        text_config=None,
        vision_config=None,
    )
    assert qwen4_exp_uses_mrope(hf) is True
    assert (
        qwen4_exp_uses_mrope(SimpleNamespace(rope_parameters={}, rope_scaling={}))
        is False
    )


def test_qwen4_exp_mm_token_id_defaults_and_root_config():
    hf = SimpleNamespace(text_config=None, vision_config=None)
    assert qwen4_exp_mm_token_id(hf, ("image_token_id",), 248056) == 248056
    hf.image_token_id = 11
    assert qwen4_exp_mm_token_id(hf, ("image_token_id",), 248056) == 11


def test_register_qwen4_exp_processor_maps_architecture_name():
    from sglang.srt.managers.multimodal_processor import PROCESSOR_MAPPING

    register_qwen4_exp_processor()
    names = [cls.__name__ for cls in PROCESSOR_MAPPING]
    assert "Qwen4ExpForConditionalGeneration" in names
    assert PROCESSOR_MAPPING[ProcessorArch] is Qwen4ExpMultimodalProcessor


def test_qwen4_exp_config_keeps_root_vision_token_ids():
    cfg = Qwen4ExpConfig(
        text_config={"hidden_size": 8, "num_hidden_layers": 1},
        vision_config={"spatial_merge_size": 2},
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
    )
    assert cfg.image_token_id == 248056
    assert cfg.video_token_id == 248057
    assert cfg.vision_start_token_id == 248053
    assert cfg.vision_end_token_id == 248054
    assert int(cfg.vision_config.spatial_merge_size) == 2


def test_sequence_positions_keep_1d_for_qsa():
    seq = torch.arange(4, dtype=torch.int64)
    assert torch.equal(_sequence_positions(seq), seq)
    mrope = torch.stack([seq + 10, seq, seq + 1])
    assert torch.equal(_sequence_positions(mrope), seq + 10)
    assert _num_tokens_from_positions(mrope) == 4
    logical = _sequence_index_positions(mrope, 4)
    assert logical.tolist() == [10, 11, 12, 13]
    assert len(logical) == 4


def test_lm_positions_prefer_3d_mrope():
    seq = torch.arange(3, dtype=torch.int64)
    mrope = torch.stack([seq, seq + 1, seq + 2])
    runtime = SimpleNamespace(
        positions=seq,
        _is_dummy_run=False,
        forward_batch=SimpleNamespace(mrope_positions=mrope),
    )
    out = _lm_positions_for_runtime(runtime)
    assert out.shape == (3, 3)
    assert torch.equal(out, mrope)
    runtime._is_dummy_run = True
    assert torch.equal(_lm_positions_for_runtime(runtime), seq)


def test_should_embed_mm_only_on_prefill_with_inputs():
    prefill = SimpleNamespace(
        forward_mode=_PrefillMode(),
        contains_mm_inputs=lambda: True,
    )
    decode = SimpleNamespace(
        forward_mode=_DecodeMode(),
        contains_mm_inputs=lambda: True,
    )
    text = SimpleNamespace(
        forward_mode=_PrefillMode(),
        contains_mm_inputs=lambda: False,
        mm_inputs=[],
    )
    assert _should_embed_mm(prefill) is True
    assert _should_embed_mm(decode) is False
    assert _should_embed_mm(text) is False


def test_cat_mm_field_concatenates_features():
    items = [
        SimpleNamespace(
            feature=torch.ones(2, 3), image_grid_thw=torch.tensor([[1, 2, 2]])
        ),
        SimpleNamespace(
            feature=torch.zeros(1, 3), image_grid_thw=torch.tensor([[1, 2, 2]])
        ),
    ]
    feat = _cat_mm_field(items, ("feature",))
    assert feat.shape == (3, 3)
    grid = _cat_mm_field(items, ("image_grid_thw",))
    assert grid.shape == (2, 3)


def test_skip_weight_prefixes_cover_mapped_visual_when_no_tower():
    skip = _skip_visual_prefixes(["mtp.", "model.visual."], has_visual=False)
    assert "visual." in skip
    assert "model.visual." in skip
    assert _skip_visual_prefixes(["mtp."], has_visual=True) == ["mtp."]
