"""SGLang-ATOM Flash MTP wiring: draft rewrite, HC flatten, QSA pool mapping."""

from types import SimpleNamespace

import pytest
import torch

from atom.plugin.config import _build_atom_speculative_config_from_sglang
from atom.plugin.sglang.models.qwen4_exp import (
    flatten_qwen4_exp_hc,
    reshape_qwen4_exp_hc,
)
from atom.plugin.sglang.patches.qwen4_exp_recognition_patch import (
    QWEN4_EXP_NEXTN_ARCH,
    apply_qwen4_exp_hc_hidden_size,
    is_qwen4_exp_nextn_arch,
    rewrite_qwen4_exp_draft_hf_config,
)
from atom.plugin.sglang.qwen4_exp_bridge import (
    _query_start_loc,
    bind_qsa_caches,
)


class _ExtendMode:
    @staticmethod
    def is_decode_or_idle():
        return False

    @staticmethod
    def is_extend():
        return True

    @staticmethod
    def is_target_verify():
        return True


def test_rewrite_qwen4_exp_draft_hf_config_shrinks_to_one_qsa_layer():
    text = SimpleNamespace(
        model_type="qwen4_exp_text",
        num_hidden_layers=48,
        mtp={"layer_types": ["full_attention"], "mtp_num_hidden_layers": 1},
        mtp_num_hidden_layers=1,
        layer_types=["full_attention"] + ["linear_attention"] * 47,
        ple_layer_ids=[1, 2, 3],
        architectures=["Qwen4ExpForConditionalGeneration"],
    )
    hf = SimpleNamespace(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=text,
        num_hidden_layers=48,
    )
    assert rewrite_qwen4_exp_draft_hf_config(hf, text)
    assert hf.architectures == [QWEN4_EXP_NEXTN_ARCH]
    assert hf.num_hidden_layers == 1
    assert text.num_hidden_layers == 1
    assert text.layer_types == ["full_attention"]
    assert text.ple_layer_ids == []
    assert is_qwen4_exp_nextn_arch(hf)


def test_rewrite_ignores_non_flash_arch():
    hf = SimpleNamespace(
        architectures=["DeepseekV3ForCausalLM"],
        model_type="deepseek_v3",
        num_hidden_layers=61,
    )
    assert not rewrite_qwen4_exp_draft_hf_config(hf)
    assert hf.architectures == ["DeepseekV3ForCausalLM"]
    assert hf.num_hidden_layers == 61


def test_hc_hidden_size_matches_flattened_bundle():
    cfg = SimpleNamespace(
        hidden_size=2560,
        spec_hidden_size=2560,
        hc_hidden_size=None,
        hf_text_config=SimpleNamespace(model_type="qwen4_exp_text", hc_count=4),
    )
    assert apply_qwen4_exp_hc_hidden_size(cfg)
    assert cfg.spec_hidden_size == 10240
    assert cfg.hc_hidden_size == 10240


def test_flatten_and_reshape_hc_roundtrip():
    hidden = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    flat = flatten_qwen4_exp_hc(hidden)
    assert tuple(flat.shape) == (2, 32)
    restored = reshape_qwen4_exp_hc(flat, hidden_size=8, hc_count=4)
    torch.testing.assert_close(restored, hidden)


def test_draft_hidden_requires_one_state_per_token():
    from atom.plugin.sglang.models.qwen4_exp_nextn_wrapper import (
        Qwen4ExpForCausalLMNextN,
    )

    # Exercise layout validation without loading model weights.
    draft = Qwen4ExpForCausalLMNextN.__new__(Qwen4ExpForCausalLMNextN)
    hidden = torch.zeros(2, 32)
    assert draft._align_spec_hidden(hidden, length=2) is hidden
    with pytest.raises(RuntimeError, match="draft hidden layout mismatch"):
        draft._align_spec_hidden(hidden, length=6)


def test_sglang_speculative_config_uses_mtp_for_eagle(monkeypatch):
    captured = {}

    class _DummySpec:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("atom.config.SpeculativeConfig", _DummySpec)
    args = SimpleNamespace(
        speculative_algorithm="EAGLE",
        model_path="/models/Qwen3.8-Flash-Next-FP8",
        speculative_num_steps=2,
    )
    spec = _build_atom_speculative_config_from_sglang(
        args, SimpleNamespace(model_type="qwen4_exp_text")
    )
    assert spec is not None
    assert captured["method"] == "mtp"
    assert captured["num_speculative_tokens"] == 2
    assert captured["model"] == "/models/Qwen3.8-Flash-Next-FP8"

    assert (
        _build_atom_speculative_config_from_sglang(
            SimpleNamespace(speculative_algorithm=None, model_path=args.model_path),
            SimpleNamespace(model_type="qwen4_exp_text"),
        )
        is None
    )


def test_query_start_loc_verify_fallback_without_extend_fields():
    device = torch.device("cpu")
    fb = SimpleNamespace(
        forward_mode=_ExtendMode(),
        batch_size=2,
        num_padding=0,
        spec_info=SimpleNamespace(draft_token_num=3),
    )
    loc = _query_start_loc(fb, num_tokens=6, device=device)
    assert torch.equal(loc, torch.tensor([0, 3, 6], dtype=torch.int32))


class _QsaLayer:
    def __init__(self, layer_num):
        self.layer_num = layer_num
        self.is_qsa_attention = True
        self.num_kv_heads = 2
        self.head_dim = 4
        self.bound = None

    def bind_caches(self, k, v, raw, compressed, rope):
        self.bound = (k, v, raw, compressed)


class _Pool:
    def __init__(self, layers=1, tokens=64, heads=2, dim=4):
        self.size = tokens
        self._bufs = {
            i: (
                torch.zeros(tokens, heads, dim),
                torch.zeros(tokens, heads, dim),
            )
            for i in range(layers)
        }

    def get_kv_buffer(self, layer_id):
        return self._bufs[int(layer_id)]


@pytest.mark.parametrize(
    "model_type,pool_layer", [("qwen4_exp_mtp", 0), ("qwen4_exp_text", 48)]
)
def test_bind_qsa_caches_uses_explicit_layer_mapping(model_type, pool_layer):
    layer = _QsaLayer(layer_num=48)
    model = SimpleNamespace(
        modules=lambda: [layer],
        parameters=lambda: iter([torch.zeros(1)]),
    )
    pool = _Pool(layers=49)
    fb = SimpleNamespace(token_to_kv_pool=pool, page_size=64)
    atom_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type=model_type,
            indexer_compress_ratio=4,
            indexer_head_dim=8,
        )
    )
    bind_qsa_caches(model, fb, atom_config)
    assert layer.bound is not None
    k, v, raw, compressed = layer.bound
    assert tuple(k.shape) == (1, 64, 2, 4)
    assert k.data_ptr() == pool._bufs[pool_layer][0].data_ptr()
    assert v.data_ptr() == pool._bufs[pool_layer][1].data_ptr()
    assert raw.shape[0] == 1
    assert compressed.shape == (1, 16, 1, 8)
    assert model._atom_qwen4_exp_qsa_bound is True


def test_target_qsa_missing_layer_does_not_fall_back_to_zero():
    layer = _QsaLayer(layer_num=48)
    model = SimpleNamespace(
        modules=lambda: [layer], parameters=lambda: iter([torch.zeros(1)])
    )
    fb = SimpleNamespace(token_to_kv_pool=_Pool(layers=1), page_size=64)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp_text",
            indexer_compress_ratio=4,
            indexer_head_dim=8,
        )
    )
    with pytest.raises(RuntimeError, match="KV mapping missing for layer 48"):
        bind_qsa_caches(model, fb, config)
    assert layer.bound is None
    assert not getattr(model, "_atom_qwen4_exp_qsa_bound", False)


@pytest.mark.parametrize("shape", [(2, 8), (2, 16), (2, 2, 8), (2, 4, 7), (32,)])
def test_draft_rejects_wrong_hc_bundle(shape):
    with pytest.raises(ValueError, match="hidden_states must have shape"):
        reshape_qwen4_exp_hc(torch.zeros(shape), hidden_size=8, hc_count=4)


def test_non_flash_speculative_and_hc_config_unchanged():
    hf = SimpleNamespace(model_type="deepseek_v3", hc_count=4)
    assert (
        _build_atom_speculative_config_from_sglang(
            SimpleNamespace(speculative_algorithm="EAGLE3"), hf
        )
        is None
    )
    cfg = SimpleNamespace(hf_text_config=hf, hidden_size=8, spec_hidden_size=8)
    assert not apply_qwen4_exp_hc_hidden_size(cfg)
    assert cfg.spec_hidden_size == 8
    assert not is_qwen4_exp_nextn_arch(
        SimpleNamespace(architectures=["DeepseekV3ForCausalLMNextN"])
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"speculative_algorithm": "EAGLE3"},
        {"speculative_eagle_topk": 2},
        {"speculative_draft_model_path": "/other"},
    ],
)
def test_flash_rejects_unsupported_speculation(overrides):
    args = {
        "speculative_algorithm": "EAGLE",
        "model_path": "/model",
        "speculative_eagle_topk": 1,
    }
    args.update(overrides)
    with pytest.raises(ValueError, match="Flash MTP requires"):
        _build_atom_speculative_config_from_sglang(
            SimpleNamespace(**args), SimpleNamespace(model_type="qwen4_exp_text")
        )


def test_draft_default_layer_does_not_inherit_target_gdn():
    hf = SimpleNamespace(model_type="qwen4_exp_text", layer_types=["linear_attention"])
    assert rewrite_qwen4_exp_draft_hf_config(hf)
    assert hf.layer_types == ["full_attention"]


@pytest.mark.parametrize(
    "layers,types",
    [(2, ["full_attention", "full_attention"]), (1, ["linear_attention"]), (1, [])],
)
def test_draft_rewrite_rejects_unsupported_native_layout(layers, types):
    hf = SimpleNamespace(
        model_type="qwen4_exp_text",
        mtp_num_hidden_layers=layers,
        mtp={"layer_types": types},
        num_hidden_layers=48,
    )
    with pytest.raises(ValueError, match="exactly one QSA"):
        rewrite_qwen4_exp_draft_hf_config(hf)
    assert hf.num_hidden_layers == 48
