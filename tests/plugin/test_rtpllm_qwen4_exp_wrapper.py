"""Registration and config tests for the Qwen3.8-Flash-Next rtp-llm wrapper."""

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_FLASH_CONFIG = Path("/data/pretrained_model/Qwen/Qwen3.8-Flash-Next-FP8/config.json")


class _Enum:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name

    def __repr__(self):
        return self.name


class _HybridAttentionType:
    NONE = _Enum("NONE")
    LINEAR = _Enum("LINEAR")


class _RopeConfig:
    def __init__(self):
        self.style = 0
        self.base = 0
        self.dim = 0


class _AttnConfig:
    def __init__(self):
        self.head_num = 0
        self.kv_head_num = 0
        self.size_per_head = 0
        self.rope_config = _RopeConfig()


class _HybridAttentionConfig:
    def __init__(self):
        self.enable_hybrid_attention = False
        self.hybrid_attention_types = []


class _LinearAttentionConfig:
    def __init__(self):
        self.linear_conv_kernel_dim = 0
        self.linear_key_head_dim = 0
        self.linear_num_key_heads = 0
        self.linear_num_value_heads = 0
        self.linear_value_head_dim = 0


class _FakeModelConfig:
    def __init__(self):
        self.attn_config = _AttnConfig()
        self.hybrid_attention_config = _HybridAttentionConfig()
        self.linear_attention_config = _LinearAttentionConfig()
        self.ckpt_path = ""
        self.num_layers = 0
        self.hidden_size = 0
        self.vocab_size = 0
        self.max_seq_len = 0
        self.tie_word_embeddings = False
        self.partial_rotary_factor = 0
        self.layernorm_eps = 0
        self.norm_type = ""
        self.has_pre_decoder_layernorm = True
        self.has_post_decoder_layernorm = True
        self.qk_norm = False
        self.activation_type = ""
        self.moe_k = 0
        self.expert_num = 0
        self.moe_inter_size = 0
        self.inter_size = 0
        self.has_moe_norm = False
        self.moe_style = 0
        self.moe_layer_index = []


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _install_fake_rtp_modules() -> dict[str, ModuleType]:
    fake_config_mod = ModuleType("rtp_llm.config.model_config")
    fake_config_mod.ModelConfig = _FakeModelConfig

    fake_factory_register_mod = ModuleType("rtp_llm.model_factory_register")
    fake_factory_register_mod.register_model = MagicMock()
    fake_factory_register_mod._model_factory = {}
    fake_factory_register_mod._hf_architecture_2_ft = {}

    fake_base_mod = ModuleType("rtp_llm.models.base_model")

    class _FakeBaseModel:
        def _get_device_str(self):
            return "cpu"

    fake_base_mod.BaseModel = _FakeBaseModel

    fake_weight_info_mod = ModuleType("rtp_llm.model_loader.model_weight_info")

    class _FakeModelWeights:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.global_weights = {}

        def set_global_weight(self, name, tensor):
            self.global_weights[name] = tensor

    class _FakeModelDeployWeightInfo:
        pass

    fake_weight_info_mod.ModelDeployWeightInfo = _FakeModelDeployWeightInfo
    fake_weight_info_mod.ModelWeights = _FakeModelWeights

    fake_module_base_mod = ModuleType("rtp_llm.models_py.model_desc.module_base")

    class _FakeGptModelBase:
        def __init__(self, *args, **kwargs):
            self.init_args = args
            self.init_kwargs = kwargs

    fake_module_base_mod.GptModelBase = _FakeGptModelBase

    fake_ops_mod = ModuleType("rtp_llm.ops")
    fake_ops_mod.HybridAttentionType = _HybridAttentionType
    fake_ops_mod.ParallelismConfig = type("ParallelismConfig", (), {})

    fake_compute_ops_mod = ModuleType("rtp_llm.ops.compute_ops")
    fake_compute_ops_mod.PyModelInputs = type("PyModelInputs", (), {})
    fake_compute_ops_mod.PyModelOutputs = type("PyModelOutputs", (), {})

    fake_weight_mod = ModuleType("rtp_llm.utils.model_weight")
    fake_weight_mod.W = SimpleNamespace(
        lm_head="lm_head",
        embedding="embedding",
        final_ln_gamma="final_ln_gamma",
    )

    return {
        "rtp_llm": _package("rtp_llm"),
        "rtp_llm.config": _package("rtp_llm.config"),
        "rtp_llm.config.model_config": fake_config_mod,
        "rtp_llm.model_factory_register": fake_factory_register_mod,
        "rtp_llm.models": _package("rtp_llm.models"),
        "rtp_llm.models.base_model": fake_base_mod,
        "rtp_llm.model_loader": _package("rtp_llm.model_loader"),
        "rtp_llm.model_loader.model_weight_info": fake_weight_info_mod,
        "rtp_llm.models_py": _package("rtp_llm.models_py"),
        "rtp_llm.models_py.model_desc": _package("rtp_llm.models_py.model_desc"),
        "rtp_llm.models_py.model_desc.module_base": fake_module_base_mod,
        "rtp_llm.ops": fake_ops_mod,
        "rtp_llm.ops.compute_ops": fake_compute_ops_mod,
        "rtp_llm.utils": _package("rtp_llm.utils"),
        "rtp_llm.utils.model_weight": fake_weight_mod,
    }


@pytest.fixture
def qwen4_exp_module():
    fake_modules = _install_fake_rtp_modules()
    with patch.dict(sys.modules, fake_modules):
        sys.modules.pop("atom.plugin.rtpllm.models.qwen4_exp", None)
        module = importlib.import_module("atom.plugin.rtpllm.models.qwen4_exp")
        yield module
        sys.modules.pop("atom.plugin.rtpllm.models.qwen4_exp", None)


def test_create_config_from_flash_text_config(qwen4_exp_module, tmp_path):
    text = {
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_hidden_layers": 8,
        "hidden_size": 2560,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": False,
        "rope_parameters": {
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
        "rms_norm_eps": 1e-6,
        "num_experts_per_tok": 10,
        "num_experts": 512,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "full_attention_interval": 4,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "ple_layer_ids": [2],
        "hc_count": 4,
    }
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "text_config": text,
            }
        )
    )

    config = qwen4_exp_module.ATOMQwen4Exp._create_config(str(ckpt))
    assert config.num_layers == 8
    assert config.attn_config.head_num == 24
    assert config.attn_config.kv_head_num == 2
    assert config.attn_config.size_per_head == 256
    assert config.hidden_size == 2560
    assert config.expert_num == 512
    assert config.moe_k == 10
    assert config.has_pre_decoder_layernorm is False
    assert config.has_post_decoder_layernorm is False
    assert config.hybrid_attention_config.enable_hybrid_attention is True
    types = config.hybrid_attention_config.hybrid_attention_types
    assert [t.name for t in types] == [
        "LINEAR",
        "LINEAR",
        "LINEAR",
        "NONE",
        "LINEAR",
        "LINEAR",
        "LINEAR",
        "NONE",
    ]
    assert config.linear_attention_config.linear_num_value_heads == 48
    assert config.linear_attention_config.linear_key_head_dim == 128


def test_create_config_falls_back_to_full_attention_interval(qwen4_exp_module, tmp_path):
    text = {
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_hidden_layers": 4,
        "hidden_size": 2560,
        "vocab_size": 10,
        "max_position_embeddings": 128,
        "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 0.25},
        "rms_norm_eps": 1e-6,
        "num_experts_per_tok": 2,
        "num_experts": 8,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
    }
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"text_config": text}))
    config = qwen4_exp_module.ATOMQwen4Exp._create_config(str(ckpt))
    assert [t.name for t in config.hybrid_attention_config.hybrid_attention_types] == [
        "LINEAR",
        "LINEAR",
        "LINEAR",
        "NONE",
    ]


@pytest.mark.skipif(not _FLASH_CONFIG.exists(), reason="Flash checkpoint not mounted")
def test_create_config_matches_released_flash_checkpoint(qwen4_exp_module):
    config = qwen4_exp_module.ATOMQwen4Exp._create_config(str(_FLASH_CONFIG.parent))
    raw = json.loads(_FLASH_CONFIG.read_text())["text_config"]
    assert config.num_layers == raw["num_hidden_layers"] == 48
    assert config.hidden_size == raw["hidden_size"] == 2560
    assert config.expert_num == raw["num_experts"] == 512
    types = config.hybrid_attention_config.hybrid_attention_types
    assert len(types) == 48
    assert sum(t.name == "NONE" for t in types) == 12
    assert sum(t.name == "LINEAR" for t in types) == 36
    assert types[1].name == "LINEAR"
    assert types[3].name == "NONE"
    assert raw["ple_layer_ids"] == [2]
    assert raw["hc_count"] == 4
    assert raw["full_attention_interval"] == 4
