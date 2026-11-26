# SPDX-License-Identifier: MIT
from copy import deepcopy
from types import SimpleNamespace

from atom import config as atom_config
from atom.config import flatten_dsa_config
from atom.models.registry import mono_decode_installers, support_model_arch_dict
from atom.multimodal.registry import _MULTIMODAL_ARCH_TO_INPUT_BUILDER
from atom.utils.selector import Family, attn_family, has_mla_indexer


def raw_config():
    return {
        "model_type": "kimi_k31",
        "architectures": ["KimiK31ForConditionalGeneration"],
        "text_config": {
            "model_type": "kimi_linear",
            "num_hidden_layers": 4,
            "kv_lora_rank": 512,
            "linear_attn_config": {"full_attn_layers": [4], "kda_layers": [1, 2, 3]},
            "dsa_config": {
                "dsa_layers": [4],
                "index_topk": 2048,
                "index_num_heads": 32,
                "index_head_dim": 128,
                "index_rope_dim": 0,
                "index_k_norm_type": "layernorm",
                "indexer_use_fp8": False,
                "index_topk_freq": 1,
                "index_skip_topk_offset": 1,
            },
        },
        "vision_config": {"vt_hidden_size": 1024},
    }


def test_real_loader_flattens_before_backend_selection(monkeypatch):
    raw = raw_config()
    monkeypatch.setattr(
        atom_config.PretrainedConfig,
        "get_config_dict",
        lambda *_a, **_k: (deepcopy(raw), {}),
    )
    full = SimpleNamespace(vision_config=raw["vision_config"])
    monkeypatch.setattr(
        atom_config.AutoConfig, "from_pretrained", lambda *_a, **_k: full
    )
    cfg = atom_config.get_hf_config("unused", trust_remote_code=True)
    assert attn_family(cfg) is Family.KIMI_MLA
    assert has_mla_indexer(cfg)
    assert cfg.text_config is cfg
    assert cfg._multimodal_config is full
    assert cfg.dsa_layer_indices == (3,)
    assert cfg.index_topk == 2048
    assert cfg.index_n_heads == 32
    assert cfg.index_rope_dim == 0
    assert cfg.index_skip_topk_offset == 1


def test_k31_loads_as_k3_for_every_architecture_table(monkeypatch):
    raw = raw_config()
    monkeypatch.setattr(
        atom_config.PretrainedConfig,
        "get_config_dict",
        lambda *_a, **_k: (deepcopy(raw), {}),
    )
    monkeypatch.setattr(
        atom_config.AutoConfig, "from_pretrained", lambda *_a, **_k: None
    )
    cfg = atom_config.get_hf_config("unused", trust_remote_code=True)
    arch = cfg.architectures[0]
    assert arch == "KimiK3ForConditionalGeneration"
    assert arch in support_model_arch_dict
    assert arch in mono_decode_installers
    assert arch in _MULTIMODAL_ARCH_TO_INPUT_BUILDER


def test_config_without_dsa_acquires_no_sparse_declaration():
    cfg = SimpleNamespace(model_type="kimi_linear")
    flatten_dsa_config(cfg)
    assert not has_mla_indexer(cfg)
    assert not hasattr(cfg, "dsa_layer_indices")
