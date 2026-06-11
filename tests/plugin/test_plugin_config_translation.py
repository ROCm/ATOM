import pytest

import atom.plugin.config as plugin_config


class _Obj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeCompilationConfig:
    def __init__(self, level, use_cudagraph, cudagraph_mode):
        self.level = level
        self.use_cudagraph = use_cudagraph
        self.cudagraph_mode = cudagraph_mode


def _patch_atom_config_module(monkeypatch):
    import atom.config as atom_config_module

    monkeypatch.setattr(atom_config_module, "Config", _FakeConfig, raising=False)
    monkeypatch.setattr(
        atom_config_module, "CompilationConfig", _FakeCompilationConfig, raising=False
    )


def test_generate_from_vllm_translates_core_fields(monkeypatch):
    _patch_atom_config_module(monkeypatch)

    vllm_cfg = _Obj(
        model_config=_Obj(model="m1", max_model_len=4096),
        scheduler_config=_Obj(max_num_batched_tokens=2048, max_num_seqs=8),
        cache_config=_Obj(
            gpu_memory_utilization=0.5,
            block_size=16,
            num_gpu_blocks=1024,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        parallel_config=_Obj(
            rank=1, tensor_parallel_size=2, enable_expert_parallel=False
        ),
        compilation_config=_Obj(mode=3),
        quant_config=_Obj(name="q"),
    )

    cfg = plugin_config._generate_atom_config_from_vllm_config(vllm_cfg)

    assert cfg.model == "m1"
    assert cfg.max_num_batched_tokens == 2048
    assert cfg.max_num_seqs == 8
    assert cfg.max_model_len == 4096
    assert cfg.tensor_parallel_size == 2
    assert cfg.enforce_eager is True
    assert cfg.compilation_config.level == 3
    assert cfg.plugin_config.is_plugin_mode is True
    assert cfg.plugin_config.is_vllm is True
    assert cfg.plugin_config.is_sglang is False


def _vllm_cfg(additional_config=None):
    return _Obj(
        model_config=_Obj(model="m1", max_model_len=4096),
        scheduler_config=_Obj(max_num_batched_tokens=2048, max_num_seqs=8),
        cache_config=_Obj(
            gpu_memory_utilization=0.5,
            block_size=16,
            num_gpu_blocks=1024,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        parallel_config=_Obj(
            rank=1, tensor_parallel_size=2, enable_expert_parallel=False
        ),
        compilation_config=_Obj(mode=3),
        quant_config=_Obj(name="q"),
        additional_config=additional_config if additional_config is not None else {},
    )


def test_online_quant_config_unset_is_none(monkeypatch):
    _patch_atom_config_module(monkeypatch)

    cfg = plugin_config._generate_atom_config_from_vllm_config(_vllm_cfg())

    assert cfg.online_quant_config is None


def test_online_quant_config_from_additional_config(monkeypatch):
    _patch_atom_config_module(monkeypatch)

    oqc = {
        "global_quant_config": "ptpc_fp8",
        "layer_quant_config": {"*expert*": "mxfp4"},
        "exclude_layer": ["lm_head", "*.gate.*"],
    }
    cfg = plugin_config._generate_atom_config_from_vllm_config(
        _vllm_cfg({"online_quant_config": oqc})
    )

    assert cfg.online_quant_config == oqc


def test_generate_atom_config_requires_plugin_mode(monkeypatch):
    import atom.plugin.config as config_module
    import atom.plugin as plugin_module
    import atom.config as atom_config_module

    monkeypatch.setattr(plugin_module, "is_vllm", lambda: False, raising=False)
    monkeypatch.setattr(plugin_module, "is_sglang", lambda: False, raising=False)
    monkeypatch.setattr(
        atom_config_module, "set_current_atom_config", lambda _cfg: None, raising=False
    )

    with pytest.raises(ValueError, match="running in plugin mode"):
        config_module.generate_atom_config_for_plugin_mode(config=None)
