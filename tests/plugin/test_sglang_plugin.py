"""Tests for ATOM's sglang plugin integration.

Covers config translation, model dict selection, and framework mode status.
All tests mock sglang dependencies so they run without sglang installed.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch

from atom.plugin import prepare as plugin_prepare
import atom.plugin.config as plugin_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Obj:
    """Minimal attribute bag for faking nested configs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_fake_server_args(**overrides):
    """Return a minimal mock of sglang's ServerArgs."""
    defaults = dict(
        model_path="/fake/model",
        tp_size=2,
        dp_size=1,
        ep_size=1,
        context_length=8192,
        max_running_requests=64,
        mem_fraction_static=0.85,
        kv_cache_dtype="bf16",
        modelopt_quant=None,
        modelopt_checkpoint_restore_path=None,
        modelopt_checkpoint_save_path=None,
        modelopt_export_path=None,
        nccl_port=29500,
        dist_init_addr="127.0.0.1:29500",
        load_format="auto",
        download_dir=None,
        model_loader_extra_config=None,
        remote_instance_weight_loader_seed_instance_ip=None,
        remote_instance_weight_loader_seed_instance_service_port=None,
        remote_instance_weight_loader_send_weights_group_ports=None,
        remote_instance_weight_loader_backend=None,
        rl_quant_profile=None,
        enable_torch_compile=False,
        disable_cuda_graph=False,
        enable_dp_attention=False,
    )
    defaults.update(overrides)
    return _Obj(**defaults)


class _FakeConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeCompilationConfig:
    def __init__(self, level, use_cudagraph, cudagraph_mode):
        self.level = level
        self.use_cudagraph = use_cudagraph
        self.cudagraph_mode = cudagraph_mode


class _FakeParallelConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_framework_state():
    plugin_prepare._set_framework_backbone("atom")
    yield
    plugin_prepare._set_framework_backbone("atom")


# ---------------------------------------------------------------------------
# Config translation
# ---------------------------------------------------------------------------


def _make_sglang_sys_modules(
    mock_server_args_mod,
    mock_distributed_mod=None,
    mock_model_config_mod=None,
    mock_modelopt_config_mod=None,
    mock_load_config_mod=None,
):
    """Build the sys.modules dict needed to mock sglang imports."""
    mods = {
        "sglang": MagicMock(),
        "sglang.srt": MagicMock(),
        "sglang.srt.server_args": mock_server_args_mod,
    }
    if mock_distributed_mod is not None:
        mods["sglang.srt.distributed"] = mock_distributed_mod
    if mock_model_config_mod is not None:
        mods["sglang.srt.configs"] = MagicMock()
        mods["sglang.srt.configs.model_config"] = mock_model_config_mod
    if mock_modelopt_config_mod is not None:
        mods["sglang.srt.configs.modelopt_config"] = mock_modelopt_config_mod
    if mock_load_config_mod is not None:
        mods["sglang.srt.configs.load_config"] = mock_load_config_mod
    return mods


def test_generate_sglang_config_translates_core_fields(monkeypatch):
    """Verify that sglang ServerArgs are correctly mapped to ATOM Config."""
    import atom.config as atom_config_module

    monkeypatch.setattr(atom_config_module, "Config", _FakeConfig, raising=False)
    monkeypatch.setattr(
        atom_config_module, "ParallelConfig", _FakeParallelConfig, raising=False
    )
    monkeypatch.setattr(
        atom_config_module, "CompilationConfig", _FakeCompilationConfig, raising=False
    )

    fake_server_args = _make_fake_server_args(
        tp_size=4,
        dp_size=2,
        kv_cache_dtype="fp8_e4m3fn",
        context_length=16384,
        max_running_requests=128,
        mem_fraction_static=0.9,
    )

    mock_sglang_server_args = MagicMock()
    mock_sglang_server_args.get_global_server_args.return_value = fake_server_args

    mock_port_args_cls = MagicMock()
    mock_port_args_cls.init_new.return_value = _Obj()
    mock_sglang_server_args.PortArgs = mock_port_args_cls

    mock_sglang_distributed = MagicMock()
    mock_sglang_distributed.get_tensor_model_parallel_rank.return_value = 0

    mock_sglang_model_config = MagicMock()
    fake_model_config = _Obj(hf_config=_Obj())
    mock_sglang_model_config.ModelConfig.from_server_args.return_value = (
        fake_model_config
    )

    mock_sglang_modelopt_config = MagicMock()
    mock_sglang_modelopt_config.ModelOptConfig.return_value = _Obj()

    mock_sglang_load_config = MagicMock()
    mock_sglang_load_config.LoadConfig.return_value = _Obj()

    sgl_mods = _make_sglang_sys_modules(
        mock_sglang_server_args,
        mock_sglang_distributed,
        mock_sglang_model_config,
        mock_sglang_modelopt_config,
        mock_sglang_load_config,
    )

    with (
        patch.dict(sys.modules, sgl_mods),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        cfg = plugin_config._generate_atom_config_from_sglang_config(
            config=_Obj(architectures=["DeepseekV3ForCausalLM"])
        )

    assert cfg.model == "/fake/model"
    assert cfg.tensor_parallel_size == 4
    assert cfg.kv_cache_dtype == "fp8_e4m3fn"
    assert cfg.max_model_len == 16384
    assert cfg.max_num_seqs == 128
    assert cfg.enforce_eager is True
    assert cfg.max_num_batched_tokens == 16384
    assert cfg.plugin_config.is_plugin_mode is True
    assert cfg.plugin_config.is_sglang is True
    assert cfg.plugin_config.is_vllm is False


def test_generate_sglang_config_raises_on_none_server_args(monkeypatch):
    """Verify clear error when sglang global args are not initialized."""
    import atom.config as atom_config_module

    monkeypatch.setattr(atom_config_module, "Config", _FakeConfig, raising=False)
    monkeypatch.setattr(
        atom_config_module, "ParallelConfig", _FakeParallelConfig, raising=False
    )
    monkeypatch.setattr(
        atom_config_module, "CompilationConfig", _FakeCompilationConfig, raising=False
    )

    mock_sglang_server_args = MagicMock()
    mock_sglang_server_args.get_global_server_args.return_value = None
    mock_sglang_server_args.PortArgs = _Obj
    mock_sglang_distributed = MagicMock()

    sgl_mods = _make_sglang_sys_modules(
        mock_sglang_server_args,
        mock_sglang_distributed,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    with patch.dict(sys.modules, sgl_mods):
        with pytest.raises(RuntimeError, match="not initialized"):
            plugin_config._generate_atom_config_from_sglang_config(
                config=_Obj(architectures=["DeepseekV3ForCausalLM"])
            )


def test_generate_sglang_config_raises_on_server_args_exception(monkeypatch):
    """Verify clear error when get_global_server_args() raises."""
    import atom.config as atom_config_module

    monkeypatch.setattr(atom_config_module, "Config", _FakeConfig, raising=False)
    monkeypatch.setattr(
        atom_config_module, "ParallelConfig", _FakeParallelConfig, raising=False
    )
    monkeypatch.setattr(
        atom_config_module, "CompilationConfig", _FakeCompilationConfig, raising=False
    )

    mock_sglang_server_args = MagicMock()
    mock_sglang_server_args.get_global_server_args.side_effect = RuntimeError("boom")
    mock_sglang_server_args.PortArgs = _Obj
    mock_sglang_distributed = MagicMock()

    sgl_mods = _make_sglang_sys_modules(
        mock_sglang_server_args,
        mock_sglang_distributed,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    with patch.dict(sys.modules, sgl_mods):
        with pytest.raises(RuntimeError, match="Failed to retrieve"):
            plugin_config._generate_atom_config_from_sglang_config(
                config=_Obj(architectures=["DeepseekV3ForCausalLM"])
            )


# ---------------------------------------------------------------------------
# Config edge cases
# ---------------------------------------------------------------------------


def _run_sglang_config_test(
    monkeypatch,
    server_args_overrides=None,
    distributed_rank=0,
    tp_rank=0,
):
    """Helper: run _generate_atom_config_from_sglang_config with full mocks.

    Returns the resulting Config-like object for assertion.
    """
    import atom.config as atom_config_module

    monkeypatch.setattr(atom_config_module, "Config", _FakeConfig, raising=False)
    monkeypatch.setattr(
        atom_config_module, "ParallelConfig", _FakeParallelConfig, raising=False
    )
    monkeypatch.setattr(
        atom_config_module, "CompilationConfig", _FakeCompilationConfig, raising=False
    )

    fake_server_args = _make_fake_server_args(**(server_args_overrides or {}))

    mock_sglang_server_args = MagicMock()
    mock_sglang_server_args.get_global_server_args.return_value = fake_server_args

    mock_port_args_cls = MagicMock()
    mock_port_args_cls.init_new.return_value = _Obj()
    mock_sglang_server_args.PortArgs = mock_port_args_cls

    mock_sglang_distributed = MagicMock()
    mock_sglang_distributed.get_tensor_model_parallel_rank.return_value = tp_rank

    mock_sglang_model_config = MagicMock()
    mock_sglang_model_config.ModelConfig.from_server_args.return_value = _Obj(
        hf_config=_Obj()
    )
    mock_sglang_modelopt_config = MagicMock()
    mock_sglang_modelopt_config.ModelOptConfig.return_value = _Obj()
    mock_sglang_load_config = MagicMock()
    mock_sglang_load_config.LoadConfig.return_value = _Obj()

    sgl_mods = _make_sglang_sys_modules(
        mock_sglang_server_args,
        mock_sglang_distributed,
        mock_sglang_model_config,
        mock_sglang_modelopt_config,
        mock_sglang_load_config,
    )

    with (
        patch.dict(sys.modules, sgl_mods),
        patch("torch.distributed.get_rank", return_value=distributed_rank),
    ):
        return plugin_config._generate_atom_config_from_sglang_config(
            config=_Obj(architectures=["DeepseekV3ForCausalLM"])
        )


def test_sglang_config_expert_parallel_enabled(monkeypatch):
    """ep_size > 1 should set enable_expert_parallel=True."""
    cfg = _run_sglang_config_test(monkeypatch, {"ep_size": 4})
    assert cfg.enable_expert_parallel is True


def test_sglang_config_expert_parallel_disabled(monkeypatch):
    """ep_size == 1 should set enable_expert_parallel=False."""
    cfg = _run_sglang_config_test(monkeypatch, {"ep_size": 1})
    assert cfg.enable_expert_parallel is False


def test_sglang_config_dp_attention_propagates(monkeypatch):
    """enable_dp_attention=True should propagate to both plugin_config and top level."""
    cfg = _run_sglang_config_test(monkeypatch, {"enable_dp_attention": True})
    assert cfg.enable_dp_attention is True
    assert cfg.plugin_config.sglang_enable_dp_attention is True


def test_sglang_config_dp_attention_disabled(monkeypatch):
    """enable_dp_attention=False should propagate correctly."""
    cfg = _run_sglang_config_test(monkeypatch, {"enable_dp_attention": False})
    assert cfg.enable_dp_attention is False
    assert cfg.plugin_config.sglang_enable_dp_attention is False


def test_sglang_config_derives_data_parallel_rank(monkeypatch):
    """dp_size > 1 should derive ATOM's data_parallel_rank from TP-local rank."""
    cfg = _run_sglang_config_test(
        monkeypatch,
        {"tp_size": 8, "dp_size": 2},
        distributed_rank=5,
        tp_rank=1,
    )
    assert cfg.plugin_config.rank == 5
    assert cfg.parallel_config.data_parallel_size == 2
    assert cfg.parallel_config.data_parallel_rank == 0


def test_sglang_config_derives_data_parallel_rank_with_higher_tp_rank(monkeypatch):
    """Higher TP-local ranks should map into the correct DP shard."""
    cfg = _run_sglang_config_test(
        monkeypatch,
        {"tp_size": 8, "dp_size": 2},
        distributed_rank=13,
        tp_rank=5,
    )
    assert cfg.plugin_config.rank == 13
    assert cfg.parallel_config.data_parallel_size == 2
    assert cfg.parallel_config.data_parallel_rank == 1


def test_sglang_config_dist_init_addr_none(monkeypatch):
    """dist_init_addr=None should be stored in plugin_config."""
    cfg = _run_sglang_config_test(monkeypatch, {"dist_init_addr": None})
    assert cfg.plugin_config.sglang_dist_init_addr is None


def test_sglang_config_torch_compile_flags(monkeypatch):
    """Torch compile and cuda graph flags should propagate to plugin_config."""
    cfg = _run_sglang_config_test(
        monkeypatch,
        {"enable_torch_compile": True, "disable_cuda_graph": True},
    )
    assert cfg.plugin_config.sglang_enable_torch_compile is True
    assert cfg.plugin_config.sglang_disable_cuda_graph is True
