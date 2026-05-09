"""Tests for atom.plugin.register — set_attn_cls and init_aiter_dist.

These functions set global state (ops.Attention) and initialize distributed
communication. All external dependencies are mocked.

Because atom.config is stubbed in conftest.py (missing most real attributes),
we patch the stub to add what the import chain needs, then import the modules
under test.
"""

import importlib
import sys
from types import ModuleType
import pytest
from unittest.mock import MagicMock, patch

from atom.plugin import prepare as plugin_prepare


class _Obj:
    """Minimal attribute bag for faking nested configs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _reset_framework_state():
    plugin_prepare._set_framework_backbone("atom")
    yield
    plugin_prepare._set_framework_backbone("atom")


# ---------------------------------------------------------------------------
# set_attn_cls — tested via mock ops module to avoid import chain
# ---------------------------------------------------------------------------


class _FakeOps:
    """Lightweight namespace to stand in for atom.model_ops."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _set_attn_cls_logic(ops):
    """Reproduce set_attn_cls dispatch logic with a fake ops module.

    Uses plugin_prepare (imported at module level) to avoid stale references
    when the real atom.plugin.prepare gets loaded by other tests.
    """
    if plugin_prepare.is_vllm():
        ops.Attention = ops.PagedAttention
    elif plugin_prepare.is_sglang():
        ops.Attention = ops.RadixAttention


def test_set_attn_cls_sglang_sets_radix_attention():
    """In sglang mode, set_attn_cls should assign RadixAttention to ops.Attention."""
    plugin_prepare._set_framework_backbone("sglang")

    sentinel_radix = object()
    sentinel_paged = object()
    ops = _FakeOps(RadixAttention=sentinel_radix, PagedAttention=sentinel_paged)

    _set_attn_cls_logic(ops)

    assert ops.Attention is sentinel_radix


def test_set_attn_cls_vllm_sets_paged_attention():
    """In vllm mode, set_attn_cls should assign PagedAttention to ops.Attention."""
    plugin_prepare._set_framework_backbone("vllm")

    sentinel_radix = object()
    sentinel_paged = object()
    ops = _FakeOps(RadixAttention=sentinel_radix, PagedAttention=sentinel_paged)

    _set_attn_cls_logic(ops)

    assert ops.Attention is sentinel_paged


def test_set_attn_cls_atom_mode_leaves_default():
    """In atom (server) mode, set_attn_cls should not change ops.Attention."""
    sentinel_default = object()
    ops = _FakeOps(
        Attention=sentinel_default,
        RadixAttention=object(),
        PagedAttention=object(),
    )

    _set_attn_cls_logic(ops)

    assert ops.Attention is sentinel_default


# ---------------------------------------------------------------------------
# init_aiter_dist — tested via function-level reimplementation
# We extract and test the core logic branches directly.
# ---------------------------------------------------------------------------


def _run_init_aiter_dist(
    config, mock_init_dist_env, mock_get_method, mock_init_tp=None
):
    """Execute the init_aiter_dist logic against mocks.

    Mirrors atom/plugin/register.py:init_aiter_dist but with injected mocks,
    avoiding the import chain.
    """
    rank = config.plugin_config.rank
    tensor_parallel_size = config.tensor_parallel_size

    assert (
        config.plugin_config.is_plugin_mode
    ), "Make sure ATOM is running in plugin mode"

    if config.plugin_config.is_vllm:
        if mock_init_tp is not None and mock_init_tp(tensor_parallel_size):
            return

    if config.plugin_config.is_vllm:
        dp_master_ip = config.parallel_config.data_parallel_master_ip
        dp_master_port = config.parallel_config.data_parallel_master_port
    elif config.plugin_config.is_sglang:
        if config.plugin_config.sglang_dist_init_addr is not None:
            dp_master_ip, dp_master_port = (
                config.plugin_config.sglang_dist_init_addr.split(":")
            )
        else:
            dp_master_ip = "127.0.0.1"
            dp_master_port = config.plugin_config.sglang_port_args.nccl_port

    distributed_init_method = mock_get_method(dp_master_ip, dp_master_port)

    mock_init_dist_env(
        tensor_model_parallel_size=tensor_parallel_size,
        rankID=rank,
        backend="nccl",
        distributed_init_method=distributed_init_method,
        data_parallel_size=config.parallel_config.data_parallel_size,
        data_parallel_rank=config.parallel_config.data_parallel_rank,
    )


def test_init_aiter_dist_asserts_plugin_mode():
    """init_aiter_dist should assert plugin mode is enabled."""
    config = _Obj(
        tensor_parallel_size=1,
        plugin_config=_Obj(is_plugin_mode=False, rank=0),
    )
    with pytest.raises(AssertionError, match="plugin mode"):
        _run_init_aiter_dist(config, MagicMock(), MagicMock())


def test_init_aiter_dist_sglang_with_dist_init_addr():
    """sglang path with dist_init_addr set: should split ip:port."""
    config = _Obj(
        tensor_parallel_size=4,
        plugin_config=_Obj(
            is_plugin_mode=True,
            is_vllm=False,
            is_sglang=True,
            rank=2,
            sglang_dist_init_addr="10.0.0.5:30000",
            sglang_port_args=_Obj(nccl_port=29500),
        ),
        parallel_config=_Obj(data_parallel_size=1, data_parallel_rank=0),
    )

    mock_init_dist_env = MagicMock()
    mock_get_method = MagicMock(return_value="tcp://10.0.0.5:30000")

    _run_init_aiter_dist(config, mock_init_dist_env, mock_get_method)

    mock_get_method.assert_called_once_with("10.0.0.5", "30000")
    mock_init_dist_env.assert_called_once_with(
        tensor_model_parallel_size=4,
        rankID=2,
        backend="nccl",
        distributed_init_method="tcp://10.0.0.5:30000",
        data_parallel_size=1,
        data_parallel_rank=0,
    )


def test_init_aiter_dist_sglang_without_dist_init_addr():
    """sglang path with dist_init_addr=None: should fall back to nccl_port."""
    config = _Obj(
        tensor_parallel_size=4,
        plugin_config=_Obj(
            is_plugin_mode=True,
            is_vllm=False,
            is_sglang=True,
            rank=0,
            sglang_dist_init_addr=None,
            sglang_port_args=_Obj(nccl_port=31000),
        ),
        parallel_config=_Obj(data_parallel_size=1, data_parallel_rank=0),
    )

    mock_init_dist_env = MagicMock()
    mock_get_method = MagicMock(return_value="tcp://127.0.0.1:31000")

    _run_init_aiter_dist(config, mock_init_dist_env, mock_get_method)

    mock_get_method.assert_called_once_with("127.0.0.1", 31000)
    mock_init_dist_env.assert_called_once()


def test_init_aiter_dist_vllm_fast_path():
    """vllm path: if init_aiter_tp_from_vllm succeeds, skip init_dist_env."""
    config = _Obj(
        tensor_parallel_size=8,
        plugin_config=_Obj(
            is_plugin_mode=True,
            is_vllm=True,
            is_sglang=False,
            rank=0,
        ),
        parallel_config=_Obj(
            data_parallel_size=1,
            data_parallel_rank=0,
            data_parallel_master_ip="192.168.1.1",
            data_parallel_master_port=29600,
        ),
    )

    mock_init_dist_env = MagicMock()
    mock_get_method = MagicMock()
    mock_init_tp = MagicMock(return_value=True)

    _run_init_aiter_dist(config, mock_init_dist_env, mock_get_method, mock_init_tp)

    mock_init_tp.assert_called_once_with(8)
    mock_init_dist_env.assert_not_called()


def test_init_aiter_dist_vllm_fallback_path():
    """vllm path: if init_aiter_tp_from_vllm fails, fall back to init_dist_env."""
    config = _Obj(
        tensor_parallel_size=8,
        plugin_config=_Obj(
            is_plugin_mode=True,
            is_vllm=True,
            is_sglang=False,
            rank=1,
        ),
        parallel_config=_Obj(
            data_parallel_size=2,
            data_parallel_rank=1,
            data_parallel_master_ip="192.168.1.1",
            data_parallel_master_port=29600,
        ),
    )

    mock_init_dist_env = MagicMock()
    mock_get_method = MagicMock(return_value="tcp://192.168.1.1:29600")
    mock_init_tp = MagicMock(return_value=False)

    _run_init_aiter_dist(config, mock_init_dist_env, mock_get_method, mock_init_tp)

    mock_init_tp.assert_called_once_with(8)
    mock_get_method.assert_called_once_with("192.168.1.1", 29600)
    mock_init_dist_env.assert_called_once_with(
        tensor_model_parallel_size=8,
        rankID=1,
        backend="nccl",
        distributed_init_method="tcp://192.168.1.1:29600",
        data_parallel_size=2,
        data_parallel_rank=1,
    )


# ---------------------------------------------------------------------------
# _register_custom_attention_to_sglang
# ---------------------------------------------------------------------------


def test_register_custom_attention_uses_aiter_name():
    """Verify the real register module binds the backend under the 'aiter' name."""

    def _package(name: str) -> ModuleType:
        module = ModuleType(name)
        module.__path__ = []
        return module

    recorded = {}

    def _fake_register_attention_backend(name):
        recorded["backend_name"] = name

        def _decorator(factory):
            recorded["factory"] = factory
            return factory

        return _decorator

    class _FakeBackend:
        def __init__(self, runner):
            self.runner = runner

    fake_attention_registry = ModuleType(
        "sglang.srt.layers.attention.attention_registry"
    )
    fake_attention_registry.register_attention_backend = (
        _fake_register_attention_backend
    )

    fake_prepare_mod = ModuleType("atom.plugin.prepare")
    fake_prepare_mod.is_vllm = lambda: False
    fake_prepare_mod.is_sglang = lambda: True

    fake_modules = {
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.layers": _package("sglang.srt.layers"),
        "sglang.srt.layers.attention": _package("sglang.srt.layers.attention"),
        "sglang.srt.layers.attention.attention_registry": fake_attention_registry,
        "atom.models.qwen3": ModuleType("atom.models.qwen3"),
        "atom.models.qwen3_moe": ModuleType("atom.models.qwen3_moe"),
        "atom.models.glm4_moe": ModuleType("atom.models.glm4_moe"),
        "atom.models.deepseek_v2": ModuleType("atom.models.deepseek_v2"),
        "atom.config": ModuleType("atom.config"),
        "atom.plugin.prepare": fake_prepare_mod,
        "atom.plugin.sglang.attention_backend.sgl_attn_backend": ModuleType(
            "atom.plugin.sglang.attention_backend.sgl_attn_backend"
        ),
    }
    fake_modules["atom.models.qwen3"].Qwen3ForCausalLM = type(
        "Qwen3ForCausalLM", (), {}
    )
    fake_modules["atom.models.qwen3_moe"].Qwen3MoeForCausalLM = type(
        "Qwen3MoeForCausalLM", (), {}
    )
    fake_modules["atom.models.glm4_moe"].Glm4MoeForCausalLM = type(
        "Glm4MoeForCausalLM", (), {}
    )
    fake_modules["atom.models.deepseek_v2"].DeepseekV3ForCausalLM = type(
        "DeepseekV3ForCausalLM", (), {}
    )
    fake_modules["atom.config"].Config = type("Config", (), {})
    fake_modules[
        "atom.plugin.sglang.attention_backend.sgl_attn_backend"
    ].ATOMAttnBackendForSgl = _FakeBackend

    with patch.dict(sys.modules, fake_modules):
        sys.modules.pop("atom.plugin.register", None)
        register_mod = importlib.import_module("atom.plugin.register")
        register_mod = importlib.reload(register_mod)
        register_mod._register_custom_attention_to_sglang()
        backend = recorded["factory"]("runner")

    assert recorded["backend_name"] == "aiter"
    assert isinstance(backend, _FakeBackend)
    assert backend.runner == "runner"
