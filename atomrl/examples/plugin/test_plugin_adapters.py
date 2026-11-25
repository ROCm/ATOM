# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import ast
import asyncio
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import tomllib
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_verl_entry_point_and_all_plugin_packages_are_declared():
    config = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert config["project"]["entry-points"]["verl.plugins"] == {
        "atomrl": "atomrl.plugin.verl.register:register"
    }

    for framework in ("verl", "vime", "miles", "slime"):
        package = PROJECT_ROOT / "atomrl" / "plugin" / framework
        assert (package / "__init__.py").is_file()
        assert (package / "lifecycle.py").is_file()


def test_plugin_examples_include_miles_without_restoring_legacy_trees():
    manifest = yaml.safe_load(
        (PROJECT_ROOT / "examples" / "plugin" / "examples.yaml").read_text(
            encoding="utf-8"
        )
    )
    ids = {example["id"] for example in manifest["examples"]}
    assert "miles-plugin-qwen3-8b" in ids
    assert "slime-plugin-contract" in ids
    assert not any("trl" in example_id for example_id in ids)
    assert not (PROJECT_ROOT / "lumen" / "examples" / "rl" / "trl").exists()
    assert not (PROJECT_ROOT / "lumen" / "examples" / "rl" / "verl").exists()


def test_plugin_docker_sources_require_explicit_revisions():
    dockerfile = (PROJECT_ROOT / "examples" / "plugin" / "Dockerfile.plugin").read_text(
        encoding="utf-8"
    )
    for revision in (
        "0.9.1",
        "8707a5a7156223ce1d92c65279e2eed748addb82",
        "fe7037f561910f26ac4eccf9082bc5f4410488f8",
        "8e43d38528aac22fef29d30db7a709041bf257cc",
        "79aa41a884ab0bf65222c94418f6357d3c31192b",
        "rocm/atom-dev:sglang-latest",
    ):
        assert revision in dockerfile
    assert (
        "pip install --no-build-isolation --no-deps -e /workspace/ATOM/atomrl"
        in dockerfile
    )
    test_script = (PROJECT_ROOT / "examples" / "plugin" / "test.sh").read_text(
        encoding="utf-8"
    )
    assert "--build-arg" not in test_script


def test_lumenrl_plugin_provenance_is_preserved():
    notice = (PROJECT_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "1e42e9238517e1675447f21c9d90c31959b4e488" in notice
    assert "Copyright (c) 2026 ZhangDanyang-AMD" in notice
    for relative in (
        "atomrl/plugin/_lifecycle.py",
        "atomrl/plugin/_megatron_hooks.py",
        "atomrl/plugin/_sglang_host_rollout.py",
        "atomrl/plugin/verl/lifecycle.py",
        "atomrl/plugin/vime/hooks.py",
        "atomrl/plugin/vime/lifecycle.py",
        "atomrl/plugin/miles/hooks.py",
        "atomrl/plugin/miles/lifecycle.py",
        "atomrl/plugin/slime/hooks.py",
        "atomrl/plugin/slime/lifecycle.py",
        "examples/plugin/Dockerfile.plugin",
        "examples/plugin/test.sh",
        "examples/plugin/verify_installed_plugin.py",
        "examples/plugin/vime/Dockerfile",
        "examples/plugin/vime/run.sh",
        "examples/plugin/miles/Dockerfile",
        "examples/plugin/miles/run.sh",
        "examples/plugin/slime/run.sh",
    ):
        header = "\n".join(
            (PROJECT_ROOT / relative).read_text(encoding="utf-8").splitlines()[:8]
        )
        assert "Copyright (c) 2026 ZhangDanyang-AMD" in header, relative


@pytest.mark.parametrize(
    "module_name",
    (
        "atomrl.plugin.vime.hooks",
        "atomrl.plugin.miles.hooks",
        "atomrl.plugin.slime.hooks",
    ),
)
def test_megatron_hook_modules_import_without_optional_framework(module_name):
    framework = module_name.split(".")[-2]
    for name in tuple(sys.modules):
        if name == framework or name.startswith(f"{framework}."):
            sys.modules.pop(name)

    module = importlib.import_module(module_name)
    assert callable(module.custom_megatron_init)
    assert callable(module.custom_model_provider)
    assert callable(module.generate_rollout)
    assert callable(module.custom_update_weight_post_write)


class _FakeManager:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.created = False
        self.stopped = False
        self.reloaded = []
        self.num_replicas = 2
        self.servers = [object(), object()]
        self.__class__.instances.append(self)

    def create(self):
        self.created = True

    def shutdown(self):
        self.stopped = True

    def reload_weights_from_path(self, path):
        assert Path(path).exists()
        self.reloaded.append(str(path))

    def wake_all(self, tags=None):
        self.wake_tags = tags

    def sleep_all(self):
        self.slept = True


@pytest.mark.parametrize(
    "framework,tp_attr,tp_value,max_attr,max_value",
    (
        ("vime", "tensor_model_parallel_size", 2, "max_num_seqs", 64),
        ("miles", "tensor_model_parallel_size", 4, "max_num_seqs", 96),
        ("slime", "sglang_tp_size", 8, "sglang_server_concurrency", 128),
    ),
)
def test_framework_lifecycle_uses_public_atom_manager(
    monkeypatch, framework, tp_attr, tp_value, max_attr, max_value
):
    atom_server = importlib.import_module("atomrl.engine.inference.atom_ray_server")
    monkeypatch.setattr(atom_server, "ATOMReplicaManager", _FakeManager)
    _FakeManager.instances.clear()

    args = SimpleNamespace(
        hf_checkpoint="/models/qwen",
        rollout_max_prompt_len=1024,
        rollout_max_response_len=2048,
        dtype="bfloat16",
        seed=17,
        **{tp_attr: tp_value, max_attr: max_value},
    )
    lifecycle = importlib.import_module(f"atomrl.plugin.{framework}.lifecycle")
    manager = lifecycle.setup_atom_manager(args, actor_wg=SimpleNamespace())

    assert manager is args._atomrl_atom_manager
    assert manager.created
    assert manager.kwargs["model_name"] == "/models/qwen"
    assert manager.kwargs["engine_kwargs"]["tensor_parallel_size"] == tp_value
    assert manager.kwargs["engine_kwargs"]["max_num_seqs"] == max_value
    assert manager.kwargs["engine_kwargs"]["max_model_len"] == 3072

    lifecycle.teardown_atom_manager(args)
    assert manager.stopped
    assert args._atomrl_atom_manager is None


def test_lifecycle_cleans_up_when_create_or_injection_fails(monkeypatch):
    lifecycle = importlib.import_module("atomrl.plugin._lifecycle")

    class FailingCreateManager(_FakeManager):
        def create(self):
            self.created = True
            raise RuntimeError("launch failed")

    atom_server = importlib.import_module("atomrl.engine.inference.atom_ray_server")
    monkeypatch.setattr(atom_server, "ATOMReplicaManager", FailingCreateManager)
    with pytest.raises(RuntimeError, match="launch failed"):
        lifecycle.setup_args_manager(
            SimpleNamespace(hf_checkpoint="/models/qwen"),
            SimpleNamespace(),
            framework="vime",
        )
    assert FailingCreateManager.instances[-1].stopped

    monkeypatch.setattr(atom_server, "ATOMReplicaManager", _FakeManager)
    adapter = SimpleNamespace(
        set_manager=lambda manager: (_ for _ in ()).throw(RuntimeError("inject failed"))
    )
    with pytest.raises(RuntimeError, match="inject failed"):
        lifecycle.setup_adapter_manager(
            adapter,
            SimpleNamespace(),
            "/models/qwen",
            {},
        )
    assert _FakeManager.instances[-1].stopped


@pytest.mark.parametrize("framework", ("vime", "miles", "slime"))
def test_checkpoint_reload_runs_on_rank_zero_only(monkeypatch, framework):
    hooks = importlib.import_module(f"atomrl.plugin.{framework}.hooks")
    manager = _FakeManager()
    args = SimpleNamespace(_atomrl_atom_manager=manager)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    hooks.custom_update_weight_post_write(args, "/weights/v1", [])
    assert manager.reloaded == []

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    hooks.custom_update_weight_post_write(args, "/weights/v1", [])
    assert manager.reloaded == ["/weights/v1"]


@pytest.mark.parametrize("framework", ("vime", "miles", "slime"))
def test_eval_rollout_returns_framework_native_type(monkeypatch, framework):
    hooks = importlib.import_module(f"atomrl.plugin.{framework}.hooks")
    result_type = type(f"{framework.title()}EvalOutput", (), {})

    base_types = types.ModuleType(f"{framework}.rollout.base_types")
    base_types.RolloutFnEvalOutput = lambda data: result_type()
    rollout = types.ModuleType(f"{framework}.rollout")
    rollout.base_types = base_types
    package = types.ModuleType(framework)
    package.__path__ = []
    package.rollout = rollout
    monkeypatch.setitem(sys.modules, framework, package)
    monkeypatch.setitem(sys.modules, f"{framework}.rollout", rollout)
    monkeypatch.setitem(sys.modules, f"{framework}.rollout.base_types", base_types)

    args = SimpleNamespace(_atomrl_atom_manager=_FakeManager())
    output = hooks.generate_rollout(
        args,
        rollout_id=3,
        data_source=SimpleNamespace(),
        evaluation=True,
    )
    assert isinstance(output, result_type)


@pytest.mark.parametrize("framework", ("vime", "miles", "slime"))
def test_training_rollout_converts_chat_samples_and_always_sleeps(
    monkeypatch, framework
):
    hooks = importlib.import_module(f"atomrl.plugin.{framework}.hooks")

    class Sample:
        class Status:
            PENDING = "pending"
            COMPLETED = "completed"
            TRUNCATED = "truncated"

        def __init__(self, index):
            self.prompt = [{"role": "user", "content": "hello"}]
            self.tokens = []
            self.response = ""
            self.response_length = 0
            self.rollout_log_probs = None
            self.status = self.Status.PENDING
            self.rollout_id = None
            self.index = index

    class Tokenizer:
        def apply_chat_template(self, prompt, **kwargs):
            assert prompt[0]["content"] == "hello"
            return [11, 12]

        def encode(self, prompt):
            raise AssertionError("chat prompts must use apply_chat_template")

        def decode(self, tokens, **kwargs):
            return f"decoded:{tokens}"

    class RemoteMethod:
        def __init__(self, fn):
            self.fn = fn

        def remote(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    outputs = [
        {"token_ids": [21, 22, 23], "logprobs": [-1.0] * 3, "text": ""},
        {"token_ids": [31], "logprobs": [-0.5], "text": "done"},
    ]
    output_iter = iter(outputs)

    def unexpected_generate_group(*args, **kwargs):
        raise AssertionError("host data source already expands sample groups")

    server = SimpleNamespace(
        generate=RemoteMethod(lambda **kwargs: next(output_iter)),
        generate_group=RemoteMethod(unexpected_generate_group),
    )
    manager = _FakeManager()
    manager.servers = [server]
    args = SimpleNamespace(
        _atomrl_atom_manager=manager,
        hf_checkpoint="/models/qwen",
        rollout_batch_size=1,
        rollout_max_response_len=3,
        n_samples_per_prompt=2,
    )

    package = types.ModuleType(framework)
    package.__path__ = []
    rollout = types.ModuleType(f"{framework}.rollout")
    rollout.__path__ = []
    base_types = types.ModuleType(f"{framework}.rollout.base_types")
    base_types.RolloutFnTrainOutput = lambda samples: SimpleNamespace(samples=samples)
    base_types.RolloutFnEvalOutput = lambda data: SimpleNamespace(data=data)
    utils = types.ModuleType(f"{framework}.utils")
    utils.__path__ = []
    processing = types.ModuleType(f"{framework}.utils.processing_utils")
    processing.load_tokenizer = lambda *args, **kwargs: Tokenizer()
    framework_types = types.ModuleType(f"{framework}.utils.types")
    framework_types.Sample = Sample
    for name, module in (
        (framework, package),
        (f"{framework}.rollout", rollout),
        (f"{framework}.rollout.base_types", base_types),
        (f"{framework}.utils", utils),
        (f"{framework}.utils.processing_utils", processing),
        (f"{framework}.utils.types", framework_types),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    ray = types.ModuleType("ray")
    ray.get = lambda value: value
    monkeypatch.setitem(sys.modules, "ray", ray)
    samples = [Sample(101), Sample(102)] if framework == "slime" else [Sample(101)]
    data_source = SimpleNamespace(get_samples=lambda selector: [samples])

    result = hooks.generate_rollout(args, 7, data_source, evaluation=False)
    generated = result.samples[0]
    expected = 2 if framework == "slime" else 1
    assert len(generated) == expected
    assert generated[0].tokens == [11, 12, 21, 22, 23]
    assert generated[0].status == Sample.Status.TRUNCATED
    if framework == "slime":
        assert generated[1].status == Sample.Status.COMPLETED
        assert generated[0] is samples[0]
        assert generated[1] is samples[1]
    if framework in {"miles", "slime"}:
        assert [item.rollout_id for item in generated] == [
            item.index for item in samples
        ]
    else:
        assert generated[0].rollout_id == 7
    assert manager.wake_tags == ["weights", "kv_cache"]
    assert manager.slept


def test_rollout_sleeps_when_generation_raises(monkeypatch):
    hooks = importlib.import_module("atomrl.plugin.slime.hooks")

    class RemoteMethod:
        def remote(self, *args, **kwargs):
            raise RuntimeError("generation failed")

    manager = _FakeManager()
    manager.servers = [
        SimpleNamespace(generate=RemoteMethod(), generate_group=RemoteMethod())
    ]
    args = SimpleNamespace(
        _atomrl_atom_manager=manager,
        hf_checkpoint="/models/qwen",
        rollout_batch_size=1,
        rollout_max_response_len=3,
        n_samples_per_prompt=2,
    )
    sample = SimpleNamespace(prompt="hello", tokens=[1], index=1)
    data_source = SimpleNamespace(get_samples=lambda count: [[sample]])

    package = types.ModuleType("slime")
    package.__path__ = []
    rollout = types.ModuleType("slime.rollout")
    rollout.__path__ = []
    base_types = types.ModuleType("slime.rollout.base_types")
    base_types.RolloutFnTrainOutput = lambda samples: SimpleNamespace(samples=samples)
    utils = types.ModuleType("slime.utils")
    utils.__path__ = []
    processing = types.ModuleType("slime.utils.processing_utils")
    processing.load_tokenizer = lambda *args, **kwargs: SimpleNamespace()
    framework_types = types.ModuleType("slime.utils.types")
    framework_types.Sample = SimpleNamespace
    for name, module in (
        ("slime", package),
        ("slime.rollout", rollout),
        ("slime.rollout.base_types", base_types),
        ("slime.utils", utils),
        ("slime.utils.processing_utils", processing),
        ("slime.utils.types", framework_types),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    ray = types.ModuleType("ray")
    ray.get = lambda value: value
    monkeypatch.setitem(sys.modules, "ray", ray)

    with pytest.raises(RuntimeError, match="generation failed"):
        hooks.generate_rollout(args, 1, data_source)
    assert manager.slept


def test_verl_weight_sync_uses_scoped_checkpoint(monkeypatch):
    register = importlib.import_module("atomrl.plugin.verl.register")
    adapter = register.ATOMRolloutAdapter.__new__(register.ATOMRolloutAdapter)
    adapter._manager = _FakeManager()
    adapter._weight_version = 0
    written_paths = []

    safetensors_torch = types.ModuleType("safetensors.torch")

    def save_file(state_dict, path):
        assert set(state_dict) == {"weight"}
        Path(path).touch()
        written_paths.append(Path(path))

    safetensors_torch.save_file = save_file
    safetensors = types.ModuleType("safetensors")
    safetensors.__path__ = []
    safetensors.torch = safetensors_torch
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)

    asyncio.run(
        adapter.update_weights(
            iter((("weight", torch.ones(2)),)),
            wire_format="named_tensors",
        )
    )

    assert adapter._manager.reloaded
    assert written_paths and not written_paths[0].parent.exists()


def test_verl_lifecycle_injects_and_detaches_manager(monkeypatch):
    atom_server = importlib.import_module("atomrl.engine.inference.atom_ray_server")
    monkeypatch.setattr(atom_server, "ATOMReplicaManager", _FakeManager)
    lifecycle = importlib.import_module("atomrl.plugin.verl.lifecycle")
    adapter = SimpleNamespace(_manager=None, set_manager=lambda manager: None)
    injected = []
    adapter.set_manager = injected.append

    manager = lifecycle.setup_atom_manager(
        adapter,
        actor_wg=SimpleNamespace(),
        model_path="/models/qwen",
        engine_kwargs={"tensor_parallel_size": 2},
    )
    adapter._manager = manager
    assert injected == [manager]
    assert manager.created

    lifecycle.teardown_atom_manager(adapter)
    assert manager.stopped
    assert adapter._manager is None


def test_legacy_replica_abort_signature_accepts_reject_request():
    source = (
        PROJECT_ROOT / "atomrl" / "plugin" / "verl" / "atom_replica.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "abort_all_requests"
    )
    assert [arg.arg for arg in method.args.args][-1] == "reject_request"
    assert isinstance(method.args.defaults[-1], ast.Constant)
    assert method.args.defaults[-1].value is False
