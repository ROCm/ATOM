import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


ATOM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ATOM_ROOT / ".github/scripts/atom_vllm_benchmark_trigger_server.py"


def _load_module():
    assert SCRIPT_PATH.exists(), "Expected local trigger server script to exist."
    spec = importlib.util.spec_from_file_location(
        "atom_vllm_benchmark_trigger_server", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_request():
    return {
        "repository": "ROCm/ATOM",
        "ref": "zejun/refine_atom_vllm_benchmark_0509",
        "benchmark_client": "InferenceMax bench",
        "model_selections": [
            {"family": "Qwen3.5-397B-A17B-FP8", "tp_sizes": [4, 8]},
            {"family": "MiniMax-M2.5", "tp_sizes": [2]},
        ],
        "isl_osl_pairs": [[1024, 1024], [1000, 100]],
        "concurrency_values": [4, 8],
        "random_range_ratios": ["0.8", "1.0"],
        "oot_image": "",
        "publish_to_dashboard": False,
        "upload_to_custom_dashboard": True,
    }


def test_build_workflow_inputs_maps_selections_into_existing_slots():
    module = _load_module()
    payload = _sample_request()

    inputs = module.build_workflow_inputs(payload, module.load_catalog())

    assert inputs["manual_selection_guide"] == module.MANUAL_SELECTION_GUIDE_DEFAULT
    assert inputs["benchmark_client"] == "InferenceMax bench"
    assert inputs["model_slot_1"] == "Qwen3.5-397B-A17B-FP8"
    assert inputs["tp_sizes_slot_1"] == "4,8"
    assert inputs["model_slot_2"] == "MiniMax-M2.5"
    assert inputs["tp_sizes_slot_2"] == "2"
    assert inputs["model_slot_8"] == "none"
    assert inputs["isl_osl_pairs"] == "1024,1024;1000,100"
    assert inputs["concurrency_values"] == "4,8"
    assert inputs["random_range_ratios"] == "0.8,1.0"
    assert inputs["publish_to_dashboard"] == "false"
    assert inputs["upload_to_custom_dashboard"] == "true"


def test_build_workflow_inputs_rejects_more_than_eight_model_selections():
    module = _load_module()
    payload = _sample_request()
    payload["model_selections"] = [
        {"family": "DeepSeek-R1 FP8", "tp_sizes": [8]},
        {"family": "DeepSeek-R1 MXFP4", "tp_sizes": [8]},
        {"family": "DeepSeek-V3.2 FP8", "tp_sizes": [8]},
        {"family": "MiniMax-M2.5", "tp_sizes": [2]},
        {"family": "gpt-oss-120b", "tp_sizes": [1]},
        {"family": "GLM-5.1-FP8", "tp_sizes": [8]},
        {"family": "Kimi-K2-Thinking-MXFP4", "tp_sizes": [4]},
        {"family": "Kimi-K2.5-MXFP4", "tp_sizes": [2]},
        {"family": "Qwen3.5-397B-A17B-FP8", "tp_sizes": [4]},
    ]

    with pytest.raises(ValueError, match="supports at most 8 model selections"):
        module.build_workflow_inputs(payload, module.load_catalog())


def test_build_workflow_inputs_rejects_unsupported_tp_selection():
    module = _load_module()
    payload = _sample_request()
    payload["model_selections"] = [{"family": "DeepSeek-R1 FP8", "tp_sizes": [4]}]

    with pytest.raises(
        ValueError, match="DeepSeek-R1 FP8 does not support TP sizes 4"
    ):
        module.build_workflow_inputs(payload, module.load_catalog())


def test_build_gh_workflow_run_command_uses_existing_workflow_inputs():
    module = _load_module()
    payload = _sample_request()
    inputs = module.build_workflow_inputs(payload, module.load_catalog())

    command = module.build_gh_workflow_run_command(
        repository=payload["repository"],
        ref=payload["ref"],
        inputs=inputs,
    )

    assert command[:5] == [
        "gh",
        "workflow",
        "run",
        "atom-vllm-benchmark.yaml",
        "--repo",
    ]
    assert "ROCm/ATOM" in command
    assert "--ref" in command
    assert "zejun/refine_atom_vllm_benchmark_0509" in command
    assert "-f" in command
    assert "benchmark_client=InferenceMax bench" in command
    assert "model_slot_1=Qwen3.5-397B-A17B-FP8" in command
    assert "tp_sizes_slot_1=4,8" in command


def test_catalog_for_ui_exposes_supported_tp_sizes():
    module = _load_module()

    families = module.catalog_for_ui(module.load_catalog())
    qwen = next(family for family in families if family["family"] == "Qwen3.5-397B-A17B-FP8")

    assert qwen["supported_tp_sizes"] == [4, 8]


def test_dispatch_workflow_runs_gh_command(monkeypatch):
    module = _load_module()
    payload = _sample_request()

    completed = Mock(returncode=0, stdout="triggered", stderr="")
    run_mock = Mock(return_value=completed)
    monkeypatch.setattr(module.subprocess, "run", run_mock)

    result = module.dispatch_workflow(payload, module.load_catalog())

    assert result["stdout"] == "triggered"
    invoked_command = run_mock.call_args.args[0]
    assert invoked_command[:4] == ["gh", "workflow", "run", "atom-vllm-benchmark.yaml"]
    assert "model_slot_1=Qwen3.5-397B-A17B-FP8" in invoked_command
