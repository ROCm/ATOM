import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / ".github" / "benchmark" / "sglang_benchmark_models.json"
WORKFLOW = REPO / ".github" / "workflows" / "atom-sglang-benchmark.yaml"
MODEL_PREFIX = "glm-5-2-fp4-tp4-mtp3"
MODEL_DISPLAY = "GLM-5.2 FP4 TP4 MTP3"


def test_glm52_mtp3_benchmark_uses_required_spec_and_graph_settings():
    catalog = json.loads(CATALOG.read_text())
    model = next(model for model in catalog["models"] if model["prefix"] == MODEL_PREFIX)
    templates = catalog["templates"]["extra_args"]
    extra_args = " ".join(
        templates.get(arg, arg) for arg in model["extra_args"]
    )
    env_vars = "\n".join(
        catalog["templates"]["env_vars"].get(env, env) for env in model["env_vars"]
    )

    assert model["source_path"] == "amd/GLM-5.2-MXFP4"
    assert model["supported_input_output_pairs"] == ["1024x1024", "8192x1024"]
    assert model["supported_concurrency_values_by_pair"]["1024x1024"][-1] == 256
    assert model["supported_concurrency_values_by_pair"]["8192x1024"][-1] == 256
    assert "--enable-return-hidden-states" in extra_args
    assert "--max-running-requests 256" in extra_args
    assert "--cuda-graph-max-bs-decode 256" in extra_args
    assert "--speculative-algorithm EAGLE" in extra_args
    assert "--speculative-num-steps 3" in extra_args
    assert "--speculative-num-draft-tokens 4" in extra_args
    assert "--cuda-graph-backend-decode full" in extra_args
    assert "ATOM_GLM52_MTP_FUSED_INPUT=1" in env_vars
    assert "ATOM_SGLANG_V4_ENABLE_TARGET_VERIFY_CG=1" in env_vars


def test_glm52_mtp3_benchmark_is_selectable_in_the_workflow():
    workflow = WORKFLOW.read_text()

    assert f"- {MODEL_DISPLAY}" in workflow
    assert f'"{MODEL_DISPLAY}": "{MODEL_PREFIX}"' in workflow
    assert f'"{MODEL_PREFIX}",' in workflow
