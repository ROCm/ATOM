import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / ".github" / "benchmark" / "sglang_models_accuracy.json"
WORKFLOW = REPO / ".github" / "workflows" / "atom-sglang-accuracy-validation.yaml"

TP4_AITER_MODELS = {
    "DeepSeek-R1-FP8 TP4",
    "DeepSeek-R1-FP8 TP4 Online Quant",
    "DeepSeek-R1-FP4 TP4",
}


def _catalog_rows() -> dict[str, dict]:
    rows = json.loads(CATALOG.read_text())
    return {row["model_name"]: row for row in rows}


def _catalog_args() -> dict[str, str]:
    return {name: row["extraArgs"] for name, row in _catalog_rows().items()}


def _workflow_block(model_name: str) -> str:
    lines = WORKFLOW.read_text().splitlines()
    marker = f'"model_name": "{model_name}"'
    model_line = next(i for i, line in enumerate(lines) if marker in line)
    return "\n".join(lines[model_line : model_line + 10])


def _workflow_args(model_name: str) -> str:
    block = _workflow_block(model_name).splitlines()
    args_line = next(line for line in block[1:] if '"extra_args":' in line)
    return args_line


def test_deepseek_r1_tp4_accuracy_uses_aiter_attention():
    catalog_args = _catalog_args()
    for model_name in TP4_AITER_MODELS:
        assert "--attention-backend aiter" in catalog_args[model_name]
        assert "--attention-backend triton" not in catalog_args[model_name]

        workflow_args = _workflow_args(model_name)
        assert "--attention-backend aiter" in workflow_args
        assert "--attention-backend triton" not in workflow_args


def test_deepseek_r1_tp8_accuracy_keeps_aiter_attention():
    model_name = "DeepSeek-R1-FP8 TP8"
    assert "--attention-backend aiter" in _catalog_args()[model_name]
    assert "--attention-backend aiter" in _workflow_args(model_name)


def test_deepseek_r1_fp4_tp4_accuracy_loads_experts_serially():
    model_name = "DeepSeek-R1-FP4 TP4"
    assert "ATOM_LOADER_NUM_THREADS=1" in _catalog_rows()[model_name]["env_vars"]
    assert "ATOM_LOADER_NUM_THREADS=1" in _workflow_block(model_name)
