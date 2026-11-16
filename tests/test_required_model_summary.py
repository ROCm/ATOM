import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / ".github" / "scripts"

sys.path.insert(0, str(SCRIPTS))

from required_model_summary import (  # noqa: E402
    load_catalog_models,
    markdown_table,
    parse_workflow_matrix_text,
)


def test_catalog_models_filter_by_event_level(tmp_path):
    catalog = tmp_path / "models.json"
    catalog.write_text(
        json.dumps(
            [
                {"model_name": "pr-model", "test_level": "pr", "runner": "runner-a"},
                {"model_name": "main-model", "test_level": "main", "runner": "runner-b"},
                {"model_name": "nightly-model", "runner": "runner-c"},
            ]
        ),
        encoding="utf-8",
    )

    pr_models, pr_level, total = load_catalog_models(catalog, "pull_request")
    push_models, push_level, _ = load_catalog_models(catalog, "push")
    nightly_models, nightly_level, _ = load_catalog_models(catalog, "schedule")

    assert pr_level == "pr"
    assert [m["model_name"] for m in pr_models] == ["pr-model"]
    assert push_level == "main"
    assert [m["model_name"] for m in push_models] == ["pr-model", "main-model"]
    assert nightly_level == "nightly"
    assert [m["model_name"] for m in nightly_models] == [
        "pr-model",
        "main-model",
        "nightly-model",
    ]
    assert total == 3


def test_parse_workflow_matrix_text_handles_block_scalars():
    workflow = """
name: Example
jobs:
  other:
    runs-on: ubuntu-latest
  target-job:
    strategy:
      fail-fast: false
      matrix:
        include:
          - display_name: "Model A TP4"
            model_name: "model-a"
            model_path: "org/model-a"
            extra_args: "--flag value"
            env_vars: |
              A=1
              B=2
            accuracy_test_threshold: 0.91
            runner: linux-a
          - model_name: "Model B"
            runner: linux-b
  after:
    runs-on: ubuntu-latest
"""

    rows = parse_workflow_matrix_text(workflow, "target-job")

    assert rows == [
        {
            "display_name": "Model A TP4",
            "model_name": "model-a",
            "model_path": "org/model-a",
            "accuracy_test_threshold": 0.91,
            "runner": "linux-a",
        },
        {"model_name": "Model B", "runner": "linux-b"},
    ]


def test_required_summary_can_parse_current_inline_workflow_matrices():
    vllm_rows = parse_workflow_matrix_text(
        (REPO / ".github" / "workflows" / "atom-vllm-test.yaml").read_text(
            encoding="utf-8"
        ),
        "atom-vllm-oot",
    )
    sglang_rows = parse_workflow_matrix_text(
        (REPO / ".github" / "workflows" / "atom-sglang-test.yaml").read_text(
            encoding="utf-8"
        ),
        "atom-sglang-accuracy",
    )

    assert any(row.get("display_name") == "DeepSeek-V4-Flash TP4" for row in vllm_rows)
    assert any(row.get("model_name") == "GLM-5.2-FP8 TP4" for row in sglang_rows)


def test_markdown_table_escapes_pipe_characters():
    markdown = markdown_table(
        "Required models",
        [{"model_name": "A | B", "runner": "runner|one", "accuracy_threshold": 0.9}],
        source="source.yaml",
    )

    assert "A \\| B" in markdown
    assert "runner\\|one" in markdown
