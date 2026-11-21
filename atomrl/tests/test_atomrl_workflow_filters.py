from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "atomrl-unit-tests.yaml"


def _filter_block(source: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^            {re.escape(name)}:\n(?P<body>.*?)(?=^            \w+:|\Z)",
        source,
    )
    assert match is not None, f"missing filter {name!r}"
    return match.group("body")


def test_atomrl_workflow_uses_exclusive_path_filters() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "predicate-quantifier: 'every'" in source

    atomrl = _filter_block(source, "atomrl")
    assert "              - 'atomrl/**'" in atomrl
    assert "              - '!atomrl/lumen/**'" in atomrl
    assert "              - '!atomrl/atomrl/lumen/**'" in atomrl
    assert "              - '!atomrl/tests/lumen/**'" in atomrl

    assert "              - 'atomrl/lumen/**'" in _filter_block(source, "lumen_shell")
    assert "              - 'atomrl/atomrl/lumen/**'" in _filter_block(
        source, "lumen_package"
    )
    assert "              - 'atomrl/tests/lumen/**'" in _filter_block(
        source, "lumen_tests"
    )

    dapo = _filter_block(source, "dapo_examples")
    assert "              - 'atomrl/**'" in dapo
    assert "              - '!atomrl/lumen/**'" in dapo
    assert "              - '!atomrl/atomrl/lumen/**'" in dapo
    assert "              - '!atomrl/tests/lumen/**'" in dapo


def test_atomrl_workflow_does_not_load_external_case_matrices() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")

    for name in ("lumen_shard", "dapo_shard"):
        assert f"            {name}:" in source

    assert "lumen-readme-cases.json" not in source
    assert "atomrl-dapo-gpu-cases.json" not in source
    assert "case_matrix_json" not in source
    assert "run_atomrl=true" in source
    assert "run_lumen=true" in source


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_gpu_filters_use_every_and_split_or_paths():
    workflow = _workflow()
    classify = next(
        step
        for step in workflow["jobs"]["changes"]["steps"]
        if step.get("id") == "classify"
    )
    filters = yaml.safe_load(classify["with"]["filters"])

    assert classify["with"]["predicate-quantifier"] == "every"
    assert filters["atomrl"] == [
        "atomrl/**",
        "!atomrl/lumen/**",
        "!atomrl/atomrl/lumen/**",
        "!atomrl/tests/lumen/**",
    ]
    assert filters["lumen_shell"] == ["atomrl/lumen/**"]
    assert filters["lumen_package"] == ["atomrl/atomrl/lumen/**"]
    assert filters["lumen_tests"] == ["atomrl/tests/lumen/**"]
    assert filters["dapo_examples"] == [
        "atomrl/**",
        "!atomrl/lumen/**",
        "!atomrl/atomrl/lumen/**",
        "!atomrl/tests/lumen/**",
    ]
    assert filters["dapo_shard"] == [".github/workflows/atomrl-dapo-gpu-shard.yaml"]


def test_packaging_changes_do_not_enable_gpu_matrices():
    workflow = _workflow()
    result = next(
        step
        for step in workflow["jobs"]["changes"]["steps"]
        if step.get("id") == "result"
    )
    script = result["run"]

    assert "PACKAGING" not in script.split("run_atomrl=true", 1)[1].split("fi", 1)[0]
    assert "PACKAGING" not in script.split("run_lumen=true", 1)[1].split("fi", 1)[0]
    assert result["env"]["LUMEN_SHELL"] == ("${{ steps.classify.outputs.lumen_shell }}")
    assert result["env"]["LUMEN_PACKAGE"] == (
        "${{ steps.classify.outputs.lumen_package }}"
    )
    assert result["env"]["LUMEN_TESTS"] == ("${{ steps.classify.outputs.lumen_tests }}")
    assert result["env"]["ATOMRL"] == "${{ steps.classify.outputs.atomrl }}"
    assert result["env"]["DAPO_SHARD"] == "${{ steps.classify.outputs.dapo_shard }}"
    assert "run_dapo" not in workflow["jobs"]["changes"]["outputs"]
    assert "run_packaging" in workflow["jobs"]["changes"]["outputs"]
    assert workflow["jobs"]["packaging"]["if"] == (
        "needs.changes.outputs.run_packaging == 'true'"
    )
    assert workflow["jobs"]["atomrl-unit"]["if"] == (
        "needs.changes.outputs.run_atomrl == 'true'"
    )
    assert workflow["jobs"]["dapo-gpu"]["if"].startswith(
        "needs.changes.outputs.run_atomrl == 'true'"
    )


def test_packaging_checks_wheel_contents_without_importing_no_deps_install():
    workflow = _workflow()
    script = workflow["jobs"]["packaging"]["steps"][-1]["run"]
    assert "importlib.import_module" not in script
    assert "pip install --no-deps --target" not in script
    for package_path in (
        "atom/model_loader/",
        "atom/utils/distributed/",
        "atom/plugin/vllm/models/",
        "atom/entrypoints/atomesh/",
    ):
        assert package_path in script


def test_gpu_shards_embed_their_case_matrices():
    lumen = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "atomrl-lumen-gpu-shard.yaml").read_text(
            encoding="utf-8"
        )
    )
    dapo = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "atomrl-dapo-gpu-shard.yaml").read_text(
            encoding="utf-8"
        )
    )

    lumen_cases = lumen["jobs"]["lumen-case"]["strategy"]["matrix"]["include"]
    dapo_cases = dapo["jobs"]["dapo-case"]["strategy"]["matrix"]["include"]
    assert [case["id"] for case in lumen_cases] == list(range(1, 14))
    assert [str(case["id"]) for case in dapo_cases] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "9",
    ]


def test_gpu_images_are_built_once_with_run_unique_tags():
    for filename, image_env in (
        ("atomrl-dapo-gpu-shard.yaml", "IMAGE_TAG"),
        ("atomrl-lumen-gpu-shard.yaml", "LUMEN_IMAGE"),
    ):
        source = (REPO_ROOT / ".github" / "workflows" / filename).read_text(
            encoding="utf-8"
        )
        builds = len(re.findall(r'(?:docker|"\$engine") build ', source))
        assert builds == 1
        assert "${{ github.run_id }}" in source
        assert "${{ github.run_attempt }}" in source
        assert f"{image_env}:" in source


def test_30b_dapo_cases_disable_ray_page_cache_monitor():
    source = (
        REPO_ROOT / ".github" / "workflows" / "atomrl-dapo-gpu-shard.yaml"
    ).read_text(encoding="utf-8")
    assert '"${{ matrix.id }}" == "6"' in source
    assert '"${{ matrix.id }}" == "9"' in source
    assert "--env RAY_memory_monitor_refresh_ms=0" in source


def test_pages_deploy_does_not_touch_retired_atomrl_directory():
    source = (REPO_ROOT / ".github" / "workflows" / "deploy-pages.yml").read_text(
        encoding="utf-8"
    )
    assert 'page_name" == "atomrl"' not in source
    assert "rm -rf atomrl" not in source
    assert "git add -u atomrl" not in source


def test_atomrl_images_reuse_the_published_atom_vllm_base():
    for name in ("Dockerfile.dev", "Dockerfile.release"):
        source = (REPO_ROOT / "atomrl" / "docker" / name).read_text(encoding="utf-8")
        assert "ARG BASE_IMAGE=rocm/atom-dev:vllm-latest" in source
        assert "VLLM_REPO" not in source
        assert "pip install vllm" not in source
