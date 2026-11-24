# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTICES = REPO_ROOT / "atomrl" / "THIRD_PARTY_NOTICES.md"
ATOMRL_ROOT = REPO_ROOT / "atomrl"

APACHE_DERIVED = (
    "atomrl/atomrl/algorithms/advantage_estimators.py",
    "atomrl/atomrl/algorithms/dapo_sampling.py",
    "atomrl/atomrl/algorithms/kl_controller.py",
    "atomrl/atomrl/algorithms/loss_functions.py",
    "atomrl/atomrl/algorithms/rollout_correction.py",
    "atomrl/atomrl/data/sft_dataset.py",
    "atomrl/atomrl/engine/inference/bucketed_weight_transfer.py",
    "atomrl/atomrl/engine/inference/vllm_colocate_worker_ext.py",
    "atomrl/atomrl/engine/training/base_engine.py",
    "atomrl/atomrl/engine/training/fsdp_engine.py",
    "atomrl/atomrl/engine/training/vllm_colocate_worker_ext.py",
    "atomrl/atomrl/rewards/math_reward.py",
    "atomrl/atomrl/trainer/metric_utils.py",
    "atomrl/atomrl/utils/fsdp_utils.py",
    "atomrl/atomrl/utils/lr_scheduler.py",
    "atomrl/atomrl/utils/seqlen_balancing.py",
    "atomrl/atomrl/utils/torch_functional.py",
    "atomrl/atomrl/utils/ulysses.py",
    "atomrl/atomrl/utils/vocab_parallel.py",
    "atomrl/atomrl/workers/base_worker.py",
    "atomrl/lumen/examples/dsv4/dsv4_flash_megatron_args.sh",
    "atomrl/lumen/examples/dsv4/tools/param_name_remap.py",
    "atomrl/lumen/examples/perf-veomni/qwen_image_1024.yaml",
    "atomrl/lumen/examples/perf-veomni/qwen_image_smoke.yaml",
    "atomrl/lumen/examples/perf-veomni/wan_video.yaml",
)


def test_third_party_notice_lists_imported_sources():
    text = NOTICES.read_text(encoding="utf-8")

    for path in (
        "atomrl/atomrl/data/sft_dataset.py",
        "atomrl/atomrl/rewards/math_reward.py",
        "atomrl/atomrl/engine/inference/bucketed_weight_transfer.py",
        "atomrl/atomrl/lumen/kernels/dsv4/quant/act_quant.py",
    ):
        assert path in text

    assert "https://github.com/verl-project/verl" in text
    assert "https://github.com/vllm-project/vllm" in text
    assert "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro" in text


def test_derived_sources_keep_file_level_license_provenance():
    for relative in APACHE_DERIVED:
        path = REPO_ROOT / relative
        header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:30])
        assert "SPDX-License-Identifier: Apache-2.0" in header, relative
        assert "SPDX-License-Identifier: MIT" not in header, relative
        assert "Advanced Micro Devices, Inc." in header, relative

    expected_upstream_copyrights = {
        "atomrl/atomrl/algorithms/advantage_estimators.py": (
            "Copyright 2024 Bytedance Ltd. and/or its affiliates",
            "Copyright 2022 The HuggingFace Team. All rights reserved.",
        ),
        "atomrl/atomrl/algorithms/kl_controller.py": (
            "Copyright 2024 Bytedance Ltd. and/or its affiliates",
            "Copyright 2022 The HuggingFace Team. All rights reserved.",
        ),
        "atomrl/atomrl/algorithms/rollout_correction.py": (
            "Copyright 2025 Bytedance Ltd. and/or its affiliates",
        ),
        "atomrl/atomrl/algorithms/loss_functions.py": (
            "Copyright 2024 Bytedance Ltd. and/or its affiliates",
            "Copyright 2022 The HuggingFace Team. All rights reserved.",
        ),
        "atomrl/atomrl/algorithms/dapo_sampling.py": (
            "Copyright 2024 Bytedance Ltd. and/or its affiliates",
        ),
        "atomrl/atomrl/engine/inference/vllm_colocate_worker_ext.py": (
            "Copyright 2024 Bytedance Ltd. and/or its affiliates",
        ),
        "atomrl/atomrl/engine/training/vllm_colocate_worker_ext.py": (
            "Copyright 2024 Bytedance Ltd. and/or its affiliates",
        ),
        "atomrl/atomrl/trainer/metric_utils.py": (
            "Copyright 2024 Bytedance Ltd. and/or its affiliates",
        ),
        "atomrl/atomrl/utils/seqlen_balancing.py": (
            "Copyright 2024 Bytedance Ltd. and/or its affiliates",
        ),
        "atomrl/atomrl/utils/vocab_parallel.py": (
            "Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.",
        ),
        "atomrl/atomrl/engine/inference/bucketed_weight_transfer.py": (
            "SPDX-FileCopyrightText: Copyright contributors to the vLLM project",
        ),
        "atomrl/atomrl/workers/base_worker.py": (
            "Copyright 2024 Bytedance Ltd. and/or its affiliates",
        ),
        "atomrl/lumen/examples/dsv4/tools/param_name_remap.py": (
            "Copyright 2023-2024 SGLang Team",
        ),
        "atomrl/lumen/examples/dsv4/dsv4_flash_megatron_args.sh": (
            "Copyright 2025 RadixArk, Inc.",
            "Copyright 2025 Zhipu AI",
        ),
    }
    for relative, copyrights in expected_upstream_copyrights.items():
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for copyright_line in copyrights:
            assert copyright_line in text, relative

    deepseek = (
        REPO_ROOT
        / "atomrl"
        / "atomrl"
        / "lumen"
        / "kernels"
        / "dsv4"
        / "quant"
        / "act_quant.py"
    )
    header = "\n".join(deepseek.read_text(encoding="utf-8").splitlines()[:6])
    assert "SPDX-License-Identifier: MIT" in header
    assert "DeepSeek" in header


def test_notices_cover_every_derived_file_and_pinned_source():
    text = NOTICES.read_text(encoding="utf-8")
    for relative in APACHE_DERIVED:
        assert relative in text
    for revision in (
        "8357f1e0a7ce4c06dc970c2c637d49b33815f5cc",
        "246cac4ca6b0e3ce1009bc48aaa798689e0600af",
        "ca439aa5afcfe913c075570a9cb3f0f83730904d",
        "fad09e8a1f51b31eba1f42ff5d651256c77a734d",
        "573848a00fcd7329c2411346c6f4a983e9f67e3f",
        "88c10654aedd7d3cc8ae7f16699020aff53efee0",
        "9437366e0aa3a25294720f70d18b081067595f85",
        "b5968e9190ef611bbf34a7229255be88a0e937c1",
    ):
        assert revision in text


def test_atomrl_distribution_packages_apache_terms():
    apache = ATOMRL_ROOT / "LICENSE-APACHE"
    assert apache.read_bytes() == (REPO_ROOT / "atom" / "mesh" / "LICENSE").read_bytes()
    pyproject = (ATOMRL_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"LICENSE-APACHE"' in pyproject
    assert '"THIRD_PARTY_NOTICES.md"' in pyproject


def test_non_derived_wrappers_and_contract_tests_are_mit_owned():
    wrapper = (
        ATOMRL_ROOT / "atomrl" / "quantization" / "rollout_correction.py"
    ).read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: MIT" in wrapper
    assert "Re-exports the Apache-2.0-derived implementation" in wrapper
    assert "Derived from verl" not in wrapper

    contract = (
        ATOMRL_ROOT
        / "tests"
        / "lumen"
        / "models"
        / "dsv4"
        / "test_miles_qat_forward_semantics.py"
    ).read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: MIT" in contract
    assert "Behavioral compatibility" in contract
    assert "Contracts copied" not in contract


def test_bootstrap_preserves_distribution_license_metadata():
    script = (
        ATOMRL_ROOT / "lumen" / "examples" / "dsv4" / "prepare_bootstrap.sh"
    ).read_text(encoding="utf-8")
    assert "packages_distributions" in script
    assert 'docker cp "${cname}:${metadata_path}"' in script
    assert "THIRD_PARTY_IMAGE_SOURCES" in script


def test_build_py_removes_stale_deleted_package_sources(tmp_path):
    build_lib = tmp_path / "build"
    stale = build_lib / "atomrl" / "trainer" / "bf16_optimizer.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("must not ship\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "setup.py",
            "build_py",
            "--build-lib",
            str(build_lib),
        ],
        cwd=ATOMRL_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not stale.exists()
    assert (build_lib / "atomrl" / "trainer" / "rl_trainer.py").is_file()


def test_blocked_unapproved_sources_are_not_distributed():
    blocked = (
        REPO_ROOT / "atomrl" / "atomrl" / "trainer" / "bf16_optimizer.py",
        REPO_ROOT / "atomrl" / "atomrl" / "data" / "kimi_k25_parser.py",
    )
    assert not any(path.exists() for path in blocked)

    kernel_root = REPO_ROOT / "atomrl" / "atomrl" / "lumen" / "kernels" / "dsv4"
    assert not tuple(kernel_root.rglob("tilelang_*.py"))

    text = NOTICES.read_text(encoding="utf-8")
    assert "must not be redistributed" not in text
    assert "AMD OSS sign-off" not in text


def test_example_tokenizers_are_not_distributed():
    examples = REPO_ROOT / "atomrl" / "lumen" / "examples"
    assert not tuple(examples.glob("*/tokenizer/*"))


def _fake_curl(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "curl.log"
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
out=""
url=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    -H) shift 2 ;;
    *) url="$1"; shift ;;
  esac
done
printf '%s\\n' "$url" >> "$CURL_LOG"
if [ -n "${CURL_FAIL_ON:-}" ] && [[ "$url" == *"$CURL_FAIL_ON"* ]]; then
  exit 22
fi
mkdir -p "$(dirname "$out")"
printf '{}\\n' > "$out"
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    return fake_bin, log


def test_example_tokenizer_downloads_pinned_revisions_atomically(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required")
    fake_bin, log = _fake_curl(tmp_path)
    script = (
        REPO_ROOT / "atomrl" / "lumen" / "examples" / "scripts" / "ensure_tokenizer.sh"
    )
    llama2 = tmp_path / "llama2"
    llama31 = tmp_path / "llama31"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CURL_LOG": str(log),
    }
    command = (
        f'source "{script}"; '
        f'example_tokenizer llama2 "{llama2}"; '
        f'example_tokenizer llama31 "{llama31}"'
    )
    subprocess.run([bash, "-c", command], env=env, check=True)

    urls = log.read_text(encoding="utf-8").splitlines()
    assert len(urls) == 8
    assert all(
        "/NousResearch/Llama-2-7b-hf/resolve/"
        "8efe6c9b93655b934e27bd9981e3ec13e55aee9d/" in url
        for url in urls[:4]
    )
    assert all(
        "/NousResearch/Meta-Llama-3.1-8B/resolve/"
        "1f47e50cdbe801ad8a5174156ec3a0655108fb9f/" in url
        for url in urls[4:]
    )
    assert not tuple(tmp_path.rglob("*.part"))
    assert (
        llama2.joinpath(".atomrl-tokenizer-revision")
        .read_text(encoding="utf-8")
        .strip()
        == "NousResearch/Llama-2-7b-hf@8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
    )

    subprocess.run([bash, "-c", command], env=env, check=True)
    assert log.read_text(encoding="utf-8").splitlines() == urls

    refresh = (
        f'source "{script}"; ensure_tokenizer "{llama2}" '
        "NousResearch/Llama-2-7b-hf new-revision "
        "config.json tokenizer.json tokenizer.model tokenizer_config.json"
    )
    subprocess.run([bash, "-c", refresh], env=env, check=True)
    refreshed_urls = log.read_text(encoding="utf-8").splitlines()
    assert len(refreshed_urls) == 12
    assert all("/resolve/new-revision/" in url for url in refreshed_urls[-4:])
    assert (
        llama2.joinpath(".atomrl-tokenizer-revision")
        .read_text(encoding="utf-8")
        .strip()
        == "NousResearch/Llama-2-7b-hf@new-revision"
    )


def test_example_tokenizer_removes_partial_file_on_download_failure(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required")
    fake_bin, log = _fake_curl(tmp_path)
    script = (
        REPO_ROOT / "atomrl" / "lumen" / "examples" / "scripts" / "ensure_tokenizer.sh"
    )
    target = tmp_path / "llama2"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CURL_LOG": str(log),
        "CURL_FAIL_ON": "tokenizer.json",
    }
    result = subprocess.run(
        [
            bash,
            "-c",
            f'source "{script}"; example_tokenizer llama2 "{target}"',
        ],
        env=env,
        check=False,
    )
    assert result.returncode != 0
    assert not target.exists()
    assert not tuple(target.glob("*.part"))
    assert not tuple(tmp_path.glob("llama2.tmp.*"))
