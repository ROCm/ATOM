# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTICES = REPO_ROOT / "atomrl" / "THIRD_PARTY_NOTICES.md"


def test_third_party_notice_lists_imported_sources():
    text = NOTICES.read_text(encoding="utf-8")

    for path in (
        "atomrl/atomrl/data/sft_dataset.py",
        "atomrl/atomrl/rewards/math_reward.py",
        "atomrl/atomrl/engine/inference/bucketed_weight_transfer.py",
        "atomrl/atomrl/lumen/kernels/dsv4/quant/act_quant.py",
    ):
        assert path in text

    assert "https://github.com/volcengine/verl" in text
    assert "https://github.com/vllm-project/vllm" in text
    assert "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro" in text


def test_derived_sources_keep_file_level_license_provenance():
    apache_files = (
        REPO_ROOT / "atomrl" / "atomrl" / "data" / "sft_dataset.py",
        REPO_ROOT / "atomrl" / "atomrl" / "rewards" / "math_reward.py",
        REPO_ROOT
        / "atomrl"
        / "atomrl"
        / "engine"
        / "inference"
        / "bucketed_weight_transfer.py",
    )

    for path in apache_files:
        header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:8])
        assert (
            "SPDX-License-Identifier: Apache-2.0" in header
            or "Licensed under the Apache License, Version 2.0" in header
        )

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
