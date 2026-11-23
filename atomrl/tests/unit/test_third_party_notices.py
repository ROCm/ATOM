from pathlib import Path

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


def test_gated_example_tokenizers_are_downloaded_not_distributed():
    examples = REPO_ROOT / "atomrl" / "lumen" / "examples"
    assert not tuple(examples.glob("*/tokenizer/*"))

    ensure_script = (examples / "scripts" / "ensure_tokenizer.sh").read_text(
        encoding="utf-8"
    )
    assert "meta-llama/Llama-2-7b-hf" in ensure_script
    assert "meta-llama/Llama-3.1-8B" in ensure_script
    assert "Qwen/Qwen3-8B-Base" in ensure_script
