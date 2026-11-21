from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTICES = REPO_ROOT / "atomrl" / "THIRD_PARTY_NOTICES.md"
TE_KERNEL = (
    REPO_ROOT
    / "atomrl"
    / "lumen"
    / "examples"
    / "perf-veomni"
    / "aiter_overlay"
    / "aiter"
    / "ops"
    / "triton"
    / "_triton_kernels"
    / "cross_entropy.py"
)


def test_third_party_notice_lists_imported_sources():
    text = NOTICES.read_text(encoding="utf-8")

    for path in (
        "atomrl/atomrl/data/sft_dataset.py",
        "atomrl/atomrl/rewards/math_reward.py",
        "atomrl/atomrl/engine/inference/bucketed_weight_transfer.py",
        "atomrl/atomrl/lumen/kernels/dsv4/indexer/tilelang_*.py",
        "atomrl/atomrl/lumen/kernels/dsv4/sparse_mla/tilelang_*.py",
        "atomrl/atomrl/lumen/kernels/dsv4/quant/act_quant.py",
        "atomrl/atomrl/trainer/bf16_optimizer.py",
        "atomrl/atomrl/data/kimi_k25_parser.py",
    ):
        assert path in text

    assert "https://github.com/volcengine/verl" in text
    assert "https://github.com/vllm-project/vllm" in text
    assert "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro" in text
    assert "TorchSpec" in text
    assert "MILES" in text


def test_transformer_engine_derived_kernel_keeps_apache_provenance():
    header = "\n".join(TE_KERNEL.read_text(encoding="utf-8").splitlines()[:6])

    assert "SPDX-License-Identifier: Apache-2.0" in header
    assert "NVIDIA CORPORATION & AFFILIATES" in header
    assert "MIT" not in header


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
        REPO_ROOT / "atomrl" / "atomrl" / "trainer" / "bf16_optimizer.py",
        REPO_ROOT / "atomrl" / "atomrl" / "data" / "kimi_k25_parser.py",
    )
    apache_files += tuple(
        path
        for directory in (
            REPO_ROOT / "atomrl" / "atomrl" / "lumen" / "kernels" / "dsv4" / "indexer",
            REPO_ROOT
            / "atomrl"
            / "atomrl"
            / "lumen"
            / "kernels"
            / "dsv4"
            / "sparse_mla",
        )
        for path in directory.glob("tilelang_*.py")
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
