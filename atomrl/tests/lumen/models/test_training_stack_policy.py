# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Policy checks for the Lumen training stack touched by parity work."""

import re
from pathlib import Path

ATOMRL_ROOT = Path(__file__).resolve().parents[3]
LUMEN_SHELL = ATOMRL_ROOT / "lumen"
LUMEN_PACKAGE = ATOMRL_ROOT / "atomrl" / "lumen"

POLICY_TARGETS = [
    LUMEN_SHELL / "examples" / "llama31" / "pretrain_llama31.py",
    LUMEN_SHELL / "examples" / "llama2" / "finetune_llama2.py",
    LUMEN_PACKAGE / "models" / "training_contract.py",
    LUMEN_PACKAGE / "models" / "experiment_ops.py",
    LUMEN_PACKAGE / "models" / "fsdp.py",
    LUMEN_PACKAGE / "models" / "llama31" / "fsdp" / "pretrain.py",
    LUMEN_PACKAGE / "models" / "llama2" / "fsdp" / "sft.py",
]


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestTrainingStackPolicy:
    def test_policy_targets_have_no_direct_nemo_imports(self):
        pattern = re.compile(r"^\s*(?:from|import)\s+nemo\b", re.MULTILINE)
        for path in POLICY_TARGETS:
            assert (
                pattern.search(_source(path)) is None
            ), f"unexpected nemo import in {path}"

    def test_policy_targets_have_no_direct_transformer_engine_imports(self):
        pattern = re.compile(
            r"^\s*(?:from|import)\s+transformer_engine\b", re.MULTILINE
        )
        for path in POLICY_TARGETS:
            assert (
                pattern.search(_source(path)) is None
            ), f"unexpected transformer_engine import in {path}"
