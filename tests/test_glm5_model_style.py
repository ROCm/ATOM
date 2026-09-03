# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Keep model declarations separate from pooled-indexer runtime tooling."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
MODEL = ROOT / "atom/models/glm5_next.py"
INDEXER = ROOT / "atom/model_ops/glm5_next/indexer.py"


def test_glm5_model_has_no_runtime_dump_or_host_sync():
    source = MODEL.read_text(encoding="utf-8")

    assert "torch.save" not in source
    assert ".item()" not in source
    assert "torch.nn.functional.linear" not in source
    assert "direct_register_custom_op" not in source


def test_glm5_pooled_indexer_has_no_device_to_host_scalar_read():
    source = INDEXER.read_text(encoding="utf-8")

    assert ".item()" not in source
    assert "torch.save" not in source
