# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Versioning guarantees for DAPO model preparation."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[2] / "examples" / "DAPO" / "tools" / "prepare_data.py"


def _load_prepare_data():
    spec = importlib.util.spec_from_file_location("dapo_prepare_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_model_revisions_are_immutable_commits():
    prepare_data = _load_prepare_data()

    for model in prepare_data.MODELS.values():
        assert re.fullmatch(r"[0-9a-f]{40}", model.revision("hf"))
        assert re.fullmatch(r"[0-9a-f]{40}", model.revision("modelscope"))


def test_snapshot_download_uses_and_records_model_revision(tmp_path, monkeypatch):
    prepare_data = _load_prepare_data()
    model = prepare_data.MODELS["Qwen3-8B-Base"]
    calls = []

    def fake_snapshot_download(repo, **kwargs):
        calls.append((repo, kwargs))
        (Path(kwargs["local_dir"]) / "config.json").write_text("{}")
        (Path(kwargs["local_dir"]) / "model.safetensors").touch()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    prepare_data.snapshot(
        "hf",
        model.repo,
        tmp_path,
        prepare_data.MODEL_FILES,
        dataset=False,
        revision=model.revision("hf"),
    )

    assert calls[0][1]["revision"] == model.revision("hf")
    assert (tmp_path / prepare_data.DONE).read_text() == f"hf@{model.revision('hf')}"
