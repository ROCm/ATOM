# SPDX-License-Identifier: MIT
"""Exercise real experiment expansion and model preparation without GPUs/NFS."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load_script("prepare_m3_gqa_draft")
guard = load_script("check_m3_gqa_store")


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    path = tmp_path_factory.mktemp("matrix") / "matrix.json"
    env = dict(
        os.environ,
        ATOMESH_SLURM_SUBMIT_RUNNER="atomesh-cicd",
        ATOMESH_SLURM_ACCOUNT="",
        ATOMESH_SLURM_PARTITION="amd-spur",
        ATOMESH_MODEL_ROOT="/mnt/models",
        ATOMESH_LOG_ROOT="/tmp/m3-logs",
        ATOMESH_1P1D_NODES="",
        ATOMESH_SINGLE_NODE="auto",
        ATOMESH_PD_RANK_MAPPING_POLICY="none",
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "pd_matrix.py"),
            "--suite",
            "gluon_gqa_store",
            "--model",
            "MiniMax-M3-MXFP4",
            "--output",
            str(path),
        ],
        env=env,
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return json.loads(path.read_text())["include"]


def test_approved_topologies_and_budgets_expand(matrix):
    assert {c["name"] for c in matrix} == guard.CASES.keys()
    for cell in matrix:
        guard.validate(cell)


@pytest.mark.parametrize("change", ["tp", "pp", "al", "pool", "draft", "indexer"])
def test_guard_rejects_unsupported_or_confounding_changes(matrix, change):
    cell = copy.deepcopy(next(c for c in matrix if c["concurrency"] == [72]))
    if change == "tp":
        cell["service"]["decode"]["tp"] = 4
    elif change == "pp":
        cell["service"]["decode"]["extra_args"] = "--pipeline-parallel-size 2"
    elif change == "al":
        cell["server_args"]["spec_decode_acceptance_length"] = 3
    elif change == "pool":
        cell["env"]["prefill"]["MOONCAKE_STORE_OWNERS"] = "0:1536:rdma0"
    elif change == "draft":
        cell["server_args"]["draft_model"] = "Inferact/MiniMax-M3-EAGLE3"
    else:
        cell["env"]["decode"]["ATOM_M3_INDEXER_CP"] = "1"
    with pytest.raises(ValueError):
        guard.validate(cell)


@pytest.fixture
def model(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "num_hidden_layers": 1,
        "head_dim": 128,
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors").write_bytes(b"synthetic weight fixture")
    siblings = []
    for p in source.iterdir():
        lfs = p.suffix == ".safetensors"
        siblings.append(
            SimpleNamespace(
                rfilename=p.name,
                size=p.stat().st_size,
                lfs=SimpleNamespace(sha256=prep.file_hash(p, True)) if lfs else None,
                blob_id=None if lfs else prep.file_hash(p, False),
            )
        )
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha=prep.REVISION, siblings=siblings)

    def copy_download(**kw):
        assert kw["repo_id"] == prep.REPO and kw["revision"] == prep.REVISION
        dest = Path(kw["local_dir"])
        dest.mkdir(parents=True, exist_ok=True)
        for name in kw["allow_patterns"]:
            (dest / name).write_bytes((source / name).read_bytes())

    return tmp_path / "destination", source, api, Mock(side_effect=copy_download)


def test_missing_directory_downloads_via_sdk_and_pins_receipt(model):
    dest, _, api, download = model
    prep.prepare(dest, api, download)
    download.assert_called_once()
    assert prep.check_ready(dest)["revision"] == prep.REVISION
    # A second submission reuses the validated snapshot without SDK traffic.
    prep.prepare(dest, api, download)
    api.model_info.assert_called_once()
    download.assert_called_once()


def test_complete_existing_directory_does_not_download(model):
    dest, source, api, download = model
    dest.mkdir()
    for p in source.iterdir():
        (dest / p.name).write_bytes(p.read_bytes())
    prep.prepare(dest, api, download)
    download.assert_not_called()


def test_incomplete_and_same_size_corrupt_weights_are_repaired(model):
    dest, source, api, download = model
    prep.prepare(dest, api, download)
    weight = dest / "model.safetensors"
    weight.write_bytes(b"x" * weight.stat().st_size)
    with pytest.raises(ValueError):
        prep.check_ready(dest)
    prep.prepare(dest, api, download)
    assert download.call_count == 2
    assert weight.read_bytes() == (source / weight.name).read_bytes()
    weight.unlink()
    with pytest.raises(OSError):
        prep.check_ready(dest)
    prep.prepare(dest, api, download)
    assert download.call_count == 3


def test_failed_download_never_publishes_ready(model):
    dest, _, api, _ = model
    with pytest.raises(ValueError, match="after SDK download"):
        prep.prepare(dest, api, Mock())
    assert not (dest / prep.RECEIPT).exists()
