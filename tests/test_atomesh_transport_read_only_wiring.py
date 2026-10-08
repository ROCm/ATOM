"""CPU-only derived NIXL READ selection and original transport contracts."""

import copy
import hashlib
import json
import os
import subprocess

import pytest
import yaml
from test_atomesh_native_nixl_wiring import exported
from test_atomesh_transport_inventory_wiring import BASE, ROOT, SCRIPTS, load

CASE = "survey-transport-nixl-read-only-2node"
INVENTORY = "survey-transport-inventory-only-2node"
FLAGS = {
    "ATOMESH_TRANSPORT_GPU": "1",
    "ATOMESH_TRANSPORT_NIXL_READ_ONLY": "1",
    "ATOMESH_TRANSPORT_UCX_SELECTION": "rdma0:1@1",
}


@pytest.fixture
def cells(monkeypatch):
    for key, value in {
        "ATOMESH_SLURM_ACCOUNT": "amd-frameworks",
        "ATOMESH_SLURM_PARTITION": "amd-spur",
        "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
        "ATOMESH_LOG_ROOT": "/logs",
        "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
        "ATOMESH_MODEL_ROOT": "/models",
        "ATOMESH_1P1D_NODES": "pit2-p03-g10,pit2-p03-g20",
    }.items():
        monkeypatch.setenv(key, value)
    matrix = load("pd_matrix")
    return {
        cell["name"]: cell
        for cell in matrix.build_cells(
            matrix.load_config(ROOT / ".github/benchmark/models_atomesh.yaml"),
            suite="vllm",
            model_filter={"Transport-vLLM-Survey"},
            case_filter={BASE, INVENTORY, CASE},
            benchmark_kind_filter=None,
            override_image=None,
            override_benchmark_concurrency=None,
            override_eval_concurrency=None,
        )
    }


def test_read_only_case_preserves_parent_except_exact_opt_in(cells):
    assert set(cells) == {BASE, INVENTORY, CASE}
    derived, parent = copy.deepcopy(cells[CASE]), copy.deepcopy(cells[BASE])
    for role in ("common", "prefill", "decode", "router"):
        for key, value in FLAGS.items():
            assert derived["env"][role][key] == value
        for key in FLAGS.keys() - {"ATOMESH_TRANSPORT_GPU"}:
            derived["env"][role].pop(key)
    for cell in (derived, parent):
        for key in ("id", "name"):
            cell.pop(key)
    assert derived == parent


def test_original_default_and_inventory_semantics_remain_unchanged(cells):
    original = yaml.safe_load(
        (ROOT / ".github/benchmark/models_atomesh.yaml").read_text()
    )["models"]["Transport-vLLM-Survey"]
    original["suites"]["vllm"] = [
        cell for cell in original["suites"]["vllm"] if cell["name"] in (BASE, INVENTORY)
    ]
    assert hashlib.sha256(
        json.dumps(original, sort_keys=True).encode()
    ).hexdigest() == (
        "333f19ddc977db8ff9e212a81fb57ec23c204b7e255352e1d7f1cdf24441beab"
    )
    for name, gpu in ((BASE, "1"), (INVENTORY, "0")):
        for env in cells[name]["env"].values():
            assert env["ATOMESH_TRANSPORT_GPU"] == gpu
            assert "ATOMESH_TRANSPORT_NIXL_READ_ONLY" not in env
            assert "ATOMESH_TRANSPORT_UCX_SELECTION" not in env


@pytest.mark.parametrize("name", [BASE, INVENTORY, CASE])
@pytest.mark.parametrize("role", ["common", "prefill", "decode", "router"])
def test_actual_exports_preserve_exact_role_selection_without_fallback(
    cells, name, role
):
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    start = source.index("apply_prefixed_env() {")
    function = source[start : source.index("\n}", start) + 2]
    commands = "\napply_prefixed_env ATOMESH_ENV_ 192.0.2.1\n"
    if role in ("prefill", "decode"):
        commands += f"apply_prefixed_env ATOMESH_{role.upper()}_ENV_ 192.0.2.1\n"
    # Router inherits common exports; only P/D have dedicated role prefixes.
    result = subprocess.check_output(
        ["bash", "-c", exported(cells[name]) + "\n" + function + commands + "env -0"],
        env={"PATH": os.environ["PATH"]},
    )
    env = dict(item.decode().split("=", 1) for item in result.split(b"\0") if item)
    assert env["ATOMESH_TRANSPORT_ONLY"] == "1"
    assert env["ATOMESH_TRANSPORT_MORI_BYTES"] == "4096,2493186048"
    assert env["ATOMESH_PRE_CLEAN_GPU"] == "1"
    if name == CASE:
        for key, value in FLAGS.items():
            assert env[key] == value
    else:
        assert env["ATOMESH_TRANSPORT_GPU"] == ("0" if name == INVENTORY else "1")
        assert "ATOMESH_TRANSPORT_NIXL_READ_ONLY" not in env
        assert "ATOMESH_TRANSPORT_UCX_SELECTION" not in env
    assert not {"UCX_NET_DEVICES", "UCX_IB_GID_INDEX", "UCX_TLS"} & env.keys()
