"""CPU-only pre-submission pool qualification and scheduler selection."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / ".github/scripts/atomesh/pd_prefix_pool.py"
spec = importlib.util.spec_from_file_location("pool", PATH)
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def metrics(now=1000):
    lines = [
        f"node_time_seconds {now}",
        'node_scrape_collector_success{collector="filesystem"} 1',
        f"node_memory_MemAvailable_bytes {pool.REQUIRED_MEMORY}",
    ]
    for key, value in {
        "avail_bytes": 250 * 1024**3,
        "files_free": 200000,
        "files": 300000,
        "readonly": 0,
        "device_error": 0,
    }.items():
        lines.append(
            f'node_filesystem_{key}{{mountpoint="/",device="/dev/root",fstype="ext4"}} {value}'
        )
    return "\n".join(lines)


def nodes(count=3):
    return {
        f"n{i}": {
            "NodeName": f"n{i}",
            "Partitions": "amd-spur",
            "State": "IDLE",
            "Gres": "gpu:mi355x:8",
        }
        for i in range(count)
    }


@pytest.mark.parametrize("fault", ["stale", "collector", "disk", "mount", "ram", "nan"])
def test_invalid_metrics(fault):
    value = metrics()
    if fault == "stale":
        value = metrics(800)
    elif fault == "collector":
        value = value.replace(
            'node_scrape_collector_success{collector="filesystem"} 1', ""
        )
    elif fault == "disk":
        value = value.replace(str(250 * 1024**3), "0")
    elif fault == "mount":
        value = value.replace('mountpoint="/"', 'mountpoint="/other"')
    elif fault == "ram":
        value = value.replace(str(pool.REQUIRED_MEMORY), "1")
    else:
        value = value.replace(str(250 * 1024**3), "NaN")
    with pytest.raises(ValueError):
        pool.resource_report(
            value, 1000, ["/var/lib/docker", "/var/lib/containerd", "/tmp", "/logs/run"]
        )


def test_all_qualified_nodes_not_fixed_pair():
    report = pool.qualify(
        nodes(), "amd-spur", ["/logs/run"], fetch=lambda n: metrics(), now=lambda: 1000
    )
    assert report["eligible"] == ["n0", "n1", "n2"]
    assert report["num_nodes"] == 2
    records = nodes()
    records["n1"]["Gres"] = "gpu:mi300x:8"
    records["n2"]["Partitions"] = "other"
    report = pool.qualify(
        records, "amd-spur", ["/logs/run"], fetch=lambda n: metrics(), now=lambda: 1000
    )
    assert report["status"] == "BLOCKED"
    assert report["eligible"] == ["n0"]
    assert all(row.get("reason") for row in report["nodes"][1:])


def test_missing_nodes_and_repeated_gres():
    records = pool.parse_nodes(
        "NodeName=n0 State=IDLE Partitions=amd-spur Gres="
        + ",".join(["gpu:mi355x:1"] * 8)
    )
    assert pool.qualify(
        records, "amd-spur", ["/logs"], fetch=lambda n: metrics(), now=lambda: 1000
    )["eligible"] == ["n0"]
    with pytest.raises(ValueError):
        pool.parse_nodes("")


def test_selector_failure_prevents_submission(tmp_path):
    # Execute the exact new-case shell block with a rejecting selector and a
    # sentinel sbatch; do not bypass the real readiness gate in production.
    script = (ROOT / ".github/scripts/atomesh/pd_submit.sh").read_text()
    block = script.split("# PREFIX_READ_POOL_BEGIN\n", 1)[1].split(
        "# PREFIX_READ_POOL_END", 1
    )[0]
    sentinel = tmp_path / "sbatch-called"
    code = f"""set -euo pipefail
REPO_ROOT={ROOT}
RESULT_DIR={tmp_path}
ATOMESH_CELL_ID=case
SLURM_PARTITION=amd-spur
LOG_ROOT={tmp_path}/logs
USES_SPUR_CONTROLLER=0
SPUR_CONTROLLER_ADDR=''
python3() {{ return 7; }}
{block}
touch {sentinel}
"""
    result = subprocess.run(["bash", "-c", code], check=False, capture_output=True)
    assert result.returncode == 7
    assert not sentinel.exists()


def test_successful_shell_block_exports_entire_pool(tmp_path):
    import json
    import sys

    script = (ROOT / ".github/scripts/atomesh/pd_submit.sh").read_text()
    block = script.split("# PREFIX_READ_POOL_BEGIN\n", 1)[1].split(
        "# PREFIX_READ_POOL_END", 1
    )[0]
    cell = {"name": "kimi-k3-vllm-c48-prefix-read-fixed", "nodes": [], "num_nodes": 2}
    (tmp_path / "case-qualified-cell.json").write_text(
        json.dumps(dict(cell, nodes=["n0", "n1", "n2"]))
    )
    code = f"""set -euo pipefail
REPO_ROOT={ROOT}
RESULT_DIR={tmp_path}
ATOMESH_CELL_ID=case
SLURM_PARTITION=amd-spur
LOG_ROOT={tmp_path}/logs
USES_SPUR_CONTROLLER=0
SPUR_CONTROLLER_ADDR=''
NUM_NODES=2
export CELL_JSON='{json.dumps(cell)}'
python3() {{ if [[ "$1" == *pd_prefix_pool.py ]]; then return 0; else {sys.executable} "$@"; fi; }}
{block}
[[ "$NODE_LIST" == n0,n1,n2 && "$NUM_NODES" == 2 ]]
"""
    subprocess.run(["bash", "-c", code], check=True, capture_output=True)


def test_pool_scheduler_arguments():
    helper = ROOT / ".github/scripts/slurm_submit_helpers.sh"
    for spur, expected in [(0, ["--exclude", "n3"]), (1, ["-w", "n0,n1,n2"])]:
        command = f"""source {helper}
USES_SPUR_CONTROLLER={spur}
run_slurm_query() {{ printf 'n0\\nn1\\nn2\\nn3\\n'; }}
slurm_node_selection_args n0,n1,n2 2
printf '%s\\n' "${{SLURM_NODE_SELECTION_ARGS[@]}}"
"""
        assert (
            subprocess.check_output(["bash", "-c", command], text=True).split()
            == expected
        )


def test_cli_controller_partition_and_evidence(monkeypatch, tmp_path):
    import json
    import sys

    output, cell_output = tmp_path / "eligibility.json", tmp_path / "cell.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PATH),
            "--controller",
            "http://controller:6817",
            "--run-dir",
            "/actual/log/root/new-job",
            "--output",
            str(output),
            "--cell-output",
            str(cell_output),
        ],
    )
    monkeypatch.setenv(
        "CELL_JSON",
        json.dumps(
            {"name": "kimi-k3-vllm-c48-prefix-read-fixed", "nodes": [], "num_nodes": 2}
        ),
    )
    commands = []

    def query(command, **kwargs):
        commands.append(command)
        if command[-1] == "partition":
            return "PartitionName=amd-spur Default=YES"
        return "\n".join(
            " ".join(f"{key}={value}" for key, value in record.items())
            for record in nodes().values()
        )

    monkeypatch.setattr(pool.subprocess, "check_output", query)
    original = pool.qualify
    monkeypatch.setattr(
        pool,
        "qualify",
        lambda records, partition, paths: original(
            records, partition, paths, fetch=lambda _: metrics(pool.time.time())
        ),
    )
    pool.main()
    report = json.loads(output.read_text())
    assert report["status"] == "QUALIFIED_POOL"
    assert report["storage_paths"][-1] == "/actual/log/root/new-job"
    assert len(json.loads(cell_output.read_text())["nodes"]) == 3
    assert all(
        command[:3] == ["scontrol", "--controller", "http://controller:6817"]
        for command in commands
    )


def test_pool_updates_cell_without_changing_node_count(tmp_path):
    cell = {"name": "kimi-k3-vllm-c48-prefix-read-fixed", "nodes": [], "num_nodes": 2}
    updated = pool.update_cell(
        cell, {"status": "QUALIFIED_POOL", "eligible": ["n0", "n1", "n2"]}
    )
    assert updated == dict(cell, nodes=["n0", "n1", "n2"])
    with pytest.raises(ValueError):
        pool.update_cell(cell, {"status": "BLOCKED", "eligible": ["n0"]})
