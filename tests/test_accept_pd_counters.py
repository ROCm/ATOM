"""The PD acceptance topology's RDMA counter read (tests/accept/atomesh_mm/topologies/pd.sh).

Runs its hca_bytes / topology_counters in bash against a fake sysfs tree: an
HCA whose counter file cannot be read must fail the read, not shrink the sum.
Runs its topology_findings on counter deltas: KV off RDMA fails the run.
"""

import json
import subprocess
from pathlib import Path

import pytest

ACCEPT = Path(__file__).parent / "accept" / "atomesh_mm"


def counters(tmp_path, call, missing=()):
    """Run `call` after sourcing lib.sh and pd.sh; GPUs 0-7 sit beside rdma0-7."""
    (tmp_path / "hca").mkdir()
    (tmp_path / "hca" / "node.map").write_text(
        "".join(f"{g} rdma{g}\n" for g in range(8))
    )
    for g in range(8):
        if g in missing:
            continue
        d = tmp_path / "sys" / f"rdma{g}" / "ports" / "1" / "hw_counters"
        d.mkdir(parents=True)
        (d / "tx_rdma_ucast_bytes").write_text(f"{100 + g}\n")
        (d / "rx_rdma_ucast_bytes").write_text(f"{200 + g}\n")
    script = f"""
        set -euo pipefail
        ACCEPT_DIR={tmp_path} ROOT={tmp_path}
        source {ACCEPT}/lib.sh
        source {ACCEPT}/topologies/pd.sh
        HERE={tmp_path} PREFILL_NODE=node PREFILL_GPUS=0,1,2,3
        DECODE_NODE=node DECODE_GPUS=4,5,6,7
        on_node() {{ bash -c "${{2//\\/sys\\/class\\/infiniband/{tmp_path}/sys}}"; }}
        {call}
    """
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )


def test_counters_sum_every_hca(tmp_path):
    out = counters(tmp_path, "topology_counters")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == (
        '{"prefill_hca_tx_bytes": 406, "prefill_hca_rx_bytes": 806, '
        '"decode_hca_tx_bytes": 422, "decode_hca_rx_bytes": 822}'
    )


@pytest.mark.parametrize("call", ["hca_bytes prefill tx", "topology_counters"])
def test_missing_counter_file_fails_the_read(tmp_path, call):
    out = counters(tmp_path, call, missing=(2,))
    assert out.returncode != 0
    assert "cannot read the tx counters of rdma0,rdma1,rdma2,rdma3" in out.stderr


def test_missing_map_fails_the_read(tmp_path):
    out = counters(tmp_path, "PREFILL_NODE=other; hca_bytes prefill tx")
    assert out.returncode != 0
    assert "no HCA map hca/other.map" in out.stderr


@pytest.mark.parametrize(
    "prefill_tx, decode_rx, failures",
    [
        (5, 5, []),
        (0, 5, ["P's HCAs sent no RDMA bytes: the KV left RDMA"]),
        (5, 0, ["D's HCAs received no RDMA bytes: the KV left RDMA"]),
    ],
)
def test_findings_fail_kv_off_rdma(tmp_path, prefill_tx, decode_rx, failures):
    deltas = json.dumps(
        {
            "prefill_hca_tx_bytes": prefill_tx,
            "prefill_hca_rx_bytes": 0,
            "decode_hca_tx_bytes": 0,
            "decode_hca_rx_bytes": decode_rx,
        }
    )
    out = counters(tmp_path, f"topology_findings '{deltas}'")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {"failures": failures}
