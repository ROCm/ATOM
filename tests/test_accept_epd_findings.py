"""The E/P/D acceptance topology's counters and findings
(tests/accept/atomesh_mm/topologies/epd.sh).

Runs its topology_counters against a fake sysfs tree and its
topology_findings against fake role logs, in bash as accept.sh does.
"""

import json
import subprocess
from pathlib import Path

import pytest

ACCEPT = Path(__file__).parent / "accept" / "atomesh_mm"

DECISION = (
    "\x1b[2m2026-11-13\x1b[0m routing decision \x1b[3mpool\x1b[0m\x1b[2m=\x1b[0m"
    '"{pool}" \x1b[3mmedia_digests\x1b[0m\x1b[2m=\x1b[0m{media}\n'
)
SERVED = 'INFO:     10.0.0.1:1 - "POST /v1/chat/completions HTTP/1.1" 200 OK\n'
FELL_BACK = (
    "[atom 19:00:00] Image embeddings from 10.0.0.2:1 are unavailable; "
    "running the vision tower\n"
)


def epd(tmp_path, call):
    """Run `call` after sourcing lib.sh and epd.sh: E on GPU 0, P on 0-3 and
    D on 4-7, beside rdma0-7, each HCA's counters 100 + GPU sent and 200 + GPU
    received."""
    (tmp_path / "hca").mkdir(exist_ok=True)
    (tmp_path / "hca" / "node.map").write_text(
        "".join(f"{g} rdma{g}\n" for g in range(8))
    )
    for g in range(8):
        d = tmp_path / "sys" / f"rdma{g}" / "ports" / "1" / "hw_counters"
        d.mkdir(parents=True, exist_ok=True)
        (d / "tx_rdma_ucast_bytes").write_text(f"{100 + g}\n")
        (d / "rx_rdma_ucast_bytes").write_text(f"{200 + g}\n")
    script = f"""
        set -euo pipefail
        ACCEPT_DIR={tmp_path} ROOT={tmp_path} HERE={ACCEPT}
        source {ACCEPT}/lib.sh
        source {ACCEPT}/topologies/epd.sh
        HERE={tmp_path} RUN_DIR={tmp_path}
        MESH_NODE=node ENCODE_NODE=node ENCODE_GPUS=0 PREFILL_NODE=node PREFILL_GPUS=0,1,2,3
        DECODE_NODE=node DECODE_GPUS=4,5,6,7
        on_node() {{ bash -c "${{2//\\/sys\\/class\\/infiniband/{tmp_path}/sys}}"; }}
        {call}
    """
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )


def test_counters_cover_e_p_and_d(tmp_path):
    out = epd(tmp_path, "topology_counters")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {
        "encode_hca_tx_bytes": 100,
        "encode_hca_rx_bytes": 200,
        "prefill_hca_tx_bytes": 406,
        "prefill_hca_rx_bytes": 806,
        "decode_hca_tx_bytes": 422,
        "decode_hca_rx_bytes": 822,
    }


def findings(tmp_path, served, fell_back, encode_tx, kv_bytes=5):
    """topology_findings over 3 image requests and a text one, E having
    answered `served`, P's ranks having fallen back `fell_back` times, P's
    HCAs having sent and D's received `kv_bytes`."""
    (tmp_path / "mesh.log").write_text(
        DECISION.format(pool="prefill", media=4) * 3
        + DECISION.format(pool="prefill", media=0)
        + DECISION.format(pool="encode", media=4) * 3
    )
    (tmp_path / "encode.log").write_text(
        'INFO:     10.0.0.1:1 - "GET /health HTTP/1.1" 200 OK\n' + SERVED * served
    )
    (tmp_path / "prefill.log").write_text(FELL_BACK * fell_back)
    deltas = json.dumps(
        {
            "encode_hca_tx_bytes": encode_tx,
            "prefill_hca_tx_bytes": kv_bytes,
            "decode_hca_rx_bytes": kv_bytes,
        }
    )
    out = epd(tmp_path, f"topology_findings '{deltas}'")
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_findings_count_requests_and_fallbacks(tmp_path):
    assert findings(tmp_path, served=2, fell_back=4, encode_tx=5) == {
        "image_requests": 3,
        "encode_served": 2,
        "encode_fallbacks": 1,
        "read_fallbacks": 4,
        "failures": [],
    }


@pytest.mark.parametrize(
    "served, fell_back, encode_tx, failure",
    [
        (0, 0, 0, "E served no request"),
        (3, 12, 5, "every read of E's embeddings fell back (12)"),
        (3, 0, 0, "E's HCAs sent no RDMA bytes: the embeddings left RDMA"),
    ],
)
def test_findings_fail_the_run(tmp_path, served, fell_back, encode_tx, failure):
    assert findings(tmp_path, served, fell_back, encode_tx)["failures"] == [failure]


def test_findings_fail_kv_off_rdma(tmp_path):
    assert findings(tmp_path, served=3, fell_back=0, encode_tx=5, kv_bytes=0)[
        "failures"
    ] == [
        "P's HCAs sent no RDMA bytes: the KV left RDMA",
        "D's HCAs received no RDMA bytes: the KV left RDMA",
    ]


def test_unreadable_log_fails_the_findings(tmp_path):
    findings(tmp_path, served=3, fell_back=0, encode_tx=5)
    (tmp_path / "encode.log").unlink()
    out = epd(tmp_path, """topology_findings '{"encode_hca_tx_bytes": 5}'""")
    assert out.returncode != 0
    assert "encode.log" in out.stderr
