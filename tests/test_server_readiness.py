"""Startup gates must distinguish an API from a DP worker's metrics exporter."""

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PD_SCRIPT = ROOT / ".github/scripts/atomesh/pd_server_atom.sh"
MODEL_LIST = json.dumps(
    {"object": "list", "data": [{"id": "test-model", "object": "model"}]}
)


@pytest.fixture
def endpoint():
    replies = []
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            status, body = replies.pop(0) if len(replies) > 1 else replies[0]
            self.send_response(status)
            # Some Mesh model-list responses use text/plain.
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server.server_port, replies, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_gate(gate, port, tmp_path, *, retry=False):
    # Keep the real shell checks and HTTP traffic; skip polling delays only.
    command = "sleep() { :; }; export -f sleep;\n"
    if gate == "local":
        log = tmp_path / "server.log"
        log.touch()
        command += 'exec bash "$1" "$2" 1 "$3" "$4"'
        args = [ROOT / "scripts/wait_server_ready.sh", port, 30 if retry else 60, log]
    else:
        source = PD_SCRIPT.read_text()
        function = source.split("wait_api_ready() {", 1)[1].split(
            "\nwait_router_closed()", 1
        )[0]
        command += "set -euo pipefail\nwait_api_ready() {" + function
        command += '\nATOMESH_SCRIPT_DIR="$1"\nwait_api_ready "$2" test "$3"'
        args = [
            PD_SCRIPT.parent,
            f"http://127.0.0.1:{port}/v1/models",
            5 if retry else 0,
        ]
    return subprocess.run(
        ["bash", "-c", command, "readiness-test", *map(str, args)],
        cwd=tmp_path,
        env={**os.environ, "NO_PROXY": "127.0.0.1,localhost"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize("gate", ["local", "ci"])
@pytest.mark.parametrize(
    "status,body,ready",
    [
        (200, "# HELP atom_demo Demo\n# TYPE atom_demo gauge\natom_demo 1\n", False),
        (200, "<html>Starting up</html>", False),
        (200, '{"status": "ok"}', False),
        (200, '{"object": "list", "data": []}', False),
        (200, '{"object": "list", "data": [null]}', False),
        (200, '{"object": "list", "data": [{"id": ""}]}', False),
        (200, '{"object": "list",', False),
        (503, MODEL_LIST, False),
        (200, MODEL_LIST, True),
    ],
)
def test_readiness_response(gate, status, body, ready, endpoint, tmp_path):
    port, replies, requests = endpoint
    replies.append((status, body))
    result = run_gate(gate, port, tmp_path)
    assert (result.returncode == 0) == ready, result.stdout + result.stderr
    assert requests == ["/v1/models"]


@pytest.mark.parametrize("gate", ["local", "ci"])
def test_waits_for_model_list_after_metrics_response(gate, endpoint, tmp_path):
    port, replies, requests = endpoint
    replies.extend([(200, "# TYPE demo gauge\ndemo 1\n"), (200, MODEL_LIST)])
    result = run_gate(gate, port, tmp_path, retry=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert requests == ["/v1/models", "/v1/models"]
