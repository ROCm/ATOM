#!/bin/bash
# setup_client.sh: install the MLPerf Endpoints client on the host (not in a container; it needs no GPU):
# uv, mlcommons/endpoints at 4b385eb, unmodified, and the agentic dataset, checked against its sha256.
# The same client, commit and dataset as the Kimi-K3 runs (/data/gyu/k3-mlperf-repro).
set -euo pipefail
cd "$(dirname "$0")" && . ./config.env
export PATH=$HOME/.local/bin:$PATH
EP=$CLIENT/endpoints
mkdir -p "$CLIENT/data" "$CLIENT/results"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
uv --version
[ -d "$EP/.git" ] || git clone -q https://github.com/mlcommons/endpoints.git "$EP"
git -C "$EP" checkout -q --detach 4b385eb
git -C "$EP" log --oneline -1
# UV_FROZEN on every uv command: a bare `uv run` may rewrite a stale uv.lock, and then the client is no longer unmodified
(cd "$EP" && UV_FROZEN=1 uv sync --frozen && UV_FROZEN=1 uv run inference-endpoint --version)
[ -z "$(git -C "$EP" status --short)" ] || { echo "$EP is modified:" >&2; git -C "$EP" status --short >&2; exit 2; }
F=$CLIENT/data/mlperf_agentic_inference_dataset.jsonl
[ -f "$F" ] || curl -fsSL -o "$F" \
  https://endpoints.mlcommons-storage.org/mlperf_agentic_inference_dataset/mlperf_agentic_inference_dataset.jsonl
echo "1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae  $F" | sha256sum -c -
echo "CLIENT READY"
