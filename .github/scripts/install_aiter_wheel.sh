#!/usr/bin/env bash
# Install the downloaded aiter wheel into the running CI container.
# De-inlined from atom-test.yaml / atomesh-accuracy-validation.yaml (identical
# blocks). Inputs via env: CONTAINER_NAME (required); AITER_WHL_DIR
# (default /tmp/aiter-whl). Behavior matches the previous inline block: no
# outer set -e, so a missing wheel still hits the explicit error+ls below.
AITER_WHL_DIR="${AITER_WHL_DIR:-/tmp/aiter-whl}"
AITER_WHL=$(ls -t ${AITER_WHL_DIR}/amd_aiter*.whl 2>/dev/null | head -1)
if [ -z "$AITER_WHL" ]; then
  echo "ERROR: No amd_aiter wheel found"
  ls -la ${AITER_WHL_DIR}/
  exit 1
fi

echo "=== Copying wheel into container ==="
WHL_NAME=$(basename "$AITER_WHL")
docker cp "$AITER_WHL" "$CONTAINER_NAME:/tmp/$WHL_NAME"

docker exec "$CONTAINER_NAME" bash -lc "
  set -euo pipefail
  echo '=== Uninstalling existing amd-aiter ==='
  pip uninstall -y amd-aiter || true

  echo '=== Installing amd-aiter from wheel ==='
  pip install /tmp/$WHL_NAME

  echo '=== Installed amd-aiter version ==='
  pip show amd-aiter
"
