#!/usr/bin/env bash
# Install exactly one downloaded AITER wheel in the CI container. Do not change
# PyTorch/ROCm dependencies supplied by the image. Fail before benchmarking if
# the copy, install, or import identity check fails.
set -euo pipefail
: "${CONTAINER_NAME:?CONTAINER_NAME is required}"
AITER_WHL_DIR="${AITER_WHL_DIR:-/tmp/aiter-whl}"
shopt -s nullglob
wheels=("$AITER_WHL_DIR"/amd_aiter*.whl)
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "ERROR: Expected exactly one amd_aiter wheel in $AITER_WHL_DIR" >&2
  exit 1
fi
WHL_NAME=$(basename "${wheels[0]}")
echo "Copying AITER wheel: $WHL_NAME"
docker cp "${wheels[0]}" "$CONTAINER_NAME:/tmp/$WHL_NAME"
docker exec "$CONTAINER_NAME" bash -lc '
  set -euo pipefail
  python3 -m pip uninstall -y amd-aiter
  python3 -m pip install --no-deps --no-cache-dir "$1"
  python3 -m pip show amd-aiter
  python3 - <<"PY"
import importlib.metadata
from pathlib import Path
import aiter

distribution = importlib.metadata.distribution("amd-aiter")
expected = Path(distribution.locate_file("aiter/__init__.py")).resolve()
actual = Path(aiter.__file__).resolve()
if actual != expected:
    raise SystemExit(f"AITER imports from {actual}, expected installed wheel at {expected}")
print(f"Using amd-aiter {distribution.version} from {actual}")
PY
' -- "/tmp/$WHL_NAME"
