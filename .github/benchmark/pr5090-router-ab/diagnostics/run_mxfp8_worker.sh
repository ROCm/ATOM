#!/usr/bin/env bash
set -euo pipefail
uv venv --system-site-packages /tmp/mxfp8-validation
uv pip install --python /tmp/mxfp8-validation/bin/python --no-index --find-links /bench/assets/wheels pytest==9.0.2
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONHASHSEED=0
exec /tmp/mxfp8-validation/bin/python /bench/tp1_mxfp8_benchmark.py "${1:?base or candidate}"
