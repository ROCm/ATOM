#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# One-time preparation inside the container: VeOmni only.
# Idempotent, so re-running is harmless.
#
#   bash $EXAMPLE_DIR/setup_env.sh
#
set -e
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

echo "=== VeOmni editable install ==="
if python3 -c "import veomni" 2>/dev/null; then
    python3 -c "import veomni; print('veomni already importable from', veomni.__file__)"
else
    # --no-deps is mandatory: VeOmni's dependency set would pull CUDA wheels and
    # replace the image's ROCm torch.
    pip install -e "$VEOMNI_DIR" --no-deps
    python3 -c "import veomni; print('veomni ->', veomni.__file__)"
fi

echo
echo "=== check ==="
python3 - <<'PY'
import torch
print("torch      ", torch.__version__)
print("torch.hip  ", torch.version.hip)
print("torch.cuda ", torch.version.cuda, "(must be None on ROCm)")
print("devices    ", torch.cuda.device_count())
assert torch.version.hip and torch.version.cuda is None, "not a ROCm torch -- stop here"
PY
PYTHONPATH="$EXAMPLE_DIR:$LUMEN_PYTHONPATH" python3 -c "
from atomrl.lumen.ops.conv import conv
fn = conv
print('Lumen convolution:', 'available' if fn else 'MISSING', '(must be available)')
assert fn is not None
"
echo
echo "SETUP OK"
