#!/bin/bash
# One-time preparation inside the container: sidecar flydsl + aiter overlay.
# Idempotent, so re-running is harmless.
#
#   bash $EXAMPLE_DIR/setup_env.sh
#
set -e
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

echo "=== 1/3  flydsl $FLYDSL_VERSION sidecar -> $FLYDSL_SIDECAR ==="
# The image ships flydsl 0.1.6, whose JIT cannot link. Upgrading in place would
# also break aiter, which imports a symbol 0.3.2 removed. Installing to a
# separate directory and putting it on PYTHONPATH only for this example leaves
# the image's own environment untouched.
if [ -d "$FLYDSL_SIDECAR/flydsl" ]; then
    echo "already present, skipping"
else
    pip install --target "$FLYDSL_SIDECAR" --no-deps --no-cache-dir "flydsl==$FLYDSL_VERSION"
fi
PYTHONPATH="$FLYDSL_SIDECAR" python3 -c "import flydsl; print('sidecar flydsl', flydsl.__version__)"
python3 -c "import flydsl; print('image flydsl  ', flydsl.__version__, '(unchanged)')"

echo
echo "=== 2/3  aiter overlay ==="
# Two sets of files aiter does not ship: the triton modules `import atomrl.lumen` needs
# from Lumen's aiter fork, and the FlyDSL convolution of ROCm/aiter#5370 with
# its tuned tables (plus Lumen's rows for conv_pad_in_kernel). Only files that
# do not already exist are written -- nothing aiter ships is modified.
python3 "$EXAMPLE_DIR/overlay_aiter.py" --apply

echo
echo "=== 3/3  VeOmni editable install ==="
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
import sys
from veomni_patches import aiter_conv
fn = aiter_conv.load()
rows = len(sys.modules['aiter.ops.flydsl.conv_kernels']._load_tuned_table()) if fn else 0
print('aiter flydsl_conv_implicit:', 'available' if fn else 'MISSING', '(must be available)')
print('aiter conv3d tuned rows   :', rows, '(86 = 76 from aiter + 10 for conv_pad_in_kernel)')
assert fn is not None and rows > 0
"
echo
echo "SETUP OK"
