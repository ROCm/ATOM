#!/usr/bin/env bash
# Drop a GLM-5.x BF16 tuned GEMM row that references a gfx942 OPUS kernel id
# missing from current AITER lookup tables.
set -euo pipefail

if [[ "${ATOM_PATCH_AITER_GLM52_BAD_OPUS_GEMM:-1}" == "0" ]]; then
  echo "AITER GLM-5.2 bad OPUS GEMM config patch disabled by ATOM_PATCH_AITER_GLM52_BAD_OPUS_GEMM=0"
  exit 0
fi

python3 - <<'PY'
import glob
import os
import site
from pathlib import Path

BAD_MARKER = "gfx942,80,768,256,6144,False,torch.bfloat16,torch.bfloat16,False,False,opus,10211,"

search_roots = [Path("/app/aiter-test")]
for site_dir in site.getsitepackages():
    search_roots.append(Path(site_dir))
for path_entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    if path_entry:
        search_roots.append(Path(path_entry))

candidates = []
for root in search_roots:
    if not root.exists():
        continue
    candidates.extend(
        Path(p)
        for p in glob.glob(
            str(root / "**" / "aiter" / "configs" / "model_configs" / "glm5_bf16_tuned_gemm.csv"),
            recursive=True,
        )
    )
    candidates.extend(
        Path(p)
        for p in glob.glob(
            str(root / "**" / "aiter" / "configs" / "bf16_tuned_gemm.csv"),
            recursive=True,
        )
    )

tmp_merged = Path("/tmp/aiter_configs/bf16_tuned_gemm.csv")
if tmp_merged.exists():
    candidates.append(tmp_merged)

seen = set()
patched = False
for candidate in candidates:
    candidate = candidate.resolve()
    if candidate in seen or not candidate.exists():
        continue
    seen.add(candidate)

    lines = candidate.read_text().splitlines()
    kept = [line for line in lines if not line.startswith(BAD_MARKER)]
    if len(kept) == len(lines):
        continue

    candidate.write_text("\n".join(kept) + ("\n" if kept else ""))
    patched = True
    print(f"Removed unsupported AITER OPUS GEMM tuned row from {candidate}")

if not patched:
    print("AITER GLM-5.2 bad OPUS GEMM config patch found no unsupported rows")
PY
