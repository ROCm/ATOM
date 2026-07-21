#!/usr/bin/env bash
# Patch AITER's CK two-stage MoE generator so GLM-5.2's per-token FP8 tuned
# config can dispatch its stage2 block_m=16 kernel.
set -euo pipefail

if [[ "${ATOM_PATCH_AITER_GLM52_MOE_B16:-1}" == "0" ]]; then
  echo "AITER GLM-5.2 MoE b16 patch disabled by ATOM_PATCH_AITER_GLM52_MOE_B16=0"
  exit 0
fi

python3 - <<'PY'
import glob
import os
import shutil
import site
from pathlib import Path

needle = """        if (block_m == 32)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 128, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}"""

replacement = """        if (block_m == 16)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 16, 128, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}
        else if (block_m == 32)
        {{
            return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 128, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {Quant} == static_cast<int>(QuantType::per_Tensor), {MulRoutedWeight}, {ActOP}>;
        }}"""

search_roots = [Path("/app/aiter-test")]
for site_dir in site.getsitepackages():
    search_roots.append(Path(site_dir))
for path_entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    if path_entry:
        search_roots.append(Path(path_entry))

candidates = []
for root in search_roots:
    if root.exists():
        candidates.extend(
            Path(p)
            for p in glob.glob(
                str(root / "**" / "ck_gemm_moe_2stages_codegen" / "gen_instances.py"),
                recursive=True,
            )
        )

seen = set()
patched = False
already_patched = False
for candidate in candidates:
    candidate = candidate.resolve()
    if candidate in seen:
        continue
    seen.add(candidate)

    text = candidate.read_text()
    if replacement in text:
        already_patched = True
        print(f"AITER GLM-5.2 MoE b16 patch already present in {candidate}")
        continue
    if needle not in text:
        continue

    candidate.write_text(text.replace(needle, replacement, 1))
    patched = True
    print(f"Patched AITER GLM-5.2 MoE b16 dispatch in {candidate}")

    # Remove stale JIT outputs generated before the dispatch table was patched.
    for parent in candidate.parents:
        jit_dir = parent / "aiter" / "jit"
        if jit_dir.exists():
            for artifact in jit_dir.glob(
                "module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_token_mulWeightStage2*"
            ):
                if artifact.is_file() or artifact.is_symlink():
                    artifact.unlink()
                    print(f"Removed stale AITER JIT artifact {artifact}")
                elif artifact.is_dir():
                    shutil.rmtree(artifact, ignore_errors=True)
                    print(f"Removed stale AITER JIT artifact {artifact}")
            build_dir = jit_dir / "build"
            if build_dir.exists():
                for artifact in build_dir.glob(
                    "module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_token_mulWeightStage2*"
                ):
                    shutil.rmtree(artifact, ignore_errors=True)
                    print(f"Removed stale AITER JIT build {artifact}")

if not patched and not already_patched:
    print("WARNING: AITER GLM-5.2 MoE b16 patch target was not found; continuing without patch")
PY
