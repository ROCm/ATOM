#!/usr/bin/env bash
# Stage 1 (v2), INSIDE a GPU-less container of the base nightly (rocm/atom-dev:nightly_202611240650):
# move the image's own aiter checkout (/app/aiter-test, a ROCm/aiter clone at 82b6b4f19 with CK f93c029e) to the
# combined tree from a git bundle, prebuild it the way the nightly builds its own (PREBUILD_KERNELS=1 python3 setup.py
# develop), for gfx950 only, and write /app/aiter-test (minus the jit/build intermediates) as a tar.
# Usage: aiter_prebuild_inner_v2.sh SHA BUNDLE OUT_TAR JOBS [SEED_DIR]
#  SEED_DIR: holds flydsl_cache/ and flydsl_aot/ of an earlier build; they seed this build's FlyDSL caches. Safe: FlyDSL
#    keys entries by the traced IR and aiter's flydsl_aot checks each entry against the hashes of the sources that built
#    it, so a changed kernel misses and rebuilds; unchanged kernels skip the AOT compile.
#  AGX_TOLERATE_AOT_8WAVE_OVERFLOW=1: setup.py's FlyDSL AOT aborts the whole build on any failed replay. The PP4 PR
#    Jasen/m31-pp-m31-tp1-mxfp8-rows makes two GEMM bisect probes (N=24576 K=6144, M=45056 / 49153) reach the MXFP8 8-wave
#    kernel's 32-bit-indexing refusal. With this set, aiter/aot/flydsl/common.py is patched IN THE BUILD CONTAINER ONLY to
#    log that error instead of raising, the build goes on, and the file is restored to the SHA right after; the build
#    still fails unless every [FAIL] line is that known refusal. The tree in the tar is exactly the SHA.
set -euo pipefail
SHA=$1 BUNDLE=$2 OUT_TAR=$3 JOBS=${4:-48} SEED_DIR=${5:-}
t0=$(date +%s)
git config --global --add safe.directory '*'
cd /app/aiter-test
git fetch -q "${BUNDLE}" "+refs/heads/*:refs/remotes/agx/*"
git checkout -q -B m31-agentx "${SHA}"
git submodule update --init --recursive   # CK pin f93c029e is the image's: no fetch
[[ "$(git -C 3rdparty/composable_kernel rev-parse HEAD)" == "$(git ls-tree HEAD 3rdparty/composable_kernel | awk '{print $3}')" ]]
git clean -ffdxq
if [[ -n "${SEED_DIR}" ]]; then
  cp -a "${SEED_DIR}/flydsl_cache" aiter/jit/flydsl_cache
  cp -a "${SEED_DIR}/flydsl_aot" aiter/jit/flydsl_aot
  echo "[agx] seeded FlyDSL caches from ${SEED_DIR}: $(du -sh aiter/jit/flydsl_cache | cut -f1) / $(du -sh aiter/jit/flydsl_aot | cut -f1)"
fi
echo "[agx] tree $(git log -1 --format='%H %s' | cut -c1-120); status: $(git status --short --untracked-files=no | wc -l) changes"
export AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=1 GPU_ARCHS=gfx950 AITER_MAX_JOBS=${JOBS} MAX_JOBS=${JOBS}
python3 -m pip install -q -r requirements.txt
before=$(python3 -c "import importlib.metadata as m;print(m.version('triton'), m.version('flydsl'), m.version('torch'))")
if [[ "${AGX_TOLERATE_AOT_8WAVE_OVERFLOW:-0}" == "1" ]]; then
  python3 - <<'PY'
p = "aiter/aot/flydsl/common.py"
s = open(p).read()
old = """        if errors:
            raise AssertionError(f"[aiter] FlyDSL AOT failures: {'; '.join(errors)}")"""
new = """        if errors:
            print(f"[agx] TOLERATED FlyDSL AOT failures (build-container patch): {'; '.join(errors)}", flush=True)"""
assert s.count(old) == 1, "run_aot raise not found"
open(p, "w").write(s.replace(old, new))
print("[agx] build-container patch: run_aot logs AOT failures instead of raising")
PY
fi
rc=0
nice -n 19 python3 setup.py develop > /tmp/aiter_setup_develop.log 2>&1 || rc=$?
if [[ "${AGX_TOLERATE_AOT_8WAVE_OVERFLOW:-0}" == "1" ]]; then
  git checkout -- aiter/aot/flydsl/common.py && rm -rf aiter/aot/flydsl/__pycache__
  echo "[agx] restored aiter/aot/flydsl/common.py; tracked changes now: $(git status --short --untracked-files=no | wc -l)"
  grep "\[agx\] TOLERATED" /tmp/aiter_setup_develop.log || true
  nfail=$(grep -c "^  \[FAIL\]" /tmp/aiter_setup_develop.log || true)
  nknown=$(grep "^  \[FAIL\]" /tmp/aiter_setup_develop.log | grep -c "N=24576 K=6144 overflows 32-bit buffer indexing" || true)
  echo "[agx] AOT [FAIL] lines: ${nfail}, of which the known 8-wave overflow refusal: ${nknown}"
  grep "^  \[FAIL\]" /tmp/aiter_setup_develop.log | cut -c1-300 || true
  [[ "${nfail}" == "${nknown}" ]] || { echo "[agx] ERROR: an AOT failure other than the known one"; exit 1; }
fi
[[ ${rc} -eq 0 ]] || { tail -80 /tmp/aiter_setup_develop.log; exit 1; }
[[ "$(git status --short --untracked-files=no | wc -l)" == "0" ]] || { echo "[agx] ERROR: tracked files differ from ${SHA}"; git status --short --untracked-files=no; exit 1; }
tail -5 /tmp/aiter_setup_develop.log
after=$(python3 -c "import importlib.metadata as m;print(m.version('triton'), m.version('flydsl'), m.version('torch'))")
echo "[agx] versions before: ${before} | after: ${after}"
[[ "${before}" == "${after}" ]] || { echo "[agx] ERROR: setup.py changed triton/flydsl/torch"; exit 1; }
n_so=$(ls aiter/jit/*.so | wc -l)
echo "[agx] prebuilt modules: ${n_so}; flydsl_cache: $(du -sh aiter/jit/flydsl_cache | cut -f1); flydsl_aot: $(du -sh aiter/jit/flydsl_aot | cut -f1); build dir: $(du -sh aiter/jit/build | cut -f1)"
for m in module_aiter_core module_m3_tp1_norm module_m3_tp1_mx8_quant module_m3_tp1_moe_g2 module_m3_tp1_qknorm module_m3_fused_index_topk module_m3_tt_sparse_prefill module_fused_qknorm_idxrqknorm; do
  if python3 -c "import json,sys; sys.exit(0 if sys.argv[1] in json.load(open('aiter/jit/optCompilerConfig.json')) else 1)" "$m"; then cfg=yes; else cfg=no; fi
  [[ -f aiter/jit/${m}.so ]] && echo "[agx] ${m}: in config=${cfg}, .so built" || echo "[agx] ${m}: in config=${cfg}, .so NOT prebuilt"
done
cp /tmp/aiter_setup_develop.log aiter/jit/agx_setup_develop.log
git log -1 --format=%H > AGX_AITER_COMMIT
chmod -R a+rX /app/aiter-test
tar -C /app --exclude=aiter-test/aiter/jit/build -cf "${OUT_TAR}.part" aiter-test
mv "${OUT_TAR}.part" "${OUT_TAR}"
echo "[agx] wrote ${OUT_TAR} ($(du -sh "${OUT_TAR}" | cut -f1)) in $(( ($(date +%s) - t0) / 60 )) min"
