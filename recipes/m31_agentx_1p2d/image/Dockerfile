# MiniMax-M3.1 AgentX campaign image (local only, never pushed): the base atom-dev nightly with the combined
# aiter tree (main + ganyi's p1-integration + our PP4 aiter PRs) prebuilt in place of the nightly's own aiter.
#
# Context: aiter-tree-<short>.tar, the stage-1 output (build_aiter_tree.sh builds it once, in a GPU-less
# container of this same base, the way the nightly builds its aiter: PREBUILD_KERNELS=1 python3 setup.py develop,
# here GPU_ARCHS=gfx950, from a git bundle of the branch), and aot/ with extra tuned GEMM tables to AOT-compile.
# Kept from the base: torch 2.10.0+rocm7.2.4, triton 3.8.0, FlyDSL 0.5.3 (= aiter's pin), Mooncake 206d634 with
# the Store (= ATOM main's pin), MORI, LMCache, the atomesh binaries.
# Not baked: ATOM. Runs mount the combined ATOM tree at /workspace/ATOM and put it on PYTHONPATH (the CI
# launcher pd_slurm_job.sh and the recipe's containers.sh both do); /app/ATOM stays the base's dc6cd23bb.
ARG BASE_IMAGE=rocm/atom-dev:nightly_202611240650
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG AITER_SHA
ARG AITER_TAR
ARG AOT_JOBS=32
RUN rm -rf /app/aiter-test
ADD ${AITER_TAR} /app/
# The nightly's final stage: re-register the editable install over the prebuilt tree. AITER_USE_SYSTEM_TRITON
# keeps the base's triton 3.8.0 (aiter's setup.py would otherwise try to install its own triton 3.9).
RUN cd /app/aiter-test && test "$(cat AGX_AITER_COMMIT)" = "${AITER_SHA}" && \
    AITER_USE_SYSTEM_TRITON=1 pip install -e . --no-build-isolation && pip show amd-aiter && \
    python3 -c "import importlib.metadata as m; print('triton', m.version('triton'), 'flydsl', m.version('flydsl'), 'torch', m.version('torch'))" && \
    test "$(python3 -c 'import importlib.metadata as m; print(m.version("flydsl"))')" = "0.5.3" && \
    test "$(python3 -c 'import importlib.metadata as m; print(m.version("triton"))')" = "3.8.0+amd.rocm7.2.0.git7cb7b059" && \
    ls /app/aiter-test/aiter/jit/module_m3_tp1_norm.so /app/aiter-test/aiter/jit/module_fused_qknorm_idxrqknorm.so
# Optional: re-point the checkout to RETARGET_SHA when it has the SAME tree as AITER_SHA (e.g. the campaign branch's
# merge of the final PP4 heads, whose tree equals the trial merge the prebuild ran on). No file changes; enforced.
ARG RETARGET_SHA=""
RUN --mount=type=bind,source=aot,target=/tmp/agx-aot \
    if [ -n "${RETARGET_SHA}" ]; then cd /app/aiter-test && \
      git fetch -q /tmp/agx-aot/retarget.bundle "+refs/heads/*:refs/remotes/agx/*" && \
      test "$(git rev-parse "${RETARGET_SHA}^{tree}")" = "$(git rev-parse HEAD^{tree})" && \
      git checkout -q -B m31-agentx "${RETARGET_SHA}" && git log -1 --format=%H > AGX_AITER_COMMIT && \
      echo "re-pointed to $(cat AGX_AITER_COMMIT) (tree $(git rev-parse HEAD^{tree}))"; fi
# Pre-warm: the FlyDSL kernels of the decode recipe's own tuned GEMM tables (ganyi's C17 sets AITER_CONFIG_GEMM_BF16 /
# AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE to them), next to the default tables' kernels stage 1 compiled.
# aot/prewarm.tar (c17_aot_prewarm_inner.sh, run beside stage 1 on the same SHA) is unpacked over the image's
# caches; without it the GPU-less AOT replay runs here. Failures are logged, not fatal: pre-warm only.
RUN --mount=type=bind,source=aot,target=/tmp/agx-aot \
    mkdir -p /opt/m31agx-aot && cp /tmp/agx-aot/*.csv /tmp/agx-aot/SOURCE.txt /opt/m31agx-aot/ 2>/dev/null; \
    cd /app/aiter-test/aiter/jit && \
    if [ -f /tmp/agx-aot/prewarm.tar ]; then \
      tar -xf /tmp/agx-aot/prewarm.tar && \
      grep -E "replay gfx950|\[FAIL\]|BISECT|Total time" flydsl_cache/agx_c17_aot.log | cut -c1-240; \
    elif [ -f /opt/m31agx-aot/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm_m31c17.csv ]; then \
      cd /app/aiter-test && \
      AITER_AOT_IMPORT=1 GPU_ARCHS=gfx950 AITER_MAX_JOBS=${AOT_JOBS} \
      FLYDSL_RUNTIME_CACHE_DIR=/app/aiter-test/aiter/jit/flydsl_cache \
      AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE=/opt/m31agx-aot/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm_m31c17.csv \
      AITER_CONFIG_GEMM_BF16=/opt/m31agx-aot/bf16_tuned_gemm_m31c17.csv \
      nice -n 19 python3 -m aiter.aot.flydsl.gemm --csv /opt/m31agx-aot/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm_m31c17.csv /opt/m31agx-aot/bf16_tuned_gemm_m31c17.csv \
        > /opt/m31agx-aot/aot.log 2>&1 || echo "AOT replay of the decode tables reported failures (pre-warm only)"; \
      grep -E "replay gfx950|\[FAIL\]|BISECT|Total time" /opt/m31agx-aot/aot.log | cut -c1-240; \
    fi; \
    chmod -R a+rX /app/aiter-test/aiter/jit/flydsl_cache /app/aiter-test/aiter/jit/flydsl_aot; \
    du -sh /app/aiter-test/aiter/jit/flydsl_cache /app/aiter-test/aiter/jit/flydsl_aot
LABEL com.amd.m31agx.aiter.commit="${AITER_SHA}" com.amd.m31agx.base="${BASE_IMAGE}" \
      com.amd.m31agx.note="local campaign build, ATOM mounted at /workspace/ATOM"
