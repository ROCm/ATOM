# MiniMax-M3.1 1P1D (P = PP4 x TP1, D = DP4/EP4) with a MORI-UMBP KV tier on the prefill node.
# = lirui927's PD image (ATOM m31-dpep-opt1, aiter 9c3f0ef, mori c5bdc391, recipe /app/m31_pd_recipe)
#   + mori c5bdc391 rebuilt with UMBP and UMBP-only fixes (MORI-IO and EP unchanged, so P and D stay compatible)
#   + ATOM zejun/m31-pd-umbp-store (this branch: umbp_store on the stack, save_policy lazy, prefetched lookups)
#   + the kit's PD_UMBP_GB mode (kit.patch, umbp_server.sh).
# Build from this directory: docker build -t <tag> .
ARG BASE=rocm/fw-bringup:gfx1250-minimax-m3-20261125-v2
FROM ${BASE}

# UMBP's control plane needs gRPC; without it mori builds with UMBP off.
RUN apt-get update && apt-get install -y --no-install-recommends libgrpc++-dev libgrpc-dev protobuf-compiler-grpc \
 && rm -rf /var/lib/apt/lists/*

# ROCm/mori: #929 (no full-buffer rescan per key), #1049 (word-wise page bitmap search), #1048 (transfer buffers on
# the heap), #1086 (io_uring EAGAIN), #1023 (group eviction of a chunk's w<r>of<N> keys), #941 (re-register after a
# server restart). All under src/umbp; none touches src/io, src/ops or shmem.
ARG MORI_BASE=c5bdc3913660f6b663ae09ba7c7176291d6f5343
ARG MORI_PICKS="8016e664654431a1fe322df6983e974478cf4639 d6608db81ba57e52bfd18f1d74828e790a409bba d9b9109cbed6edf6f8da26afc16d816a672d1802 20455c2fc99c040704c4a51d34f856f425bc6000 fadcc8c56ea95d9b0375d265c94a473ada974087 614089ffb3468c37cd15e469224a278e6948ca75 eed40da55a8824f5cc27847cc7fdd6f5f05989b9 70b771b7ec2def0d9ee5b218471d68a279c5a8eb c3c6e7d479598b8ed368d623a838b3d834733a56"
RUN cd /app/mori \
 && git -c advice.detachedHead=false checkout -q ${MORI_BASE} \
 && git fetch -q origin main pull/1049/head pull/1048/head pull/1023/head pull/941/head \
 && git -c user.name=build -c user.email=build@localhost cherry-pick -x ${MORI_PICKS} \
 && rm -rf build \
 && ROCM_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel \
    HIP_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel \
    GPU_ARCHS=gfx1250 GPU_TARGETS=gfx1250 MORI_GPU_ARCHS=gfx1250 BUILD_UMBP=ON \
    python3 -m pip install --break-system-packages --no-build-isolation --no-deps -e /app/mori \
 && python3 -c "import mori.io, mori.umbp as u; u.UMBPStandaloneProcessConfig().lookup_timeout_ms = 1"

# ATOM: this branch (the image's /app/ATOM is an editable install, so a checkout is enough).
ARG ATOM_REPO=https://github.com/ROCm/ATOM.git
ARG ATOM_REF=zejun/m31-pd-umbp-store
RUN cd /app/ATOM && git fetch -q ${ATOM_REPO} ${ATOM_REF} && git -c advice.detachedHead=false checkout -q FETCH_HEAD \
 && rm -rf /root/.cache/atom/* \
 && python3 -c "import atom.kv_transfer.offload.umbp_store.scheduler"

# The kit's PD_UMBP_GB mode (also adds umbp_server.sh).
COPY kit.patch /app/m31_pd_recipe/kit.umbp.patch
RUN cd /app/m31_pd_recipe/kit && patch -p1 < ../kit.umbp.patch && chmod +x umbp_server.sh
