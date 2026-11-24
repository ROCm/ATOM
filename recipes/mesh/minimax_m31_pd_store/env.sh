# Shared settings of the M3.1 1P1D recipe, sourced inside each container.
export MODEL=${MODEL:-/share_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4}
# The address on the interface that owns the default route (eno0 on pit2-p03; eno1 is firewalled).
export NODE_IP=${NODE_IP:-$(ip route get 1.1.1.1 | awk '{for (i = 1; i < NF; i++) if ($i == "src") {print $(i + 1); exit}}')}
NODE_IF=$(ip -o -4 addr show | awk -v ip="${NODE_IP}" '{split($4, a, "/")} a[1] == ip {print $2; exit}')
export ATOM_HOST_IP=${NODE_IP}
export GLOO_SOCKET_IFNAME=${NODE_IF} MORI_SOCKET_IFNAME=${NODE_IF} NCCL_SOCKET_IFNAME=${NODE_IF}
# Both roles sit on one node: RCCL needs no IB (its probe of the host's ionic provider can fault).
export NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none
export AITER_LOG_LEVEL=WARNING PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 HF_HUB_OFFLINE=1
# One P/D transfer engine QP per endpoint on both sides (the Store's clients need it;
# endpoints with different QP counts cannot connect), and RDMA rather than HIP IPC.
export MC_NUM_QP_PER_EP=1 MC_DISABLE_HIP=1
# Numerics: shared expert as its own MXFP8 MLP, bf16 all-reduce.
export ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=0
unset AITER_QUICK_REDUCE_QUANTIZATION
export STORE_MASTER_PORT=${STORE_MASTER_PORT:-26051}
export STORE_METADATA_PORT=${STORE_METADATA_PORT:-26080}
export STORE_METRICS_PORT=${STORE_METRICS_PORT:-26090}
export P_PORT=8010 D_PORT=8020 ROUTER_PORT=8000
# KV and index cache dtype, the same on both roles: nvfp4 (default) or fp8.
export KV_DTYPE=${KV_DTYPE:-nvfp4}
export SPEC_ARGS="--method dspark --draft-model ${MODEL}/dspark --num-speculative-tokens 7"
# FORCE_AL=4.04 for benchmark runs (InferenceX golden AL for DSpark k=7); empty = real acceptance.
export FORCE_AL=${FORCE_AL:-}
AL_ARGS=""
[[ -n "${FORCE_AL}" ]] && AL_ARGS="--spec-decode-acceptance-length ${FORCE_AL}"
export AL_ARGS
export RUN=${RUN:-/runs/${TAG:-manual}}
mkdir -p "${RUN}"
