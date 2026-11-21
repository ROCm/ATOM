#!/usr/bin/env bash
# Shared NFS preflight for 2-node DSV4 GRPO finetune launches.
#
# Ensures head and worker use identical torchrun args before any rank enters
# Megatron (avoids NCCL hang from GBS / rollout path mismatch across nodes).
#
# Called from run_dsv4.sh when DSV4_PROFILE=flash and NNODES > 1.
# Set SKIP_PREFLIGHT=1 to bypass (not recommended).

preflight_dsv4_multinode() {
    local timeout_sec="${PREFLIGHT_TIMEOUT_SEC:-300}"
    local preflight_root="${LOG_DIR}/.dsv4_preflight"
    local latest_id_file="${preflight_root}/latest_id"
    local launch_id=""

    if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
        echo "[preflight] SKIP_PREFLIGHT=1 — skipping multinode config check"
        return 0
    fi

    if [[ ! -d "${LOG_DIR}" ]]; then
        echo "[preflight][ERROR] LOG_DIR not found: ${LOG_DIR}"
        return 1
    fi

    mkdir -p "${preflight_root}"

    if [[ "${NODE_RANK}" == "0" ]]; then
        launch_id="${PREFLIGHT_ID:-$(date +%Y%m%d_%H%M%S)}"
        echo "${launch_id}" > "${latest_id_file}"
        echo "[preflight] head published PREFLIGHT_ID=${launch_id}"
    else
        echo "[preflight] worker waiting for head PREFLIGHT_ID (timeout ${timeout_sec}s) ..."
        local waited=0
        while [[ ! -f "${latest_id_file}" && "${waited}" -lt "${timeout_sec}" ]]; do
            sleep 2
            waited=$((waited + 2))
        done
        if [[ ! -f "${latest_id_file}" ]]; then
            echo "[preflight][ERROR] timed out waiting for ${latest_id_file}"
            echo "  Start head (NODE_RANK=0) first, or set the same PREFLIGHT_ID on both nodes."
            return 1
        fi
        launch_id="$(tr -d '[:space:]' < "${latest_id_file}")"
        echo "[preflight] worker using PREFLIGHT_ID=${launch_id}"
    fi

    if [[ -z "${launch_id}" ]]; then
        echo "[preflight][ERROR] empty PREFLIGHT_ID"
        return 1
    fi

    export PREFLIGHT_ID="${launch_id}"
    local run_dir="${preflight_root}/runs/${launch_id}"
    mkdir -p "${run_dir}"

    local _train_iters="${TRAIN_ITERS:-${NUM_ROLLOUT:-10}}"
    local _eval_iters="${EVAL_ITERS:-0}"
    local _megatron_no_batch_p2p="${MEGATRON_NO_BATCH_P2P_COMM:-1}"
    local _rollout_path="${FAKE_ROLLOUT_DATA:-}"
    if [[ -z "${_rollout_path}" && "${NNODES}" -gt 1 && -n "${DATA_ROOT:-}" ]]; then
        _rollout_path="${DATA_ROOT}/models/fake_rollout.pt"
    elif [[ -z "${_rollout_path}" ]]; then
        _rollout_path="/root/models/fake_rollout.pt"
    fi
    local manifest="${run_dir}/node${NODE_RANK}.manifest"
    local tmp_manifest="${manifest}.$$"
    cat > "${tmp_manifest}" <<EOF
HOSTNAME=$(hostname)
NODE_RANK=${NODE_RANK}
IMAGE=${IMAGE}
NNODES=${NNODES}
NPROC_PER_NODE=${NPROC_PER_NODE}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT=${MASTER_PORT}
FAKE_ROLLOUT_DATA=${_rollout_path}
GBS=${GBS}
MBS=${MBS}
SEQ_LEN=${SEQ_LEN}
TRAIN_ITERS=${_train_iters}
EVAL_ITERS=${_eval_iters}
MODEL_NAME=${MODEL_NAME}
DSV4_HC_MULT=${DSV4_HC_MULT}
SKIP_PREPARE=${SKIP_PREPARE}
TP=${TP:-4}
PP=${PP:-4}
EP=${EP:-4}
ETP=${ETP:-1}
DECODER_FIRST_PP_LAYERS=${DECODER_FIRST_PP_LAYERS:-11}
DECODER_LAST_PP_LAYERS=${DECODER_LAST_PP_LAYERS:-10}
MEGATRON_NO_BATCH_P2P_COMM=${_megatron_no_batch_p2p}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
NCCL_IB_GDR_LEVEL=${NCCL_IB_GDR_LEVEL:-}
NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens14np0}
V4_SPARSE_MLA_BACKEND=${V4_SPARSE_MLA_BACKEND}
V4_INDEXER_IMPL=${V4_INDEXER_IMPL}
LUMEN_DIR=${LUMEN_DIR}
MEGATRON_PATH=${MEGATRON_PATH}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION}
DISTRIBUTED_TIMEOUT_MINUTES=${DISTRIBUTED_TIMEOUT_MINUTES}
DSV4_ENABLE_RECOMPUTE=${DSV4_ENABLE_RECOMPUTE:-1}
LUMEN_DSV4_LINEAR_FP8=${LUMEN_DSV4_LINEAR_FP8}
EOF
    mv "${tmp_manifest}" "${manifest}"
    echo "[preflight] wrote ${manifest}"

    echo "[preflight] waiting for ${NNODES} node manifest(s) in ${run_dir} ..."
    local waited=0
    while [[ "${waited}" -lt "${timeout_sec}" ]]; do
        local count=0
        local r=0
        for ((r = 0; r < NNODES; r++)); do
            [[ -f "${run_dir}/node${r}.manifest" ]] && count=$((count + 1))
        done
        if [[ "${count}" -eq "${NNODES}" ]]; then
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done

    local missing=()
    local r=0
    for ((r = 0; r < NNODES; r++)); do
        if [[ ! -f "${run_dir}/node${r}.manifest" ]]; then
            missing+=("${r}")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "[preflight][ERROR] missing manifest(s) for node rank(s): ${missing[*]}"
        echo "  Launch the worker/head with the same PREFLIGHT_ID=${launch_id} within ${timeout_sec}s."
        return 1
    fi

    local ref="${run_dir}/node0.manifest"
    local diffs=0
    for ((r = 1; r < NNODES; r++)); do
        local other="${run_dir}/node${r}.manifest"
        local ref_cfg tmp_cfg
        ref_cfg="$(grep -v '^HOSTNAME=' "${ref}" | grep -v '^NODE_RANK=')"
        tmp_cfg="$(grep -v '^HOSTNAME=' "${other}" | grep -v '^NODE_RANK=')"
        if [[ "${ref_cfg}" != "${tmp_cfg}" ]]; then
            echo "[preflight][ERROR] config mismatch: node0 vs node${r}"
            diff -u <(printf '%s\n' "${ref_cfg}") <(printf '%s\n' "${tmp_cfg}") | sed 's/^/  /' || true
            diffs=$((diffs + 1))
        fi
    done
    if [[ "${diffs}" -gt 0 ]]; then
        echo "[preflight][ERROR] fix mismatched env on all nodes, then relaunch with a new PREFLIGHT_ID."
        return 1
    fi

    if [[ "${NODE_RANK}" == "0" ]]; then
        date -Iseconds > "${run_dir}/.ok"
    else
        waited=0
        while [[ ! -f "${run_dir}/.ok" && "${waited}" -lt 30 ]]; do
            sleep 1
            waited=$((waited + 1))
        done
    fi

    echo "[preflight] OK — all ${NNODES} nodes agree (PREFLIGHT_ID=${launch_id})"
    echo "[preflight]   rollout=${_rollout_path} GBS=${GBS} MLA=${V4_SPARSE_MLA_BACKEND} MHC=aiter"
    return 0
}
