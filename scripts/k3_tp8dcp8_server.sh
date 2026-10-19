#!/bin/bash
# Kimi-K3 on two HeliosM trays (2 x 4 MI455, gfx1250 B0) as one tensor-parallel group of 8: attention TP8 with
# decode context parallel 8, MoE expert-parallel over the same 8 GPUs, RCCL for every collective.
#
# Self-contained: it needs only a container of rocm/fw-bringup:gfx1250-atom-k3-20261016 (aiter, mori, Triton,
# torch) and the Kimi-K3 checkpoint. It fetches the ATOM it serves, ROCm/ATOM branch gyu/k3-455-tp8 at the commit
# below (the image's /app/ATOM, i.e. main d82f906b + #2557 + #2554, plus TP across nodes and DCP decode on
# gfx1250's persistent PS1 kernel), patches the image's aiter with ROCm/aiter branch gyu/k3-455-tp8's cc8d4ff12
# (embedded below; the grouped MoE GEMM's k-tile), and sets every environment variable itself.
#
# One container per tray, both from that image (--network=host is required: the ranks meet on data-plane IPs):
#   docker run -d --name k3tp8 -w /tmp --device=/dev/kfd --device=/dev/dri --network=host --ipc=host --pid=host \
#     --group-add $(getent group video | cut -d: -f3) --group-add $(getent group render | cut -d: -f3) \
#     --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --shm-size=128g \
#     -v <dir holding Kimi-K3/>:/models:ro -v <work>/cache:/root/.cache -v <work>/triton:/root/.triton \
#     --entrypoint sleep rocm/fw-bringup:gfx1250-atom-k3-20261016 infinity
#   docker cp k3_tp8dcp8_server.sh k3tp8:/root/
# Then on each tray, in either order (each waits for the other):
#   node0, TP ranks 0-3, serves :8000   docker exec -d k3tp8 bash -c 'bash /root/k3_tp8dcp8_server.sh 0 <node0 IP> > /root/.cache/tp8.log 2>&1'
#   node1, TP ranks 4-7, workers only   docker exec -d k3tp8 bash -c 'bash /root/k3_tp8dcp8_server.sh 4 <node0 IP> > /root/.cache/tp8.log 2>&1'
#   until curl -sf -m 5 http://<node0 IP>:8000/health; do sleep 15; done    # ~3 min; ~10 min while aiter JIT-builds
# <node0 IP> is node0's data-plane (enp1s0f1) address, the one you ssh to on these trays.
# Stop it on each tray with only this container's processes: --pid=host makes a plain pkill reach other people's.
#   docker top k3tp8 -eo pid,args | awk '/openai_server|ATOM::/ {print $1}' | xargs -r kill
#
# Knobs, as environment variables (the same on both trays):
#   K3_MODE      accuracy (default: no prefix caching, 16384 batched tokens) | agentx (the TP8 + DCP8 server options
#                of ATOM's recipes/Agentic-Kimi-K3.md: prefix caching, 8192 batched tokens, KDA state checkpoints
#                only where a request resumes, a CUDA graph for every batch size up to K3_MAX_SEQS)
#   K3_MAX_SEQS  --max-num-seqs of the whole server (default 64). The recipe uses 2 x concurrency for AgentX.
#   K3_MOE       ep (default) | tp. tp splits every expert 8 ways as the MI355 recipe does (inter_dim 384 per rank).
#                Without aiter cc8d4ff12 the gfx1250 grouped GEMM dropped the last 128 of that K, which cost 7 points
#                of GSM8K (strict-match 0.8923 against 0.9575 with ep); with it the MoE matches aiter's reference
#                (rel_l2 0.0027). End-to-end GSM8K with tp is not measured yet.
#   K3_ATOM_DIR  the ATOM checkout (default /root/.cache/atom-gyu-k3-455-tp8); fetched from GitHub on first use
#   K3_MODEL     checkpoint directory (default /models/Kimi-K3)
#   K3_PORT      API port on node0 (default 8000)
#
# Validated 2026-10-16 on CT11 (node0) + CT10, driver loaded without perf setup: gate PASS, full GSM8K 5-shot
# strict-match 0.9575 +-0.0056 (dp8ep8 on the same image family: 0.9651). Not supported here: DSpark speculative
# decoding (ATOM allows it with DCP only on gfx950) and the recipe's --block-size 128 (untested; 16 is used).
# Keep aiter's custom all-reduce off across the trays (AITER_CUSTOM_AR_USE_SYMM_MEM, ATOM_USE_CUSTOM_ALL_GATHER):
# both times it was on, both trays took a fatal CPU error (Processor Context Corrupt) during startup and rebooted.
set -euo pipefail
RANK0=${1:-}
MASTER=${2:-}
case $RANK0 in 0 | 4) ;; *) echo "usage: k3_tp8dcp8_server.sh <first TP rank of this tray: 0|4> <node0 IP>" >&2; exit 2 ;; esac
[ -n "$MASTER" ] || { echo "usage: k3_tp8dcp8_server.sh <first TP rank of this tray: 0|4> <node0 IP>" >&2; exit 2; }

ATOM_REPO=https://github.com/ROCm/ATOM.git
# Branch gyu/k3-455-tp8's code; the branch's later commits only add or edit this script. Both trays must serve the same.
ATOM_COMMIT=571d22de7175af319844027553348c33012e599e
ATOM_DIR=${K3_ATOM_DIR:-/root/.cache/atom-gyu-k3-455-tp8}
MODEL=${K3_MODEL:-/models/Kimi-K3}
MAX_SEQS=${K3_MAX_SEQS:-64}
[ -f "$MODEL/config.json" ] || { echo "no checkpoint at $MODEL" >&2; exit 2; }

# --- the ATOM this serves
git_() { git -c safe.directory='*' -C "$ATOM_DIR" "$@"; }
if [ "$(git_ rev-parse HEAD 2>/dev/null)" != "$ATOM_COMMIT" ]; then
  if [ ! -e "$ATOM_DIR/.git" ]; then
    mkdir -p "$ATOM_DIR" && git_ init -q && git_ remote add origin "$ATOM_REPO"
  fi
  [ -z "$(git_ status --porcelain --untracked-files=no)" ] \
    || { echo "$ATOM_DIR has local changes; not replacing them with $ATOM_COMMIT" >&2; exit 2; }
  echo "fetching ATOM $ATOM_COMMIT into $ATOM_DIR"
  git_ fetch -q --depth 1 origin "$ATOM_COMMIT"
  git_ -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
fi
[ -z "$(git_ status --porcelain --untracked-files=no)" ] \
  || { echo "$ATOM_DIR has local changes; this script serves $ATOM_COMMIT as committed" >&2; exit 2; }
# Ahead of the image's editable /app/ATOM, whose finder comes after the path search
export PYTHONPATH=$ATOM_DIR${PYTHONPATH:+:$PYTHONPATH}
cd /tmp   # from /app, the image's aiter checkout would shadow the aiter package
origin=$(python3 -c 'import importlib.util as u; print(u.find_spec("atom").origin)')
[ "$origin" = "$ATOM_DIR/atom/__init__.py" ] || { echo "atom resolves to $origin, not $ATOM_DIR" >&2; exit 2; }

# --- aiter: the image's checkout (main dee019f6 + #6525 + #6530) plus ROCm/aiter cc8d4ff12. The gfx1250 TDM grouped
# GEMM runs K // tile_k k-tiles with no tail, and the default tile_k of a shape without a tuned row, 256, does not
# divide K3's TP8 expert slice (inter_dim 384): stage 2 dropped the last 128 of K. Only K3_MOE=tp has such a shape.
AITER_DIR=$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("aiter").origin)))')
AITER_PATCH=$(cat <<'EOF'
diff --git a/aiter/ops/flydsl/grouped_gemm_mxfp4.py b/aiter/ops/flydsl/grouped_gemm_mxfp4.py
index 40ed66f28..c4c76f0e0 100644
--- a/aiter/ops/flydsl/grouped_gemm_mxfp4.py
+++ b/aiter/ops/flydsl/grouped_gemm_mxfp4.py
@@ -156,6 +156,9 @@ def flydsl_grouped_gemm_a8w4_masked(
             raise ValueError(f"situ_beta must be > 0, got {situ_beta!r}")
         if float(situ_linear_beta) <= 0.0:
             raise ValueError(f"situ_linear_beta must be > 0, got {situ_linear_beta!r}")
+    if K % tile_k:
+        # The kernel runs K // tile_k k-tiles and has no tail: the rest of K would be dropped.
+        raise ValueError(f"[grouped gemm] K={K} is not a multiple of tile_k={tile_k}")
     num_buffers = min(num_buffers, max(1, K // tile_k))
     has_bias = 1 if bias is not None else 0
     bias_ptr = ptr_arg(bias) if bias is not None else ptr_arg(a)
diff --git a/aiter/ops/flydsl/grouped_moe_gfx1250.py b/aiter/ops/flydsl/grouped_moe_gfx1250.py
index 5959c343d..74601367a 100644
--- a/aiter/ops/flydsl/grouped_moe_gfx1250.py
+++ b/aiter/ops/flydsl/grouped_moe_gfx1250.py
@@ -56,6 +56,31 @@ def _grouped_weight_uint8(w: torch.Tensor) -> torch.Tensor:
     return out
 
 
+@functools.lru_cache(maxsize=None)
+def _fit_tile_k(tile_k: int, k: int) -> int:
+    """The widest k-tile, no wider than ``tile_k``, that the TDM GEMM can run for K = ``k``.
+
+    The GEMM runs K // tile_k k-tiles and has no tail, so a tile that does not
+    divide K silently drops the rest of it. Every tuned row divides its K; the
+    defaults (256) do not for a tensor-parallel expert slice such as Kimi-K3's
+    3072 / 8 = 384. Only the widths the tuned rows use are tried.
+    """
+    tile_k = int(tile_k)
+    for t in (tile_k, 512, 256, 128):
+        if t <= tile_k and t % 128 == 0 and k % t == 0:
+            if t != tile_k:
+                logger.info(
+                    "[grouped-moe tdm] tile_k=%d does not divide K=%d; using tile_k=%d",
+                    tile_k,
+                    k,
+                    t,
+                )
+            return t
+    raise ValueError(
+        f"[grouped-moe tdm] no k-tile of 512, 256 or 128 divides K={k} (tile_k={tile_k})"
+    )
+
+
 def _as_bool(value, default: bool) -> bool:
     if value is None or str(value).strip() == "":
         return default
@@ -629,6 +654,8 @@ def _grouped_a8w4_tdm_moe(
         tile_n2 = tile_n
     if tile_k2 is None:
         tile_k2 = tile_k
+    tile_k = _fit_tile_k(tile_k, model_dim)
+    tile_k2 = _fit_tile_k(tile_k2, inter_dim)
     if num_buffers2 is None:
         num_buffers2 = num_buffers
     if waves_per_tensor_tdm2 is None:
EOF
)
aiter_git() { git -c safe.directory='*' -C "$AITER_DIR" "$@"; }
if aiter_git apply --check -R <<<"$AITER_PATCH" 2>/dev/null; then
  :   # applied by an earlier start
elif aiter_git apply --check <<<"$AITER_PATCH"; then
  aiter_git apply <<<"$AITER_PATCH"
  echo "patched $AITER_DIR with aiter cc8d4ff12 (grouped MoE k-tile)"
else
  echo "aiter cc8d4ff12 neither applies to nor is already in $AITER_DIR: not the K3 image's aiter?" >&2; exit 2
fi

# --- architecture (B0 runs the image's code objects as they are)
export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 MORI_GPU_ARCHS=gfx1250 GPU_TARGETS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0 ENABLE_CK=0
# mori opens libamd_smi.so; TheRock's wheels ship only libamd_smi.so.27
export MORI_AMD_SMI_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamd_smi.so.27
# --- attention: MLA decode on aiter's persistent PS1 FP8 kernel, which takes the 96 heads the DCP gather makes
export ATOM_USE_TRITON_MLA=0 AITER_MLA_DECODE_PS1_FLYDSL=1 ATOM_USE_TRITON_MLA_SHUFFLE_KV=0 ATOM_USE_UNIFIED_ATTN=1
# --- MoE
export ATOM_MOE_GU_ITLV=1             # required on gfx1250
export ATOM_USE_TRITON_MOE_DECODE=0   # K3's activation is situ, not SiLU
export AITER_USE_GROUPED_GEMM=1
# --- GEMM and quantization: the online FP8 layers take E8M0 block scales (#2554) on preshuffled weights
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1 ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1
export AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1
# --- communication: RCCL over UALink (without NCCL_MNNVL_ENABLE it falls back to TCP); no custom collectives
export NCCL_MNNVL_ENABLE=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=1
export MORI_SOCKET_IFNAME=enp1s0f1 NCCL_SOCKET_IFNAME=enp1s0f1 GLOO_SOCKET_IFNAME=enp1s0f1   # the image says lo
export ATOM_USE_CUSTOM_ALL_GATHER=0 ATOM_DP_LM_HEAD_MODE=default
unset AITER_CUSTOM_AR_USE_SYMM_MEM
# --- loading
export HSA_USE_SVM=1 HSA_ENABLE_SDMA=1 ATOM_LOADER_NUM_THREADS=4
export HSA_XNACK=$([ "$(cat /sys/module/amdgpu/parameters/noretry 2>/dev/null)" = 1 ] && echo 0 || echo 1)
export TRITON_CACHE_DIR=/root/.triton/cache
# A GPU memory fault dumps ~100 GB; the default, the cwd, is the container layer on the tray's root disk
export HSA_COREDUMP_PATTERN=/root/.cache/gpucore.%p
# --- the rest of what the image's /opt/k3/env_k3.sh exports for its dp8ep8 launch, kept so that this reproduces
# the validated runs exactly. At DP 1 nothing reads the MegaMoE and mori ones; ATOM main reads neither of the last two.
export ATOM_MEGA_STAGE1_FUSED=1 ATOM_MEGA_STAGE2_FUSED=1 ATOM_MORI_V2=1 ATOM_MORI_V2_FUSED=1 MEGA_DISPATCH=flydsl \
  MEGA_DISPATCH_WIRE=fp4 ATOM_MEGA_COMBINE_WIRE=fp4 ATOM_MEGA_DECODE_FAST_PATH=0 ATOM_MEGA_MASK_PAD_ROWS=0 \
  MORI_EP_TOKOFF_EXT=0 ATOM_UNFUSED_GATHER_KV_B_PROJ=1 ATOM_MLA_QH128_ASM_PAGE_SIZE=0
# --- torch profiler: per-rank trace directories; recording runs only between /start_profile and /stop_profile
export ATOM_PROFILER_TIMEOUT=3600 ROCPROFILER_REGISTER_FORCE_LOAD=0

case "${K3_MODE:-accuracy}" in
  accuracy) MODE_ARGS=(--max-num-batched-tokens 16384 --no-enable_prefix_caching) ;;
  agentx)   export ATOM_DP_SESSION_AFFINITY=1   # read only with DP > 1
            MODE_ARGS=(--max-num-batched-tokens 8192 --enable_prefix_caching --state-checkpoint-interval-tokens -1
                       --cudagraph-capture-sizes "[$(seq -s, 1 "$MAX_SEQS")]") ;;
  *) echo "K3_MODE must be accuracy or agentx, got '$K3_MODE'" >&2; exit 2 ;;
esac
case "${K3_MOE:-ep}" in
  ep) MOE_ARGS=(--enable-expert-parallel) ;;
  tp) MOE_ARGS=() ;;
  *) echo "K3_MOE must be ep or tp, got '$K3_MOE'" >&2; exit 2 ;;
esac

# Online FP8 for the checkpoint's BF16 layers at load (the image's own launch passes the same)
ONLINE_QUANT='{"global_quant_config": "per_block_fp8", "layer_quant_config": {"*self_attn.kv_b_proj": "ptpc_fp8"}, "exclude_layer": ["*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*"]}'

echo "[k3_tp8dcp8_server] $(date '+%F %T') $(hostname): TP ranks $RANK0-$((RANK0 + 3)) of 8, master $MASTER," \
  "K3_MODE=${K3_MODE:-accuracy} K3_MOE=${K3_MOE:-ep} max-num-seqs $MAX_SEQS HSA_XNACK=$HSA_XNACK, ATOM $ATOM_COMMIT"
exec python3 -m atom.entrypoints.openai_server \
  --model "$MODEL" --served-model-name moonshotai/Kimi-K3 --trust-remote-code --server-port "${K3_PORT:-8000}" \
  -tp 8 --tensor-parallel-size-local 4 --tensor-parallel-rank "$RANK0" --decode-context-parallel-size 8 \
  --data-parallel-master-ip "$MASTER" --data-parallel-base-port 29700 "${MOE_ARGS[@]}" \
  --kv_cache_dtype fp8 --cudagraph-mode FULL --max-num-seqs "$MAX_SEQS" --gpu-memory-utilization 0.90 \
  "${MODE_ARGS[@]}" --online_quant_config "$ONLINE_QUANT" --torch-profiler-dir /root/.cache/torch_profile \
  --disable_uvicorn_access_log
