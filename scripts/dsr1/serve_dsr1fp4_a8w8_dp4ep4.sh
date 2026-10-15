rm -r ~/.cache/atom
# TRACE=1: collect torch profiler traces. Has to go through the CLI flag:
# EngineArgs.torch_profiler_dir defaults to a hard None that _get_engine_kwargs
# forwards into the Config, overriding the field's ATOM_TORCH_PROFILER_DIR
# default_factory -- so the env var alone silently traces nothing. Kept off the
# / overlay, which fills up; each rank writes its own subdir and traces get
# large fast.
TRACE="${TRACE:-0}"
TRACE_DIR="${TRACE_DIR:-/app/traces}"
TRACE_ARGS=()
FAKE_EPLB="${FAKE_EPLB:-1}"
FAKE_EPLB_ARGS=()
if [ "$FAKE_EPLB" = "1" ]; then
  FAKE_EPLB_ARGS=(--fake-eplb --hf-overrides '{"num_hash_layers": 0}')
fi
if [ "$TRACE" = "1" ]; then
  mkdir -p "$TRACE_DIR"
  TRACE_ARGS=(--torch-profiler-dir "$TRACE_DIR")
  echo "[trace] torch profiler dir: $TRACE_DIR (client needs --profile)"
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="${1:-/data/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4/}"
SERVER_ENV=(
  ATOM_MLA_ABSORB_BF16_FUSE_QOUT=1
  # --- ASM MLA page64 + MoE pad-row masking 默认开启 (20261006) ---
  ATOM_MLA_KV_SHUFFLED=1
  ATOM_MEGA_MASK_PAD_ROWS=1

  "MEGA_COMBINE_WIRE=${MEGA_COMBINE_WIRE:-bf16}"
  AITER_DISABLE_KERNARG_PRELOAD=1
  HSA_ENABLE_SDMA=1
  HSA_USE_SVM=1
  HSA_XNACK=1
  ATOM_DP_LM_HEAD_MODE=allgather
  ATOM_USE_UNIFIED_ATTN=1
  ATOM_USE_TRITON_MLA=0
  ATOM_MLA_PAGE_SIZE=1
  AITER_MLA_DECODE_PS1_FLYDSL=1
  AITER_ROPE_TRITON_BACKEND=1
  ATOM_ENABLE_DS_QKNORM_FUSION=0
  ATOM_LOADER_NUM_THREADS=4
  ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=0
  ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0
  ATOM_USE_TRITON_GEMM=1
  ATOM_USE_TRITON_MLA_SHUFFLE_KV=0
  ENABLE_CK=0
  AITER_USE_GROUPED_GEMM=1
  AITER_FORCE_A8W4=0
  AITER_USE_OPUS_MOE_SORTING=1
  ATOM_MOE_GU_ITLV=1
  ATOM_MORI_V2=1
  ATOM_MORI_V2_FUSED=1
  ATOM_MEGA_STAGE1_FUSED=0
  MEGA_DISPATCH=mori
  MORI_GPU_ARCHS=gfx1250
  FLYDSL_GPU_ARCH=gfx1250
  AITER_FLYDSL_GATHER_REDUCE_WAVE_MIN_TOKENS=-1
  ATOM_GC_THRESHOLD=20000,50,50
  MEGA_DISPATCH_WIRE=fp4
  # Same extra flags as the 1014 server.yaml runs.
  ATOM_MLA_ABSORB_BF16=1
  ATOM_MEGA_COMBINE_WIRE=bf16
  MORI_AMD_SMI_PATH=/opt/rocm/lib/libamd_smi.so.27
  HIP_FORCE_DEV_KERNARG=0
  ATOM_LOG_LEVEL=INFO
  ATOM_LOG_ENGINE_STATS=1
  ATOM_ENGINE_STATS_INTERVAL=40
  ATOM_RESERVE_DECODE_TOKENS=5312
  ATOM_PREFILL_DECODE_INTERVAL=0
  ATOM_ENABLE_PREFILL_DELAYER=0
  ATOM_DP_DISPATCH_DEPTH=1860
  HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0
)
if [ "$TRACE" = "1" ]; then
  SERVER_ENV+=(
    DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=0
    DEBUG_HIP_GRAPH_CLASSIC_PATH=1
  ) 
fi
# Per-GPU batch. Client concurrency is 4x this (4 GPUs).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
CAPTURE_SIZES="[1,2,4,8,16,32,48,64,128,256,512]"
if [ "$MAX_NUM_SEQS" != "512" ]; then
  CAPTURE_SIZES="[1,2,4,8,16,32,48,64,128,256,512,${MAX_NUM_SEQS}]"
fi

# DeepSeek-R1 gfx1250 FP4 MoE decode tuning (K7168 + K2048).
# Keep the 64x256x256/b3 kernel at 512 seq/GPU; it wins for the smaller M.
# At 764/1024 seq/GPU, 128x256x256/b2 is scatter-safe and reduced the two
# GEMMs by 23.4% / 13.3% combined in the exact four-rank EP benchmark.
TDM_TUNE="baseline-64x256x256-b3"
if [ "$MAX_NUM_SEQS" -ge 764 ]; then
  SERVER_ENV+=(
    AITER_TDM_TILE_M=128
    AITER_TDM_TILE_N=256
    AITER_TDM_TILE_K=256
    AITER_TDM_M_WARP=1
    AITER_TDM_N_WARP=4
    AITER_TDM_NUM_BUFFERS=2
    AITER_TDM_TILE_M2=128
    AITER_TDM_TILE_N2=256
    AITER_TDM_TILE_K2=256
    AITER_TDM_M_WARP2=1
    AITER_TDM_N_WARP2=4
    AITER_TDM_NUM_BUFFERS2=2
  )
  TDM_TUNE="tuned-128x256x256-b2"
fi

echo "[serve] max_num_seqs=${MAX_NUM_SEQS} (per GPU) graphs=${CAPTURE_SIZES} tdm=${TDM_TUNE}"
env "${SERVER_ENV[@]}" \
  python -m atom.entrypoints.openai_server -tp 4 --model "$MODEL_PATH" --kv_cache_dtype fp8 --block-size 64 --no-enable_prefix_caching --no-enable-chunked-prefill --enable-expert-parallel --enable-dp-attention --online_quant_config '{"layer_quant_config":{"*.self_attn.q_b_proj":"mxfp4"}}' --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization 0.975 --cudagraph-capture-sizes "$CAPTURE_SIZES" "${FAKE_EPLB_ARGS[@]}" "${TRACE_ARGS[@]}" 2>&1 | tee server.log

