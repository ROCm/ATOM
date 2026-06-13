set -x

export ATOM_USE_TRITON_MOE=1
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export AITER_LOG_LEVEL=WARNING
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_OPT_USE_FUSED_COMPRESS=true
export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_HACK_FLASHMLA_BACKEND=triton
export SGLANG_OPT_FP8_WO_A_GEMM=false
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_OPT_USE_AITER_INDEXER=true
export SGLANG_OPT_USE_TILELANG_INDEXER=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
export SGLANG_ROCM_USE_MULTI_STREAM=false
export SGLANG_DSV4_FP4_EXPERTS=true

model_path=/workspace/shared/data/amd_int/models/deepseek-ai/DeepSeek-V4-Pro

NUM_GPU=8
if [ $NUM_GPU -eq 8 ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    TP_SIZE=8
elif [ $NUM_GPU -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,5"
    TP_SIZE=4
fi

export ATOM_SGLANG_V4_DEBUG=${ATOM_SGLANG_V4_DEBUG:-0}
# Use the environment-provided SGLang.  Only prepend atom-main so the external
# model package resolves to the local ATOM plugin code.
export PYTHONPATH=/home/qichu_qle/zhiwei/dsv4/atom-main:${PYTHONPATH:-}
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path $model_path \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size $TP_SIZE \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.9 \
    --swa-full-tokens-ratio 0.1 \
    --max-running-requests 256 \
    --page-size 256 \
    --disable-radix-cache \
    --tool-call-parser deepseekv4 \
    --disable-shared-experts-fusion \
    --reasoning-parser deepseek-v4 \
    2>&1 | tee log.deepseek_v4.serve.log
    # --attention-backend dsv4 \
    # --chunked-prefill-size 8192 \
    # >log.deepseek_v4.serve.log 2>&1
