#!/bin/bash
export AITER_LOG_LEVEL=WARNING
export AITER_QUICK_REDUCE_QUANTIZATION=INT4 AITER_USE_FLYDSL_MOE_SORTING=1
export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 HSA_OVERRIDE_GFX_VERSION=12.5.0
export ATOM_MOE_GU_ITLV=1 AITER_MLA_DECODE_PS1_FLYDSL=1 AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1
export ATOM_LOADER_NUM_THREADS=4         # 默认 16；曾有 tray 在加载权重时整机报 CPU ProcessorError，降到 4 后没再出现
cd /tmp
python3 -m atom.entrypoints.openai_server \
  --model /models/GLM-5.2-MXFP4 --server-port 8000 -tp 4 \
  --kv_cache_dtype fp8 --no-enable_prefix_caching \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "*expert*"]}'
