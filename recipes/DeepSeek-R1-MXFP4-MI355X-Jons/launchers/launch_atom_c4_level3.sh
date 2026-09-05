#!/bin/bash
# c4 + --level 3 (max compilation level) + tight cudagraph + small max-num-seqs.
# Hypothesis: torch.compile level 3 may help small-batch decode kernels.

export AITER_ROOT_DIR=/projects/teamK/aiter_cache
export HF_HOME=/projects/teamK/hf_home
export HF_MODULES_CACHE=/projects/teamK/hf_home/modules
export TRITON_CACHE_DIR=/projects/teamK/triton_cache
export TVM_FFI_CACHE_DIR=/projects/teamK/tvm_cache
export TMPDIR=/projects/teamK/tmp
export OMP_NUM_THREADS=1
export AMDGCN_USE_BUFFER_OPS=1
export VLLM_CACHE_ROOT=/projects/teamK/atom_cache
export HOME=/projects/teamK/home_atom

python3 -m atom.entrypoints.openai_server \
  --model /share4/teamK/DeepSeek-R1-0528-MXFP4 \
  --server-port 8888 -tp 8 \
  --kv_cache_dtype fp8 \
  --max-model-len 10240 \
  --method mtp --num-speculative-tokens 3 \
  --level 3 \
  --cudagraph-capture-sizes "[1,2,4,8]" \
  2>&1 | tee /projects/teamK/server_c4_level3.log
