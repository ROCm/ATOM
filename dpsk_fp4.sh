export AITER_LOG_LEVEL=WARNING
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export MORI_SHMEM_MODE=ISOLATION
model_path=/models/DeepSeek-R1-0528-MXFP4/
export PYTHONPATH=/workspace/dpsk-fp4/ATOM/ATOM_zejun/ATOM
rm -rf logs/*
python3 -m atom.entrypoints.openai_server \
    --model "$model_path" \
    --host localhost \
    --server-port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --enable-dp-attention \
    --enable-expert-parallel \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.85