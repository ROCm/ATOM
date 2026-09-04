FROM rocm/atom-dev:nightly_202608201458@sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976

COPY . /app/ATOM-dsv4-0731
WORKDIR /app/ATOM-dsv4-0731

RUN python recipes/DeepSeek-V4-Flash-0731-MI308X/apply_aiter_optimizations.py \
      --aiter-root /app/aiter-test \
      --assets /app/ATOM-dsv4-0731/recipes/DeepSeek-V4-Flash-0731-MI308X/assets \
 && python -m pip install --no-deps -e /app/ATOM-dsv4-0731 \
 && chmod +x recipes/DeepSeek-V4-Flash-0731-MI308X/entrypoint.sh \
 && python -m py_compile \
      atom/model_ops/linear.py \
      atom/model_ops/moe.py \
      atom/model_ops/v4_kernels/paged_prefill.py \
      atom/models/deepseek_v2.py \
      atom/models/deepseek_v4.py \
      atom/plugin/vllm/attention/layer_sparse_mla.py \
      atom/quantization/quark/utils.py

ENV NCCL_IB_GID_INDEX=3 \
    NCCL_P2P_DISABLE=0 \
    ATOM_DSV4_0731_OPTIMIZATIONS=1 \
    ATOM_USE_TRITON_GEMM=1 \
    ATOM_USE_TRITON_MOE=1 \
    ATOM_ENABLE_MXFP4_SOURCE_ONLINE_QUANT=1 \
    FLYDSL_FP8_MQA_LOGITS_VARIANT=mfma_r4_adaptive \
    ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB=2048 \
    ATOM_INDEXER_PREFILL_ROW_SHARD=1 \
    AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
    ATOM_PARALLEL_DISABLE_CUSTOM_AR=0 \
    AITER_DISABLE_MULTI_INSTANCE_AR=0 \
    ATOM_ENABLE_PREFILL_DELAYER=1 \
    ATOM_PREFILL_DELAYER_TARGET_FILL=0.9 \
    ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS=200 \
    ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS=100 \
    ATOM_PREFILL_DELAYER_STALL_TICKS=10

ENTRYPOINT ["/app/ATOM-dsv4-0731/recipes/DeepSeek-V4-Flash-0731-MI308X/entrypoint.sh"]
