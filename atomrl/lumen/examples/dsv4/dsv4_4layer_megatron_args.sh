#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# DSV4 Flash 4-layer Megatron model + training args (sourced by run_dsv4_4layer_pretrain_inner.sh).
#
# Lumen pretrain smoke default GBS=8 (Miles GRPO smoke uses GBS=256 via rollout 32×8).

NLAYERS=4
DSV4_HC_MULT="${DSV4_HC_MULT:-2}"
COMPRESS_RATIOS=(0 0 4 128)
LUMEN_DSV4_SPEC_MODULE="${LUMEN_DSV4_SPEC_MODULE:-atomrl.lumen.models.dsv4.megatron.spec}"
LUMEN_DSV4_SPEC_FN="${LUMEN_DSV4_SPEC_FN:-get_dsv4_spec}"

MOE_LAYER_FREQ="[1,1,1,1]"

GBS="${GBS:-8}"
MBS="${MBS:-1}"
# Mock pretrain uses fixed seq_len (default 2048).
SEQ_LEN="${SEQ_LEN:-2048}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"

DSV4_MODEL_ARGS=(
    --disable-bias-linear
    --num-layers "${NLAYERS}"
    --hidden-size 4096
    --ffn-hidden-size 2048
    --num-attention-heads 64
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 129280
    --hidden-dropout 0.0
    --attention-dropout 0.0

    --multi-latent-attention
    --q-lora-rank 1024
    --kv-lora-rank 512
    --qk-head-dim 512
    --qk-pos-emb-head-dim 64
    --v-head-dim 512
    --qk-layernorm
    --rotary-scaling-factor 16
    --rotary-base 10000
    --original-max-position-embeddings 65536
    --beta-fast 32
    --beta-slow 1
    --attention-softmax-in-fp32
    --no-rope-fusion

    --num-experts 256
    --moe-layer-freq "${MOE_LAYER_FREQ}"
    --moe-ffn-hidden-size 2048
    --moe-router-topk 6
    --moe-shared-expert-intermediate-size 2048
    --moe-router-pre-softmax
    --moe-router-score-function sqrtsoftplus
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0
    --moe-grouped-gemm
    --moe-router-topk-scaling-factor 1.5

    --experimental-attention-variant dsv4
    --dsv4-hc-mult "${DSV4_HC_MULT}"
    --dsv4-hc-sinkhorn-iters 20
    --dsv4-compress-ratios "${COMPRESS_RATIOS[@]}"
    --dsv4-compress-rope-theta 160000
    --dsv4-o-groups 8
    --dsv4-o-lora-rank 1024
    --dsv4-n-hash-layers 3
    --dsv4-window-size 128

    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk 512

    --spec "${LUMEN_DSV4_SPEC_MODULE}" "${LUMEN_DSV4_SPEC_FN}"
    --activation-func-clamp-value 10
    --no-bias-swiglu-fusion
    --no-activation-func-clamp-shared-expert
)

# 4-layer torch_dist from Miles convert omits router expert_bias shards.
if [[ "${DSV4_ENABLE_EXPERT_BIAS:-0}" == "1" ]]; then
    DSV4_MODEL_ARGS+=(--moe-router-enable-expert-bias)
fi

# Parallel layout for 8×MI308X smoke (torchrun flags in run_dsv4_4layer_pretrain_inner.sh).
TP=8
PP=1
CP=1
EP=8
ETP=1
