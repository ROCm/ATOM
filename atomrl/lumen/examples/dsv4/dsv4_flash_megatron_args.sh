#!/usr/bin/env bash
# DSV4 Flash full model (43 layers) Megatron args — sourced by run_dsv4_flash_pretrain_inner.sh.
#
# Ported from miles/scripts/models/deepseek-v4-flash.sh with Lumen spec module.

NLAYERS=43
DSV4_HC_MULT="${DSV4_HC_MULT:-4}"
COMPRESS_RATIOS=(
    0 0 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128
    4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 0
)
LUMEN_DSV4_SPEC_MODULE="${LUMEN_DSV4_SPEC_MODULE:-atomrl.lumen.models.dsv4.megatron.spec}"
LUMEN_DSV4_SPEC_FN="${LUMEN_DSV4_SPEC_FN:-get_dsv4_spec}"

MOE_LAYER_FREQ="[$(printf '1%.0s,' $(seq 1 "${NLAYERS}") | sed 's/,$//')]"

# Lumen pretrain smoke default GBS=8 (fast iteration; override via env GBS=...).
GBS="${GBS:-8}"
MBS="${MBS:-1}"
SEQ_LEN="${SEQ_LEN:-2048}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-1024}"

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
    --make-vocab-size-divisible-by 32
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
    --moe-router-enable-expert-bias
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
