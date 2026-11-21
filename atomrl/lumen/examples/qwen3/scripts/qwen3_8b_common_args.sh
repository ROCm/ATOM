###############################################################################
# Qwen3-8B arguments shared by the Lumen and the stock-Megatron+TE launchers.
#
# Sourced, not executed. Populates the COMMON_ARGS array from variables the
# caller has already set: NUM_LAYERS MBS GBS SEQ_LEN TRAIN_STEPS SEED
# TOKENIZER_PATH TRAIN_JSONL VALID_JSONL SPLIT EVAL_ITERS EVAL_INTERVAL.
#
# The two launchers exist to compare quantization stacks, so every
# hyperparameter that is not the thing under test lives here — duplicating this
# list per launcher is how the two arms silently drift apart and produce a
# comparison that means nothing. For the same reason the model-independent half
# now lives in examples/scripts/pretrain_common_args.sh, shared with the other
# benchmark models; only the architecture below is Qwen3's own.
###############################################################################

COMMON_ARGS=(
    --num-layers "${NUM_LAYERS}"
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --qk-layernorm
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings "${SEQ_LEN}"
    --use-rotary-position-embeddings
    --rotary-base 1000000
    --no-position-embedding
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/pretrain_common_args.sh"
