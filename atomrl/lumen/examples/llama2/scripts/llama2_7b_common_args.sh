###############################################################################
# Llama2-7B architecture arguments for the pretrain benchmark.
#
# Sourced, not executed. Populates COMMON_ARGS from variables the caller has
# already set: NUM_LAYERS MBS GBS SEQ_LEN TRAIN_STEPS SEED TOKENIZER_PATH
# TRAIN_JSONL VALID_JSONL SPLIT EVAL_ITERS EVAL_INTERVAL.
#
# Architecture only; everything model-independent comes from
# examples/scripts/pretrain_common_args.sh so the precision arms of every model
# share one definition. Matches the config the BF16 / FP8 reference runs used:
# 32 layers, hidden 4096, FFN 11008, plain MHA (no GQA), rope base 1e4.
###############################################################################

COMMON_ARGS=(
    --num-layers "${NUM_LAYERS}"
    --hidden-size 4096
    --ffn-hidden-size 11008
    --num-attention-heads 32
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings "${SEQ_LEN}"
    --use-rotary-position-embeddings
    --rotary-base 10000
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
