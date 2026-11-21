###############################################################################
# Training arguments that are identical for every pretrain benchmark model.
#
# Sourced, not executed, and always *after* a model's architecture args have
# opened the COMMON_ARGS array — this file only appends. Splitting it out is the
# point: the architecture is the only thing that may differ between models in a
# precision comparison, so everything else has exactly one definition.
#
# Reads: MBS GBS TRAIN_STEPS SEED TOKENIZER_PATH TRAIN_JSONL VALID_JSONL SPLIT
#        EVAL_ITERS EVAL_INTERVAL
###############################################################################

COMMON_ARGS+=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --train-iters "${TRAIN_STEPS}"
    --lr 1.0e-5 --min-lr 0.0
    --lr-decay-style cosine
    --lr-warmup-iters 2
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8
    --bf16
    --no-gradient-accumulation-fusion
    --use-distributed-optimizer
    --overlap-grad-reduce
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_PATH}"
    --train-data-path "${TRAIN_JSONL}"
    --valid-data-path "${VALID_JSONL}"
    --test-data-path "${VALID_JSONL}"
    --split "${SPLIT}"
    --seed "${SEED}"
    --eval-iters "${EVAL_ITERS}"
    --eval-interval "${EVAL_INTERVAL}"
    --save-interval 1000000
    --log-interval 1
)
