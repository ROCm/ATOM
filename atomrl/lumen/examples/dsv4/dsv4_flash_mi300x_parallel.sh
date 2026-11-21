#!/usr/bin/env bash
# Parallel layout for DeepSeek-V4-Flash full model on 2×8 MI300X/MI308X (16 GPUs).
#
# Defaults: TP=4 PP=4 EP=4 (11+11+11+10 layers). Override via env for bisect, e.g.:
#   TP=4 PP=2 EP=1 DECODER_FIRST_PP_LAYERS=22 DECODER_LAST_PP_LAYERS=21

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

TP="${TP:-4}"
PP="${PP:-4}"
CP="${CP:-1}"
EP="${EP:-4}"
ETP="${ETP:-1}"

DECODER_FIRST_PP_LAYERS="${DECODER_FIRST_PP_LAYERS:-11}"
DECODER_LAST_PP_LAYERS="${DECODER_LAST_PP_LAYERS:-10}"
