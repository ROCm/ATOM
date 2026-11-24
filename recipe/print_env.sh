#!/bin/bash
# Prints what an arm's server gets (no GPU, starts nothing): repo check, PYTHONPATH as serve.sh builds it, every
# ATOM_/AITER_ variable (incl. the three serve.sh sets itself), MEM, EXTRA and the server command line for CONC=32.
# usage: print_env.sh <A|B|B0|B80|C|C-noSP|C-noWIDE|C2|C3|C4|C4T|C8|C12|C13|C14|C14b|C15|C15-noback|C16|C17|C17b> [CONC=32] [AL=4.04]
# (milestone2: the AL argument is kept in PAL because env_arm*.sh unset AL; MEM is printed and used in CMD.)
set -u
ARM=$1; N=${2:-32}; PAL=${3:-4.04}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh" >/dev/null
source $RCP/env_arm$ARM.sh
echo "# arm $ARM"
check_repo
echo "REPO=$REPO"
echo "PYTHONPATH=${PYPATH_PRE:+$PYPATH_PRE:}$REPO"
( export AITER_LOG_LEVEL=WARNING ATOM_FORCE_ATTN_TRITON=1 ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=0   # serve.sh's own
  env | grep -E '^(ATOM_|AITER_)' | grep -v '^ATOM_BASE_IMAGE=' | sort )
echo "MEM=${MEM:-0.85}"
echo "EXTRA=$EXTRA"
spec="--method dspark --draft-model /shared_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4/dspark --num-speculative-tokens 7"
[ "$PAL" != real ] && spec="$spec --spec-decode-acceptance-length $PAL"
echo "CMD=python3 -u -m atom.entrypoints.openai_server --model /shared_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4" \
  "--served-model-name minimax-m3.1 --host 0.0.0.0 --port \$((PORT + 6)) --server-port \$PORT --tensor-parallel-size 4" \
  "--trust-remote-code --default-chat-template-kwargs '{\"thinking_mode\": \"enabled\"}' --gpu-memory-utilization ${MEM:-0.85}" \
  "--block-size 128 --max-num-batched-tokens 32768 --max-num-seqs $((2 * N)) --kv_cache_dtype nvfp4" \
  "--index-cache-dtype nvfp4 --enable-prefix-caching $spec $EXTRA"
