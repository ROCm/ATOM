#!/bin/bash
# Accuracy gate of one arm (run INSIDE the p1 container under a gpurun lease; use launch.sh).
# Port of /shared_nfs/ygan/max/int1/gate_job.sh: gate.sh (copy, outputs moved) with GATE_LEASED=1 GATE_NOLOCK=1 LONG=1
# GSM_REPEATS=2: serve.sh AL=real (REAL DSpark-7 acceptance, max-num-seqs 16), GSM8K the InferenceX way twice
# (max/bin/gsm8k_ix.sh: lm-eval b315ef3, 1319 q, 5-shot chat, greedy, max_tokens 16384, conc 16), real acceptance length
# over the GSM8K window from the server log (max/bin/al_stats.py: "[MTP Stats Interval]" lines, AL = 1 + accepted/steps),
# needle check at 20K/80K/130K (max/bin/long_check.py), verdict vs /shared_nfs/ygan/max/acc/baseline.json
# (gate_verdict.py copy): PASS needs GSM8K strict mean of the 2 runs >= 0.9621, real AL >= 4.608, needle 3/3.
# usage: gate_job.sh <A|B> <GTAG>           -> /shared_nfs/ygan/p1/c17/gates/<GTAG>/{gate.log,verdict.txt,...}
set -u
ARM=$1; G=$2
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source $RCP/env_arm$ARM.sh || exit 2
check_repo || exit 2
export FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR:-$NLOC/flydsl_arm$ARM}; seed_flydsl
OUT=$MS/gates/$G
[ -e "$OUT" ] && { echo "gate $G exists: pick a new GTAG"; exit 2; }
mkdir -p $OUT
VARS=()
while IFS= read -r kv; do VARS+=("$kv"); done < <(env | grep -E '^(ATOM_|AITER_)' | grep -v '^ATOM_BASE_IMAGE=' | sort)
VARS+=("EXTRA=$EXTRA")
echo "### $(date -Is) p1 milestone gate $G arm=$ARM on $(hostname -s) GPUs $HIP_VISIBLE_DEVICES (GATE_LEASED=1 GATE_NOLOCK=1)" >> $OUT/gate.log
GTAG=$G GATE_LEASED=1 GATE_NOLOCK=1 LONG=1 GSM_REPEATS=2 PORT=${PORT:-18940} bash $RCP/gate.sh "$REPO" "$PYPATH_PRE" "${VARS[@]}" >> $OUT/gate.log 2>&1
rc=$?
echo "### $(date -Is) gate rc=$rc" >> $OUT/gate.log
pkill -9 -f "server-port ${PORT:-18940}( |$)" 2>/dev/null
gpus_free >> $OUT/gate.log 2>&1
cat $OUT/verdict.txt 2>/dev/null
exit $rc
