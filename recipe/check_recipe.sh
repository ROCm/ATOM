#!/bin/bash
# CPU-only validation of the milestone recipe (no GPU, no server). Run on the login node or in the p1 container.
set -u
R=/shared_nfs/ygan/p1/c17/recipe; fail=0
ok() { echo "OK   $*"; }; bad() { echo "FAIL $*"; fail=1; }
# 1. shell syntax and python byte-compile of every recipe file
for f in $R/*.sh; do bash -n "$f" && ok "bash -n $(basename $f)" || bad "bash -n $f"; done
for f in $R/*.py; do python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" && ok "py parse $(basename $f)" || bad "py parse $f"; done
# 2. arm A's environment == the env header of the previous team's int1_both_c32 server log
LOGA=/shared_nfs/ygan/max/logs/server_int1_both_c32.log
bash $R/print_env.sh A > /tmp/pe_A.$$ 2>&1
grep -E '^(ATOM_|AITER_)' /tmp/pe_A.$$ > /tmp/envA.$$
grep '^### env ' $LOGA | sed 's/^### env //' | grep -v '^ATOM_BASE_IMAGE=' | sort > /tmp/envA_ref.$$
if diff /tmp/envA_ref.$$ /tmp/envA.$$ > /tmp/envA_diff.$$; then ok "arm A env == $LOGA env ($(wc -l < /tmp/envA.$$) vars)"
else bad "arm A env differs from $LOGA:"; cat /tmp/envA_diff.$$; fi
xa=$(grep '^EXTRA=' /tmp/pe_A.$$ | sed 's/^EXTRA=//'); xr=$(grep '^### extra: ' $LOGA | sed 's/^### extra: //')
[ "$(echo $xa)" = "$(echo $xr)" ] && ok "arm A EXTRA == log extra" || bad "arm A EXTRA '$xa' vs log '$xr'"
grep -q "tag=int1_both_c32 conc=32 max_seqs=64 port=18950 k=7 al=4.04 mem=0.85" $LOGA && ok "arm A conc/max_seqs/k/al/mem as logged" || bad "log header"
grep -q "^check_repo OK" /tmp/pe_A.$$ && ok "arm A repo pinned" || bad "arm A repo: $(grep check_repo /tmp/pe_A.$$)"
# 3. arm B: repo pinned, every ATOM_* name it sets is read by the int5 tree, capture list covers 1..24 and the ladder
bash $R/print_env.sh B > /tmp/pe_B.$$ 2>&1
grep -q "^check_repo OK" /tmp/pe_B.$$ && ok "arm B repo pinned" || bad "arm B repo: $(grep check_repo /tmp/pe_B.$$)"
REPO=/shared_nfs/ygan/work/p1-integrated5
for v in $(grep -oE '^ATOM_[A-Z0-9_]+' /tmp/pe_B.$$); do
  grep -rqs --include=*.py "\"$v\"\|'$v'" $REPO/atom || bad "arm B sets $v but $REPO/atom never reads it"
done; ok "arm B flag names checked against $REPO/atom"
python3 - /tmp/pe_B.$$ <<'P' && ok "arm B capture list covers 1..24 + ladder; ladder = production list" || bad "arm B capture/ladder"
import re, sys
t = open(sys.argv[1]).read()
cg = [int(x) for x in re.search(r"--cudagraph-capture-sizes \[([\d,]+)\]", t).group(1).split(",")]
lad = [int(x) for x in re.search(r"^ATOM_P1_DRAFT_BS_LADDER=([\d,]+)$", t, re.M).group(1).split(",")]
prod = [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64]
assert set(range(1, 25)) <= set(cg) and set(lad) <= set(cg) and lad == prod and max(cg) == 64, (cg, lad)
for k, v in [("ATOM_MONO_ENABLE", "1"), ("ATOM_P1_MONO_MAX_ROWS", "8"), ("ATOM_PREFILL_DECODE_INTERVAL", "8"),
             ("ATOM_P1_SCHED_FINISH_HOLD_LEFT", "128"), ("ATOM_P1_SCHED_FINISH_HOLD_MAX_MS", "500"),
             ("ATOM_P1_SCHED_FINISH_HOLD_MAX_TOKENS", "400"), ("ATOM_SCHED_PREFILL_DUTY", "0.3"),
             ("ATOM_SCHED_DECODE_PREFILL_CAP", "8192"), ("ATOM_PREFILL_DELAYER_TARGET_FILL", "0.225"),
             ("ATOM_P1_INDEXER_SPLIT4", "1"), ("ATOM_P1_INDEXER_SPLIT4_WIDE", "1")]:
    assert re.search(rf"^{k}={re.escape(v)}$", t, re.M), k
assert "draft-fusion/overlay" in re.search(r"^PYTHONPATH=(.*)$", t, re.M).group(1)
P
# 4. A and B differ only where intended (printed for review)
echo "---- arm A -> arm B variable diff"
diff <(grep -E '^(ATOM_|AITER_)' /tmp/pe_A.$$) <(grep -E '^(ATOM_|AITER_)' /tmp/pe_B.$$) | grep '^[<>]'
# 4b. milestone2: arm B == the milestone's recorded arm B (flags_armB.txt; its trailing AL error line was a print_env bug)
diff <(grep -v -E '^(MEM|CMD)=' /tmp/pe_B.$$) <(grep -v 'AL: unbound variable' /shared_nfs/ygan/p1/milestone/recipe/flags_armB.txt) >/dev/null \
  && grep -q '^MEM=0.85$' /tmp/pe_B.$$ && ok "arm B == milestone flags_armB.txt (int5 5f26263a8, MEM 0.85)" || bad "arm B differs from milestone flags_armB.txt"
# 5. the code checks of the int5 tree (CPU): draft pin test
python3 -I $R/../../build-int5/tests/cpu_pin_test.py >/dev/null && ok "cpu_pin_test (int5, prefill pinned)" || bad cpu_pin_test
# 6. milestone2 arms C / C-noSP / C-noWIDE on p1/integrated6: repo pinned, every ATOM_* name set is read by the int6 tree,
#    values as intended, and each component arm differs from C by exactly its component
R6=/shared_nfs/ygan/work/p1-integrated6
for a in C C-noSP C-noWIDE; do
  bash $R/print_env.sh $a > /tmp/pe_$a.$$ 2>&1
  grep -q "^check_repo OK" /tmp/pe_$a.$$ && ok "arm $a repo pinned" || bad "arm $a repo: $(grep check_repo /tmp/pe_$a.$$)"
  grep -qiE 'unbound|error|No such file' /tmp/pe_$a.$$ && bad "arm $a print_env errors: $(grep -iE 'unbound|error|No such' /tmp/pe_$a.$$)"
  n=0; for v in $(grep -oE '^ATOM_[A-Z0-9_]+' /tmp/pe_$a.$$); do
    n=$((n+1)); grep -rqs --include=*.py "\"$v\"\|'$v'" $R6/atom || bad "arm $a sets $v but $R6/atom never reads it"
  done; ok "arm $a: $n ATOM_* names checked against $R6/atom"
done
python3 - /tmp/pe_B.$$ /tmp/pe_C.$$ /tmp/pe_C-noSP.$$ /tmp/pe_C-noWIDE.$$ <<'P2' && ok "arm C = B + P2 sched + AK_WIDE + SP(2048) + MEM 0.80 + watchdog; C-noSP/C-noWIDE = C minus one component" || bad "arm C/C-noSP/C-noWIDE values"
import re, sys
def kv(p):
    d = {}
    for l in open(p):
        m = re.match(r"^((?:ATOM_|AITER_)[A-Z0-9_]+|MEM|EXTRA|PYTHONPATH)=(.*)$", l.rstrip("\n"))
        if m: d[m.group(1)] = m.group(2)
    return d
B, C, CnS, CnW = (kv(p) for p in sys.argv[1:5])
I6 = "/shared_nfs/ygan/work/p1-integrated6"
SP = dict(ATOM_M3_SP_PREFILL="1", ATOM_M3_SP_PREFILL_COMM="pull", ATOM_M3_SP_PREFILL_OPROJ="full",
          ATOM_M3_SP_PREFILL_QGATHER="1", ATOM_P1_SP_HANDOFFS="1", ATOM_P1_SP_QG_AK="1", ATOM_P1_SP_A2A_UNPACK="1",
          ATOM_P1_SP_OPROJ_AK="1", ATOM_P1_SP_MOE_SLOTS="1", ATOM_M3_SP_PREFILL_MIN_TOKENS="2048",
          ATOM_TOKEN_GRAPH_SIZES="256,512,1024")
exp = dict(B)
del exp["ATOM_SCHED_PREFILL_DUTY"]
exp.update(ATOM_PREFILL_DECODE_INTERVAL="1", ATOM_P1_SCHED_ITL_RISK_MS="4.5", ATOM_P1_SCHED_ITL_RISK_MAX_MS="300",
           ATOM_P1_AK_WIDE="1", MEM="0.80", ATOM_STEP_WATCHDOG_S="5", ATOM_STEP_KICK_S="5", **SP)
exp["PYTHONPATH"] = B["PYTHONPATH"].replace("p1-integrated5", "p1-integrated6")
assert exp["PYTHONPATH"].endswith(I6) and "draft-fusion/overlay" in exp["PYTHONPATH"]
assert C == exp, sorted(set(C.items()) ^ set(exp.items()))
# unchanged from B on purpose: fill, finish hold, cap, mono cap 8, ladder, capture list, fp8 draft KV
for k in ("ATOM_PREFILL_DELAYER_TARGET_FILL", "ATOM_P1_SCHED_FINISH_HOLD_LEFT", "ATOM_P1_SCHED_FINISH_HOLD_MAX_MS",
          "ATOM_P1_SCHED_FINISH_HOLD_MAX_TOKENS", "ATOM_SCHED_DECODE_PREFILL_CAP", "ATOM_P1_MONO_MAX_ROWS", "EXTRA",
          "ATOM_P1_DRAFT_BS_LADDER", "ATOM_MONO_ENABLE", "ATOM_P1_INDEXER_SPLIT4", "ATOM_P1_INDEXER_SPLIT4_WIDE"):
    assert C[k] == B[k], k
noSP = {k: v for k, v in C.items() if k not in SP}
assert CnS == noSP, sorted(set(CnS.items()) ^ set(noSP.items()))
noW = {k: v for k, v in C.items() if k != "ATOM_P1_AK_WIDE"}
assert CnW == noW, sorted(set(CnW.items()) ^ set(noW.items()))
P2
python3 -I /shared_nfs/ygan/p1/build-int6/tests/cpu_pin_test.py >/dev/null && ok "cpu_pin_test (int6)" || bad "cpu_pin_test int6"
grep -qE '^case "\$ARM" in A\|B\|B0\|B80\|C\|C-noSP\|C-noWIDE\|C2\|C3\|C4\|C4T\|C8\|C12\|C13\|C14\|C14b\|C15\|C15-noback\|C16\|C17\|C17b\)' $R/launch.sh && ok "launch.sh accepts C..C3, C4, C4T, C8, C12, C13, C14, C14b, C15, C15-noback, C16, C17, C17b" || bad "launch.sh arm list"
rm -f /tmp/pe_C.$$ /tmp/pe_C-noSP.$$ /tmp/pe_C-noWIDE.$$
rm -f /tmp/pe_A.$$ /tmp/pe_B.$$ /tmp/envA.$$ /tmp/envA_ref.$$ /tmp/envA_diff.$$

# 7. milestone3: arm C2 on p1/integrated7 = C + SPLIT4_MIN8 6 + decode-prefill cap 16384 + ITL-risk max 150 ms;
#    arm B80 = B with MEM 0.80
R7=/shared_nfs/ygan/work/p1-integrated7
for a in C2 B80; do bash $R/print_env.sh $a > /tmp/pe_$a.$$ 2>&1
  grep -q "^check_repo OK" /tmp/pe_$a.$$ && ok "arm $a repo pinned" || bad "arm $a repo: $(grep check_repo /tmp/pe_$a.$$)"
  grep -qiE 'unbound|error|No such file' /tmp/pe_$a.$$ && bad "arm $a print_env errors"; done
n=0; for v in $(grep -oE '^ATOM_[A-Z0-9_]+' /tmp/pe_C2.$$); do n=$((n+1)); grep -rqs --include=*.py "\"$v\"\|'$v'" $R7/atom || bad "arm C2 sets $v but $R7/atom never reads it"; done; ok "arm C2: $n names checked against $R7/atom"
bash $R/print_env.sh C > /tmp/pe_C.$$ 2>&1; bash $R/print_env.sh B > /tmp/pe_B.$$ 2>&1
python3 - /tmp/pe_C.$$ /tmp/pe_C2.$$ /tmp/pe_B.$$ /tmp/pe_B80.$$ <<'P3' && ok "arm C2 = C + MIN8 6 + CAP 16384 + ITL max 150 (int7); B80 = B + MEM 0.80" || bad "arm C2/B80 values"
import re, sys
def kv(p):
    d = {}
    for l in open(p):
        m = re.match(r"^((?:ATOM_|AITER_)[A-Z0-9_]+|MEM|EXTRA|PYTHONPATH)=(.*)$", l.rstrip("\n"))
        if m: d[m.group(1)] = m.group(2)
    return d
C, C2, B, B80 = (kv(p) for p in sys.argv[1:5])
exp = dict(C); exp.update(ATOM_P1_INDEXER_SPLIT4_MIN8="6", ATOM_SCHED_DECODE_PREFILL_CAP="16384", ATOM_P1_SCHED_ITL_RISK_MAX_MS="150")
exp["PYTHONPATH"] = C["PYTHONPATH"].replace("p1-integrated6", "p1-integrated7")
assert C2 == exp, sorted(set(C2.items()) ^ set(exp.items()))
e2 = dict(B); e2["MEM"] = "0.80"
assert B80 == e2, sorted(set(B80.items()) ^ set(e2.items()))
P3
bash -n $R/series.sh && ok "series.sh syntax"
rm -f /tmp/pe_C2.$$ /tmp/pe_B80.$$ /tmp/pe_C.$$ /tmp/pe_B.$$
# 8. milestone4: arm C3 on p1/integrated8 = C2 + SP MoE compact hand-off (+ AUX_NEXT if kept); C2 stays the int7 control
R8=/shared_nfs/ygan/work/p1-integrated8
for a in C2 C3; do bash $R/print_env.sh $a > /tmp/pe_$a.$$ 2>&1
  grep -q "^check_repo OK" /tmp/pe_$a.$$ && ok "arm $a repo pinned" || bad "arm $a repo: $(grep check_repo /tmp/pe_$a.$$)"
  grep -qiE 'unbound|error|No such file' /tmp/pe_$a.$$ && bad "arm $a print_env errors"; done
n=0; for v in $(grep -oE '^ATOM_[A-Z0-9_]+' /tmp/pe_C3.$$); do n=$((n+1)); grep -rqs --include=*.py "\"$v\"\|'$v'" $R8/atom || bad "arm C3 sets $v but $R8/atom never reads it"; done; ok "arm C3: $n names checked against $R8/atom"
python3 - /tmp/pe_C2.$$ /tmp/pe_C3.$$ $R/env_armC3.sh <<'P4' && ok "arm C3 = C2 + COMPACT(4096) [+ AUX_NEXT 64] on int8; C2 unchanged on int7" || bad "arm C3 values"
import re, sys
def kv(p):
    d = {}
    for l in open(p):
        m = re.match(r"^((?:ATOM_|AITER_)[A-Z0-9_]+|MEM|EXTRA|PYTHONPATH)=(.*)$", l.rstrip("\n"))
        if m: d[m.group(1)] = m.group(2)
    return d
C2, C3 = kv(sys.argv[1]), kv(sys.argv[2])
aux = "ATOM_P1_DSPARK_AUX_NEXT=64" in open(sys.argv[3]).read()
exp = dict(C2); exp.update(ATOM_P1_SP_MOE_COMPACT="1", ATOM_P1_SP_MOE_COMPACT_MIN_TOKENS="4096")
if aux: exp["ATOM_P1_DSPARK_AUX_NEXT"] = "64"
exp["PYTHONPATH"] = C2["PYTHONPATH"].replace("p1-integrated7", "p1-integrated8")
assert C2["PYTHONPATH"].endswith("p1-integrated7") and exp["PYTHONPATH"].endswith("p1-integrated8")
assert C3 == exp, sorted(set(C3.items()) ^ set(exp.items()))
print("C3 extra flags:", sorted(set(C3) - set(C2)))
P4
bash -c "source $R/common.sh >/dev/null; source $R/env_armC3.sh; [ \"\$FLYDSL_RUNTIME_CACHE_DIR\" = \"\$NLOC/flydsl_armC3\" ] && [ \"\$FLYDSL_SEED\" = \"\$NLOC/flydsl_armC\" ]" && ok "arm C3 flydsl cache flydsl_armC3 seeded from flydsl_armC" || bad "arm C3 flydsl cache"
python3 -I $R/score_diag.py /shared_nfs/ygan/p1/milestone3/results/m3_C_035_1 --json 2>/dev/null | tail -1 > /tmp/sd.$$; rm -f /shared_nfs/ygan/p1/milestone3/results/m3_C_035_1/score_diag.json
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert round(d['RS'])==59575 and round(d['PRO900'])==59660 and d['grace_timeout'] and d['abandoned_joins']==1 and d['inflight_at_cutoff']==3" /tmp/sd.$$ \
  && ok "score_diag on m3_C_035_1: RS 59575, PRO900 59660, grace timeout + leaked join (== aiperf-anomaly table)" || bad "score_diag"
python3 -c "import sys; sys.path.insert(0,sys.argv[1]); import ladder_stats as L; L.R='/shared_nfs/ygan/p1/milestone3/results'; a=L.agg_fallback('m3_C_035_1')['tput_ix']; o=L.official('m3_C_035_1')['tput_ix']; assert abs(a-59574.7)<0.1 and abs(o-a)<0.01, (a,o)" $R \
  && ok "ladder_stats: summarize_ix path real, fallback == record span (59574.7)" || bad "ladder_stats official/fallback"
rm -f /tmp/pe_C2.$$ /tmp/pe_C3.$$ /tmp/sd.$$
# 9. milestone5 (m5-smoke): arm C8 on p1/integrated12 = C3 + scheduler-owned SP TBO (10240..24575, snap 10%), nothing else
R12=/shared_nfs/ygan/work/p1-integrated12
for a in C3 C8; do bash $R/print_env.sh $a > /tmp/pe_$a.$$ 2>&1
  grep -q "^check_repo OK" /tmp/pe_$a.$$ && ok "arm $a repo pinned" || bad "arm $a repo: $(grep check_repo /tmp/pe_$a.$$)"
  grep -qiE 'unbound|error|No such file' /tmp/pe_$a.$$ && bad "arm $a print_env errors"; done
n=0; for v in $(grep -oE '^ATOM_[A-Z0-9_]+' /tmp/pe_C8.$$); do n=$((n+1)); grep -rqs --include=*.py "\"$v\"\|'$v'" $R12/atom || bad "arm C8 sets $v but $R12/atom never reads it"; done; ok "arm C8: $n names checked against $R12/atom"
diff <(grep -v '^# arm' /tmp/pe_C3.$$) <(grep -v '^# arm' /shared_nfs/ygan/p1/milestone4/recipe/flags_armC3.txt) >/dev/null && ok "arm C3 == milestone4 flags_armC3.txt (untouched)" || bad "arm C3 differs from milestone4 flags_armC3.txt"
python3 - /tmp/pe_C3.$$ /tmp/pe_C8.$$ <<'P5' && ok "arm C8 = C3 + SP_TBO(gen, pair, 10240) + BALANCE_TBO 10240..24575 snap 10 on int12; no BALANCE / SCHED_PAIR" || bad "arm C8 values"
import re, sys
def kv(p):
    d = {}
    for l in open(p):
        m = re.match(r"^((?:ATOM_|AITER_)[A-Z0-9_]+|MEM|EXTRA|PYTHONPATH)=(.*)$", l.rstrip("\n"))
        if m: d[m.group(1)] = m.group(2)
    return d
C3, C8 = kv(sys.argv[1]), kv(sys.argv[2])
exp = dict(C3); exp.update(ATOM_P1_SP_TBO="1", ATOM_P1_SP_TBO_PAIR="1", ATOM_P1_SP_TBO_MODE="gen", ATOM_P1_SP_TBO_MIN_TOKENS="10240",
    ATOM_P1_SCHED_BALANCE_TBO_MIN="10240", ATOM_P1_SCHED_BALANCE_TBO_MAX="24575", ATOM_P1_SCHED_BALANCE_SNAP_PCT="10")
exp["PYTHONPATH"] = C3["PYTHONPATH"].replace("p1-integrated8", "p1-integrated12")
assert C3["PYTHONPATH"].endswith("p1-integrated8") and exp["PYTHONPATH"].endswith("p1-integrated12")
assert C8 == exp, sorted(set(C8.items()) ^ set(exp.items()))
assert not any(k.startswith(("ATOM_P1_SCHED_PAIR", "ATOM_P1_SCHED_BALANCE_SCAN")) or k in ("ATOM_P1_SCHED_BALANCE", "ATOM_P1_SCHED_BALANCE_MIN") for k in C8)
assert int(C8["ATOM_SCHED_DECODE_PREFILL_CAP"]) == 16384 and int(C8["ATOM_P1_SCHED_BALANCE_TBO_MAX"]) < 24576
P5
bash -c "source $R/common.sh >/dev/null; source $R/env_armC8.sh; [ \"\$FLYDSL_RUNTIME_CACHE_DIR\" = \"\$NLOC/flydsl_armC8\" ] && case \"\$FLYDSL_SEED\" in \"\$NLOC\"/flydsl_armC3|\"\$NLOC\"/flydsl_armC3a) ;; *) false ;; esac" && ok "arm C8 flydsl cache flydsl_armC8 seeded from flydsl_armC3 (or C3a on 034)" || bad "arm C8 flydsl cache"
grep -q '^MS=/shared_nfs/ygan/p1/c17;' $R/series.sh && ok "series.sh writes to p1/c17" || bad "series.sh MS path"
rm -f /tmp/pe_C3.$$ /tmp/pe_C8.$$
# 12. prefix-fix: arm C12 = C3 + ATOM_P1_PREFIX_DEMOTE_DECODE=1 on p1/prefix-cache; C13 = C12 + MEM 0.85
#     + ATOM_P1_SCHED_IDLE_PREFILL_CAP=16384 (same tree); the tree's CPU tests for both flags pass
RPC=/shared_nfs/ygan/work/p1-prefix-cache
for a in C3 C12 C13; do bash $R/print_env.sh $a > /tmp/pe_$a.$$ 2>&1
  grep -q "^check_repo OK" /tmp/pe_$a.$$ && ok "arm $a repo pinned" || bad "arm $a repo: $(grep check_repo /tmp/pe_$a.$$)"
  grep -qiE 'unbound|error|No such file' /tmp/pe_$a.$$ && bad "arm $a print_env errors"; done
for a in C12 C13; do n=0; for v in $(grep -oE '^ATOM_[A-Z0-9_]+' /tmp/pe_$a.$$); do n=$((n+1)); grep -rqs --include=*.py "\"$v\"\|'$v'" $RPC/atom || bad "arm $a sets $v but $RPC/atom never reads it"; done; ok "arm $a: $n names checked against $RPC/atom"; done
python3 - /tmp/pe_C3.$$ /tmp/pe_C12.$$ /tmp/pe_C13.$$ <<'P6' && ok "arm C12 = C3 + DEMOTE_DECODE on p1-prefix-cache; C13 = C12 + MEM 0.85 + IDLE_PREFILL_CAP 16384" || bad "arm C12/C13 values"
import re, sys
def kv(p):
    d = {}
    for l in open(p):
        m = re.match(r"^((?:ATOM_|AITER_)[A-Z0-9_]+|MEM|EXTRA|PYTHONPATH)=(.*)$", l.rstrip("\n"))
        if m: d[m.group(1)] = m.group(2)
    return d
C3, C12, C13 = (kv(p) for p in sys.argv[1:4])
assert C3["MEM"] == "0.80" and C3["PYTHONPATH"].endswith("p1-integrated8")
exp = dict(C3); exp["ATOM_P1_PREFIX_DEMOTE_DECODE"] = "1"
exp["PYTHONPATH"] = C3["PYTHONPATH"].replace("p1-integrated8", "p1-prefix-cache")
assert C12 == exp, sorted(set(C12.items()) ^ set(exp.items()))
e13 = dict(C12); e13.update(MEM="0.85", ATOM_P1_SCHED_IDLE_PREFILL_CAP="16384")
assert C13 == e13, sorted(set(C13.items()) ^ set(e13.items()))
assert int(C13["ATOM_SCHED_DECODE_PREFILL_CAP"]) == 16384  # with the idle cap: no step above 16384 tokens
assert "--max-num-batched-tokens 32768" in open(sys.argv[3]).read()
assert not any(k.startswith(("ATOM_P1_SP_TBO", "ATOM_P1_SCHED_BALANCE")) for k in C13)
P6
diff <(grep -v '^# arm' /tmp/pe_C3.$$) <(grep -v '^# arm' /shared_nfs/ygan/p1/milestone4/recipe/flags_armC3.txt) >/dev/null && ok "arm C3 == milestone4 flags_armC3.txt (control untouched)" || bad "arm C3 differs from milestone4"
bash -c "source $R/common.sh >/dev/null; source $R/env_armC13.sh; [ \"\$FLYDSL_RUNTIME_CACHE_DIR\" = \"\$NLOC/flydsl_armC12\" ] && [ \"\$FLYDSL_SEED\" = \"\$NLOC/flydsl_armC3\" ]" && ok "arms C12/C13 flydsl cache flydsl_armC12 seeded from flydsl_armC3" || bad "arm C12/C13 flydsl cache"
grep -q '^MS=/shared_nfs/ygan/p1/c17' $R/common.sh && ! grep -lq 'p1/official''-3600' $(ls $R/*.sh | grep -v check_recipe.sh) && ok "recipe writes only under p1/c17" || bad "recipe paths"
if python3 -c "import torch" 2>/dev/null; then
  (cd $RPC && python3 -m pytest -q -p no:cacheprovider tests/test_prefix_demote.py tests/test_sched_decode_prefill_policy.py tests/test_block_pool.py tests/test_block_manager.py > /tmp/pt.$$ 2>&1) \
    && ok "p1-prefix-cache CPU tests: $(tail -1 /tmp/pt.$$)" || bad "p1-prefix-cache CPU tests: $(tail -3 /tmp/pt.$$)"; rm -f /tmp/pt.$$
else echo "SKIP CPU tests (no torch here: run check_recipe in the p1 container)"; fi
rm -f /tmp/pe_C3.$$ /tmp/pe_C12.$$ /tmp/pe_C13.$$
# 13. c14: arm C14 = C13 + the five C10 flags + campaign overlay on p1/c14 (= int14 + prefix_fix.patch);
#     C14b = C14 + amdk 3d62453 (in place) + SPLIT4_MIN8 4 + aiter 7772 bitonly CSV. CPU only.
W14=/shared_nfs/ygan/work/p1-c14; G14="git -c safe.directory=*"
for a in C13 C14 C14b; do bash $R/print_env.sh $a > /tmp/pe_$a.$$ 2>&1
  grep -q "^check_repo OK" /tmp/pe_$a.$$ && ok "arm $a repo pinned" || bad "arm $a repo: $(grep check_repo /tmp/pe_$a.$$)"
  grep -qiE 'unbound|error|No such file' /tmp/pe_$a.$$ && bad "arm $a print_env errors"; done
for a in C14 C14b; do n=0; for v in $(grep -oE '^ATOM_[A-Z0-9_]+' /tmp/pe_$a.$$); do n=$((n+1)); grep -rqs --include=*.py "\"$v\"\|'$v'" $W14/atom || bad "arm $a sets $v but $W14/atom never reads it"; done; ok "arm $a: $n ATOM_* names, all read by $W14/atom"; done
python3 - /tmp/pe_C13.$$ /tmp/pe_C14.$$ /tmp/pe_C14b.$$ /shared_nfs/ygan/p1/campaign/recipe/flags_armC10.txt /shared_nfs/ygan/p1/campaign/recipe/flags_armC3.txt <<'P13' && ok "C14 = C13 + 5 C10 flags + campaign overlay + REPO p1-c14 (= C10 + DEMOTE + IDLE cap + MEM 0.85); C14b = C14 + amdk 3d62453 + MIN8 4 + bitonly CSV" || bad "arm C14/C14b values"
import re, sys
def kv(p):
    d = {}
    for l in open(p):
        m = re.match(r"^((?:ATOM_|AITER_)[A-Z0-9_]+|MEM|EXTRA|PYTHONPATH|REPO)=(.*)$", l.rstrip("\n"))
        if m: d[m.group(1)] = m.group(2)
    return d
C13, C14, C14b, C10, C3c = (kv(p) for p in sys.argv[1:6])
F = dict(ATOM_P1_L2_D2="1", ATOM_P1_L4_PULLQ="1", ATOM_P1_FAST_CONCAT="1", ATOM_P1_LMHEAD_RING="1", ATOM_P1_DRAFT_FC_AK="1")
exp = dict(C13); exp.update(F); exp["REPO"] = "/shared_nfs/ygan/work/p1-c14"
exp["PYTHONPATH"] = C13["PYTHONPATH"].rsplit(":", 1)[0] + ":/shared_nfs/ygan/p1/campaign/overlay:/shared_nfs/ygan/work/p1-c14"
assert C14 == exp, sorted(set(C14.items()) ^ set(exp.items()))
# independent route: C14 == campaign C10 (as gated/run there) + the C13 deltas, paths aside
e2 = {k: v for k, v in C10.items() if k not in ("PYTHONPATH", "REPO")}
e2.update(ATOM_P1_PREFIX_DEMOTE_DECODE="1", ATOM_P1_SCHED_IDLE_PREFILL_CAP="16384", MEM="0.85")
got = {k: v for k, v in C14.items() if k not in ("PYTHONPATH", "REPO")}
assert got == e2, sorted(set(got.items()) ^ set(e2.items()))
assert C10["PYTHONPATH"].replace("p1-integrated14", "p1-c14") == C14["PYTHONPATH"], (C10["PYTHONPATH"], C14["PYTHONPATH"])
for k in ("ATOM_DG_LMHEAD_AK", "ATOM_P1_AR_QKV_CODES", "ATOM_P1_FRONT_QKV_CHAIN", "ATOM_M3_AK_DECODE"): assert C14[k] == "1", k
assert int(C14["ATOM_SCHED_DECODE_PREFILL_CAP"]) == 16384 and C14["MEM"] == "0.85"
assert not any(k.startswith(("ATOM_P1_SP_SHARED_LOCAL", "ATOM_P1_SP_TBO", "ATOM_P1_SCHED_BALANCE")) for k in C14), "sp-local-shared / TBO must not be in C14"
eb = dict(C14); eb.update(ATOM_P1_INDEXER_SPLIT4_MIN8="4",
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE="/shared_nfs/ygan/p1/aiter-7772-bench/mx8_tuned_gemm_m31_pr7772_bitonly.csv")
eb["PYTHONPATH"] = C14["PYTHONPATH"].replace("/shared_nfs/ygan/attn-prefill/amdk_snapshot", "/shared_nfs/ygan/p1/upstream-leads/amdk_3d62453")
assert eb["PYTHONPATH"] != C14["PYTHONPATH"] and eb["PYTHONPATH"].startswith("/shared_nfs/ygan/p1/upstream-leads/amdk_3d62453:")
assert C14b == eb, sorted(set(C14b.items()) ^ set(eb.items()))
assert not any(k.startswith("ATOM_P1_SP_SHARED_LOCAL") for k in C14b)
P13
[ "$($G14 -C $W14 rev-parse HEAD^)" = "$($G14 -C $W14 rev-parse 5bf0aff51)" ] && ok "p1-c14 parent == int14 5bf0aff51" || bad "p1-c14 parent"
[ "$($G14 -C $W14 log -1 --format=%B | grep -ci co-authored || true)" = 0 ] && ok "p1-c14 commit: no Co-authored-by" || bad "p1-c14 trailers"
cmp -s <($G14 -C $W14 diff a52b845cb 9bfa1c6c4) <($G14 -C $W14 diff 5bf0aff51 HEAD) && ok "p1-c14 - int14 == prefix-cache - int8 (byte-identical diff)" || bad "p1-c14 prefix diff"
cmp -s <($G14 -C $W14 diff a52b845cb 5bf0aff51) <($G14 -C $W14 diff 9bfa1c6c4 HEAD) && ok "p1-c14 - prefix-cache == int14 - int8 (byte-identical diff)" || bad "p1-c14 int14 diff"
(cd /shared_nfs/ygan/p1/campaign/overlay/p1l2 && sha256sum -c --quiet /shared_nfs/ygan/p1/verify-L2-attn-decode/ovl/SHA256) && ok "overlay p1l2 == verified L2 copy" || bad "p1l2 sha"
l4=0; for f in /shared_nfs/ygan/p1/L4-comm/overlay/l4c_y/*.py; do cmp -s $f /shared_nfs/ygan/p1/campaign/overlay/l4c_y/$(basename $f) || { bad "l4c_y $(basename $f) differs"; l4=1; }; done; [ $l4 = 0 ] && ok "overlay l4c_y == L4-comm overlay"
bash -c "source $R/common.sh >/dev/null; source $R/env_armC14.sh; [ \"\$FLYDSL_RUNTIME_CACHE_DIR\" = \"\$NLOC/flydsl_armC14\" ] && [ \"\${FLYDSL_SEED%% *}\" = \"\$NLOC/flydsl_armC12\" ]" && ok "arm C14 flydsl cache flydsl_armC14, first seed flydsl_armC12" || bad "arm C14 flydsl cache"
grep -q 'VRAM_MAX_GIB=${VRAM_MAX_GIB:-282}' $R/point.sh && grep -q 'STALL_MAX=${STALL_MAX:-1}' $R/point.sh && grep -q "OutOfMemory|Traceback" $R/point.sh && grep -q 'reported running long' $R/point.sh && ok "point.sh strict abort watch (VRAM 282 GiB, alloc/OOM/Traceback, any long forward)" || bad "point.sh watch"
! grep -l '/shared_nfs/ygan/p1/prefix''-cache' $(ls $R/*.sh $R/*.py | grep -v check_recipe.sh) >/dev/null 2>&1 && ok "no recipe file points at p1/prefix-cache outputs" || bad "recipe still references p1/prefix-cache"
if python3 -c "import torch" 2>/dev/null; then
  (cd $W14 && python3 -m pytest -q -p no:cacheprovider tests/test_prefix_demote.py tests/test_sched_decode_prefill_policy.py tests/test_block_pool.py tests/test_block_manager.py > /tmp/pt14.$$ 2>&1) \
    && ok "p1-c14 prefix/scheduler CPU tests: $(tail -1 /tmp/pt14.$$)" || bad "p1-c14 CPU tests: $(tail -3 /tmp/pt14.$$)"
  (cd /tmp && ATOM_P1_PREFIX_DEMOTE_DECODE=1 python3 -m pytest -q -p no:cacheprovider $W14/tests/test_prefix_demote.py $W14/tests/test_block_pool.py $W14/tests/test_block_manager.py --rootdir $W14 > /tmp/pt14d.$$ 2>&1) \
    && ok "p1-c14 block tests with DEMOTE on: $(tail -1 /tmp/pt14d.$$)" || bad "p1-c14 DEMOTE-on tests: $(tail -3 /tmp/pt14d.$$)"
  python3 /shared_nfs/ygan/p1/campaign/tests/cpu_int14_test.py $W14 > /tmp/ct14.$$ 2>&1 && ok "campaign cpu_int14_test on p1-c14: $(tail -1 /tmp/ct14.$$)" || bad "cpu_int14_test on p1-c14: $(tail -5 /tmp/ct14.$$)"
  for a in C14 C14b; do pp=$(grep '^PYTHONPATH=' /tmp/pe_$a.$$ | cut -d= -f2-)
    (cd /tmp && python3 -I /shared_nfs/ygan/p1/k2-compose/bin/import_check.py "$pp" > /shared_nfs/ygan/p1/k2-compose/logs/import_check_$a.txt 2>&1)
    grep -q '^IMPORT_CHECK OK' /shared_nfs/ygan/p1/k2-compose/logs/import_check_$a.txt && ! grep '^  MULTI\|^  SHADOW' /shared_nfs/ygan/p1/k2-compose/logs/import_check_$a.txt | grep -qv '^  SHADOW atom: PYTHONPATH entry /shared_nfs/ygan/work/p1-c14 shadows container module /app/ATOM/atom/__init__.py' \
      && ok "arm $a import resolution: no multi-provided / shadowing module except atom -> p1-c14 over /app/ATOM (by design, every arm) (logs/import_check_$a.txt)" || bad "arm $a import resolution: $(grep -E 'MULTI|SHADOW|FAIL|ERROR' /shared_nfs/ygan/p1/k2-compose/logs/import_check_$a.txt | head -5)"; done
  rm -f /tmp/pt14.$$ /tmp/pt14d.$$ /tmp/ct14.$$
else echo "SKIP C14 CPU tests + import check (no torch here: run check_recipe in the p1 container)"; fi
rm -f /tmp/pe_C13.$$ /tmp/pe_C14.$$ /tmp/pe_C14b.$$
# 14. c15: arm C15 = C14 + ATOM_P1_MOE_IDROWS + ATOM_P1_MBF_ASM + ATOM_P1_MBF_BACK on p1/c15 (= p1/c14 + the p1/mbf-compose
#     diff); C15-noback = C15 without ATOM_P1_MBF_BACK. CPU only.
W15=/shared_nfs/ygan/work/p1-c15; G15="git -c safe.directory=*"
for a in C14 C15 C15-noback; do bash $R/print_env.sh $a > /tmp/pe15_$a.$$ 2>&1
  grep -q "^check_repo OK" /tmp/pe15_$a.$$ && ok "arm $a repo pinned" || bad "arm $a repo: $(grep check_repo /tmp/pe15_$a.$$)"
  grep -qiE 'unbound|error|No such file' /tmp/pe15_$a.$$ && bad "arm $a print_env errors"; done
for a in C15 C15-noback; do n=0; for v in $(grep -oE '^ATOM_[A-Z0-9_]+' /tmp/pe15_$a.$$); do n=$((n+1)); grep -rqs --include=*.py "\"$v\"\|'$v'" $W15/atom || bad "arm $a sets $v but $W15/atom never reads it"; done; ok "arm $a: $n ATOM_* names, all read by $W15/atom"; done
python3 - /tmp/pe15_C14.$$ /tmp/pe15_C15.$$ /tmp/pe15_C15-noback.$$ <<'P14' && ok "C15 = C14 + IDROWS + MBF_ASM + MBF_BACK + REPO p1-c15 (C14 had none of them; MOE_TILES=1 inherited); C15-noback = C15 - MBF_BACK; no TBO" || bad "arm C15/C15-noback values"
import re, sys
def kv(p):
    d = {}
    for l in open(p):
        m = re.match(r"^((?:ATOM_|AITER_)[A-Z0-9_]+|MEM|EXTRA|PYTHONPATH|REPO)=(.*)$", l.rstrip("\n"))
        if m: d[m.group(1)] = m.group(2)
    return d
C14, C15, NB = (kv(p) for p in sys.argv[1:4])
F = dict(ATOM_P1_MOE_IDROWS="1", ATOM_P1_MBF_ASM="1", ATOM_P1_MBF_BACK="1")
assert not set(F) & set(C14) and "ATOM_P1_MOE_ASM" not in C14 and C14["ATOM_P1_MOE_TILES"] == "1"
exp = dict(C14); exp.update(F); exp["REPO"] = "/shared_nfs/ygan/work/p1-c15"
exp["PYTHONPATH"] = C14["PYTHONPATH"].replace("/shared_nfs/ygan/work/p1-c14", "/shared_nfs/ygan/work/p1-c15")
assert exp["PYTHONPATH"].endswith(":/shared_nfs/ygan/work/p1-c15") and exp["PYTHONPATH"].count("p1-c15") == 1
assert C15 == exp, sorted(set(C15.items()) ^ set(exp.items()))
e2 = dict(C15); del e2["ATOM_P1_MBF_BACK"]
assert NB == e2, sorted(set(NB.items()) ^ set(e2.items()))
for d in (C15, NB):
    assert "ATOM_P1_MOE_ASM" not in d
    assert not any(k.startswith(("ATOM_P1_SP_TBO", "ATOM_P1_SCHED_BALANCE", "ATOM_P1_SP_SHARED_LOCAL")) for k in d), "TBO must stay off"
    assert d["MEM"] == "0.85" and d["ATOM_P1_SCHED_IDLE_PREFILL_CAP"] == "16384"
P14
[ "$($G15 -C $W15 rev-parse HEAD^)" = "$($G15 -C $W15 rev-parse 1121e1d52)" ] && ok "p1-c15 parent == p1/c14 1121e1d52" || bad "p1-c15 parent"
[ "$($G15 -C $W15 log -1 --format=%B | grep -ciE 'co-authored|signed-off' || true)" = 0 ] && ok "p1-c15 commit: no trailers" || bad "p1-c15 trailers"
nd() { grep -v '^index ' | sed -E 's/^@@ -[0-9,]+ \+[0-9,]+ @@/@@/'; }
cmp -s <($G15 -C $W15 diff 1121e1d52 HEAD | nd) <($G15 -C $W15 diff a52b845cb 1c4af4413 | nd) && ok "p1-c15 - p1-c14 == mbf-compose - int8 (identical modulo hunk line numbers / index lines)" || bad "p1-c15 mbf diff"
cmp -s <($G15 -C $W15 diff 1c4af4413 HEAD | nd) <($G15 -C $W15 diff a52b845cb 1121e1d52 | nd) && ok "p1-c15 - mbf-compose == p1-c14 - int8 (identical modulo hunk line numbers / index lines)" || bad "p1-c15 c14 diff"
[ -z "$($G15 -C $W15 status --short --untracked-files=no)" ] && ok "p1-c15 clean" || bad "p1-c15 dirty"
bash -c "source $R/common.sh >/dev/null; source $R/env_armC15.sh; [ \"\$FLYDSL_RUNTIME_CACHE_DIR\" = \"\$NLOC/flydsl_armC15\" ] && [ \"\${FLYDSL_SEED%% *}\" = \"\$NLOC/flydsl_armC14\" ]" && ok "arm C15 flydsl cache flydsl_armC15, first seed flydsl_armC14" || bad "arm C15 flydsl cache"
bash -c "source $R/common.sh >/dev/null; source $R/env_armC15-noback.sh; [ \"\$FLYDSL_RUNTIME_CACHE_DIR\" = \"\$NLOC/flydsl_armC15nb\" ] && [ \"\${FLYDSL_SEED%% *}\" = \"\$NLOC/flydsl_armC15\" ] && [ -z \"\${ATOM_P1_MBF_BACK:-}\" ]" && ok "arm C15-noback flydsl cache flydsl_armC15nb, MBF_BACK unset" || bad "arm C15-noback env"
if python3 -c "import torch" 2>/dev/null; then
  (cd $W15 && python3 -m pytest -q -p no:cacheprovider tests/test_prefix_demote.py tests/test_sched_decode_prefill_policy.py tests/test_block_pool.py tests/test_block_manager.py > /tmp/pt15.$$ 2>&1) \
    && ok "p1-c15 prefix/scheduler CPU tests: $(tail -1 /tmp/pt15.$$)" || bad "p1-c15 CPU tests: $(tail -3 /tmp/pt15.$$)"
  (cd /tmp && ATOM_P1_PREFIX_DEMOTE_DECODE=1 python3 -m pytest -q -p no:cacheprovider $W15/tests/test_prefix_demote.py $W15/tests/test_block_pool.py $W15/tests/test_block_manager.py --rootdir $W15 > /tmp/pt15d.$$ 2>&1) \
    && ok "p1-c15 block tests with DEMOTE on: $(tail -1 /tmp/pt15d.$$)" || bad "p1-c15 DEMOTE-on tests: $(tail -3 /tmp/pt15d.$$)"
  python3 /shared_nfs/ygan/p1/campaign/tests/cpu_int14_test.py $W15 > /tmp/ct15.$$ 2>&1 && ok "campaign cpu_int14_test on p1-c15: $(tail -1 /tmp/ct15.$$)" || bad "cpu_int14_test on p1-c15: $(tail -5 /tmp/ct15.$$)"
  pp=$(grep '^PYTHONPATH=' /tmp/pe15_C15.$$ | cut -d= -f2-)
  (cd /tmp && PYTHONPATH=$pp python3 /shared_nfs/ygan/p1/k2-compose/tests/cpu_c15_test.py $W15 > /shared_nfs/ygan/p1/k2-compose/logs/cpu_c15_test.txt 2>&1) && ok "cpu_c15_test (mbf flags: tiling tables, prerequisites, dispatch order, amd_kernels symbols): $(tail -1 /shared_nfs/ygan/p1/k2-compose/logs/cpu_c15_test.txt)" || bad "cpu_c15_test: $(tail -5 /shared_nfs/ygan/p1/k2-compose/logs/cpu_c15_test.txt)"
  for a in C15 C15-noback; do pp=$(grep '^PYTHONPATH=' /tmp/pe15_$a.$$ | cut -d= -f2-)
    (cd /tmp && python3 -I /shared_nfs/ygan/p1/k2-compose/bin/import_check.py "$pp" > /shared_nfs/ygan/p1/k2-compose/logs/import_check_$a.txt 2>&1)
    grep -q '^IMPORT_CHECK OK' /shared_nfs/ygan/p1/k2-compose/logs/import_check_$a.txt && ! grep '^  MULTI\|^  SHADOW' /shared_nfs/ygan/p1/k2-compose/logs/import_check_$a.txt | grep -qv '^  SHADOW atom: PYTHONPATH entry /shared_nfs/ygan/work/p1-c15 shadows container module /app/ATOM/atom/__init__.py' \
      && ok "arm $a import resolution: no multi-provided / shadowing module except atom -> p1-c15 over /app/ATOM (logs/import_check_$a.txt)" || bad "arm $a import resolution: $(grep -E 'MULTI|SHADOW|FAIL|ERROR' /shared_nfs/ygan/p1/k2-compose/logs/import_check_$a.txt | head -5)"; done
  rm -f /tmp/pt15.$$ /tmp/pt15d.$$ /tmp/ct15.$$
else echo "SKIP C15 CPU tests + import check (no torch here: run check_recipe in the p1 container)"; fi
rm -f /tmp/pe15_C14.$$ /tmp/pe15_C15.$$ /tmp/pe15_C15-noback.$$
# 15. k2-compose: arm C16 = C15 + ATOM_P1_K2_SCORER=rot2_wr2_h1024 + ATOM_P1_DRAFT_MARKOV_PERSIST=1 + overlay
#     /shared_nfs/ygan/p1/k2-compose/overlay on p1/k2-compose (= p1/c15 + k2-scorer diff + k2-draft diff without the
#     harness-only model_runner.py layout fix). CPU only.
W16=/shared_nfs/ygan/work/p1-k2-compose; K2C=/shared_nfs/ygan/p1/k2-compose
for a in C15 C16; do bash $R/print_env.sh $a > /tmp/pe16_$a.$$ 2>&1
  grep -q "^check_repo OK" /tmp/pe16_$a.$$ && ok "arm $a repo pinned" || bad "arm $a repo: $(grep check_repo /tmp/pe16_$a.$$)"
  grep -qiE 'unbound|error|No such file' /tmp/pe16_$a.$$ && bad "arm $a print_env errors"; done
n=0; for v in $(grep -oE '^ATOM_[A-Z0-9_]+' /tmp/pe16_C16.$$); do n=$((n+1)); grep -rqs --include=*.py "\"$v\"\|'$v'" $W16/atom || bad "arm C16 sets $v but $W16/atom never reads it"; done; ok "arm C16: $n ATOM_* names, all read by $W16/atom"
python3 - /tmp/pe16_C15.$$ /tmp/pe16_C16.$$ <<'P15' && ok "C16 = C15 + K2_SCORER rot2_wr2_h1024 + DRAFT_MARKOV_PERSIST + k2-compose overlay + REPO p1-k2-compose (C15 had neither); no TBO" || bad "arm C16 values"
import re, sys
def kv(p):
    d = {}
    for l in open(p):
        m = re.match(r"^((?:ATOM_|AITER_)[A-Z0-9_]+|MEM|EXTRA|PYTHONPATH|REPO)=(.*)$", l.rstrip("\n"))
        if m: d[m.group(1)] = m.group(2)
    return d
C15, C16 = (kv(p) for p in sys.argv[1:3])
F = dict(ATOM_P1_K2_SCORER="rot2_wr2_h1024", ATOM_P1_DRAFT_MARKOV_PERSIST="1")
assert not set(F) & set(C15)
exp = dict(C15); exp.update(F); exp["REPO"] = "/shared_nfs/ygan/work/p1-k2-compose"
pre = C15["PYTHONPATH"][: -len(":/shared_nfs/ygan/work/p1-c15")]
assert C15["PYTHONPATH"].endswith(":/shared_nfs/ygan/work/p1-c15")
exp["PYTHONPATH"] = pre + ":/shared_nfs/ygan/p1/k2-compose/overlay:/shared_nfs/ygan/work/p1-k2-compose"
assert C16 == exp, sorted(set(C16.items()) ^ set(exp.items()))
assert not any(k.startswith(("ATOM_P1_SP_TBO", "ATOM_P1_SCHED_BALANCE", "ATOM_P1_SP_SHARED_LOCAL")) for k in C16), "TBO must stay off"
assert C16["MEM"] == "0.85" and all(C16[k] == "1" for k in ("ATOM_P1_MOE_IDROWS", "ATOM_P1_MBF_ASM", "ATOM_P1_MBF_BACK", "ATOM_P1_INDEXER_SPLIT4"))
P15
G16="git -c safe.directory=*"
[ "$($G16 -C $W16 rev-parse HEAD^)" = "$($G16 -C $W16 rev-parse 82e41951d)" ] && ok "p1-k2-compose parent == p1/c15 82e41951d" || bad "p1-k2-compose parent"
[ "$($G16 -C $W16 log -1 --format=%B | grep -ciE 'co-authored|signed-off' || true)" = 0 ] && ok "p1-k2-compose commit: no trailers" || bad "p1-k2-compose trailers"
TI=$(mktemp); ( cd $W16 && GIT_INDEX_FILE=$TI $G16 read-tree 82e41951d && GIT_INDEX_FILE=$TI $G16 apply --cached $K2C/patches/k2_scorer.diff && GIT_INDEX_FILE=$TI $G16 apply --cached $K2C/patches/k2_draft_markov.diff && [ "$(GIT_INDEX_FILE=$TI $G16 write-tree)" = "$($G16 rev-parse HEAD^{tree})" ] ) \
  && ok "p1-k2-compose tree == p1/c15 + patches/k2_scorer.diff + patches/k2_draft_markov.diff exactly" || bad "p1-k2-compose tree != c15 + the two lane patches"; rm -f $TI
cmp -s <(grep -v '^index' $K2C/patches/k2_scorer.diff) <(grep -v '^index' /shared_nfs/ygan/p1/verify-k2-scorer/k2_lane.diff) && ok "k2_scorer.diff == verify-k2-scorer's k2_lane.diff (the verified diff)" || bad "scorer diff differs from the verified one"
python3 -c "
import sys; s=open('/shared_nfs/ygan/p1/k2-draft/markov_persist.patch').read().split('diff --git ')[1:]
k=''.join('diff --git '+p for p in s if not p.startswith('a/atom/model_engine/model_runner.py'))
sys.exit(0 if k==open('$K2C/patches/k2_draft_markov.diff').read() else 1)" && ok "k2_draft_markov.diff == k2-draft markov_persist.patch minus model_runner.py (layout fix excluded)" || bad "draft diff"
[ -z "$($G16 -C $W16 diff 82e41951d HEAD -- atom/model_engine/model_runner.py)" ] && ok "model_runner.py unchanged vs p1/c15 (no harness layout fix in the recipe tree)" || bad "model_runner.py changed"
[ -z "$($G16 -C $W16 status --short --untracked-files=no)" ] && ok "p1-k2-compose clean" || bad "p1-k2-compose dirty"
[ ! -e $K2C/overlay/p1k2/atom_patch.py ] && bad "k2-compose overlay missing"; for m in $(ls $K2C/overlay); do for d in $(grep '^PYTHONPATH=' /tmp/pe16_C16.$$ | cut -d= -f2- | tr : ' '); do [ "$d" != $K2C/overlay ] && [ -e "$d/$m" ] && bad "overlay package $m also provided by $d"; done; done; ok "overlay packages ($(ls $K2C/overlay | tr '\n' ' ')) provided by no other C16 PYTHONPATH entry"
bash -c "source $R/common.sh >/dev/null; source $R/env_armC16.sh; [ \"\$FLYDSL_RUNTIME_CACHE_DIR\" = \"\$NLOC/flydsl_armC16\" ] && [ \"\${FLYDSL_SEED%% *}\" = \"\$NLOC/flydsl_armC15\" ]" && ok "arm C16 flydsl cache flydsl_armC16, first seed flydsl_armC15" || bad "arm C16 flydsl cache"
if python3 -c "import torch" 2>/dev/null; then
  pp=$(grep '^PYTHONPATH=' /tmp/pe16_C16.$$ | cut -d= -f2-)
  (cd /tmp && PYTHONPATH=$pp python3 $K2C/tests/cpu_k2c_test.py $W16 > $K2C/logs/cpu_k2c_test.txt 2>&1) && ok "cpu_k2c_test (flags, overlay pin, arm parse, wrapper signature, markov module): $(tail -1 $K2C/logs/cpu_k2c_test.txt)" || bad "cpu_k2c_test: $(tail -5 $K2C/logs/cpu_k2c_test.txt)"
  (cd /tmp && PYTHONPATH=$pp python3 $K2C/tests/cpu_c15_test.py $W16 > $K2C/logs/cpu_c15_test_on_k2c.txt 2>&1) && ok "cpu_c15_test on p1-k2-compose: $(tail -1 $K2C/logs/cpu_c15_test_on_k2c.txt)" || bad "cpu_c15_test on k2c: $(tail -5 $K2C/logs/cpu_c15_test_on_k2c.txt)"
  python3 /shared_nfs/ygan/p1/campaign/tests/cpu_int14_test.py $W16 > /tmp/ct16.$$ 2>&1 && ok "campaign cpu_int14_test on p1-k2-compose: $(tail -1 /tmp/ct16.$$)" || bad "cpu_int14_test on p1-k2-compose: $(tail -5 /tmp/ct16.$$)"
  (cd $W16 && python3 -m pytest -q -p no:cacheprovider tests/test_prefix_demote.py tests/test_sched_decode_prefill_policy.py tests/test_block_pool.py tests/test_block_manager.py > /tmp/pt16.$$ 2>&1) \
    && ok "p1-k2-compose prefix/scheduler CPU tests: $(tail -1 /tmp/pt16.$$)" || bad "p1-k2-compose CPU tests: $(tail -3 /tmp/pt16.$$)"
  (cd /tmp && python3 -I /shared_nfs/ygan/p1/k2-compose/bin/import_check.py "$pp" > $K2C/logs/import_check_C16.txt 2>&1)
  grep -q '^IMPORT_CHECK OK' $K2C/logs/import_check_C16.txt && ! grep '^  MULTI\|^  SHADOW' $K2C/logs/import_check_C16.txt | grep -qv '^  SHADOW atom: PYTHONPATH entry /shared_nfs/ygan/work/p1-k2-compose shadows container module /app/ATOM/atom/__init__.py' \
    && ok "arm C16 import resolution: no multi-provided / shadowing module except atom -> p1-k2-compose over /app/ATOM (logs/import_check_C16.txt)" || bad "arm C16 import resolution: $(grep -E 'MULTI|SHADOW|FAIL|ERROR' $K2C/logs/import_check_C16.txt | head -5)"
  rm -f /tmp/ct16.$$ /tmp/pt16.$$
else echo "SKIP C16 CPU tests + import check (no torch here: run check_recipe in the p1 container)"; fi
rm -f /tmp/pe16_C15.$$ /tmp/pe16_C16.$$
# 16. c17: arm C17 = C16 + ATOM_P1_SP_SHARED_LOCAL{=1,_MIN_TOKENS=2048,_SIDE_ROWS=1} + sp-local-shared overlay on p1/c17
#     (= p1/k2-compose + sp-local-shared/sp_local_shared.patch, as C9/C11 use it); C17b = C17 with MEM 0.8591. CPU only.
W17=/shared_nfs/ygan/work/p1-c17; SLS=/shared_nfs/ygan/p1/sp-local-shared; G17="git -c safe.directory=*"
for a in C16 C17 C17b; do bash $R/print_env.sh $a > /tmp/pe17_$a.$$ 2>&1
  grep -q "^check_repo OK" /tmp/pe17_$a.$$ && ok "arm $a repo pinned" || bad "arm $a repo: $(grep check_repo /tmp/pe17_$a.$$)"
  grep -qiE 'unbound|error|No such file' /tmp/pe17_$a.$$ && bad "arm $a print_env errors"; done
for a in C17 C17b; do n=0; for v in $(grep -oE '^ATOM_[A-Z0-9_]+' /tmp/pe17_$a.$$); do n=$((n+1)); grep -rqs --include=*.py "\"$v\"\|'$v'" $W17/atom || bad "arm $a sets $v but $W17/atom never reads it"; done; ok "arm $a: $n ATOM_* names, all read by $W17/atom"; done
for v in ATOM_P1_SP_SHARED_LOCAL ATOM_P1_SP_SHARED_LOCAL_MIN_TOKENS ATOM_P1_SP_SHARED_LOCAL_SIDE_ROWS; do
  grep -rqs --include=*.py "envs\.$v\b" $W17/atom/models && ok "$v consumed (envs.$v) in $W17/atom/models" || bad "$v defined but never consumed"; done
python3 - /tmp/pe17_C16.$$ /tmp/pe17_C17.$$ /tmp/pe17_C17b.$$ <<'P16' && ok "C17 = C16 + 3 SP_SHARED_LOCAL flags + sls overlay + REPO p1-c17 (C16 had none; SP/SLOTS/AK_MOE companions on); C17b = C17 + MEM 0.8591; no TBO" || bad "arm C17/C17b values"
import re, sys
def kv(p):
    d = {}
    for l in open(p):
        m = re.match(r"^((?:ATOM_|AITER_)[A-Z0-9_]+|MEM|EXTRA|PYTHONPATH|REPO)=(.*)$", l.rstrip("\n"))
        if m: d[m.group(1)] = m.group(2)
    return d
C16, C17, C17b = (kv(p) for p in sys.argv[1:4])
F = dict(ATOM_P1_SP_SHARED_LOCAL="1", ATOM_P1_SP_SHARED_LOCAL_MIN_TOKENS="2048", ATOM_P1_SP_SHARED_LOCAL_SIDE_ROWS="1")
assert not any(k.startswith("ATOM_P1_SP_SHARED_LOCAL") for k in C16)
exp = dict(C16); exp.update(F); exp["REPO"] = "/shared_nfs/ygan/work/p1-c17"
assert C16["PYTHONPATH"].endswith(":/shared_nfs/ygan/work/p1-k2-compose")
pre = C16["PYTHONPATH"][: -len(":/shared_nfs/ygan/work/p1-k2-compose")]
exp["PYTHONPATH"] = pre + ":/shared_nfs/ygan/p1/sp-local-shared/overlay:/shared_nfs/ygan/work/p1-c17"
assert C17 == exp, sorted(set(C17.items()) ^ set(exp.items()))
for k in ("ATOM_M3_SP_PREFILL", "ATOM_P1_SP_MOE_SLOTS", "ATOM_M3_AK_MOE", "ATOM_P1_SP_MOE_COMPACT"):
    assert C17.get(k) == "1", k
assert not any(k.startswith(("ATOM_P1_SP_TBO", "ATOM_P1_SCHED_BALANCE")) for k in C17), "TBO must stay off"
assert C17["MEM"] == "0.85" and C17["ATOM_P1_K2_SCORER"] == "rot2_wr2_h1024" and C17["ATOM_P1_DRAFT_MARKOV_PERSIST"] == "1"
e2 = dict(C17); e2["MEM"] = "0.8591"
assert C17b == e2, sorted(set(C17b.items()) ^ set(e2.items()))
P16
[ "$($G17 -C $W17 rev-parse HEAD^)" = "$($G17 -C $W17 rev-parse 33786376b)" ] && ok "p1-c17 parent == p1/k2-compose 33786376b" || bad "p1-c17 parent"
[ "$($G17 -C $W17 log -1 --format=%B | grep -ciE 'co-authored|signed-off' || true)" = 0 ] && ok "p1-c17 commit: no trailers" || bad "p1-c17 trailers"
nd() { grep -v '^index ' | sed -E 's/^@@ -[0-9,]+ \+[0-9,]+ @@/@@/'; }
cmp -s <($G17 -C $W17 diff 33786376b HEAD | nd) <(nd < $SLS/sp_local_shared.patch) && ok "p1-c17 - p1-k2-compose == sp_local_shared.patch (modulo hunk offsets / index lines)" || bad "p1-c17 diff != sls patch"
cmp -s <($G17 -C $W17 diff a52b845cb 4a46145aa | nd) <(nd < $SLS/sp_local_shared.patch) && ok "sls patch == int13 - int8 (the C9/C11 tree's change)" || bad "sls patch != int13 diff"
[ -z "$($G17 -C $W17 status --short --untracked-files=no)" ] && ok "p1-c17 clean" || bad "p1-c17 dirty"
for m in $(ls $SLS/overlay); do for d in $(grep '^PYTHONPATH=' /tmp/pe17_C17.$$ | cut -d= -f2- | tr : ' '); do [ "$d" != $SLS/overlay ] && [ -e "$d/$m" ] && bad "overlay package $m also provided by $d"; done; done; ok "sls overlay packages ($(ls $SLS/overlay | tr '\n' ' ')) provided by no other C17 PYTHONPATH entry"
bash -c "source $R/common.sh >/dev/null; source $R/env_armC17.sh; [ \"\$FLYDSL_RUNTIME_CACHE_DIR\" = \"\$NLOC/flydsl_armC17\" ] && [ \"\${FLYDSL_SEED%% *}\" = \"\$NLOC/flydsl_armC16\" ]" && ok "arm C17 flydsl cache flydsl_armC17, first seed flydsl_armC16" || bad "arm C17 flydsl cache"
bash -c "source $R/common.sh >/dev/null; source $R/env_armC17b.sh; [ \"\$FLYDSL_RUNTIME_CACHE_DIR\" = \"\$NLOC/flydsl_armC17b\" ] && [ \"\${FLYDSL_SEED%% *}\" = \"\$NLOC/flydsl_armC17\" ] && [ \"\$MEM\" = 0.8591 ]" && ok "arm C17b flydsl cache flydsl_armC17b, first seed flydsl_armC17, MEM 0.8591" || bad "arm C17b env"
if python3 -c "import torch" 2>/dev/null; then
  pp=$(grep '^PYTHONPATH=' /tmp/pe17_C17.$$ | cut -d= -f2-)
  (cd /tmp && PYTHONPATH=$pp python3 $K2C/tests/cpu_k2c_test.py $W17 > /shared_nfs/ygan/p1/c17/logs/cpu_k2c_test_on_c17.txt 2>&1) && ok "cpu_k2c_test on p1-c17: $(tail -1 /shared_nfs/ygan/p1/c17/logs/cpu_k2c_test_on_c17.txt)" || bad "cpu_k2c_test on c17: $(tail -5 /shared_nfs/ygan/p1/c17/logs/cpu_k2c_test_on_c17.txt)"
  (cd /tmp && PYTHONPATH=$pp python3 $K2C/tests/cpu_c15_test.py $W17 > /shared_nfs/ygan/p1/c17/logs/cpu_c15_test_on_c17.txt 2>&1) && ok "cpu_c15_test on p1-c17: $(tail -1 /shared_nfs/ygan/p1/c17/logs/cpu_c15_test_on_c17.txt)" || bad "cpu_c15_test on c17: $(tail -5 /shared_nfs/ygan/p1/c17/logs/cpu_c15_test_on_c17.txt)"
  python3 /shared_nfs/ygan/p1/campaign/tests/cpu_int14_test.py $W17 > /tmp/ct17.$$ 2>&1 && ok "campaign cpu_int14_test on p1-c17: $(tail -1 /tmp/ct17.$$)" || bad "cpu_int14_test on p1-c17: $(tail -5 /tmp/ct17.$$)"
  python3 -I /shared_nfs/ygan/p1/milestone6/tests/cpu_sls_test.py $W17 $SLS/overlay > /shared_nfs/ygan/p1/c17/logs/cpu_sls_test_on_c17.txt 2>&1 && ok "milestone6 cpu_sls_test on p1-c17 (py_compile, envs defaults, TP4 shard -> full assembly byte-identical, dispatch guards): $(tail -1 /shared_nfs/ygan/p1/c17/logs/cpu_sls_test_on_c17.txt)" || bad "cpu_sls_test on p1-c17: $(tail -5 /shared_nfs/ygan/p1/c17/logs/cpu_sls_test_on_c17.txt)"
  (cd $W17 && python3 -m pytest -q -p no:cacheprovider tests/test_prefix_demote.py tests/test_sched_decode_prefill_policy.py tests/test_block_pool.py tests/test_block_manager.py > /tmp/pt17.$$ 2>&1) \
    && ok "p1-c17 prefix/scheduler CPU tests: $(tail -1 /tmp/pt17.$$)" || bad "p1-c17 CPU tests: $(tail -3 /tmp/pt17.$$)"
  (cd /tmp && PYTHONPATH=$pp python3 -c "import p1_sls_shared, sys; from p1_sls_shared import SharedFull, warm_up, sum5; print('p1_sls_shared from', p1_sls_shared.__file__)" > /shared_nfs/ygan/p1/c17/logs/sls_import.txt 2>&1) && grep -q "$SLS/overlay/p1_sls_shared/__init__.py" /shared_nfs/ygan/p1/c17/logs/sls_import.txt && ok "p1_sls_shared imports from the sls overlay on the C17 PYTHONPATH" || bad "p1_sls_shared import: $(tail -3 /shared_nfs/ygan/p1/c17/logs/sls_import.txt)"
  (cd /tmp && python3 -I /shared_nfs/ygan/p1/k2-compose/bin/import_check.py "$pp" > /shared_nfs/ygan/p1/c17/logs/import_check_C17.txt 2>&1)
  grep -q '^IMPORT_CHECK OK' /shared_nfs/ygan/p1/c17/logs/import_check_C17.txt && ! grep '^  MULTI\|^  SHADOW' /shared_nfs/ygan/p1/c17/logs/import_check_C17.txt | grep -qv '^  SHADOW atom: PYTHONPATH entry /shared_nfs/ygan/work/p1-c17 shadows container module /app/ATOM/atom/__init__.py' \
    && ok "arm C17 import resolution: no multi-provided / shadowing module except atom -> p1-c17 over /app/ATOM (logs/import_check_C17.txt)" || bad "arm C17 import resolution: $(grep -E 'MULTI|SHADOW|FAIL|ERROR' /shared_nfs/ygan/p1/c17/logs/import_check_C17.txt | head -5)"
  rm -f /tmp/ct17.$$ /tmp/pt17.$$
else echo "SKIP C17 CPU tests + import check (no torch here: run check_recipe in the p1 container)"; fi
rm -f /tmp/pe17_C16.$$ /tmp/pe17_C17.$$ /tmp/pe17_C17b.$$
[ $fail = 0 ] && echo "ALL CHECKS PASS" || echo "SOME CHECKS FAILED"
exit $fail
