#!/bin/bash
# Re-creates the recipe's copies of the previous team's scripts from the read-only originals in /shared_nfs/ygan/max,
# changing ONLY where they write (max/logs, max/results, max/home, max/acc/gates -> p1/milestone or node-local) and
# making serve.sh's `git log` header work in root containers. Writes diffs/*.diff so every change is reviewable.
set -eu
M=/shared_nfs/ygan/max; R=/shared_nfs/ygan/p1/c17/recipe; MS=/shared_nfs/ygan/p1/c17
NL=/mnt/m2m_nobackup/ygan/p1/milestone
mkdir -p $R/diffs
cp $M/int1/recipe.sh $R/int1_recipe.sh
sed -e "s#^LOG=\$D/logs/server_\${TAG}.log#LOG=$MS/logs/server_\${TAG}.log#" \
    -e "s#^export HOME=\$D/home/\$TAG#export HOME=$NL/home/\$TAG#" \
    -e "s#FLYDSL_RUNTIME_CACHE_DIR:-\$D/flydsl_cache#FLYDSL_RUNTIME_CACHE_DIR:-$NL/flydsl#" \
    -e "s#  git -C \"\$REPO\" log -1#  git -c safe.directory='*' -C \"\$REPO\" log -1#" \
    $M/bin/serve.sh > $R/serve.sh
sed -e "s#^ART=/shared_nfs/ygan/max/results/\$TAG#ART=$MS/results/\$TAG#" $M/bin/bench.sh > $R/bench.sh
sed -e "s#^OUT=\$A/gates/\$GTAG#OUT=$MS/gates/\$GTAG#" \
    -e "s#^LOG=\$D/logs/server_\$TAG.log#LOG=$MS/logs/server_\$TAG.log#" \
    -e "s#bash \$B/serve.sh#bash $R/serve.sh#" \
    -e "s#>> \$A/gates/index.tsv#>> $MS/gates/index.tsv#" \
    -e "s#python3 \$B/gate_verdict.py#python3 $R/gate_verdict.py#" \
    $M/bin/gate.sh > $R/gate.sh
sed -e "s#open(os.path.join(A, \"gates\", \"index.tsv\"), \"a\")#open(\"$MS/gates/index.tsv\", \"a\")#" \
    $M/bin/gate_verdict.py > $R/gate_verdict.py
sed -e "s#^D = \"/shared_nfs/ygan/max\"#D = \"$MS\"#" $M/bin/ladder_stats.py > $R/ladder_stats.py
chmod +x $R/serve.sh $R/bench.sh $R/gate.sh
for f in serve.sh bench.sh gate.sh gate_verdict.py ladder_stats.py; do
  diff -u $M/bin/$f $R/$f > $R/diffs/$f.diff || true
done
diff -u $M/int1/recipe.sh $R/int1_recipe.sh > $R/diffs/int1_recipe.sh.diff || true
# every copy must differ from its original exactly where intended
n() { grep -c '^[-+][^-+]' "$R/diffs/$1.diff" || true; }
echo "changed lines (-/+): serve.sh $(n serve.sh) (expect 8) bench.sh $(n bench.sh) (2) gate.sh $(n gate.sh) (10)" \
     "gate_verdict.py $(n gate_verdict.py) (2) ladder_stats.py $(n ladder_stats.py) (2) int1_recipe.sh $(n int1_recipe.sh) (0)"
sha256sum $M/int1/recipe.sh $R/int1_recipe.sh
