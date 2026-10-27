#!/bin/bash
# Fetch the MLPerf agentic dataset (613 trajectories) into $R/dataset and verify it.
# Plain curl over the published md5 list: the same files mlc-r2-downloader.sh fetches.
# usage: R=<work dir> bash get_dataset.sh
set -e
R=${R:-/work/qwen35_ep}; NAME=mlperf_agentic_inference_dataset
BASE=https://endpoints.mlcommons-storage.org
mkdir -p $R/dataset/$NAME && cd $R/dataset/$NAME
curl -sfL --retry 3 -o ../$NAME.md5 $BASE/metadata/$NAME.md5
for f in $(awk '{print $2}' ../$NAME.md5 | sed 's,^\./,,'); do
  [ -s "$f" ] || curl -sfL --retry 3 -o "$f" "$BASE/$NAME/$f"
done
md5sum -c ../$NAME.md5
echo "1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae  $NAME.jsonl" | sha256sum -c
