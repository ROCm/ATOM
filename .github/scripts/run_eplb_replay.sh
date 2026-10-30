#!/usr/bin/env bash
set -euo pipefail
out=eplb-replay-evidence
mkdir -p "$out"
image_id=$(docker inspect --format '{{.Image}}' atom-eplb-replay)
test "$image_id" = sha256:35b984a7a2fc779ba288a4bb21e3a8c41f2ef4ec8e488e382cd84d975e9b7be8
printf '%s\n' "$image_id" > "$out/image-id.txt"
git rev-parse HEAD > "$out/atom-sha.txt"
cp .github/scripts/eplb_replay_layer52.json "$out/replay-input.json"
docker exec atom-eplb-replay python3 .github/scripts/validate_eplb_replay.py
docker exec atom-eplb-replay python3 .github/scripts/sample_eplb_host.py "/workspace/$out/host.jsonl" &
sampler=$!
finish() {
  local code=$?
  trap - EXIT
  docker exec atom-eplb-replay pkill -TERM -f '[s]ample_eplb_host.py' || true
  wait "$sampler" || true
  echo "$code" > "$out/exit-code.txt"
  exit "$code"
}
trap finish EXIT
docker exec -e PYTHONPATH=/workspace -e EXPECTED_ATOM_SHA="$(git rev-parse HEAD)" \
  -e EXPECTED_AITER_SHA=6fe1bfc8fe1341179567268820e62827c19aee39 \
  atom-eplb-replay timeout --signal=TERM --kill-after=30s 12m \
  python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  .github/scripts/eplb_migration_replay.py \
  --input .github/scripts/eplb_replay_layer52.json --output "/workspace/$out" \
  2>&1 | tee "$out/replay.log"
