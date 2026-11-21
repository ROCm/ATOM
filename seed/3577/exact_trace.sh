#!/usr/bin/env bash
# One diagnostic request; the common supervisor owns collector/server cleanup.
set -euo pipefail
python3 - <<'PY'
import hashlib, json, os, pathlib, pydoc_data.topics
from transformers import AutoTokenizer
out = pathlib.Path(os.environ["EVIDENCE"])
tok = AutoTokenizer.from_pretrained(os.environ["MODEL"], trust_remote_code=True,
                                   local_files_only=True)
text = "Validation 3577 exact prefill 357702\n" + "\n".join(
    pydoc_data.topics.topics[k] for k in sorted(pydoc_data.topics.topics))
ids = tok.encode(text, add_special_tokens=False)
assert len(ids) >= 49152
payload = dict(model=os.environ["MODEL"], prompt=ids[:49152], max_tokens=64,
               temperature=0.0, ignore_eos=True, stream=False, return_token_ids=True)
raw = json.dumps(payload, separators=(",", ":")).encode()
(out / "exact-request.json").write_bytes(raw)
(out / "exact-request.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")
PY
python3 -m atom.benchmarks.results capture-config --model "$MODEL" \
  --kind random --concurrency 1 --launch /tmp/atom-benchmark-launch.json \
  --output "$EVIDENCE/trace-config.json"
curl --fail-with-body --connect-timeout 2 --max-time 5 http://localhost:8000/health
python3 -m atom.benchmarks.results collect \
  --config "$EVIDENCE/trace-config.json" --interval 0.1 \
  --output "$EVIDENCE/exact-telemetry" --server-url http://localhost:8000/metrics \
  > "$EVIDENCE/exact-collector.log" 2>&1 &
set +e
curl --fail-with-body --connect-timeout 2 --max-time 135 -X POST \
  http://localhost:8000/start_profile -o "$EVIDENCE/start-profile.json"
start_rc=$?
request_rc=1
if [ "$start_rc" -eq 0 ]; then
  curl --fail-with-body --connect-timeout 2 --max-time 300 \
    http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
    --data-binary "@$EVIDENCE/exact-request.json" \
    -D "$EVIDENCE/exact-headers.txt" -o "$EVIDENCE/exact-response.json"
  request_rc=$?
fi
curl --fail-with-body --connect-timeout 2 --max-time 135 -X POST \
  http://localhost:8000/stop_profile -o "$EVIDENCE/stop-profile.json"
stop_rc=$?
printf '{"start":%d,"request":%d,"stop":%d}\n' "$start_rc" "$request_rc" "$stop_rc" \
  > "$EVIDENCE/exact-exit.json"
set -e
test "$start_rc" -eq 0
test "$request_rc" -eq 0
test "$stop_rc" -eq 0
python3 - <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ["EVIDENCE"])
request = json.loads((out / "exact-request.json").read_text())
response = json.loads((out / "exact-response.json").read_text())
assert response["prompt_token_ids"] == request["prompt"]
assert response["usage"]["prompt_tokens"] == 49152
assert response["usage"]["completion_tokens"] == 64
assert response["choices"][0]["finish_reason"] == "length"
PY
