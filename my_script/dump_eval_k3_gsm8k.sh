#!/usr/bin/env bash
set -euo pipefail

mkdir -p accuracy_test_results
RUN_TAG=$(date +%Y%m%d%H%M%S)
OUTPUT_PATH=${OUTPUT_PATH:-accuracy_test_results/${RUN_TAG}}

MODEL=/data/models/Kimi-k2.7-xiaobing

# SHOT=${SHOT:-20}
SHOT=${SHOT:-5}
LIMIT=${LIMIT:-50}
NUM_CONCURRENT=${NUM_CONCURRENT:-16}

LIMIT_ARG=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARG+=(--limit "${LIMIT}")
fi

echo "MODEL: ${MODEL}"
echo "SHOT: ${SHOT}"
echo "LIMIT: ${LIMIT:-full}"
echo "NUM_CONCURRENT: ${NUM_CONCURRENT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

lm_eval --model local-completions \
  --model_args "model=${MODEL},base_url=http://localhost:8083/v1/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k \
  --num_fewshot ${SHOT} \
  --seed 42 \
  "${LIMIT_ARG[@]}" \
  --log_samples \
  --output_path "${OUTPUT_PATH}" 2>&1 | tee log.lmeval.log

python3 - "${OUTPUT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"[dump] output path does not exist: {root}")

sample_files = sorted(
    p
    for p in root.rglob("*")
    if p.is_file()
    and p.suffix in {".json", ".jsonl"}
    and "sample" in p.name.lower()
    and "gsm8k" in p.name.lower()
)
if not sample_files:
    raise SystemExit(f"[dump] no GSM8K sample json/jsonl found under {root}")


def load_records(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        for key in ("samples", "gsm8k"):
            value = data.get(key)
            if isinstance(value, list):
                yield from value
                return
        yield data


def first_scalar(value):
    while isinstance(value, list) and value:
        value = value[0]
    return value


def text_field(value):
    value = first_scalar(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def metric_value(sample):
    metrics = sample.get("metrics")
    candidates = []
    if isinstance(metrics, dict):
        candidates.extend(metrics.items())
    candidates.extend(sample.items())

    preferred = [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match",
    ]
    for name in preferred:
        for key, value in candidates:
            if key == name:
                return first_scalar(value), key
    for key, value in candidates:
        if "exact_match" in str(key):
            return first_scalar(value), key
    raise KeyError("no exact_match metric found in sample")


def is_correct(sample):
    value, key = metric_value(sample)
    if isinstance(value, bool):
        return value, key, value
    if isinstance(value, (int, float)):
        return float(value) > 0.5, key, value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "correct"}:
            return True, key, value
        if normalized in {"false", "no", "incorrect"}:
            return False, key, value
        try:
            return float(normalized) > 0.5, key, value
        except ValueError:
            pass
    raise ValueError(f"unsupported metric value for {key}: {value!r}")


def model_response(sample):
    for key in ("filtered_resps", "resps", "response", "model_output", "generation"):
        if key in sample:
            return text_field(sample[key])
    return ""


def normalize(sample):
    doc = sample.get("doc") if isinstance(sample.get("doc"), dict) else {}
    correct, metric_name, metric_raw = is_correct(sample)
    return {
        "doc_id": sample.get("doc_id"),
        "question": doc.get("question") or sample.get("question") or "",
        "gold_answer": doc.get("answer") or sample.get("target") or sample.get("answer") or "",
        "model_response": model_response(sample),
        "correct": correct,
        "metric_name": metric_name,
        "metric_value": metric_raw,
    }


rows_by_doc_id = {}
duplicate_doc_ids = {}
for sample_file in sample_files:
    for sample in load_records(sample_file):
        row = normalize(sample)
        doc_id = row["doc_id"]
        if doc_id is None:
            raise ValueError(f"sample without doc_id in {sample_file}: {row}")
        if doc_id in rows_by_doc_id:
            duplicate_doc_ids.setdefault(doc_id, 1)
            duplicate_doc_ids[doc_id] += 1
            continue
        rows_by_doc_id[doc_id] = row

rows = [rows_by_doc_id[doc_id] for doc_id in sorted(rows_by_doc_id)]
correct_rows = [row for row in rows if row["correct"]]
incorrect_rows = [row for row in rows if not row["correct"]]

correct_path = root / "gsm8k_correct.jsonl"
incorrect_path = root / "gsm8k_incorrect.jsonl"

with correct_path.open("w", encoding="utf-8") as f:
    for row in correct_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with incorrect_path.open("w", encoding="utf-8") as f:
    for row in incorrect_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[dump] sample files: {', '.join(str(p) for p in sample_files)}")
if duplicate_doc_ids:
    examples = ", ".join(str(doc_id) for doc_id in sorted(duplicate_doc_ids)[:10])
    print(
        f"[dump] skipped duplicate doc_id rows: "
        f"{sum(count - 1 for count in duplicate_doc_ids.values())} "
        f"across {len(duplicate_doc_ids)} docs"
        f"{' (examples: ' + examples + ')' if examples else ''}"
    )
print(f"[dump] total unique: {len(rows)}")
print(f"[dump] correct:   {len(correct_rows)} -> {correct_path}")
print(f"[dump] incorrect: {len(incorrect_rows)} -> {incorrect_path}")
PY