#!/usr/bin/env python3
"""Render the per-concurrency client yaml for the Qwen3.5 agentic sweep.

usage: mkcfg.py <base_yaml> <dataset_jsonl> <model> <concurrency> <report_dir> <out_yaml>

Starts from mlcommons/endpoints examples/10_Agentic_Inference/qwen_agentic_benchmark.yaml
and makes the same deviations as the DSv4.1 recipe (sampling params / thinking flags
untouched): perf dataset only (no SWE-bench, no inline accuracy), 613 trajectories, salt +
tool delay on, stop issuing at the first user completion, 30 min perf-issue cap.
"""

import os
import sys

import yaml

base, dataset, model, conc, report_dir, out = sys.argv[1:7]
with open(base) as f:
    cfg = yaml.safe_load(f)

cfg["model_params"]["name"] = model
cfg["model_params"]["tokenizer_name"] = model

perf = next(d for d in cfg["datasets"] if d.get("type") == "performance")
perf["path"] = dataset
perf.pop("accuracy_config", None)
ai = perf["agentic_inference"]
# trajectories: at least the dataset's 613, and TRAJ_X (default 4) per concurrency slot. The client
# repeats conversations past 613 (<id>__repeat_N, each repeat with its own system-prompt salt), so
# at c >= 613 the 30-minute cap still ends the point; with 613, every trajectory starts at once and
# stop_issuing_on_first_user_complete ends it at the first completion (c2048: 4.4 min)
ai["num_trajectories_to_issue"] = max(
    613, int(os.environ.get("TRAJ_X", "4")) * int(conc)
)
# TURN_TIMEOUT (s): a turn still unanswered after it fails and its trajectory ends. Proxy points
# (STOP_FIRST=1) default to 900, which bounds the drain after the cap at high concurrency; full runs
# keep the benchmark's 14400
ai["turn_timeout_s"] = float(
    os.environ.get(
        "TURN_TIMEOUT", "900" if os.environ.get("STOP_FIRST", "1") == "1" else "14400"
    )
)
ai["enable_salt"] = True
ai["inject_tool_delay"] = True
# STOP_FIRST=0 / CAP_MIN=600: every trajectory to its end, the compliant 10-hour cap
ai["stop_issuing_on_first_user_complete"] = os.environ.get("STOP_FIRST", "1") == "1"
cfg["datasets"] = [perf]

cfg["settings"]["load_pattern"]["target_concurrency"] = int(conc)
cfg["settings"].setdefault("runtime", {})["max_issue_duration_ms"] = int(
    float(os.environ.get("CAP_MIN", "30")) * 60 * 1000
)
cfg["endpoint_config"]["endpoints"] = ["http://localhost:8000"]
cfg["report_dir"] = report_dir

with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(out)
