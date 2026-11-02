#!/usr/bin/env python3
"""Render the per-concurrency client yaml for the Qwen3.5 agentic sweep.

usage: mkcfg.py <base_yaml> <dataset_jsonl> <model> <concurrency> <report_dir> <out_yaml>

Starts from mlcommons/endpoints examples/10_Agentic_Inference/qwen_agentic_benchmark.yaml
and makes the same deviations as the DSv4.1 recipe (sampling params / thinking flags
untouched): perf dataset only (no SWE-bench; inline accuracy with ACC_INLINE=1), a multiple of
613 trajectories, salt +
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
# ACC_INLINE=1 keeps the example's inline accuracy (agentic_inference_inline, scored from events.jsonl
# when the run finalizes), which an official submission requires on every point; SWE-bench stays out.
if os.environ.get("ACC_INLINE", "0") != "1":
    perf.pop("accuracy_config", None)
ai = perf["agentic_inference"]
# trajectories: at least the dataset's 613, and TRAJ_X (default 4) per concurrency slot. The client
# repeats conversations past 613 (<id>__repeat_N, each repeat with its own system-prompt salt), so
# at c >= 613 the 30-minute cap still ends the point; with 613, every trajectory starts at once and
# stop_issuing_on_first_user_complete ends it at the first completion (c2048: 4.4 min)
# An official run issues a positive multiple of the dataset's 613 (README, Dataset Size): NUM_TRAJ
# names the count outright, otherwise TRAJ_X * conc is rounded up to the next multiple.
DATASET_TRAJ = 613
if os.environ.get("NUM_TRAJ"):
    n_traj = int(os.environ["NUM_TRAJ"])
    if n_traj <= 0 or n_traj % DATASET_TRAJ:
        sys.exit(f"NUM_TRAJ={n_traj} is not a positive multiple of {DATASET_TRAJ}")
else:
    want = max(DATASET_TRAJ, int(os.environ.get("TRAJ_X", "4")) * int(conc))
    n_traj = -(-want // DATASET_TRAJ) * DATASET_TRAJ
ai["num_trajectories_to_issue"] = n_traj
# TURN_TIMEOUT (s): a turn still unanswered after it fails, and with it the rest of its trajectory.
# Default: the benchmark's own 14400, as the DSv4.1 30-minute points use. A short one bounds the
# drain after the cap but kills long turns at high concurrency (c2048 at 900 s: 22% of turns failed,
# at a 377 ms TPOT a 2,400-token turn already takes 15 min).
ai["turn_timeout_s"] = float(os.environ.get("TURN_TIMEOUT", "14400"))
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
