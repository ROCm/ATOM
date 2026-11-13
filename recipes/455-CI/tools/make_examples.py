#!/usr/bin/env python3
"""Build an example 455-CI data dir in ./examples and draw its charts and reports.

Made-up numbers in the real layout (perf.csv, accuracy.csv, runs/, reports/)
to show what the reports look like. Each week has two sessions on one image:
Saturday runs MLPerf Endpoint, Wednesday runs 8K/1K. In the newest Saturday
session (11-14) DeepSeek-V4.1-Flash gains 6%, Kimi-K3 recovers from a failed
11-07 and Qwen-35B-A3B fails at c64; in the Wednesday session after it
(11-18) GLM-5.2 c64 drops 4% and MiniMax-M3 loses 4.7 GSM8K points. Each
problem has a follow-up, one of each kind: a repro for Qwen, our verified PR
for GLM, someone else's open PR for MiniMax. The charts and reports say
EXAMPLE.
"""

import csv
import datetime
import json
import os
import random
import shutil

from ci455_data import ACCURACY_COLUMNS, FOLLOWUP_COLUMNS, PERF_COLUMNS
from make_charts import main as make_charts
from make_report import main as make_report

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "examples")
IMAGES = [
    "2026-09-26",
    "2026-10-03",
    "2026-10-10",
    "2026-10-17",
    "2026-10-24",
    "2026-10-31",
    "2026-11-07",
    "2026-11-14",
]
NEWEST = IMAGES[-1]
WEDNESDAY = str(datetime.date.fromisoformat(NEWEST) + datetime.timedelta(days=4))
# model: (GPUs, {concurrency: (x, tok/s per GPU)}), where x is the E2E
# interactivity (endpoint) or the mean TPOT in ms (8k1k).
ENDPOINT = {
    "deepseek-r1": (4, {4: (150.0, 160.0), 64: (58.0, 1450.0)}),
    "deepseek-v4.1-flash": (4, {4: (210.0, 230.0), 64: (95.0, 2600.0)}),
    "kimi-k3": (8, {4: (105.0, 55.0), 64: (45.0, 330.0)}),
    "qwen-35b-a3b": (1, {4: (260.0, 980.0), 64: (120.0, 6200.0)}),
}
EIGHT_K = {
    "deepseek-v4-pro": (4, {4: (14.0, 2100.0), 64: (38.0, 9800.0)}),
    "glm-5.2": (4, {4: (11.0, 2600.0), 64: (30.0, 12500.0)}),
    "minimax-m3": (4, {4: (9.0, 3100.0), 64: (24.0, 15800.0)}),
    "qwen-3.8-flash": (1, {4: (6.0, 4800.0), 64: (16.0, 26000.0)}),
}
# Parallelism as recorded from the recipe (Kimi-K3, GLM-5.2 and MiniMax-M3
# as in 455-CI-recipe.md; the others made up).
PARALLEL = {
    "deepseek-r1": "tp4",
    "deepseek-v4.1-flash": "dp4ep4",
    "kimi-k3": "tp8dcp8",
    "qwen-35b-a3b": "tp1",
    "deepseek-v4-pro": "dp4ep4",
    "glm-5.2": "tp4",
    "minimax-m3": "tp4",
    "qwen-3.8-flash": "tp1",
}
# Placeholder owners: underscores are not allowed in GitHub logins, so these
# can never tag a real person.
OWNERS = {
    "deepseek-r1": "@r1_owner",
    "deepseek-v4.1-flash": "@v41_flash_owner",
    "kimi-k3": "@k3_owner",
    "qwen-35b-a3b": "@qwen35b_owner",
    "deepseek-v4-pro": "@v4_pro_owner",
    "glm-5.2": "@glm52_owner",
    "minimax-m3": "@m3_owner",
    "qwen-3.8-flash": "@qwen38_owner",
}
GSM8K = {
    "deepseek-r1": 0.952,
    "deepseek-v4.1-flash": 0.941,
    "kimi-k3": 0.958,
    "qwen-35b-a3b": 0.902,
    "deepseek-v4-pro": 0.955,
    "glm-5.2": 0.948,
    "minimax-m3": 0.931,
    "qwen-3.8-flash": 0.893,
}
FAILED = {("kimi-k3", "2026-11-07"): {4, 64, "gsm8k"}, ("qwen-35b-a3b", NEWEST): {64}}
BASELINES = {"kimi-k3": 0.95, "glm-5.2": 0.94}
# Next steps for the example problems (455-CI-guide.md §3.7). The PR numbers
# are far beyond any real PR, so the links point at nothing.
FOLLOWUPS = [
    {
        "test_date": NEWEST,
        "model": "qwen-35b-a3b",
        "action": "repro",
        "link": f"runs/{NEWEST.replace('-', '')}/qwen-35b-a3b/repro.md",
        "note": "(mock) OOM while capturing CUDA graphs, at c64 only",
    },
    {
        "test_date": WEDNESDAY,
        "model": "glm-5.2",
        "action": "our_pr",
        "link": "https://github.com/ROCm/ATOM/pull/99901",
        "note": "(mock) verified on 455: c64 back to 12,830 tok/s/GPU",
    },
    {
        "test_date": WEDNESDAY,
        "model": "minimax-m3",
        "action": "open_pr",
        "link": "https://github.com/ROCm/aiter/pull/99902",
        "note": "(mock) the open aiter PR fixes the MoE scale rounding",
    },
]
REPRO = """\
> ⚠️ **EXAMPLE**: a made-up repro that only illustrates the format.

# Qwen-35B-A3B: c64 fails with HIP out of memory (455-CI 2026-11-14)

## Environment

- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114` (digest `sha256:0000mock`)
- ATOM `main@000000000`, aiter `main@111111111`; last good session 2026-11-07
- Machine: e15-06, 1 x MI455X; weights: `<HF repo> @ <revision>` (mock)

## Steps

1. Start the test container as in 455-CI-recipe.md.
2. Start the server:

   ```bash
   python3 -m atom.entrypoints.openai_server --model /models/<Qwen-35B-A3B> --kv_cache_dtype fp8    # (mock)
   ```

3. Run the benchmark at concurrency 64, as in `bench_cmd.sh` of this run with `CONC=64`.

## Expected vs actual

- Expected: c64 completes, about 6,345 tok/s/GPU as on 2026-11-07.
- Actual: the server exits while capturing CUDA graphs with
  `RuntimeError: HIP error: out of memory (mock)`. 2 of 2 runs; c4 passes.

## Logs

[error.log](error.log), the core excerpt of the server log.

## What we checked

- Open and recently merged PRs in ROCm/ATOM and ROCm/aiter for "out of memory",
  "cudagraph" and "Qwen": none about this.
- Not tried yet: a lower `--gpu-memory-utilization`.
"""
ERROR = "RuntimeError: HIP error: out of memory (mock)"
ERROR_LOG = """\
(mock) INFO  [model_runner] capturing CUDA graphs for batch sizes 1..64
(mock) Traceback (most recent call last):
(mock)   File "/app/ATOM/atom/model_engine/model_runner.py", line 1234, in capture_cudagraph
(mock)     self.run_model(dummy_batch)
(mock)   File "/app/ATOM/atom/model_engine/model_runner.py", line 987, in run_model
(mock)     hidden = self.model(input_ids, positions)
(mock) RuntimeError: HIP error: out of memory (mock)
----- last 30 lines -----
(mock) ERROR [engine_core] EngineCore_DP0 died, shutting down the server
"""


def scale(model, conc, image):
    """Throughput multiplier of a point on an image."""
    week = IMAGES.index(image)
    m = 1 + 0.004 * week
    if model == "deepseek-v4.1-flash" and image == NEWEST:
        m *= 1.06
    if model == "glm-5.2" and conc == 64 and image == NEWEST:
        m *= 0.96
    return m


def mock_rows():
    rng = random.Random(455)
    perf, acc = [], []
    for image in IMAGES:
        tag = f"rocm/fw-bringup-mlperf:gfx1250-atom-ci455-{image.replace('-', '')}"
        for scenario, table in (("endpoint", ENDPOINT), ("8k1k", EIGHT_K)):
            shift = (
                0 if scenario == "endpoint" else 4
            )  # 8K/1K runs midweek on the same image
            common = {
                "image_date": image,
                "scenario": scenario,
                "image": tag,
                "test_date": str(
                    datetime.date.fromisoformat(image) + datetime.timedelta(days=shift)
                ),
                "atom_branch": "main",
                "aiter_branch": "main",
            }
            for model, (gpus, base) in table.items():
                failed = FAILED.get((model, image), set())
                for conc, (x, y) in base.items():
                    row = {
                        **common,
                        "model": model,
                        "gpus": str(gpus),
                        "parallel": PARALLEL[model],
                        "concurrency": str(conc),
                    }
                    if conc in failed:
                        row.update(status="fail", error=ERROR)
                    else:
                        tput = (
                            y
                            * scale(model, conc, image)
                            * (1 + rng.uniform(-0.004, 0.004))
                        )
                        speed = 1 + rng.uniform(-0.01, 0.01)
                        row.update(status="ok", tok_s_per_gpu=f"{tput:.2f}")
                        if scenario == "endpoint":
                            row["interactivity"] = f"{x * speed:.2f}"
                        else:
                            row.update(
                                tpot_mean_ms=f"{x / speed:.3f}",
                                itl_p50_ms=f"{x / speed * 0.97:.3f}",
                            )
                    perf.append(row)
                row = {**common, "model": model, "num_fewshot": "3"}
                if "gsm8k" in failed:
                    row.update(status="fail", error=ERROR)
                else:
                    s = GSM8K[model] + rng.uniform(-0.004, 0.004)
                    if (model, image) == ("minimax-m3", NEWEST):
                        s -= 0.047
                    row.update(
                        status="ok", gsm8k=f"{s:.4f}", gsm8k_strict=f"{s - 0.008:.4f}"
                    )
                acc.append(row)
    return perf, acc


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        w.writeheader()
        w.writerows(rows)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    run = os.path.join(OUT, "runs", NEWEST.replace("-", ""))
    os.makedirs(os.path.join(run, "qwen-35b-a3b"))
    perf, acc = mock_rows()
    write_csv(os.path.join(OUT, "perf.csv"), PERF_COLUMNS, perf)
    write_csv(os.path.join(OUT, "accuracy.csv"), ACCURACY_COLUMNS, acc)
    write_csv(
        os.path.join(OUT, "owners.csv"),
        ("model", "owner"),
        [{"model": m, "owner": o} for m, o in OWNERS.items()],
    )
    write_csv(
        os.path.join(OUT, "accuracy_baseline.csv"),
        ("model", "baseline"),
        [{"model": m, "baseline": str(b)} for m, b in BASELINES.items()],
    )
    with open(
        os.path.join(run, "qwen-35b-a3b", "error.log"), "w", encoding="utf-8"
    ) as f:
        f.write(ERROR_LOG)
    with open(os.path.join(OUT, FOLLOWUPS[0]["link"]), "w", encoding="utf-8") as f:
        f.write(REPRO)
    write_csv(os.path.join(OUT, "followups.csv"), FOLLOWUP_COLUMNS, FOLLOWUPS)
    with open(os.path.join(run, "image.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": "rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114",
                "digest": "sha256:0000mock",
                "base_image": "rocm/fw-bringup-mlperf:gfx1250-atom-20261113",
                "atom": {
                    "commit": "0" * 40,
                    "date": "2026-11-14",
                    "title": "(mock) ATOM main head",
                },
                "aiter": {
                    "commit": "1" * 40,
                    "date": "2026-11-14",
                    "title": "(mock) aiter main head",
                },
                "dependency_changes": [],
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    for day in (NEWEST, WEDNESDAY):  # the two sessions on the newest image
        make_charts(["--date", day, "--data-dir", OUT, "--mock"])
        make_report(["--date", day, "--data-dir", OUT, "--mock"])


if __name__ == "__main__":
    main()
