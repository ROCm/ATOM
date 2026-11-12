#!/usr/bin/env python3
"""Render mock-data previews of the 455-CI charts into ./examples.

The numbers are made up to show the layout: a weekly gain, two runs that
failed outright (Kimi-K3 10-31, MiniMax-M3 11-07), one concurrency that
failed on the newest image (Qwen-35B-A3B c64), a GSM8K drop on the newest
image (MiniMax-M3) and two baselines. Every chart says MOCK DATA.
"""

import csv
import datetime
import os
import random
import shutil
import tempfile

from ci455_data import ACCURACY_COLUMNS, PERF_COLUMNS
from make_charts import main as make_charts

HERE = os.path.dirname(os.path.abspath(__file__))
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
# model: (GPUs, {concurrency: (x, tok/s per GPU)} on the first image), where x
# is the E2E interactivity (endpoint) or the mean TPOT in ms (8k1k).
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
FAILED = {
    ("kimi-k3", "2026-10-31"): (4, 64),
    ("minimax-m3", "2026-11-07"): (4, 64),
    ("qwen-35b-a3b", "2026-11-14"): (64,),
}
GSM8K_DROP = {("minimax-m3", "2026-11-14"): -0.047}
BASELINES = {"kimi-k3": 0.95, "glm-5.2": 0.94}


def mock_rows():
    rng = random.Random(455)
    perf, acc = [], []
    for week, image in enumerate(IMAGES):
        tag = f"rocm/fw-bringup-mlperf:gfx1250-atom-ci455-{image.replace('-', '')}"
        for scenario, table in (("endpoint", ENDPOINT), ("8k1k", EIGHT_K)):
            shift = (
                0 if scenario == "endpoint" else 4
            )  # 8K/1K runs midweek on the same image
            test_date = str(
                datetime.date.fromisoformat(image) + datetime.timedelta(days=shift)
            )
            common = {
                "image_date": image,
                "test_date": test_date,
                "scenario": scenario,
                "image": tag,
                "atom_branch": "main",
                "aiter_branch": "main",
            }
            for model, (gpus, base) in table.items():
                gain = 1 + 0.025 * week + rng.uniform(-0.01, 0.01)
                speed = 1 + 0.012 * week + rng.uniform(-0.01, 0.01)
                failed = FAILED.get((model, image), ())
                for conc, (x, y) in base.items():
                    row = {
                        **common,
                        "model": model,
                        "gpus": str(gpus),
                        "concurrency": str(conc),
                    }
                    if conc in failed:
                        row.update(
                            status="fail",
                            error="RuntimeError: HIP error: out of memory (mock)",
                        )
                    elif scenario == "endpoint":
                        row.update(
                            status="ok",
                            tok_s_per_gpu=f"{y * gain:.2f}",
                            interactivity=f"{x * speed:.2f}",
                        )
                    else:
                        tpot = x / speed
                        row.update(
                            status="ok",
                            tok_s_per_gpu=f"{y * gain:.2f}",
                            tpot_mean_ms=f"{tpot:.3f}",
                            itl_p50_ms=f"{tpot * 0.97:.3f}",
                        )
                    perf.append(row)
                row = {**common, "model": model, "num_fewshot": "3"}
                if set(failed) == set(base):
                    row.update(
                        status="fail",
                        error="RuntimeError: HIP error: out of memory (mock)",
                    )
                else:
                    score = (
                        GSM8K[model]
                        + rng.uniform(-0.006, 0.006)
                        + GSM8K_DROP.get((model, image), 0)
                    )
                    row.update(
                        status="ok",
                        gsm8k=f"{score:.4f}",
                        gsm8k_strict=f"{score - 0.008:.4f}",
                    )
                acc.append(row)
    return perf, acc


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        w.writeheader()
        w.writerows(rows)


# The charts kept in ./examples: an endpoint chart with a failed image left
# out, one with a failed concurrency on the newest image, an 8K/1K chart and
# both accuracy charts. The other models' charts look the same.
SHOWCASE = (
    "perf-kimi-k3.png",
    "perf-qwen-35b-a3b.png",
    "perf-minimax-m3.png",
    "accuracy-endpoint.png",
    "accuracy-8k1k.png",
)


def main():
    perf, acc = mock_rows()
    examples = os.path.join(HERE, "examples")
    os.makedirs(examples, exist_ok=True)
    with tempfile.TemporaryDirectory() as data:
        write_csv(os.path.join(data, "perf.csv"), PERF_COLUMNS, perf)
        write_csv(os.path.join(data, "accuracy.csv"), ACCURACY_COLUMNS, acc)
        write_csv(
            os.path.join(data, "accuracy_baseline.csv"),
            ("model", "baseline"),
            [{"model": m, "baseline": str(b)} for m, b in BASELINES.items()],
        )
        out = os.path.join(data, "charts")
        make_charts(
            ["--image-date", IMAGES[-1], "--data-dir", data, "--out", out, "--mock"]
        )
        for name in SHOWCASE:
            shutil.copyfile(os.path.join(out, name), os.path.join(examples, name))
            print("kept", os.path.join(examples, name))


if __name__ == "__main__":
    main()
