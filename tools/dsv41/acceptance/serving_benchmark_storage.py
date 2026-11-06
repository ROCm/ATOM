"""Bound serving client result temporaries inside the admitted kernel tmpfs."""

import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import run_owner_serving as serving

RESULT_LIMIT = 100_000_000
ARCHIVE_LIMIT = 30_000_000


def digest(stream):
    sha = hashlib.sha256()
    while block := stream.read(1_000_000):
        sha.update(block)
    return sha.hexdigest()


class BoundedWriter:
    def __init__(self, stream):
        self.stream, self.size = stream, 0

    def write(self, data):
        if self.size + len(data) > ARCHIVE_LIMIT:
            raise ValueError("Client result archive exceeds its bounded reservation")
        self.size += len(data)
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def benchmark(
    label,
    mode,
    server,
    *,
    count=256,
    concurrency=256,
    input_length=8192,
    output_length=1024,
    seed=2026102518,
):
    serving.CONTROL.write_text(mode + "\n")
    before = serving.routes()
    result_dir = Path(os.environ["DSV41_CLIENT_RESULT_TMP"])
    result_dir.mkdir(parents=True, exist_ok=True)
    temporary = result_dir / (label + ".json")
    if temporary.exists():
        raise ValueError("Preserve named client result")
    argv = [
        sys.executable,
        "-m",
        "atom.benchmarks.benchmark_serving",
        "--model",
        serving.MODEL,
        "--served-model-name",
        serving.MODEL,
        "--backend",
        "vllm",
        "--base-url",
        serving.URL,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_length),
        "--random-output-len",
        str(output_length),
        "--random-range-ratio",
        "0.8",
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--trust-remote-code",
        "--use-chat-template",
        "--save-result",
        "--max-concurrency",
        str(concurrency),
        "--num-prompts",
        str(count),
        "--num-warmups",
        "0",
        "--seed",
        str(seed),
        "--result-dir",
        str(result_dir),
        "--result-filename",
        temporary.name,
    ]
    print(f"OWNER_SERVING_CLIENT start {label} mode={mode} C={concurrency}", flush=True)
    serving.child(argv, label, server)
    if temporary.stat().st_size > RESULT_LIMIT:
        raise ValueError("Result exceeds 100MB ceiling inside bounded tmpfs")
    with temporary.open("rb") as source:
        raw_sha = digest(source)
    result = json.loads(temporary.read_text())
    if count >= 2560:
        archive = serving.OUTPUT / (label + ".json.gz")
        with (
            archive.open("xb") as output,
            gzip.GzipFile(
                fileobj=BoundedWriter(output), mode="wb", compresslevel=6
            ) as target,
            temporary.open("rb") as source,
        ):
            while block := source.read(1_000_000):
                target.write(block)
        with gzip.open(archive, "rb") as source:
            if digest(source) != raw_sha:
                raise ValueError("Exact formal result archive SHA differs")
        receipt = {
            "archive": str(archive),
            "compressed_bytes": archive.stat().st_size,
            "status": "complete_lossless_formal_SHA_verified",
        }
    else:
        receipt = {
            "status": "unaccepted_warm_or_capacity_details_summarized",
            "scope": "All metric scalars and routes retained; request timing arrays/generated text from warmup or capacity screen removed. Full GSM8K samples and formal results retained separately.",
        }
    receipt.update(raw_bytes=temporary.stat().st_size, decoded_sha256=raw_sha)
    serving.save(serving.OUTPUT / (label + "_archive.json"), receipt)
    temporary.unlink()
    if (
        result["completed"] != count
        or any(result.get("errors", []))
        or result["total_output_tokens"] <= 0
    ):
        raise RuntimeError("Incomplete client-visible generation")
    after = serving.routes()
    delta = [
        {
            key: rank["counts"].get(key, 0) - old.get("counts", {}).get(key, 0)
            for key in rank["counts"]
        }
        for rank, old in zip(after, before, strict=True)
    ]
    summary = {
        key: result[key]
        for key in (
            "completed",
            "duration",
            "total_input_tokens",
            "total_output_tokens",
            "mean_tpot_ms",
            "output_throughput",
        )
    }
    summary.update(mode=mode, concurrency=concurrency, route_delta=delta)
    serving.save(serving.OUTPUT / (label + "_summary.json"), summary)
    print(
        f"OWNER_SERVING_CLIENT done {label} TPOT={summary['mean_tpot_ms']:.6f}ms throughput={summary['output_throughput']:.3f}tok/s",
        flush=True,
    )
    return summary
