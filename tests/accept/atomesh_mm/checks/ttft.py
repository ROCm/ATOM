"""Median time to first token at 4 MMStar images per request and 32
concurrent streamed requests, against the deployment under test.

The PD deployment is the reference: a run on the pd topology records its
median, and a run on another topology passes when no request failed and its
median is no higher than the recorded PD one; the JSON line carries both
medians and their difference. Each request carries its own images, after a
warm-up round on images the measured requests do not use. LIMIT sets the
number of measured requests. Run by accept.sh in the client container.
"""

import json
import os
import statistics
import time
import urllib.request
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from mmstar import env, finish, reference, run_check

IMAGES = 4
CONCURRENCY = 32
REQUESTS = 256
WARMUP = CONCURRENCY
PROMPT = "Describe each image in one sentence."
MAX_TOKENS = 16
REFERENCE = "pd"


def mmstar_images():
    """MMStar's images, as the encoded bytes the dataset stores."""
    path = hf_hub_download("Lin-Chen/MMStar", "mmstar.parquet", repo_type="dataset")
    return pq.read_table(path, columns=["image"]).column("image").to_pylist()


def data_url(raw):
    mime = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
    return f"data:{mime};base64," + b64encode(raw).decode()


def ttft_ms(images):
    """Milliseconds from sending a streamed request with images to its first
    generated token (answer or reasoning). Reads the stream to its end."""
    content = [
        {"type": "image_url", "image_url": {"url": data_url(raw)}} for raw in images
    ]
    content.append({"type": "text", "text": PROMPT})
    body = {
        "model": env("SERVED_MODEL"),
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    request = urllib.request.Request(
        env("ENDPOINT") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    first = None
    sent = time.monotonic()
    with urllib.request.urlopen(request, timeout=14400) as response:
        for line in response:
            if first is not None or not line.startswith(b"data: {"):
                continue
            choices = json.loads(line[6:]).get("choices") or [{}]
            delta = choices[0].get("delta") or {}
            if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning")):
                first = time.monotonic()
    if first is None:
        raise RuntimeError("the stream carried no token")
    return (first - sent) * 1e3


def load(images, requests, offset):
    """TTFT in ms of each of requests requests, IMAGES images each from
    images[offset:], CONCURRENCY at a time; a failed request's entry is its
    error."""

    def one(n):
        start = offset + n * IMAGES
        try:
            return ttft_ms(images[start : start + IMAGES])
        except Exception as exc:  # noqa: BLE001 - a failed request is counted
            return f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(CONCURRENCY) as pool:
        return list(pool.map(one, range(requests)))


def main():
    ref = reference(REFERENCE) if env("ACCEPT_TOPOLOGY") != REFERENCE else None
    requests = int(os.environ.get("LIMIT") or REQUESTS)
    images = mmstar_images()
    if (requests + WARMUP) * IMAGES > len(images):
        raise ValueError(
            f"MMStar has {len(images)} images, too few for {requests} requests"
        )
    load(images, WARMUP, requests * IMAGES)
    results = load(images, requests, 0)
    ttfts = [r for r in results if isinstance(r, float)]
    errors = [r for r in results if isinstance(r, str)]
    median = statistics.median(ttfts) if ttfts else None
    result = {
        "images_per_request": IMAGES,
        "concurrency": CONCURRENCY,
        "requests": requests,
        "failed_requests": len(errors),
        "median_ttft_ms": median,
    }
    failures = []
    if errors:
        failures.append(f"{len(errors)} failed requests, the first: {errors[0]}")
    if ref:
        result["reference"] = {
            "topology": REFERENCE,
            "median_ttft_ms": ref["median_ttft_ms"],
            "run_dir": ref["run_dir"],
        }
        if median is not None:
            result["delta_ms"] = median - ref["median_ttft_ms"]
            if result["delta_ms"] > 0:
                failures.append(
                    f"median TTFT {median:.1f} ms is above {REFERENCE}'s "
                    f"{ref['median_ttft_ms']:.1f} ms"
                )
    finish(result, failures)


if __name__ == "__main__":
    run_check(main)
