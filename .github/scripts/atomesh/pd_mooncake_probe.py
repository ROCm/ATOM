#!/usr/bin/env python3
"""Observe Store reuse after GPU cache reset before the C16 replay."""

import argparse
import json
import time
from pathlib import Path

import requests
from prometheus_client.parser import text_string_to_metric_families


def counter(text, name, **labels):
    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name
        and all(sample.labels.get(key) == value for key, value in labels.items())
    )


def main():
    parser = argparse.ArgumentParser()
    for option in ("prefill", "decode", "router", "model", "output"):
        parser.add_argument(f"--{option}", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    result = {"cases": [], "store_reuse_observed": False, "error": None}

    def request(url, path, body):
        response = session.post(url + path, json=body, timeout=1800)
        response.raise_for_status()
        return response.json() if response.content else None

    def reset():
        for url in (args.prefill, args.decode):
            for attempt in range(60):
                response = session.post(url + "/reset_prefix_cache", timeout=10)
                response.raise_for_status()
                if response.json().get("success") is True:
                    break
                time.sleep(1)
            else:
                raise RuntimeError("GPU prefix cache reset was refused")

    def metrics(label):
        texts = {}
        for role in ("prefill", "decode"):
            response = session.get(getattr(args, role) + "/metrics", timeout=30)
            response.raise_for_status()
            texts[role] = response.text
            (out / f"{label}-{role}.metrics").write_text(response.text)
        return texts

    try:
        tokens = request(
            args.prefill,
            "/tokenize",
            {
                "model": args.model,
                "prompt": "Read this conversation history and continue. Mooncake cache test. ",
            },
        )["tokens"]
        for length in (7449, 98305):
            reset()
            case_prefix = request(
                args.prefill,
                "/tokenize",
                {"model": args.model, "prompt": f"Independent cache case {length}. "},
            )["tokens"]
            body = {
                "model": args.model,
                "prompt": (case_prefix + tokens * (length // len(tokens) + 1))[:length],
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            }
            cold = request(args.router, "/v1/completions", body)
            (out / f"{length}-cold.json").write_text(json.dumps(cold, indent=2))
            # Store save and the metric logger are asynchronous.
            time.sleep(15)
            reset()
            before = metrics(f"{length}-before-reuse")
            warm = request(args.router, "/v1/completions", body)
            (out / f"{length}-reuse.json").write_text(json.dumps(warm, indent=2))
            time.sleep(15)
            after = metrics(f"{length}-after-reuse")
            name = "vllm:mooncake_store_operation_bytes_total"
            loaded = counter(
                after["prefill"], name, operation="load_get", status="ok"
            ) - counter(before["prefill"], name, operation="load_get", status="ok")
            external = counter(
                after["prefill"], "vllm:external_prefix_cache_hits_total"
            ) - counter(before["prefill"], "vllm:external_prefix_cache_hits_total")
            row = {
                "prompt_tokens": length,
                "store_load_bytes": loaded,
                "prefill_external_hit_tokens": external,
                "same_first_token": cold["choices"][0]["text"]
                == warm["choices"][0]["text"],
            }
            result["cases"].append(row)
            result["store_reuse_observed"] |= loaded > 0 and external > 0
            print(json.dumps(row), flush=True)
            if not row["same_first_token"]:
                raise RuntimeError("Cold and CPU-cache reuse first-token outputs differ")
        reset()
        if not result["store_reuse_observed"]:
            raise RuntimeError("No CPU Store reuse after GPU reset; C16 gate failed")
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
