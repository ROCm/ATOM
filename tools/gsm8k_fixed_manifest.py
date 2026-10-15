"""Offline GSM8K manifest preparation and explicit, one-pass completions replay.

prepare NEVER uses a model/tokenizer, HTTP, HF download, or GPU. Only run sends
HTTP. Run it once per arm/pass, with the SAME prepared manifest and a NEW output
directory. PASS means measurement validity, never model accuracy/equivalence.
No model lifecycle management, retries, chat templating, or implicit warm-up.

Use --dataset-cache with an existing official GSM8K Arrow cache. No cache path,
model location, container, or server URL is assumed. A moved manifest can use
run --dataset-cache at its new location; saved dataset hashes still must match.
Client preparation/scoring was validated with lm_eval 0.4.13. Evaluation
packages are optional for importing this tool or running its transport tests.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
SCHEMA = "gsm8k-fixed-completions-v1"
MAX_TOKENS = 4096
# Transport safety limit, independent of the frozen generation budget.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
SOURCE_FILES = [
    "tasks/gsm8k/gsm8k.yaml",
    "api/task.py",
    "api/instance.py",
    "api/samplers.py",
    "api/filter.py",
    "api/metrics.py",
    "api/registry.py",
    "filters/__init__.py",
    "filters/extraction.py",
    "filters/selection.py",
    "prompts/__init__.py",
]
DEPENDENCIES = [
    "lm_eval",
    "datasets",
    "numpy",
    "torch",
    "transformers",
    "Jinja2",
    "PyYAML",
    "aiohttp",
    "prometheus_client",
    "pyarrow",
]
METRIC_NAMES = {
    "prompt_tokens": "vllm:prompt_tokens_total",
    "local_hits": "vllm:prefix_cache_hits_total",
    "external_hits": "vllm:external_prefix_cache_hits_total",
    "local_queries": "vllm:prefix_cache_queries_total",
    "external_queries": "vllm:external_prefix_cache_queries_total",
    "local_supplied_tokens": "vllm:prompt_tokens_by_source_total",
    "external_supplied_tokens": "vllm:prompt_tokens_by_source_total",
    "computed_tokens": "vllm:prompt_tokens_by_source_total",
}
SOURCE_LABELS = {
    "local_supplied_tokens": "local_cache_hit",
    "external_supplied_tokens": "external_kv_transfer",
    "computed_tokens": "local_compute",
}


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def utc_now():
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def log(message):
    print(f"{utc_now()} {message}", flush=True)


@contextlib.contextmanager
def offline_cpu():
    """Process-local guards, not changes to the container/install environment."""
    env = {
        "HIP_VISIBLE_DEVICES": "",
        "CUDA_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_EVALUATE_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    audit = {"network_attempts": 0, "gpu_init_attempts": 0}

    def no_network(*args, **kwargs):
        audit["network_attempts"] += 1
        raise RuntimeError(
            "Offline CPU phase attempted network; no download is authorized"
        )

    def no_gpu(*args, **kwargs):
        audit["gpu_init_attempts"] += 1
        raise RuntimeError("GPU initialization is forbidden in this harness")

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, env))
        for method in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, method, no_network))
        stack.enter_context(patch("socket.create_connection", no_network))
        import torch

        if torch.cuda.is_initialized():
            raise RuntimeError("GPU was already initialized before CPU phase")
        stack.enter_context(patch.object(torch.cuda, "_lazy_init", no_gpu))
        stack.enter_context(patch.object(torch.cuda, "init", no_gpu))
        stack.enter_context(
            patch.object(torch.cuda, "is_available", return_value=False)
        )
        stack.enter_context(patch.object(torch.cuda, "device_count", return_value=0))
        if hasattr(torch._C, "_cuda_init"):
            stack.enter_context(patch.object(torch._C, "_cuda_init", no_gpu))
        yield audit
        audit["cuda_is_initialized_after"] = torch.cuda.is_initialized()
        if (
            audit["cuda_is_initialized_after"]
            or audit["gpu_init_attempts"]
            or audit["network_attempts"]
        ):
            raise RuntimeError(f"CPU isolation violation: {audit}")


def installed_contract():
    import lm_eval
    import yaml

    root = Path(lm_eval.__file__).parent
    yaml_path = root / "tasks/gsm8k/gsm8k.yaml"
    cfg = yaml.safe_load(yaml_path.read_text())
    if (
        cfg["task"],
        cfg["dataset_path"],
        cfg["dataset_name"],
        cfg["test_split"],
        cfg["num_fewshot"],
    ) != ("gsm8k", "openai/gsm8k", "main", "test", 5):
        raise ValueError("Installed task is not the expected official GSM8K test/5shot")
    if {f["name"] for f in cfg["filter_list"]} != {"strict-match", "flexible-extract"}:
        raise ValueError("Unexpected official filters")
    # Also fingerprint rendering helpers without guessing their current filenames.
    files = set(SOURCE_FILES)
    for subdir in ("prompts", "api/samplers"):
        files.update(str(p.relative_to(root)) for p in (root / subdir).rglob("*.py"))
    hashes = {f: sha_file(root / f) for f in sorted(files) if (root / f).is_file()}
    return cfg, {
        "versions": {n: importlib.metadata.version(n) for n in DEPENDENCIES},
        "lm_eval_root": str(root),
        "lm_eval_source_sha256": hashes,
        "task_config_sha256": sha_bytes(dumps(cfg).encode()),
    }


def make_task(cache_dir, cfg):
    """Only dataset transport is replaced. Rendering/filter/scoring stay upstream."""
    import datasets
    from lm_eval.api.task import ConfigurableTask

    cache_dir = Path(cache_dir)
    for s in ("train", "test"):
        if not (cache_dir / f"gsm8k-{s}.arrow").is_file():
            raise FileNotFoundError(
                f"Cached GSM8K {s} unavailable at {cache_dir}; "
                "download would be required; ask the user first"
            )

    class ReadOnlyCachedGSM8K(ConfigurableTask):
        def download(self, *args, **kwargs):
            self.dataset = datasets.DatasetDict(
                {
                    s: datasets.Dataset.from_file(str(cache_dir / f"gsm8k-{s}.arrow"))
                    for s in ("train", "test")
                }
            )

    task = ReadOnlyCachedGSM8K(config=copy.deepcopy(cfg))
    if len(task.dataset["train"]) != 7473 or len(task.dataset["test"]) != 1319:
        raise ValueError("Cached dataset cardinality is not official GSM8K 7473/1319")
    return task


def describe_lengths(values):
    values = sorted(values)
    return {
        "min": values[0],
        "p50": values[math.ceil(len(values) * 0.5) - 1],
        "p90": values[math.ceil(len(values) * 0.9) - 1],
        "p99": values[math.ceil(len(values) * 0.99) - 1],
        "max": values[-1],
        "mean": statistics.mean(values),
    }


def score_records(task, rows, records):
    """Official contract: Instance.resps=[completion str], NEVER a bare str/dict."""
    from lm_eval.api.instance import Instance

    instances = []
    by_id = {r["doc_id"]: r for r in rows}
    for r in records:
        if r.get("error") is not None:
            continue
        row = by_id[r["doc_id"]]
        inst = Instance(
            request_type="generate_until",
            doc=row["doc"],
            arguments=(row["prompt"], row["lm_eval_generation_kwargs"]),
            idx=0,
            metadata=("gsm8k", row["doc_id"], 1),
        )
        inst.resps = [r["response"]["choices"][0]["text"]]
        instances.append(inst)
    task._instances = instances
    if not instances:
        return [], {"scored": 0, "exact_match": {}}
    task.apply_filters()
    items = []
    collected = {}
    for inst in instances:
        scores = {}
        for name, value in inst.filtered_resps.items():
            if not isinstance(value, str):
                raise TypeError("Official filter output must be str for generate_until")
            # Do not reproduce metric regexes or score with custom numeric extraction.
            score = float(task.process_results(inst.doc, [value])["exact_match"])
            scores[name] = score
            collected.setdefault(name, []).append(score)
        items.append(
            {
                "doc_id": inst.doc_id,
                "filtered_resps": inst.filtered_resps,
                "exact_match": scores,
            }
        )
    aggregate = task.aggregation()["exact_match"]
    return items, {
        "scored": len(items),
        "exact_match": {k: float(aggregate(v)) for k, v in collected.items()},
    }


def prepare(args):
    # Resolve only the input location; rendered prompts and request construction
    # deliberately stay identical to the fixed experimental client.
    args.dataset_cache = str(Path(args.dataset_cache).expanduser().resolve())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    with offline_cpu() as audit:
        cfg, contract = installed_contract()
        task = make_task(args.dataset_cache, cfg)
        task.set_fewshot_seed(0)
        # Capture actual selected exemplars while leaving upstream sampler intact.
        index = {dumps(doc): i for i, doc in enumerate(task.dataset["train"])}
        sampled = []
        original_sample = task.sampler.sample

        def sample(*a, **kw):
            docs = original_sample(*a, **kw)
            sampled.append([index[dumps(d)] for d in docs])
            return docs

        task.sampler.sample = sample
        task.build_all_requests(
            limit=None,
            rank=0,
            world_size=1,
            cache_requests=False,
            apply_chat_template=False,
            tokenizer_name="none",
        )
        rows = []
        for inst, examples in zip(task.instances, sampled, strict=True):
            prompt, gen = inst.args
            if not isinstance(prompt, str) or len(examples) != 5:
                raise TypeError(
                    "Expected upstream raw prompt and five training exemplars"
                )
            request = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "seed": 0,
                "stop": gen["until"],
                "n": 1,
                "stream": False,
                "echo": False,
            }
            rows.append(
                {
                    "schema": SCHEMA,
                    "task": "gsm8k",
                    "split": "test",
                    "doc_id": inst.doc_id,
                    "doc": inst.doc,
                    "gold": task.doc_to_target(inst.doc),
                    "prompt": prompt,
                    "prompt_sha256": sha_bytes(prompt.encode()),
                    "prompt_characters": len(prompt),
                    "prompt_utf8_bytes": len(prompt.encode()),
                    "fewshot_train_doc_ids": examples,
                    "fewshot_seed": 0,
                    "request": request,
                    "lm_eval_generation_kwargs": gen,
                }
            )
        if [r["doc_id"] for r in rows] != list(range(1319)):
            raise ValueError("Expected exactly GSM8K test IDs 0..1318")
        manifest = out / "manifest.jsonl"
        manifest.write_text("".join(dumps(r) + "\n" for r in rows), encoding="utf-8")
        # Full-dataset official gold-response positive control, no generated responses.
        oracle = [
            {
                "doc_id": r["doc_id"],
                "error": None,
                "response": {"choices": [{"text": r["gold"]}]},
            }
            for r in rows
        ]
        oracle_items, oracle_score = score_records(task, rows, oracle)
        if any(v != 1.0 for v in oracle_score["exact_match"].values()):
            raise ValueError(
                f"Official all-gold positive control failed: {oracle_score}"
            )
        lengths = {
            "characters": describe_lengths([len(r["prompt"]) for r in rows]),
            "whitespace_words": describe_lengths(
                [len(r["prompt"].split()) for r in rows]
            ),
            "utf8_bytes": describe_lengths([len(r["prompt"].encode()) for r in rows]),
        }
        lengths["context_risk"] = {
            "max_model_len": args.max_model_len,
            "max_completion_tokens": MAX_TOKENS,
            "tokenizer_backend": "none",
            "actual_token_lengths": None,
            "utf8_bytes_plus_output_exceeds_context_count": sum(
                len(r["prompt"].encode()) + MAX_TOKENS > args.max_model_len
                for r in rows
            ),
            "maximum_utf8_bytes_plus_output": max(
                len(r["prompt"].encode()) for r in rows
            )
            + MAX_TOKENS,
            "interpretation": "Bytes/characters/words are not model tokens. UTF-8 bytes are only a "
            "conservative proxy for byte-level tokenizers, NOT a proved bound for "
            "this server. No local tokenizer loaded; usage.prompt_tokens and "
            "HTTP context errors must be checked in run.",
        }
        cache = Path(args.dataset_cache)
        dataset = {
            "path": "openai/gsm8k",
            "name": "main",
            "train_count": 7473,
            "test_count": 1319,
            "transport": "cached Arrow read-only, no load_dataset",
            "cache_dir": str(cache),
            "files_sha256": {
                p.name: sha_file(p)
                for p in sorted(cache.iterdir())
                if p.name
                in ("gsm8k-train.arrow", "gsm8k-test.arrow", "dataset_info.json")
            },
            "fingerprints": {s: d._fingerprint for s, d in task.dataset.items()},
        }
        metadata = {
            "schema": SCHEMA,
            "created_utc": utc_now(),
            "count": len(rows),
            "manifest_sha256": sha_file(manifest),
            "model": args.model,
            "max_model_len": args.max_model_len,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "request_seed": 0,
            "fewshot_seed": 0,
            "num_fewshot": 5,
            "tokenizer_backend": "none",
            "apply_chat_template": False,
            "dataset": dataset,
            "contract": contract,
            "task_config": cfg,
            "lengths": lengths,
            "oracle_score": oracle_score,
            "filter_contract": {
                "input": "Instance.resps = [completion_text] (list[str], one repeat)",
                "output": "Instance.filtered_resps[filter_name] is str",
                "score": "task.process_results(doc, [filtered_str]); task.aggregation()['exact_match']",
                "gold_samples": [
                    {
                        "doc_id": r["doc_id"],
                        "gold": r["gold"],
                        "official_oracle": oracle_items[i],
                    }
                    for i, r in enumerate(rows[:3])
                ],
            },
            "python": sys.version,
            "harness_sha256": sha_file(__file__),
            "baseline_warning": "fewshot_seed=0; concurrency=16; max_tokens=4096. Do not inherit "
            "scores/noise floors from old CI concurrency=32/default-budget runs.",
        }
    metadata["cpu_isolation"] = audit
    write_json(out / "manifest.meta.json", metadata)
    write_json(
        out / "dependencies.json",
        {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
    )
    write_json(
        out / "prepare_summary.json",
        {
            k: metadata[k]
            for k in (
                "count",
                "manifest_sha256",
                "dataset",
                "lengths",
                "oracle_score",
                "filter_contract",
                "cpu_isolation",
            )
        },
    )
    log(
        f"PREPARED count={len(rows)} manifest_sha256={metadata['manifest_sha256']} output={out}"
    )
    return 0


def load_bundle(manifest, dataset_cache=None):
    manifest = Path(manifest)
    meta = json.loads(manifest.with_name("manifest.meta.json").read_text())
    raw = manifest.read_bytes()
    if sha_bytes(raw) != meta["manifest_sha256"]:
        raise ValueError("Manifest SHA256 mismatch; replay forbidden")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if (
        meta["schema"] != SCHEMA
        or len(rows) != 1319
        or [r["doc_id"] for r in rows] != list(range(1319))
    ):
        raise ValueError(
            "This run command requires all 1319 ordered official test rows"
        )
    cfg, current = installed_contract()
    for key in ("versions", "lm_eval_source_sha256", "task_config_sha256"):
        if current[key] != meta["contract"][key]:
            raise ValueError(f"Installed evaluation dependency/contract changed: {key}")
    if cfg != meta["task_config"]:
        raise ValueError("Saved task config mismatch")
    cache = Path(dataset_cache or meta["dataset"]["cache_dir"]).expanduser().resolve()
    for name, digest in meta["dataset"]["files_sha256"].items():
        if sha_file(cache / name) != digest:
            raise ValueError(f"Cached dataset changed: {name}")
    task = make_task(cache, cfg)
    for r in rows:
        req = r["request"]
        if (
            r["schema"] != SCHEMA
            or r["doc"] != task.dataset["test"][r["doc_id"]]
            or r["gold"] != task.doc_to_target(r["doc"])
            or req["prompt"] != r["prompt"]
            or sha_bytes(r["prompt"].encode()) != r["prompt_sha256"]
            or req["model"] != meta["model"]
            or req["max_tokens"] != MAX_TOKENS
            or req["temperature"] != 0
            or req["seed"] != 0
            or req["stop"] != cfg["generation_kwargs"]["until"]
            or req["stream"] is not False
            or req["echo"] is not False
            or req["n"] != 1
        ):
            raise ValueError(f"Manifest contract mismatch in doc {r['doc_id']}")
    return rows, meta, task


def metric_series(text, name, labels):
    from prometheus_client.parser import text_string_to_metric_families

    found = {}
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            if s.name != name or any(s.labels.get(k) != v for k, v in labels.items()):
                continue
            key = tuple(sorted(s.labels.items()))
            if key in found:
                raise ValueError(f"Duplicate metric series: {name} {key}")
            if not math.isfinite(s.value) or s.value < 0:
                raise ValueError(f"Invalid counter: {name} {key}={s.value}")
            found[key] = s.value
    return found


def metrics_delta(before, after, model_name):
    result = {}
    labels = {"model_name": model_name}
    for key, name in METRIC_NAMES.items():
        select = {
            **labels,
            **({"source": SOURCE_LABELS[key]} if key in SOURCE_LABELS else {}),
        }
        try:
            b = metric_series(before, name, select)
            a = metric_series(after, name, select)
            status = "ok"
            if not b or not a:
                status = (
                    "missing_before_and_after"
                    if not b and not a
                    else "missing_before" if not b else "missing_after"
                )
            elif b.keys() != a.keys():
                status = "label_series_changed"
            elif any(a[k] < b[k] for k in b):
                status = "counter_reset"
            result[key] = {
                "name": name,
                "selector": select,
                "status": status,
                "before_sum": sum(b.values()) if b else None,
                "after_sum": sum(a.values()) if a else None,
                "delta": sum(a[k] - b[k] for k in b) if status == "ok" else None,
                "before_series": [
                    {"labels": dict(k), "value": v} for k, v in b.items()
                ],
                "after_series": [{"labels": dict(k), "value": v} for k, v in a.items()],
            }
        except ValueError as e:
            result[key] = {
                "name": name,
                "selector": select,
                "status": "parse_error",
                "delta": None,
                "error": repr(e),
            }
    return result


def validate_response(record):
    r = record["response"]
    if not isinstance(r, dict) or r.get("error"):
        raise ValueError("JSON is not a successful completion response")
    choices = r.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one completion choice")
    c = choices[0]
    if not isinstance(c.get("text"), str) or c.get("finish_reason") not in (
        "stop",
        "length",
    ):
        raise ValueError("Missing text or unsupported finish_reason")
    usage = r.get("usage")
    if not isinstance(usage, dict):
        raise TypeError("Response usage is required")
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage.get(k)) is not int or usage[k] < 0:
            raise ValueError(f"Invalid/missing usage.{k}")
    if (
        usage["prompt_tokens"] < 1
        or usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]
    ):
        raise ValueError("Inconsistent usage counts")
    if usage["completion_tokens"] > record["request"]["max_tokens"]:
        raise ValueError("Server exceeded manifest completion budget")
    record["usage"] = usage
    record["finish_reason"] = c["finish_reason"]


async def fetch_metrics(session, url, out, which, timeout):
    import aiohttp

    info = {"url": url, "started_utc": utc_now(), "status": None, "error": None}
    body = b""
    start = time.monotonic()
    try:
        async with session.get(
            url, allow_redirects=False, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            info["status"] = resp.status
            info["headers"] = list(resp.headers.items())
            body = await resp.read()
            if resp.status != 200:
                raise ValueError(f"Metrics HTTP {resp.status}")
        body.decode("utf-8", errors="strict")
    except (aiohttp.ClientError, TimeoutError, OSError, ValueError) as e:
        info["error"] = {"type": type(e).__name__, "message": str(e)}
    finally:
        info["latency_seconds"] = time.monotonic() - start
        info["finished_utc"] = utc_now()
        (out / f"metrics.{which}.txt").write_bytes(body)
        write_json(out / f"metrics.{which}.http.json", info)
    return body.decode("utf-8", errors="replace"), info


async def request_batch(session, rows, args, out, *, clock=time.monotonic):
    import aiohttp

    sem = asyncio.Semaphore(args.concurrency)
    records = []
    started = last_completion = clock()
    deadline = started + args.pass_timeout
    failure = None
    f = (out / "responses.jsonl").open("x", encoding="utf-8")

    async def one(row):
        nonlocal last_completion
        rec = {
            "doc_id": row["doc_id"],
            "prompt_sha256": row["prompt_sha256"],
            "request": row["request"],
            "attempts": 0,
            "sent": False,
            "queued_utc": utc_now(),
            "started_utc": None,
            "http_status": None,
            "response": None,
            "response_raw": None,
            "response_headers": None,
            "usage": None,
            "finish_reason": None,
            "latency_seconds": None,
            "error": None,
        }
        start = None
        try:
            async with sem:
                rec["started_utc"] = utc_now()
                start = clock()
                rec["sent"] = True
                rec["attempts"] = 1
                async with session.post(
                    args.endpoint,
                    json=row["request"],
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=args.http_timeout),
                ) as resp:
                    rec["http_status"] = resp.status
                    rec["response_headers"] = list(resp.headers.items())
                    # Retain partial bytes even on timeout/disconnect (base64 lossless backup).
                    chunks = bytearray()
                    try:
                        async for chunk in resp.content.iter_chunked(65536):
                            remaining = MAX_RESPONSE_BYTES - len(chunks)
                            chunks.extend(chunk[:remaining])
                            if len(chunk) > remaining:
                                rec["response_body_truncated"] = True
                                raise ValueError(
                                    f"Completion body exceeds {MAX_RESPONSE_BYTES} bytes"
                                )
                    finally:
                        import base64

                        rec["response_body_base64"] = base64.b64encode(chunks).decode(
                            "ascii"
                        )
                        rec["response_raw"] = bytes(chunks).decode(
                            "utf-8", errors="replace"
                        )
                    # Replacement decoding above is evidence for humans only.
                    # Never parse/score repaired bytes as a valid completion.
                    decoded = bytes(chunks).decode("utf-8", errors="strict")
                    try:
                        rec["response"] = json.loads(decoded)
                    except json.JSONDecodeError:
                        if resp.status == 200:
                            raise
                    if resp.status != 200:
                        raise ValueError(f"Completion HTTP {resp.status}")
                validate_response(rec)
        except asyncio.CancelledError:
            rec["error"] = {
                "type": "CancelledError",
                "message": "pass deadline/stall/interruption; no retry",
            }
        except (
            aiohttp.ClientError,
            TimeoutError,
            OSError,
            ValueError,
            TypeError,
            AttributeError,
        ) as e:
            rec["error"] = {"type": type(e).__name__, "message": str(e)}
        finally:
            rec["finished_utc"] = utc_now()
            completed = clock()
            rec["latency_seconds"] = completed - start if start is not None else None
            rec["completed_monotonic_s"] = completed
            records.append(rec)
            f.write(dumps(rec) + "\n")
            f.flush()
            last_completion = completed
        return rec

    tasks = [asyncio.create_task(one(row)) for row in rows]
    pending = set(tasks)
    log(f"PROGRESS completed=0/{len(rows)} errors=0 concurrency={args.concurrency}")
    try:
        while pending:
            now = clock()
            wait_seconds = max(
                0,
                min(
                    args.progress_seconds,
                    deadline - now,
                    last_completion + args.stall_timeout - now,
                ),
            )
            _, pending = await asyncio.wait(
                pending, timeout=wait_seconds, return_when=asyncio.ALL_COMPLETED
            )
            now = clock()
            # Surface task failures (e.g. storage failure), not just transport failures.
            for t in tasks:
                if t.done() and not t.cancelled() and t.exception() is not None:
                    raise t.exception()
            errors = sum(r["error"] is not None for r in records)
            log(
                f"PROGRESS completed={len(records)}/{len(rows)} errors={errors} "
                f"elapsed_s={now-started:.1f} no_completion_s={now-last_completion:.1f}"
            )
            # Check actual completion timestamps even when ALL_COMPLETED won
            # the race with the timeout callback. Polling delay alone is not a
            # late completion when every request already finished on time.
            if (pending and now >= deadline) or any(
                r["completed_monotonic_s"] >= deadline for r in records
            ):
                failure = "pass_timeout"
                break
            if pending and now - last_completion >= args.stall_timeout:
                failure = "no_completed_request_stall"
                break
    except BaseException:
        failure = "interrupted_or_storage_failure"
        raise
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        f.close()
    return sorted(records, key=lambda r: r["doc_id"]), failure


def counter_gates(deltas, arm):
    """Validate counter availability/shape without requiring any warm supply."""
    reasons, notes = [], []
    required = [
        "prompt_tokens",
        "local_hits",
        "local_queries",
        "computed_tokens",
        "local_supplied_tokens",
        "external_supplied_tokens",
    ]
    external = ("external_hits", "external_queries")
    if arm == "off" and all(
        deltas[k]["status"] == "missing_before_and_after" for k in external
    ):
        notes.append("OFF external hits/queries absent (permitted), NOT measured zero")
    else:
        required.extend(external)
    for key in required:
        item = deltas[key]
        value = item["delta"]
        if item["status"] != "ok":
            reasons.append(f"{key}_{item['status']}")
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            reasons.append(f"{key}_invalid_delta")
    for prefix in ("local", "external"):
        hits, queries = (
            deltas[f"{prefix}_{suffix}"]["delta"] for suffix in ("hits", "queries")
        )
        if (
            all(
                type(v) in (int, float) and math.isfinite(v) and v >= 0
                for v in (hits, queries)
            )
            and hits > queries
        ):
            reasons.append(f"{prefix}_hits_exceed_queries")
    return reasons, notes


def measurement_gates(args, records, total, deltas, metric_info, score, meta):
    reasons, notes = counter_gates(deltas, args.arm)
    if (
        len(records) != total
        or score.get("scored") != total
        or any(r["error"] for r in records)
    ):
        reasons.append("incomplete_or_failed_requests")
    if any(i["error"] for i in metric_info):
        reasons.append("metrics_http_failed")
    length_count = sum(r.get("finish_reason") == "length" for r in records)
    if length_count:
        reasons.append("length_truncated_outputs")
    good = [r for r in records if r.get("error") is None]
    usage_prompt = sum(r["usage"]["prompt_tokens"] for r in good)
    usage_completion = sum(r["usage"]["completion_tokens"] for r in good)
    if any(
        r["usage"]["prompt_tokens"] + MAX_TOKENS > meta["max_model_len"] for r in good
    ):
        reasons.append("prompt_plus_requested_output_exceeds_declared_context")
    values = {
        k: (
            d["delta"]
            if d["status"] == "ok"
            and type(d["delta"]) in (int, float)
            and math.isfinite(d["delta"])
            and d["delta"] >= 0
            else None
        )
        for k, d in deltas.items()
    }
    if values["prompt_tokens"] is not None and values["prompt_tokens"] != usage_prompt:
        reasons.append("metrics_prompt_delta_differs_from_this_pass_usage")
    if values["local_hits"] not in (None, 0):
        reasons.append("unexpected_local_hbm_hits")
    if values["local_supplied_tokens"] not in (None, 0):
        reasons.append("unexpected_local_hbm_supply")
    if args.arm == "off":
        if values["external_hits"] not in (None, 0):
            reasons.append("off_external_hits_not_zero")
        if values["external_supplied_tokens"] not in (None, 0):
            reasons.append("off_external_supplied_tokens_not_zero")
    source_keys = (
        "computed_tokens",
        "local_supplied_tokens",
        "external_supplied_tokens",
    )
    if (
        all(values[k] is not None for k in (*source_keys, "prompt_tokens"))
        and sum(values[k] for k in source_keys) != values["prompt_tokens"]
    ):
        reasons.append("prompt_source_conservation_failed")
    supplied = values["external_supplied_tokens"]
    fraction = (
        supplied / usage_prompt if supplied is not None and usage_prompt > 0 else None
    )
    lookup_fraction = (
        values["external_hits"] / usage_prompt
        if values["external_hits"] is not None and usage_prompt > 0
        else None
    )
    if fraction is not None and not 0 <= fraction <= 1:
        reasons.append("external_supply_fraction_out_of_range")
    if lookup_fraction is not None and not 0 <= lookup_fraction <= 1:
        reasons.append("external_lookup_fraction_out_of_range")
    readback = "not_required_for_this_pass"
    if args.arm == "on" and args.phase == "warm":
        if (
            fraction is None
            or fraction < args.min_external_fraction
            or not (values["external_hits"] is not None and values["external_hits"] > 0)
        ):
            reasons.append("ON_warm_readback_untested_or_below_minimum")
            readback = "UNTESTED"
        else:
            readback = "effective_supply_gate_met"
    return {
        "measurement_status": "PASS" if not reasons else "FAIL",
        "failure_reasons": reasons,
        "notes": notes,
        "readback_status": readback,
        "external_supplied_fraction_of_prompt": fraction,
        "external_lookup_hits_fraction_of_prompt": lookup_fraction,
        "external_supply_formula": "delta prompt_tokens_by_source_total{source=external_kv_transfer} / sum successful response.usage.prompt_tokens",
        "minimum_external_fraction": args.min_external_fraction,
        "usage_prompt_tokens": usage_prompt,
        "usage_completion_tokens": usage_completion,
        "length_finish_count": length_count,
        "accuracy_verdict": "NOT_ASSIGNED: official scores only; no equivalence/noise-floor claim",
        "scope": "Client/metrics validity only; server engine/GPU/STATE/log/lifecycle checks belong to coordinator",
    }


async def run_pass(args, rows, meta, task):
    import aiohttp

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config.update(
        {
            "started_utc": utc_now(),
            "manifest_sha256": meta["manifest_sha256"],
            "harness_sha256": sha_file(__file__),
            "retry_count": 0,
            "request_mutation": False,
            "tokenizer_backend": "none",
            "server_isolation_requirement": "Exclusive endpoint/metric model labels during this pass; counters cannot prove exclusivity",
        }
    )
    write_json(out / "run_config.json", config)
    headers = {}
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            raise ValueError(f"Requested API key env var {args.api_key_env} is unset")
        headers["Authorization"] = f"Bearer {key}"
    before = after = ""
    infos = []
    records, failure, fatal = [], None, None
    started = time.monotonic()
    async with aiohttp.ClientSession(
        headers=headers,
        trust_env=False,
        connector=aiohttp.TCPConnector(limit=args.concurrency),
    ) as session:
        # aiohttp may retry stale GET connections internally; turn that OFF too.
        session._retry_connection = False
        before, bi = await fetch_metrics(
            session, args.metrics_url, out, "before", args.metrics_timeout
        )
        infos.append(bi)
        try:
            if bi["error"]:
                raise RuntimeError(
                    "Before metrics snapshot failed; sending no completions"
                )
            before_errors, _ = counter_gates(
                metrics_delta(before, before, args.metric_model_name or meta["model"]),
                args.arm,
            )
            if before_errors:
                raise ValueError(
                    f"Before metrics invalid; sending no completions: {before_errors}"
                )
            records, failure = await request_batch(session, rows, args, out)
        except (Exception, asyncio.CancelledError) as e:
            logger.exception(
                "Pass failed; preserving recorded responses and final metrics"
            )
            fatal = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            response_path = out / "responses.jsonl"
            if response_path.exists():
                records = [
                    json.loads(s) for s in response_path.read_text().splitlines()
                ]
        finally:
            await asyncio.sleep(args.metrics_settle_seconds)
            after, ai = await fetch_metrics(
                session, args.metrics_url, out, "after", args.metrics_timeout
            )
            infos.append(ai)
    deltas = metrics_delta(before, after, args.metric_model_name or meta["model"])
    write_json(out / "metrics.delta.json", deltas)
    with offline_cpu():
        items, score = score_records(task, rows, records)
    (out / "scores.jsonl").write_text("".join(dumps(x) + "\n" for x in items))
    gates = measurement_gates(args, records, len(rows), deltas, infos, score, meta)
    if failure or fatal:
        gates["failure_reasons"].append(failure or "fatal_error")
        gates["measurement_status"] = "FAIL"
    summary = {
        **gates,
        "arm": args.arm,
        "phase": args.phase,
        "pass_id": args.pass_id,
        "manifest_sha256": meta["manifest_sha256"],
        "expected": len(rows),
        "recorded": len(records),
        "successful": sum(r["error"] is None for r in records),
        "failed": sum(r["error"] is not None for r in records),
        "official_scores": score,
        "elapsed_seconds_including_metric_snapshots": time.monotonic() - started,
        "finished_utc": utc_now(),
        "fatal": fatal,
        "observed_prompt_tokens": (
            describe_lengths(
                [r["usage"]["prompt_tokens"] for r in records if r["error"] is None]
            )
            if any(r["error"] is None for r in records)
            else None
        ),
    }
    write_json(out / "summary.json", summary)
    log(
        f"FINISHED completed={len(records)}/{len(rows)} measurement={summary['measurement_status']} "
        f"scores={score['exact_match']} external_supply={summary['external_supplied_fraction_of_prompt']} output={out}"
    )
    return 0 if summary["measurement_status"] == "PASS" else 2


def run(args):
    # Construct scoring task now, with network blocked; NEVER build/re-sample prompts.
    with offline_cpu():
        rows, meta, task = load_bundle(args.manifest, args.dataset_cache)
    u = urlsplit(args.endpoint)
    if (
        u.scheme not in ("http", "https")
        or not u.hostname
        or u.path != "/v1/completions"
        or u.query
        or u.fragment
        or u.username
    ):
        raise ValueError(
            "--endpoint must be an explicit http(s)://host:port/v1/completions URL"
        )
    if not args.metrics_url:
        args.metrics_url = urlunsplit((u.scheme, u.netloc, "/metrics", "", ""))
    m = urlsplit(args.metrics_url)
    if (
        (m.scheme, m.netloc) != (u.scheme, u.netloc)
        or m.path != "/metrics"
        or m.query
        or m.fragment
    ):
        raise ValueError(
            "Metrics must be the same server origin at /metrics (avoid host-network cross-talk)"
        )
    if args.min_external_fraction < 0.10 or args.min_external_fraction > 1:
        raise ValueError("ON warm minimum external supply cannot be below 10%")
    for name in (
        "concurrency",
        "http_timeout",
        "metrics_timeout",
        "pass_timeout",
        "stall_timeout",
        "progress_seconds",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.metrics_settle_seconds < 0:
        raise ValueError("metrics_settle_seconds must be nonnegative")
    return asyncio.run(run_pass(args, rows, meta, task))


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = p.add_subparsers(dest="command", required=True)
    prep = sp.add_parser(
        "prepare",
        help="CPU/offline only: freeze upstream official test1319/5shot requests",
    )
    prep.add_argument("--output", required=True, help="New directory; never overwrite")
    prep.add_argument(
        "--dataset-cache",
        required=True,
        help="Existing directory containing official gsm8k-train.arrow and gsm8k-test.arrow; never downloaded",
    )
    prep.add_argument(
        "--model", required=True, help="Served model ID to freeze into every request"
    )
    prep.add_argument("--max-model-len", type=int, default=32768)
    prep.set_defaults(func=prepare)
    r = sp.add_parser(
        "run",
        help="EXPLICIT HTTP execution: one pass, fixed manifest, fresh output directory",
    )
    r.add_argument("--manifest", required=True)
    r.add_argument(
        "--dataset-cache",
        help="Relocated cache directory; saved dataset hashes must match (default: original prepare location)",
    )
    r.add_argument(
        "--endpoint",
        required=True,
        help="Full /v1/completions URL; no chat or tokenizer endpoint",
    )
    r.add_argument("--metrics-url")
    r.add_argument(
        "--metric-model-name",
        help="model_name label override only, never changes request model",
    )
    r.add_argument(
        "--output",
        required=True,
        help="New per-pass directory; cannot overwrite/resume",
    )
    r.add_argument("--arm", choices=("on", "off"), required=True)
    r.add_argument("--phase", choices=("cold", "warm"), required=True)
    r.add_argument("--pass-id", required=True)
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--http-timeout", type=float, default=180)
    r.add_argument("--metrics-timeout", type=float, default=15)
    r.add_argument("--pass-timeout", type=float, default=1500)
    r.add_argument("--stall-timeout", type=float, default=180)
    r.add_argument("--progress-seconds", type=float, default=5)
    r.add_argument("--metrics-settle-seconds", type=float, default=2)
    r.add_argument("--min-external-fraction", type=float, default=0.10)
    r.add_argument(
        "--api-key-env", help="Optional name of secret env var; secret is never saved"
    )
    r.set_defaults(func=run)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception:
        logger.exception("FATAL: GSM8K client failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
