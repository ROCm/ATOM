"""CPU HTTP-contract tests for optional NIXL cache composition evidence."""

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_m3_nixl_smoke.py"
)


@pytest.mark.parametrize("profile,boundary", [("m3", 128), ("v4", 256)])
def test_composition_phases_and_evidence(tmp_path, monkeypatch, profile, boundary):
    smoke, args, calls = setup_run(tmp_path, monkeypatch, profile)
    asyncio.run(smoke.run(args))
    report = json.loads((tmp_path / "composition.json").read_text())
    assert report["status"] == "PENDING_REVIEW"
    targets = [r for r in report["requests"] if r["kind"] == "target"]
    assert [r["tag"] for r in targets] == [
        "cold",
        "P-warm-D-cold",
        "both-warm",
        "partial-prefix",
        "boundary-minus",
        "boundary",
        "boundary-plus",
    ]
    assert [r["prompt_tokens"] for r in targets] == [
        513,
        513,
        513,
        513,
        boundary - 1,
        boundary,
        boundary + 1,
    ]
    assert all(r["generated_token_ids_equal"] for r in targets)
    assert all(r["direct_pd_text_equal"] for r in targets)
    assert all(r["accounting_checked"] for r in targets)
    assert targets[1]["apc_status"] == "APC_EXERCISED"
    assert all(r["transfer_status"] == "REMOTE_TRANSFER_OBSERVED" for r in targets)
    for target in targets:
        values = target["labeled_metric_deltas"]["decode"]

        def source(name, values=values):
            return smoke.total(
                values, "vllm:prompt_tokens_by_source_total", f'source="{name}"'
            )

        assert source("local_compute") == 0
        assert source("external_kv_transfer") == target["prompt_tokens"] - source(
            "local_cache_hit"
        )
    assert targets[2]["cache_hits"]["decode"] == 512
    assert (
        smoke.total(
            targets[2]["labeled_metric_deltas"]["decode"],
            "vllm:prompt_tokens_by_source_total",
            'source="external_kv_transfer"',
        )
        == 1
    )
    refs = [r for r in report["requests"] if r["kind"] == "reference"]
    assert len(refs) == 5
    assert [r["kind"] for r in report["requests"]] == ["reference"] * 5 + ["target"] * 7
    primary = report["prompts"]["primary"]
    branch = report["prompts"]["branch"]
    assert len(primary) == len(branch) == 513
    assert primary[:256] == branch[:256]
    assert primary[256] != branch[256]
    assert len([c for c in calls if c[1] == "/reset_prefix_cache"]) == 19
    assert report["runtime_audit"]["graph_execution"] == "NOT_VERIFIED"
    assert report["runtime_audit"]["cudagraph_metrics_requested"] is True
    assert report["scrapes"]
    assert any(
        "vllm:iteration_tokens_total 0" in s["runtime_metric_lines"]
        for s in report["scrapes"]
    )
    assert any("-after-6" in s["tag"] for s in report["scrapes"])
    for scrape in report["scrapes"]:
        assert scrape["sampled_at"] > 0
        assert (tmp_path / scrape["path"]).exists()
    assert not (tmp_path / "complete.json").exists()


@pytest.mark.parametrize(
    "fault,message",
    [
        ("reset", "Prefix cache reset failed"),
        ("no_external", "No external KV consumption"),
        ("producer_external", "Producer external tokens"),
        ("producer_bytes", "Unattributed NIXL transfer"),
        ("failure", "NIXL failure counter"),
        ("overcount", "accounting exceeds N"),
        ("nonisolated", "Non-isolated"),
        ("cold_hit", "Cold role reported"),
        ("ids", "token ID mismatch"),
        ("ids_missing", "Generated IDs missing"),
        ("prompt", "Prompt IDs changed"),
        ("text", "text mismatch"),
        ("usage", "completion_tokens"),
        ("finish", "finish_reason"),
    ],
)
def test_composition_fails_closed(tmp_path, monkeypatch, fault, message):
    smoke, args, _ = setup_run(tmp_path, monkeypatch, fault=fault)
    with pytest.raises(AssertionError, match=message):
        asyncio.run(smoke.run(args))
    report = json.loads((tmp_path / "composition.json").read_text())
    assert report["status"] == "FAIL"
    assert report["requests"][-1]["status"] == "FAIL"
    assert not report["requests"][-1]["accounting_checked"]
    assert not (tmp_path / "complete.json").exists()


@pytest.mark.parametrize(
    "fault", ["missing_metrics", "series_disappears", "missing_bytes"]
)
def test_incomplete_metrics_stop_following_requests(tmp_path, monkeypatch, fault):
    smoke, args, calls = setup_run(tmp_path, monkeypatch, fault=fault)
    asyncio.run(smoke.run(args))
    report = json.loads((tmp_path / "composition.json").read_text())
    assert report["status"] == "PENDING_REVIEW_INCOMPLETE"
    assert not report["requests"][-1]["accounting_checked"]
    if fault == "missing_metrics":
        assert not any(c[1] == "/v1/completions" for c in calls)
    elif fault == "series_disappears":
        assert any(
            v is None
            for values in report["requests"][-1]["labeled_metric_deltas"].values()
            for v in values.values()
        )
    assert not (tmp_path / "complete.json").exists()


def test_zero_warm_hits_are_not_transfer_failure(tmp_path, monkeypatch):
    smoke, args, _ = setup_run(tmp_path, monkeypatch, fault="apc_not_exercised")
    asyncio.run(smoke.run(args))
    report = json.loads((tmp_path / "composition.json").read_text())
    assert report["status"] == "PENDING_REVIEW"
    assert report["apc_status"] == "APC_NOT_EXERCISED"
    assert len(report["requests"]) == 12
    assert all(
        r["transfer_status"] == "REMOTE_TRANSFER_OBSERVED"
        for r in report["requests"]
        if r["kind"] == "target"
    )


def test_delayed_accounting(tmp_path, monkeypatch):
    smoke, args, _ = setup_run(tmp_path, monkeypatch, fault="delayed_metrics")
    asyncio.run(smoke.run(args))
    report = json.loads((tmp_path / "composition.json").read_text())
    assert report["status"] == "PENDING_REVIEW"
    assert all(r["accounting_checked"] for r in report["requests"])


@pytest.mark.parametrize("profile", ["m3", "v4"])
@pytest.mark.parametrize(
    "fault,phase",
    [
        ("warm_no_transfer", "both-warm"),
        ("partial_no_transfer", "partial-prefix"),
        ("partial_recompute", "partial-prefix"),
        ("full_cache_hit", "both-warm"),
    ],
)
def test_warm_missing_transfer_fails_closed(
    tmp_path, monkeypatch, profile, fault, phase
):
    smoke, args, _ = setup_run(tmp_path, monkeypatch, profile, fault)
    with pytest.raises(AssertionError, match="Decoder NIXL source accounting"):
        asyncio.run(smoke.run(args))
    report = json.loads((tmp_path / "composition.json").read_text())
    assert report["status"] == "FAIL"
    assert report["requests"][-1]["tag"] == phase
    assert not report["requests"][-1]["accounting_checked"]
    assert report["requests"][-1]["status"] == "FAIL"


def setup_run(tmp_path, monkeypatch, profile="m3", fault=None):
    spec = importlib.util.spec_from_file_location("composition_smoke", SCRIPT)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    counters = {
        role: {
            "local_compute": 0,
            "local_cache_hit": 0,
            "external_kv_transfer": 0,
            "success": 0,
            "bytes": 0,
            "count": 0,
            "failure": 0,
        }
        for role in ("prefill", "decode")
    }
    cached = {role: [] for role in counters}
    pending = {}
    calls = []
    tokenizations = 0
    handoff = {
        "do_remote_prefill": True,
        "remote_engine_id": "producer",
        "remote_block_ids": [[1, 2], [3]],
        "remote_host": "prefill",
        "remote_port": 15559,
        "opaque": {"unchanged": [1, 2]},
    }

    def respond(request):
        nonlocal tokenizations
        role = request.url.host
        c = counters[role]
        body = json.loads(request.content) if request.content else None
        calls.append((role, request.url.path, body))
        if request.url.path == "/tokenize":
            tokenizations += 1
            offset = 0 if tokenizations == 1 else 1000
            return httpx.Response(
                200, json={"tokens": list(range(offset, offset + 1024))}
            )
        if request.url.path == "/metrics":
            if role in pending:
                remaining, updates = pending[role]
                if remaining == 0:
                    for name, value in updates.items():
                        c[name] += value
                    del pending[role]
                else:
                    pending[role] = (remaining - 1, updates)
            text = "\n".join(
                f'vllm:prompt_tokens_by_source_total{{source="{s}"}} {c[s]}'
                for s in smoke.SOURCES
            )
            text += f'\nvllm:request_success_total {c["success"]}'
            text += f'\nvllm:nixl_bytes_transferred_sum {c["bytes"]}'
            text += f'\nvllm:nixl_bytes_transferred_count {c["count"]}'
            text += "".join(f'\n{name} {c["failure"]}' for name in smoke.FAILURES)
            text += "\nvllm:iteration_tokens_total 0\nvllm:cudagraph_test_total 0"
            if fault == "missing_metrics" or (
                fault == "series_disappears" and c["success"]
            ):
                text = "\n".join(
                    line
                    for line in text.splitlines()
                    if not line.startswith(smoke.FAILURES[0])
                )
            return httpx.Response(200, text=text)
        if request.url.path == "/reset_prefix_cache":
            assert request.url.params["reset_external"] == "false"
            assert request.url.params["reset_running_requests"] == "false"
            cached[role] = []
            return httpx.Response(200, json={"success": fault != "reset"})
        assert request.url.path == "/v1/completions"
        assert body["return_token_ids"] is True
        assert body["max_tokens"] in (1, 16)
        prompt = body["prompt"]
        n = len(prompt)
        transfer = body.get("kv_transfer_params")
        hit = 0
        if transfer and fault != "apc_not_exercised":
            for old in cached[role]:
                shared = 0
                for left, right in zip(prompt, old):
                    if left != right:
                        break
                    shared += 1
                hit = max(hit, min(shared, n - 1) // 128 * 128)
        if role == "decode" and hit == 512 and fault == "full_cache_hit":
            hit = n
        ext = n - hit if role == "decode" else 0
        if role == "decode":
            if (hit == 512 and fault == "warm_no_transfer") or (
                hit == 256 and fault == "partial_no_transfer"
            ):
                ext = 0
            elif hit == 256 and fault == "partial_recompute":
                ext -= 1
        if fault == "no_external" and role == "decode":
            ext = 0
        c["success"] += 1
        c["local_cache_hit"] += hit
        c["external_kv_transfer"] += ext
        c["local_compute"] += n - hit - ext
        if ext:
            c["bytes"] += 4096
            c["count"] += 1
        if fault == "producer_external" and transfer and role == "prefill":
            c["external_kv_transfer"] += 1
            c["local_compute"] -= 1
        if fault == "producer_bytes" and transfer and role == "prefill":
            c["bytes"] += 4096
            c["count"] += 1
        if role == "decode":
            if fault == "failure":
                c["failure"] += 1
            elif fault == "overcount":
                c["local_compute"] += 1
            elif fault == "nonisolated":
                c["success"] += 1
            elif fault == "cold_hit":
                c["external_kv_transfer"] -= 1
                c["local_cache_hit"] += 1
            elif fault == "missing_bytes":
                c["bytes"] = c["count"] = 0
        if fault == "delayed_metrics" and role == "decode":
            # Transport completes before the stats interval publishes source counters.
            updates = {
                "local_compute": n - hit - ext,
                "local_cache_hit": hit,
                "external_kv_transfer": ext,
                "success": 1,
            }
            for name, value in updates.items():
                c[name] -= value
            pending[role] = (8, updates)
        cached[role].append(prompt)
        result = {
            "choices": [
                {
                    "text": "same",
                    "finish_reason": "length",
                    "prompt_token_ids": prompt,
                    "token_ids": [7] * body["max_tokens"],
                }
            ],
            "usage": {"prompt_tokens": n, "completion_tokens": body["max_tokens"]},
        }
        if role == "decode":
            assert transfer == handoff
            if fault == "ids":
                result["choices"][0]["token_ids"][-1] = 8
            if fault == "text":
                result["choices"][0]["text"] = "different"
            if fault == "ids_missing":
                result["choices"][0].pop("token_ids")
            if fault == "prompt":
                result["choices"][0]["prompt_token_ids"] = prompt[:-1]
            if fault == "usage":
                result["usage"]["completion_tokens"] -= 1
            if fault == "finish":
                result["choices"][0]["finish_reason"] = "stop"
        elif transfer:
            result["kv_transfer_params"] = handoff
        return httpx.Response(200, json=result)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        smoke.httpx,
        "AsyncClient",
        lambda **kw: original_client(transport=httpx.MockTransport(respond), **kw),
    )

    async def no_sleep(_):
        pass

    monkeypatch.setattr(smoke.asyncio, "sleep", no_sleep)
    args = argparse.Namespace(
        prefill="http://prefill",
        decode="http://decode",
        model="fixture",
        model_profile=profile,
        output=tmp_path,
        cache_composition=True,
        cudagraph_metrics=True,
    )
    return smoke, args, calls
