"""CPU HTTP contract for the opt-in, correctness-first K3 C1 timing survey."""

import argparse
import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_vllm_profile.py"
)


def exercise(tmp_path, fault=None, calls=None):
    spec = importlib.util.spec_from_file_location("bounded_k3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = [] if calls is None else calls
    counters = {
        role: {
            "compute": 0,
            "hit": 0,
            "external": 0,
            "success": 0,
            "drafts": 0,
            "drafted": 0,
            "accepted": 0,
            "count": 10,
            "total": 1.0,
        }
        for role in ("prefill", "decode")
    }
    pd_count = 0
    current_fault = None
    reset_roles = set()
    poll_count = 0
    flipped = False

    def metrics(role):
        nonlocal poll_count
        c = counters[role]
        labels = 'engine="0",model_name="K3"'
        if flipped:
            labels = 'engine="1",model_name="K3"'
        if role == "decode" and current_fault == "mid_poll_regression":
            poll_count += 1
            if poll_count == 2:
                c["total"] -= 0.01
        lines = [
            f'vllm:prompt_tokens_by_source_total{{{labels},source="{source}"}} {c[key]}'
            for source, key in (
                ("local_compute", "compute"),
                ("local_cache_hit", "hit"),
                ("external_kv_transfer", "external"),
            )
        ]
        lines.append(
            f'vllm:request_success_total{{{labels},finished_reason="length"}} {c["success"]}'
        )
        for name, key in (
            ("num_drafts", "drafts"),
            ("num_draft_tokens", "drafted"),
            ("num_accepted_tokens", "accepted"),
        ):
            lines.append(f"vllm:spec_decode_{name}_total{{{labels}}} {c[key]}")
        lines.extend(
            [
                f'vllm:request_time_per_output_token_seconds_count{{{labels}}} {c["count"]}',
                f'vllm:request_time_per_output_token_seconds_sum{{{labels}}} {c["total"]}',
            ]
        )
        if role == "decode" and current_fault:
            if current_fault == "missing_timing":
                lines = [
                    line
                    for line in lines
                    if "request_time_per_output_token" not in line
                ]
            if current_fault == "missing_spec":
                lines = [line for line in lines if "spec_decode" not in line]
            if current_fault == "nan_timing":
                lines[-1] = lines[-1].rsplit(" ", 1)[0] + " NaN"
            if current_fault == "inf_count":
                lines[-2] = lines[-2].rsplit(" ", 1)[0] + " inf"
            if current_fault == "fractional_count":
                lines[-2] = lines[-2].rsplit(" ", 1)[0] + " 10.5"
            if current_fault == "timing_reset":
                lines[-1] = lines[-1].rsplit(" ", 1)[0] + " 0"
                lines[-2] = lines[-2].rsplit(" ", 1)[0] + " 0"
            if current_fault == "delayed_forever":
                lines[-1] = lines[-1].rsplit(" ", 1)[0] + " 1.0"
                lines[-2] = lines[-2].rsplit(" ", 1)[0] + " 10"
            if current_fault == "duplicate_timing":
                lines.append(lines[-1])
            if current_fault == "remapped_timing":
                lines = [
                    (
                        line.replace('engine="0"', 'engine="1"')
                        if "request_time_per_output_token" in line
                        else line
                    )
                    for line in lines
                ]
        return "\n".join(lines)

    def respond(request):
        nonlocal pd_count, current_fault, flipped
        role, path = request.url.host, request.url.path
        body = json.loads(request.content) if request.content else None
        calls.append((role, path, body))
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(5000))})
        if path == "/metrics":
            return httpx.Response(200, text=metrics(role))
        if path == "/reset_prefix_cache":
            assert dict(request.url.params) == {
                "reset_external": "false",
                "reset_running_requests": "false",
            }
            reset_roles.add(role)
            if fault == "between_requests_labels" and pd_count == 4:
                flipped = True
            return httpx.Response(200, json={"success": True})
        assert path == "/v1/completions"
        assert body["stream"] is False and body["return_token_ids"] is True
        assert body["temperature"] == 0 and body["seed"] == 42
        assert body["ignore_eos"] is True
        prompt, count = body["prompt"], body["max_tokens"]
        assert len(prompt) + count <= 4096
        params = body.get("kv_transfer_params")
        producer = role == "prefill" and params is not None
        if role == "prefill":
            assert reset_roles == {
                "prefill",
                "decode",
            }, "Both resets required before each direct/PD"
            reset_roles.clear()
        else:
            assert not reset_roles, "No reset may split the P/D handoff"
        if producer:
            pd_count += 1
            current_fault = fault if pd_count == 1 else None
            assert count == 1
        else:
            assert count == 128
        c = counters[role]
        c["compute"] += len(prompt) - 1 if producer else (1 if params else len(prompt))
        c["external"] += len(prompt) - 1 if role == "decode" else 0
        c["success"] += 1
        c["count"] += 1
        c["total"] += 0.0 if producer else 0.025
        if not producer:
            c["drafts"] += 40
            c["drafted"] += 120
            c["accepted"] += 80
        result = {
            "choices": [
                {
                    "prompt_token_ids": prompt,
                    "token_ids": list(range(count)),
                    "text": "x" * count,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": count,
                "total_tokens": len(prompt) + count,
            },
        }
        if fault == "null_text":
            result["choices"][0]["text"] = None
        if producer:
            if current_fault == "producer_bad_finish":
                result["choices"][0]["finish_reason"] = "stop"
            result["kv_transfer_params"] = {
                "remote_host": "prefill",
                "remote_block_ids": [[1], [2]],
                "transfer_id": params["transfer_id"],
                "extension": {"untouched": True},
            }
        if role == "decode":
            assert params["extension"] == {"untouched": True}
            journal = [
                json.loads(line)
                for line in (tmp_path / "http-responses.jsonl").read_text().splitlines()
            ]
            assert journal[-1]["url"] == "http://prefill/v1/completions"
            assert (
                json.loads(journal[-1]["response"])["kv_transfer_params"]["transfer_id"]
                == params["transfer_id"]
            )
            history_path = tmp_path / "requests.json"
            history = (
                json.loads(history_path.read_text()) if history_path.exists() else []
            )
            assert (
                len(history) == pd_count - 1
            ), "Growing history rewrite must be outside current P/D interval"
            if current_fault == "wrong_ids":
                result["choices"][0]["token_ids"][-1] = 999
            if current_fault == "bool_ids":
                result["choices"][0]["token_ids"][-1] = True
            if current_fault == "wrong_text":
                result["choices"][0]["text"] = "wrong"
            if current_fault == "wrong_usage":
                result["usage"]["completion_tokens"] = 127
            if current_fault == "wrong_finish":
                result["choices"][0]["finish_reason"] = "stop"
            if current_fault == "http_error":
                return httpx.Response(500, json={"error": "fixture failure"})
            if current_fault == "timeout":
                raise httpx.ReadTimeout("fixture timeout", request=request)
            if current_fault == "wrong_source":
                c["compute"] += 1
            if current_fault == "extra_count":
                c["count"] += 1
            if current_fault == "zero_drafts":
                c["drafts"] = c["drafted"] = c["accepted"] = 0
            if current_fault == "overaccepted":
                c["accepted"] = c["drafted"] + 1
            if current_fault == "zero_accepted":
                c["accepted"] = 0
        return httpx.Response(200, json=result)

    async def no_sleep(_):
        return None

    args = argparse.Namespace(
        output=tmp_path,
        prefill="http://prefill",
        decode="http://decode",
        model="K3",
        tokenizer="unused",
        phase="benchmark",
        mode="bounded-perf",
        tp=8,
        dcp=8,
        hybrid=True,
        cache_composition=False,
    )
    real_client = httpx.AsyncClient
    with (
        patch.object(
            module.httpx,
            "AsyncClient",
            lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw),
        ),
        patch.object(module.asyncio, "sleep", no_sleep),
    ):
        asyncio.run(module.run(args))
    return calls


def test_journal_delay_is_inclusive_not_http_and_history_is_outside_handoff(tmp_path):
    clock = [0.0]
    original_open = Path.open

    class DelayedWriter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def write(self, data):
            clock[0] += 0.1
            return self.handle.write(data)

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

    def delayed_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if (
            path.name in ("http-requests.jsonl", "http-responses.jsonl")
            and args
            and args[0] == "a"
        ):
            return DelayedWriter(handle)
        return handle

    with (
        patch("time.perf_counter", lambda: clock[0]),
        patch.object(Path, "open", delayed_open),
    ):
        calls = exercise(tmp_path)
    samples = json.loads((tmp_path / "complete.json").read_text())["samples"]
    for sample in samples:
        assert sample["client_http_total_ms"] == 0
        assert sample["client_inclusive_elapsed_ms"] == pytest.approx(400)
    assert calls


@pytest.mark.parametrize(
    "fault", ["wrong_ids", "null_text", "wrong_source", "mid_poll_regression"]
)
def test_optimized_python_keeps_runtime_validation(tmp_path, fault):
    code = (
        "import runpy, sys; from pathlib import Path; "
        "ns=runpy.run_path(sys.argv[1]); ns['exercise'](Path(sys.argv[2]), sys.argv[3])"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code, __file__, str(tmp_path), fault],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert (tmp_path / "failure.json").exists()
    assert not (tmp_path / "complete.json").exists()


def test_c1_correctness_precedes_three_cold_timing_samples(tmp_path):
    calls = exercise(tmp_path)
    requests = [body for _, path, body in calls if path == "/v1/completions"]
    assert [(len(r["prompt"]), r["max_tokens"]) for r in requests] == [
        (1025, 128),
        (1025, 1),
        (1025, 128),
        (2050, 128),
        (2050, 1),
        (2050, 128),
        (3073, 128),
        (3073, 1),
        (3073, 128),
        (1025, 1),
        (1025, 128),
        (1025, 1),
        (1025, 128),
        (1025, 1),
        (1025, 128),
        (1025, 1),
        (1025, 128),
    ]
    result = json.loads((tmp_path / "complete.json").read_text())
    assert result["status"] == "PENDING_REVIEW"
    assert result["apc_effect_evaluated"] is False
    assert result["concurrency"] == 1 and result["output_tokens"] == 128
    assert len(result["samples"]) == 3
    assert "p99" not in json.dumps(result)
    for sample in result["samples"]:
        assert sample["decode_tpot_seconds"] == pytest.approx(0.025)
        assert (
            sample["client_inclusive_elapsed_ms"] >= sample["client_http_total_ms"] >= 0
        )
        assert (
            sample["client_http_total_ms"] == sample["p_http_ms"] + sample["d_http_ms"]
        )
    evidence = json.loads((tmp_path / "correctness-1025-evidence.json").read_text())
    assert evidence["direct_pd_token_ids_equal"] is True
    assert evidence["token_counter_deltas"]["decode"] == {
        "local_compute": 1,
        "local_cache_hit": 0,
        "external_kv_transfer": 1024,
        "request_success": 1,
    }
    assert evidence["speculative_counters"]["decode"]["drafts"] == 40
    assert evidence["timing"]["decode"]["count"] == 1
    assert list(tmp_path.glob("*-before-decode.metrics.txt"))
    assert (tmp_path / "http-responses.jsonl").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_ids",
        "bool_ids",
        "wrong_text",
        "wrong_usage",
        "wrong_finish",
        "http_error",
        "timeout",
        "wrong_source",
        "extra_count",
        "zero_drafts",
        "overaccepted",
        "missing_timing",
        "missing_spec",
        "nan_timing",
        "fractional_count",
        "remapped_timing",
        "inf_count",
        "timing_reset",
        "duplicate_timing",
        "delayed_forever",
        "mid_poll_regression",
    ],
)
def test_invalid_request_stops_before_next_length_with_evidence(tmp_path, fault):
    calls = []
    with pytest.raises((AssertionError, httpx.HTTPError)):
        exercise(tmp_path, fault, calls)
    completions = [body for _, path, body in calls if path == "/v1/completions"]
    assert len(completions) == 3
    assert all(len(body["prompt"]) == 1025 for body in completions)
    assert not (tmp_path / "complete.json").exists()
    evidence = json.loads((tmp_path / "correctness-1025-evidence.json").read_text())
    assert evidence["status"] == "FAIL"
    assert evidence["timing_status"] == "UNKNOWN"
    assert (tmp_path / "failure.json").exists()
    assert (tmp_path / "reference-1025-evidence.json").exists()
    assert (tmp_path / "http-responses.jsonl").exists()
    if fault == "timeout":
        pending = [
            json.loads(line)
            for line in (tmp_path / "http-requests.jsonl").read_text().splitlines()
        ]
        assert pending[-1]["url"] == "http://decode/v1/completions"
        assert pending[-1]["request"]["max_tokens"] == 128
    if fault in ("missing_timing", "missing_spec"):
        assert evidence["timing"] is None
        assert "missing series" in evidence["error"]


def test_invalid_producer_stops_before_decode_request(tmp_path):
    calls = []
    with pytest.raises(AssertionError):
        exercise(tmp_path, "producer_bad_finish", calls)
    assert len([body for _, path, body in calls if path == "/v1/completions"]) == 2
    assert (tmp_path / "reference-1025-evidence.json").exists()
    assert (
        json.loads((tmp_path / "correctness-1025-evidence.json").read_text())["status"]
        == "FAIL"
    )


@pytest.mark.parametrize("fault", ["null_text", "between_requests_labels"])
def test_invalid_text_or_cross_request_label_change_stops(tmp_path, fault):
    with pytest.raises(AssertionError):
        exercise(tmp_path, fault)
    assert not (tmp_path / "complete.json").exists()


def test_nonempty_output_is_rejected_without_touching_old_evidence(tmp_path):
    old = tmp_path / "complete.json"
    old.write_text("old-run")
    calls = []
    with pytest.raises(ValueError):
        exercise(tmp_path, calls=calls)
    assert not calls
    assert old.read_text() == "old-run"
    assert list(tmp_path.iterdir()) == [old]


def test_zero_acceptance_is_observed_speculation_not_disabled(tmp_path):
    exercise(tmp_path, "zero_accepted")
    evidence = json.loads((tmp_path / "correctness-1025-evidence.json").read_text())
    assert evidence["speculative_counters"]["decode"]["accepted_tokens"] == 0
    assert evidence["speculative_counters"]["decode"]["drafts"] == 40
    assert (tmp_path / "complete.json").exists()
