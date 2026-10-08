"""CPU-only graph correctness HTTP and newly emitted target-log contracts."""

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
PREFIX = "(APIServer pid=123) INFO 10-08 12:00:00 [cuda_graph.py:123] "
IDLE = (
    PREFIX.replace("cuda_graph.py:123", "loggers.py:320")
    + "Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs\n"
)
TABLE = (
    "\n".join(
        PREFIX + line
        for line in [
            "**CUDAGraph Config Settings:**",
            "",
            "- Mode: FULL_DECODE_ONLY",
            "- Capture sizes: [3, 4]",
            "",
            "**CUDAGraph Stats:**",
            "",
            "| Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |",
            "|-----------------|---------------|--------------|--------------|-------|",
            "| 1               | 4             | 3            | FULL         | 2     |",
            "| 1               | 1             | 0            | NONE         | 1     |",
            "",
        ]
    )
    + "\n"
)


def exercise(root, fault=None):
    root.mkdir(exist_ok=True)
    output = root / "out"
    log = root / "decode.log"
    log.write_text(TABLE)  # stale historical table must never count
    if fault == "existing_output":
        output.mkdir()
        (output / "existing.txt").write_text("retain exactly")
    if fault == "missing_log":
        log.unlink()
    spec = importlib.util.spec_from_file_location("graph_client", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counters = {
        role: {
            "local_compute": 0,
            "local_cache_hit": 0,
            "external_kv_transfer": 0,
            "success": 0,
            "drafts": 0,
        }
        for role in ("prefill", "decode")
    }
    if fault == "used_endpoint":
        counters["decode"]["success"] = 1
    resets = set()
    calls = []
    pending = []
    sleeps = 0

    def respond(request):
        role, path = request.url.host, request.url.path
        body = json.loads(request.content) if request.content else None
        calls.append((role, path, body))
        if path == "/tokenize":
            assert body["prompt"].startswith("Explain this engineering record.\n")
            return httpx.Response(200, json={"tokens": list(range(5000))})
        if path == "/reset_prefix_cache":
            assert dict(request.url.params) == {
                "reset_external": "false",
                "reset_running_requests": "false",
            }
            assert not pending, "No reset or next request after sole D completion"
            resets.add(role)
            return httpx.Response(200, json={"success": fault != "reset_failed"})
        if path == "/metrics":
            c = counters[role]
            labels = 'engine="0",model_name="K3"'
            text = "\n".join(
                f'vllm:prompt_tokens_by_source_total{{{labels},source="{source}"}} {c[source]}'
                for source in (
                    "local_compute",
                    "local_cache_hit",
                    "external_kv_transfer",
                )
            )
            text += f'\nvllm:request_success_total{{{labels},finished_reason="length"}} {c["success"]}'
            for name, multiplier in (
                ("num_drafts", 1),
                ("num_draft_tokens", 3),
                ("num_accepted_tokens", 2),
            ):
                text += f'\nvllm:spec_decode_{name}_total{{{labels}}} {c["drafts"] * multiplier}'
            return httpx.Response(200, text=text)
        assert path == "/v1/completions"
        prompt, count = body["prompt"], body["max_tokens"]
        params = body.get("kv_transfer_params")
        if role == "prefill":
            assert resets == {"prefill", "decode"}
            resets.clear()
        else:
            assert not resets
        producer = role == "prefill" and params is not None
        assert count == (1 if producer else 16)
        c = counters[role]
        c["local_compute"] += (
            len(prompt) - 1 if producer else (1 if params else len(prompt))
        )
        c["external_kv_transfer"] += len(prompt) - 1 if role == "decode" else 0
        c["success"] += 1
        if not producer:
            c["drafts"] += 4
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
        if producer:
            assert (output / f"reference-{len(prompt)}-evidence.json").exists()
            result["kv_transfer_params"] = {
                "remote_host": "prefill",
                "remote_block_ids": [[1]],
                "transfer_id": params["transfer_id"],
            }
        if role == "decode":
            if fault == "ids":
                result["choices"][0]["token_ids"][0] = 999
            if fault == "source":
                c["local_compute"] += 1
            if fault == "stale" or fault == "capture_only":
                if fault == "capture_only":
                    with log.open("a") as handle:
                        handle.write("Capturing CUDA graphs FULL complete\n")
            else:
                table = TABLE
                if fault == "none":
                    table = table.replace("FULL         |", "NONE         |")
                if fault == "wrong_config":
                    table = table.replace("FULL_DECODE_ONLY", "FULL")
                if fault == "wrong_table_source":
                    table = table.replace("cuda_graph.py:123", "loggers.py:123")
                if fault == "ansi_crlf":
                    table = table.replace("INFO", "\x1b[32mINFO\x1b[0m").replace(
                        "\n", "\r\n"
                    )
                if fault == "different_count":
                    table = table.replace("| 2     |", "| 9     |")
                if fault == "truncated":
                    table = table[: table.index("| 1") + 8]
                if fault == "ambiguous":
                    table += TABLE.replace("pid=123", "pid=456")
                if fault == "rotated":
                    log.write_text("")
                elif fault in ("late", "unterminated"):
                    pending.append(
                        IDLE + (table if fault == "late" else table.rstrip("\n"))
                    )
                else:
                    with log.open("a") as handle:
                        handle.write(table)
                        if fault == "partial_idle":
                            handle.write("unrelated incomplete line ")
                    if fault != "no_post_observation":
                        idle = IDLE
                        if fault == "wrong_idle_source":
                            idle = idle.replace("loggers.py:320", "loggers.py:321")
                        if fault == "wrong_idle_emitter":
                            idle = idle.replace("pid=123", "pid=456")
                        pending.append(idle + idle)
        return httpx.Response(200, json=result)

    async def tick(_):
        nonlocal sleeps
        sleeps += 1
        if pending and sleeps % 10 == 0:
            with log.open("a") as handle:
                handle.write(pending.pop())
            if fault == "extra_activity":
                counters["decode"]["success"] += 1

    args = argparse.Namespace(
        output=output,
        prefill="http://prefill",
        decode="http://decode",
        model="K3",
        tokenizer="unused",
        phase="benchmark",
        mode="graph-correctness",
        tp=8,
        dcp=8,
        hybrid=True,
        cache_composition=False,
        decode_log=log,
    )
    real_client = httpx.AsyncClient
    with (
        patch.object(
            module.httpx,
            "AsyncClient",
            lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw),
        ),
        patch.object(module.asyncio, "sleep", tick),
    ):
        asyncio.run(module.run(args))
    if fault == "late_tail":
        assert counters["decode"]["success"] == 1
        with log.open("a") as handle:
            handle.write(TABLE)
    return output, calls


@pytest.mark.parametrize(
    "fault", [None, "late", "ansi_crlf", "different_count", "late_tail"]
)
def test_new_target_full_tables_follow_exact_cold_requests(tmp_path, fault):
    output, calls = exercise(tmp_path, fault)
    complete = json.loads((output / "complete.json").read_text())
    assert complete["status"] == "PENDING_REVIEW"
    assert complete["graph_status"] == "TARGET_FULL_OBSERVED"
    assert complete["output_tokens"] == 16 and complete["lengths"] == [1025]
    assert complete["decode_request_limit"] == 1
    assert complete["direct_pd_token_id_checks"] == 1
    assert complete["complete_stats_drain_verified"] is False
    assert "samples" not in complete and "summary" not in complete
    requests = [body for _, path, body in calls if path == "/v1/completions"]
    assert [(len(r["prompt"]), r["max_tokens"]) for r in requests] == [
        (1025, 16),
        (1025, 1),
        (1025, 16),
    ]
    for length in (1025,):
        evidence = json.loads(
            (output / f"correctness-{length}-evidence.json").read_text()
        )
        assert evidence["graph"]["full_count"] == (
            9 if fault == "different_count" else 2
        )
        assert evidence["graph"]["start_offset"] >= len(TABLE.encode())
        assert evidence["graph"]["end_offset"] > evidence["graph"]["start_offset"]
        assert evidence["direct_pd_token_ids_equal"]
        assert "timing" not in evidence and "timing_status" not in evidence
        assert (output / f"correctness-{length}-decode-window.log").exists()
        assert "closure" not in evidence["graph"]
        assert "pending tail statistics may remain" in evidence["graph"]["observation"]
        if fault == "late_tail":
            assert (tmp_path / "decode.log").stat().st_size > evidence["graph"][
                "end_offset"
            ]
    assert (
        sum(role == "decode" and path == "/v1/completions" for role, path, _ in calls)
        == 1
    )
    assert not (output / "reference-2050-evidence.json").exists()
    assert calls[-1][1] == "/metrics"


@pytest.mark.parametrize(
    "fault",
    [
        "stale",
        "capture_only",
        "none",
        "truncated",
        "ambiguous",
        "rotated",
        "no_post_observation",
        "wrong_idle_source",
        "wrong_idle_emitter",
        "partial_idle",
        "unterminated",
        "wrong_config",
        "wrong_table_source",
        "extra_activity",
    ],
)
def test_missing_or_ambiguous_new_full_stops_before_second_request(tmp_path, fault):
    with pytest.raises((AssertionError, ValueError)):
        exercise(tmp_path, fault)
    output = tmp_path / "out"
    assert not (output / "complete.json").exists()
    evidence = json.loads((output / "correctness-1025-evidence.json").read_text())
    assert evidence["graph"]["status"] == "GRAPH_NOT_OBSERVED"
    assert not (output / "reference-2050-evidence.json").exists()
    assert (output / "reference-1025-evidence.json").exists()


@pytest.mark.parametrize("fault", ["ids", "source", "reset_failed"])
def test_correctness_gates_stop_without_graph_success(tmp_path, fault):
    with pytest.raises(AssertionError):
        exercise(tmp_path, fault)
    assert not (tmp_path / "out" / "complete.json").exists()
    assert not (tmp_path / "out" / "reference-2050-evidence.json").exists()
    if fault != "reset_failed":
        evidence = json.loads(
            (tmp_path / "out" / "correctness-1025-evidence.json").read_text()
        )
        assert evidence["graph"]["status"] == "GRAPH_NOT_OBSERVED"
        assert (
            TABLE
            in (tmp_path / "out" / "correctness-1025-decode-window.log").read_text()
        )


def test_used_decode_endpoint_rejected_before_any_completion(tmp_path):
    with pytest.raises(AssertionError, match="fresh unused D endpoint"):
        exercise(tmp_path, "used_endpoint")
    entries = [
        json.loads(line)
        for line in (tmp_path / "out" / "http-requests.jsonl").read_text().splitlines()
    ]
    assert not any("/v1/completions" in entry["url"] for entry in entries)


def test_existing_output_is_untouched(tmp_path):
    with pytest.raises(ValueError):
        exercise(tmp_path, "existing_output")
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["existing.txt"]
    assert (tmp_path / "out" / "existing.txt").read_text() == "retain exactly"


def test_missing_log_stops_before_decode(tmp_path):
    with pytest.raises(FileNotFoundError):
        exercise(tmp_path, "missing_log")
    assert not (tmp_path / "out" / "complete.json").exists()
    entries = [
        json.loads(line)
        for line in (tmp_path / "out" / "http-requests.jsonl").read_text().splitlines()
    ]
    assert not any(
        entry.get("url") == "http://decode/v1/completions" for entry in entries
    )


@pytest.mark.parametrize("fault", ["ids", "source", "stale"])
def test_graph_gates_survive_optimized_python(tmp_path, fault):
    code = "import runpy,sys; from pathlib import Path; runpy.run_path(sys.argv[1])['exercise'](Path(sys.argv[2]),sys.argv[3])"
    result = subprocess.run(
        [sys.executable, "-O", "-c", code, __file__, str(tmp_path), fault],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not (tmp_path / "out" / "complete.json").exists()
