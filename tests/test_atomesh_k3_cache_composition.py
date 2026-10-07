"""CPU-only HTTP/metrics contract for the optional K3 cache-composition survey."""

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_vllm_profile.py"
)


def load_profile():
    spec = importlib.util.spec_from_file_location("k3_cache_composition", SCRIPT)
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    return profile


def run_composition(tmp_path, fault=None):
    profile = load_profile()
    calls = []
    sources = ("local_compute", "local_cache_hit", "external_kv_transfer")
    counters = {
        role: dict.fromkeys((*sources, "success", "draft", "accepted"), 0)
        for role in ("prefill", "decode")
    }
    warm = dict.fromkeys(counters, False)
    split_count = 0
    pending = {}
    handoff = {
        "remote_host": "prefill",
        "remote_block_ids": [[2], [5]],
        "unknown_extension": {"preserve": [17, "ticket"]},
    }

    def metrics(role):
        c = counters[role]
        lines = [
            f'vllm:prompt_tokens_by_source_total{{model_name="K3",engine="0",source="{source}"}} {c[source]}'
            for source in sources
        ]
        lines += [
            f'vllm:request_success_total{{model_name="K3",engine="0",finished_reason="length"}} {c["success"]}',
            f'vllm:spec_decode_num_draft_tokens_total{{model_name="K3",engine="0"}} {c["draft"]}',
            f'vllm:spec_decode_num_accepted_tokens_total{{model_name="K3",engine="0"}} {c["accepted"]}',
        ]
        if fault == "missing_baseline" and role == "prefill":
            lines = [line for line in lines if 'source="local_cache_hit"' not in line]
        if fault == "missing_after" and split_count and role == "decode":
            lines = [line for line in lines if 'source="local_cache_hit"' not in line]
        if fault == "missing_engine_source" and role == "prefill":
            lines += [
                line.replace('engine="0"', 'engine="1"').rsplit(" ", 1)[0] + " 0"
                for line in lines
                if 'source="local_cache_hit"' not in line
            ]
        if fault == "missing_spec":
            lines = [line for line in lines if "spec_decode" not in line]
        if fault == "remapped_spec":
            lines = [
                line.replace(
                    "spec_decode_num_draft_tokens", "diffusion_num_canvas_positions"
                ).replace(
                    "spec_decode_num_accepted_tokens", "diffusion_num_committed_tokens"
                )
                for line in lines
            ]
        return "\n".join(lines)

    def respond(request):
        nonlocal split_count
        host, path = request.url.host, request.url.path
        body = json.loads(request.content) if request.content else None
        calls.append((host, path, body, dict(request.url.params)))
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(4096))})
        if path == "/metrics":
            text = metrics(host)
            if host in pending:
                pending[host] -= 1
                if pending[host] == 0:
                    counters[host]["success"] += 1
                    del pending[host]
            return httpx.Response(200, text=text)
        if path == "/reset_prefix_cache":
            assert not pending, "Reference must drain before reset"

            assert dict(request.url.params) == {
                "reset_external": "false",
                "reset_running_requests": "false",
            }
            warm[host] = False
            return httpx.Response(200, json={"success": fault != "reset_false"})
        assert path == "/v1/completions"
        assert body["return_token_ids"] is True
        assert len(body["prompt"]) in (1026, 1153)
        assert body["max_tokens"] in (1, 16)
        params = body.get("kv_transfer_params")
        p = host == "prefill" and params is not None
        if p:
            split_count += 1
        c = counters[host]
        c["success"] += 1
        if fault == "extra_success" and split_count:
            c["success"] += 1
        if fault in ("late_reference", "delayed_reference") and params is None:
            c["success"] -= 1
            if fault == "delayed_reference":
                pending[host] = 2
        n = len(body["prompt"]) - int(p)
        if host == "decode":
            assert all(params[k] == value for k, value in handoff.items())
            c[
                (
                    "local_cache_hit"
                    if warm[host] and fault != "zero_d_hit"
                    else "external_kv_transfer"
                )
            ] += (n - 1)
            c["local_compute"] += 1
            c["draft"] += 8
        else:
            cached = min(1024, n - 1) if warm[host] and fault != "zero_p_hit" else 0
            c["local_cache_hit"] += cached
            c["local_compute"] += n - cached
        if fault == "zero_external" and host == "decode" and not warm[host]:
            c["external_kv_transfer"] -= n - 1
            c["local_compute"] += n - 1
        warm[host] = True
        choice = {
            "prompt_token_ids": body["prompt"],
            "token_ids": [999] if p else list(range(16)),
            "text": "same text",
            "finish_reason": "length",
        }
        if fault == "different_ids" and host == "decode":
            choice["token_ids"][-1] = 99
        if fault == "different_text" and host == "decode":
            choice["text"] = "different text"
        result = {"choices": [choice]}
        if p:
            result["kv_transfer_params"] = handoff
        return httpx.Response(200, json=result)

    args = argparse.Namespace(
        output=tmp_path,
        prefill="http://prefill",
        decode="http://decode",
        model="K3",
        phase="benchmark",
        mode="smoke",
        hybrid=True,
        tp=8,
        dcp=8,
        cache_composition=True,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

    async def no_sleep(seconds):
        pass

    with (
        patch.object(profile.httpx, "AsyncClient", return_value=client),
        patch.object(profile.asyncio, "sleep", side_effect=no_sleep),
    ):
        asyncio.run(profile.run(args))
    return calls


@pytest.mark.parametrize("fault", [None, "delayed_reference"])
def test_opt_in_runs_separate_bounded_composition_workload(tmp_path, fault):
    calls = run_composition(tmp_path, fault)
    requests = [
        (host, body) for host, path, body, _ in calls if path == "/v1/completions"
    ]
    assert len(requests) == 12  # Two isolated references and five P/D requests.
    assert [len(body["prompt"]) for host, body in requests] == [
        1026,
        1153,
        1026,
        1026,
        1026,
        1026,
        1026,
        1026,
        1026,
        1026,
        1153,
        1153,
    ]
    records = json.loads((tmp_path / "requests.json").read_text())
    assert [record["tag"] for record in records] == [
        "composition-cold",
        "composition-p-warm-d-cold",
        "composition-both-warm",
        "composition-branch-prime",
        "composition-branch",
    ]
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["status"] == "PENDING_REVIEW"
    assert complete["direct_pd_token_id_checks"] == 5
    workload = json.loads((tmp_path / "workload.json").read_text())
    assert workload["requested_hash_block_tokens"] == 128
    assert workload["physical_geometry"] == "UNKNOWN_PENDING_LOG_AUDIT"
    assert workload["profile"] == "hash-partial-tail-under4096"
    original, branch = [body["prompt"] for _, body in requests[:2]]
    assert original[:1024] == branch[:1024]
    assert original[1024] != branch[1024]
    # Cache choreography: references, cold, D-only, no reset, isolated prime, D-only.
    resets = [host for host, path, _, _ in calls if path == "/reset_prefix_cache"]
    assert resets == ["prefill", "decode"] * 3 + [
        "decode",
        "prefill",
        "decode",
        "decode",
    ]
    for record in records:
        evidence = json.loads((tmp_path / f"{record['tag']}-evidence.json").read_text())
        p, d = (
            evidence["token_counter_deltas"][role] for role in ("prefill", "decode")
        )
        assert (
            sum(
                p[k]
                for k in ("local_compute", "local_cache_hit", "external_kv_transfer")
            )
            == record["prompt_tokens"] - 1
        )
        assert (
            sum(
                d[k]
                for k in ("local_compute", "local_cache_hit", "external_kv_transfer")
            )
            == record["prompt_tokens"]
        )
        assert p["request_success"] == d["request_success"] == 1
        assert evidence["bytes"] == evidence["ack"] == "UNKNOWN"
        spec = evidence["speculative_counters"]["decode"]
        assert any("draft_tokens" in key and value == 8 for key, value in spec.items())
        assert any(
            "accepted_tokens" in key and value == 0 for key, value in spec.items()
        )


@pytest.mark.parametrize(
    "fault,message",
    [
        ("missing_baseline", "Missing"),
        ("missing_after", "Missing before/after"),
        ("missing_engine_source", "Missing"),
        ("extra_success", "Non-isolated"),
        ("late_reference", "Missing/delayed"),
        ("zero_external", "No external"),
        ("reset_false", "cache reset failed"),
        ("different_ids", "token IDs differ"),
        ("different_text", "text differs"),
    ],
)
def test_incomplete_or_nonisolated_evidence_fails_closed(tmp_path, fault, message):
    with pytest.raises(AssertionError, match=message):
        run_composition(tmp_path, fault)
    assert not (tmp_path / "complete.json").exists()


def test_zero_producer_hits_are_apc_not_exercised_not_unsupported(tmp_path):
    run_composition(tmp_path, "zero_p_hit")
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["status"] == "APC_NOT_EXERCISED"
    assert complete["apc_not_exercised"] == [
        "composition-p-warm-d-cold",
        "composition-both-warm",
        "composition-branch",
    ]
    assert complete["direct_pd_token_id_checks"] == 5


def test_zero_consumer_hit_is_apc_not_exercised(tmp_path):
    run_composition(tmp_path, "zero_d_hit")
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["status"] == "APC_NOT_EXERCISED"
    assert complete["apc_not_exercised"] == ["composition-both-warm"]
    evidence = json.loads(
        (tmp_path / "composition-both-warm-evidence.json").read_text()
    )
    assert evidence["consumer_cache_hit_required"] is True
    assert evidence["apc_not_exercised_roles"] == ["decode"]


def test_every_delayed_metric_scrape_is_retained(tmp_path):
    run_composition(tmp_path, "delayed_reference")
    snapshots = sorted(
        tmp_path.glob("composition-reference-base-after-*-prefill.metrics.txt")
    )
    assert len(snapshots) == 4
    assert 'finished_reason="length"} 0' in snapshots[0].read_text()
    assert 'finished_reason="length"} 1' in snapshots[-1].read_text()


@pytest.mark.parametrize("fault", ["missing_spec", "remapped_spec"])
def test_speculative_counters_are_evidence_not_certification(tmp_path, fault):
    run_composition(tmp_path, fault)
    evidence = json.loads((tmp_path / "composition-cold-evidence.json").read_text())
    assert evidence["speculative_certification"] is False
    spec = evidence["speculative_counters"]["decode"]
    if fault == "missing_spec":
        assert spec is None
    else:
        assert any("diffusion_num_canvas_positions" in key for key in spec)
