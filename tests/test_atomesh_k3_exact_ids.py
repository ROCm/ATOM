"""CPU-only checks of the K3 survey client's HTTP and evidence contract."""

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


def run_survey(tmp_path, mutate=None, mode="smoke"):
    spec = importlib.util.spec_from_file_location("k3_exact_ids", SCRIPT)
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    calls = []
    counters = {
        role: dict.fromkeys(
            (
                "local_compute",
                "local_cache_hit",
                "external_kv_transfer",
                "request_success",
            ),
            0,
        )
        for role in ("prefill", "decode")
    }
    handoff = {
        "remote_block_ids": [[1, 2], [3, 4]],
        "remote_host": "prefill",
        "opaque_extension": {"ticket": ["unchanged", 17]},
    }

    def respond(request):
        host, path = request.url.host, request.url.path
        body = json.loads(request.content) if request.content else None
        calls.append((host, path, body))
        if path == "/metrics":
            return httpx.Response(200, text=json.dumps(counters[host]))
        if path == "/tokenize":
            size = 4096 if mode == "smoke" else 40001
            return httpx.Response(200, json={"tokens": list(range(size))})
        if path == "/reset_prefix_cache":
            return httpx.Response(200, json={"success": True})
        if path in ("/start_profile", "/stop_profile"):
            return httpx.Response(200, json={})
        assert path == "/v1/completions"
        transfer = body.get("kv_transfer_params")
        role = "decode" if host == "decode" else "prefill" if transfer else "direct"
        count = body["max_tokens"]
        choice = {
            "text": "same decoded text",
            "finish_reason": "length",
            "prompt_token_ids": list(body["prompt"]),
            # P's handoff output is deliberately not D's final output.
            "token_ids": [999] if role == "prefill" else list(range(100, 100 + count)),
        }
        result = {"choices": [choice]}
        if role == "prefill":
            result["kv_transfer_params"] = handoff
        elif role == "decode":
            for key, value in handoff.items():
                assert transfer[key] == value
        c = counters[host]
        c["request_success"] += 1
        if role == "decode":
            c["external_kv_transfer"] += len(body["prompt"]) - 1
            c["local_compute"] += 1
        else:
            c["local_compute"] += len(body["prompt"]) - int(role == "prefill")
        if mutate:
            mutate(role, body, choice)
        return httpx.Response(200, json=result)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    args = argparse.Namespace(
        output=tmp_path,
        prefill="http://prefill",
        decode="http://decode",
        model="Kimi-K3",
        phase="benchmark",
        mode=mode,
        tp=8,
        dcp=8,
        hybrid=True,
    )
    with (
        patch.object(profile.httpx, "AsyncClient", return_value=client),
        patch.object(profile, "token_counters", side_effect=json.loads),
    ):
        asyncio.run(profile.run(args))
    return calls


def test_text_only_response_fails_closed(tmp_path):
    def omit_ids(role, body, choice):
        choice.pop("token_ids")
        choice.pop("prompt_token_ids")

    with pytest.raises(AssertionError, match="prompt_token_ids"):
        run_survey(tmp_path, omit_ids)
    assert not (tmp_path / "complete.json").exists()
    failure = json.loads(
        (tmp_path / "completion-validation-failures.jsonl").read_text()
    )
    assert failure["response"] == {
        "choices": [{"text": "same decoded text", "finish_reason": "length"}]
    }
    assert failure["request"]["max_tokens"] == 16
    assert failure["url"] == "http://prefill"


@pytest.mark.parametrize("mode", ["smoke", "profile"])
def test_identical_text_with_different_final_ids_fails(tmp_path, mode):
    def change_final_id(role, body, choice):
        if role == "decode":
            choice["token_ids"][-1] += 1

    with pytest.raises(AssertionError):
        run_survey(tmp_path, change_final_id, mode)
    assert not (tmp_path / "complete.json").exists()
    records = json.loads((tmp_path / "requests.json").read_text())
    assert records[-1]["direct"]["choices"][0]["token_ids"][-1] != (
        records[-1]["decode"]["choices"][0]["token_ids"][-1]
    )
    if mode == "smoke":
        evidence = json.loads((tmp_path / "correctness-127-evidence.json").read_text())
        assert evidence["status"] == "FAIL"
        assert evidence["direct_pd_text_equal"] is True
        assert evidence["direct_pd_token_ids_equal"] is False


@pytest.mark.parametrize("role", ["direct", "prefill", "decode"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_token_ids", None),
        ("prompt_token_ids", []),
        ("prompt_token_ids", "prompt"),
        ("token_ids", None),
        ("token_ids", []),
        ("token_ids", "tokens"),
    ],
)
def test_missing_or_malformed_ids_fail_closed(tmp_path, role, field, value):
    def corrupt(actual_role, body, choice):
        if actual_role == role:
            choice[field] = value

    with pytest.raises(AssertionError, match=field):
        run_survey(tmp_path, corrupt)
    assert not (tmp_path / "complete.json").exists()


@pytest.mark.parametrize("role", ["direct", "prefill", "decode"])
@pytest.mark.parametrize("field", ["prompt_token_ids", "token_ids"])
@pytest.mark.parametrize(
    "fault", ["bool", "float", "negative", "short", "long", "missing"]
)
def test_id_types_and_full_lengths_are_required(tmp_path, role, field, fault):
    def corrupt(actual_role, body, choice):
        if actual_role != role:
            return
        if fault == "bool":
            choice[field][0] = False
        elif fault == "float":
            choice[field][0] = float(choice[field][0])
        elif fault == "negative":
            choice[field][0] = -1
        elif fault == "short":
            choice[field].pop()
        elif fault == "long":
            choice[field].append(123)
        else:
            choice.pop(field)

    with pytest.raises(AssertionError, match=field):
        run_survey(tmp_path, corrupt)
    assert not (tmp_path / "complete.json").exists()
    failure = json.loads(
        (tmp_path / "completion-validation-failures.jsonl").read_text()
    )
    if fault == "negative":
        assert failure["response"]["choices"][0][field][0] == -1


@pytest.mark.parametrize("role", ["direct", "prefill", "decode"])
def test_same_length_wrong_prompt_echo_is_rejected(tmp_path, role):
    def corrupt(actual_role, body, choice):
        if actual_role == role:
            choice["prompt_token_ids"][-1] += 1

    with pytest.raises(AssertionError, match="prompt_token_ids"):
        run_survey(tmp_path, corrupt)


def test_smoke_still_requires_length_finish_reason(tmp_path):
    def stop_early(role, body, choice):
        if role == "decode":
            choice["finish_reason"] = "stop"

    with pytest.raises(AssertionError, match="Incomplete decode response"):
        run_survey(tmp_path, stop_early)
    evidence = json.loads((tmp_path / "correctness-127-evidence.json").read_text())
    assert evidence["status"] == "FAIL"


def test_smoke_keeps_handoff_metrics_and_unreferenced_warm_ids(tmp_path):
    decodes = 0

    def warm_ids(role, body, choice):
        nonlocal decodes
        if role == "decode":
            decodes += 1
            if decodes > 8:
                choice["token_ids"] = [777] * 16

    calls = run_survey(tmp_path, warm_ids)
    completions = [body for _, path, body in calls if path == "/v1/completions"]
    assert len(completions) == 28
    assert all(body["return_token_ids"] is True for body in completions)
    assert [
        len(body["prompt"]) for body in completions if "kv_transfer_params" not in body
    ] == [127, 128, 129, 1023, 1024, 1025, 2049, 2050]
    assert sum(path == "/reset_prefix_cache" for _, path, _ in calls) == 34
    assert not any(path in ("/start_profile", "/stop_profile") for _, path, _ in calls)
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["requests"] == 10
    assert complete["direct_pd_text_checks"] == 8
    assert complete["direct_pd_token_id_checks"] == 8
    assert complete["status"] == "PENDING_REVIEW"
    records = json.loads((tmp_path / "requests.json").read_text())
    for record in records:
        assert record["prefill"]["choices"][0]["token_ids"] == [999]
        assert len(record["decode"]["choices"][0]["token_ids"]) == 16
    evidence = json.loads((tmp_path / "correctness-1025-evidence.json").read_text())
    assert evidence["producer_effective_prompt_tokens_expected"] == 1024
    assert evidence["producer_effective_prompt_tokens_observed"] == 1024
    assert evidence["accounting_checked"] is True
    assert evidence["direct_pd_token_ids_equal"] is True
    assert len(records[5]["prefill"]["choices"][0]["prompt_token_ids"]) == 1025
    for record in records[-2:]:
        assert "direct" not in record
        assert record["decode"]["choices"][0]["token_ids"] == [777] * 16
        evidence = json.loads((tmp_path / f"{record['tag']}-evidence.json").read_text())
        assert "direct_pd_token_ids_equal" not in evidence


def test_profile_keeps_existing_workload_and_full_output_ids(tmp_path):
    calls = run_survey(tmp_path, mode="profile")
    completions = [body for _, path, body in calls if path == "/v1/completions"]
    assert len(completions) == 114
    assert all(body["return_token_ids"] is True for body in completions)
    assert sum(path == "/reset_prefix_cache" for _, path, _ in calls) == 28
    assert sum(path == "/start_profile" for _, path, _ in calls) == 4
    assert sum(path == "/stop_profile" for _, path, _ in calls) == 4
    records = json.loads((tmp_path / "requests.json").read_text())
    assert len(records) == 55
    assert [record["prompt_tokens"] for record in records[:4]] == [
        32767,
        32768,
        32769,
        20224,
    ]
    assert all("direct" in record for record in records[:4])
    assert all("direct" not in record for record in records[4:])
    for record in records[4:]:
        expected = 16 if record["tag"] == "profile-long-prefill" else 128
        assert len(record["decode"]["choices"][0]["token_ids"]) == expected
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["first_token_checks"] == 4
