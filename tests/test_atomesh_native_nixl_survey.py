"""HTTP-boundary tests for the bounded native CPU -> NIXL survey client."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github/scripts/atomesh/pd_native_nixl_survey.py"
)


def load_client():
    spec = importlib.util.spec_from_file_location("native_nixl_survey", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Servers:
    def __init__(self):
        self.calls = []
        self.values = {}
        for role in ("P", "D"):
            values = {
                'vllm:request_success_total{finished_reason="length"}': 0,
                'vllm:request_success_total{finished_reason="error"}': 0,
                'vllm:request_success_total{finished_reason="abort"}': 0,
                "vllm:num_preemptions_total": 0,
                "vllm:nixl_bytes_transferred_count": 0,
                "vllm:nixl_bytes_transferred_sum": 0,
                "vllm:nixl_num_failed_transfers_total": 0,
                "vllm:nixl_num_failed_notifications_total": 0,
                "vllm:nixl_num_kv_expired_reqs_total": 0,
            }
            for source in ("local_compute", "local_cache_hit", "external_kv_transfer"):
                values[f'vllm:prompt_tokens_by_source_total{{source="{source}"}}'] = 0
            if role == "P":
                for direction in ("CPU_to_GPU", "GPU_to_CPU"):
                    values[
                        "vllm:kv_offload_total_bytes_total"
                        f'{{transfer_type="{direction}"}}'
                    ] = 0
            self.values[role] = values
        self.p_requests = 0
        self.transfer = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_engine_id": "producer-engine",
            "remote_host": "10.0.0.1",
            "remote_port": 15559,
            "remote_block_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9]],
            "opaque_future_key": {"keep": [17]},
        }

    def __call__(self, request):
        role = "P" if request.url.host == "prefill.test" else "D"
        body = json.loads(request.content) if request.content else None
        self.calls.append((role, request.url.path, str(request.url.query), body))
        values = self.values[role]
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(1100))})
        if request.url.path == "/metrics":
            return httpx.Response(
                200, text="\n".join(f"{k} {v}" for k, v in values.items())
            )
        if request.url.path == "/reset_prefix_cache":
            return httpx.Response(200, json={"success": True})
        assert request.url.path == "/v1/completions"
        assert body["return_token_ids"] is True
        assert body["prompt"] == list(range(1025))
        local, external = 0, 0
        if role == "P":
            self.p_requests += 1
            if self.p_requests in (1, 4):
                local = 1025
            else:
                local, external = 1, 1024
                values[
                    'vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}'
                ] += 117440512
            if self.p_requests == 1:
                values[
                    'vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"}'
                ] += 117440512
        else:
            external = 1025
            values["vllm:nixl_bytes_transferred_count"] += 1
            values["vllm:nixl_bytes_transferred_sum"] += 132120576
        values['vllm:request_success_total{finished_reason="length"}'] += 1
        values['vllm:prompt_tokens_by_source_total{source="local_compute"}'] += local
        values[
            'vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}'
        ] += external
        count = body["max_tokens"]
        response = {
            "id": request.headers["X-Request-Id"],
            "choices": [
                {
                    "text": "first" if count == 1 else "first continued output",
                    "token_ids": [42] if count == 1 else list(range(42, 74)),
                    "prompt_token_ids": body["prompt"],
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1025, "completion_tokens": count},
        }
        if body.get("kv_transfer_params", {}).get("do_remote_decode"):
            response["kv_transfer_params"] = self.transfer
        return httpx.Response(200, json=response)


def execute(tmp_path, servers):
    module = load_client()
    args = SimpleNamespace(
        prefill="http://prefill.test",
        decode="http://decode.test",
        model="Qwen3-0.6B",
        output=tmp_path,
    )

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(servers)) as client:
            return await module.run(
                args, client, poll_interval=0, settle_polls=1, max_polls=3
            )

    return asyncio.run(go())


def test_native_restore_then_immediate_opaque_nixl_handoff(tmp_path):
    servers = Servers()
    result = execute(tmp_path, servers)
    assert result["status"] == "PENDING_REVIEW"
    assert [phase["name"] for phase in result["phases"]] == [
        "cold_store",
        "cpu_restore",
        "cpu_restore_then_nixl",
        "no_load_control",
    ]
    assert all(phase["accounting_checked"] for phase in result["phases"])
    completions = [call for call in servers.calls if call[1] == "/v1/completions"]
    assert [call[0] for call in completions] == ["P", "P", "P", "D", "P"]
    assert completions[-1][3]["kv_transfer_params"] == {"max_load_tokens": 0}
    p_handoff = servers.calls.index(completions[2])
    assert servers.calls[p_handoff + 1] == completions[3]
    assert completions[3][3]["kv_transfer_params"] == servers.transfer
    resets = [call for call in servers.calls if call[1] == "/reset_prefix_cache"]
    assert len(resets) == 3
    assert all(call[0] == "P" for call in resets)
    assert all("reset_external=false" in call[2] for call in resets)
    assert all("reset_running_requests=false" in call[2] for call in resets)
    assert (
        json.loads((tmp_path / "result.json").read_text())["status"] == "PENDING_REVIEW"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "reset",
        "missing_bytes",
        "failed_transfer",
        "allocation",
        "ids",
        "text",
        "handoff",
        "counter_reset",
        "http",
        "negative_load",
    ],
)
def test_stops_on_invalid_evidence_without_followup_work(tmp_path, fault):
    servers = Servers()

    def faulty(request):
        if fault == "reset" and request.url.path == "/reset_prefix_cache":
            return httpx.Response(200, json={"success": False})
        response = servers(request)
        if request.url.path == "/v1/completions":
            if fault == "http":
                return httpx.Response(500, text="worker failed")
            if fault in ("ids", "text") and servers.p_requests == 2:
                data = response.json()
                if fault == "ids":
                    data["choices"][0]["token_ids"][-1] += 1
                else:
                    data["choices"][0]["text"] += " changed"
                return httpx.Response(200, json=data)
            if fault == "handoff" and servers.p_requests == 3:
                data = response.json()
                data["kv_transfer_params"]["remote_block_ids"] = []
                return httpx.Response(200, json=data)
            if fault == "negative_load" and servers.p_requests == 4:
                servers.values["P"][
                    'vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}'
                ] += 1
        if request.url.path == "/metrics" and servers.p_requests:
            if fault == "missing_bytes":
                return httpx.Response(
                    200,
                    text="\n".join(
                        line
                        for line in response.text.splitlines()
                        if "kv_offload_total_bytes_total" not in line
                    ),
                )
            if fault == "failed_transfer":
                return httpx.Response(
                    200,
                    text=response.text.replace(
                        "vllm:nixl_num_failed_transfers_total 0",
                        "vllm:nixl_num_failed_transfers_total 1",
                    ),
                )
            if fault == "allocation":
                return httpx.Response(
                    200,
                    text=response.text + "\nvllm:kv_offload_allocation_failure_total 1",
                )
            if fault == "counter_reset":
                return httpx.Response(
                    200,
                    text=response.text.replace(
                        "vllm:num_preemptions_total 0", "vllm:num_preemptions_total -1"
                    ),
                )
        return response

    with pytest.raises((RuntimeError, httpx.HTTPStatusError)):
        execute(tmp_path, faulty)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "FAIL"
    assert "error" in result
    assert "failure_metrics" in result
    assert not (tmp_path / "complete.json").exists()
    if fault in (
        "reset",
        "missing_bytes",
        "failed_transfer",
        "allocation",
        "counter_reset",
        "http",
    ):
        assert servers.p_requests == 1
    elif fault in ("ids", "text"):
        assert servers.p_requests == 2
    elif fault == "handoff":
        assert not any(
            call[0] == "D" and call[1] == "/v1/completions" for call in servers.calls
        )


def test_cancellation_marks_artifact_failed(tmp_path):
    servers = Servers()

    def cancelled(request):
        if request.url.path == "/v1/completions":
            raise asyncio.CancelledError()
        return servers(request)

    with pytest.raises(asyncio.CancelledError):
        execute(tmp_path, cancelled)
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "FAIL"


def test_rejects_degenerate_repeated_token_reference(tmp_path):
    servers = Servers()

    def degenerate(request):
        response = servers(request)
        if request.url.path == "/v1/completions":
            data = response.json()
            data["choices"][0]["token_ids"] = [198] * 32
            return httpx.Response(200, json=data)
        return response

    with pytest.raises(RuntimeError, match="Degenerate"):
        execute(tmp_path, degenerate)
    assert servers.p_requests == 1


@pytest.mark.parametrize("replacement", [False, 0.0, -1])
def test_rejects_invalid_prompt_echo_types_before_next_request(tmp_path, replacement):
    servers = Servers()

    def invalid_echo(request):
        response = servers(request)
        if request.url.path == "/v1/completions":
            data = response.json()
            data["choices"][0]["prompt_token_ids"][0] = replacement
            return httpx.Response(200, json=data)
        return response

    with pytest.raises(RuntimeError, match="Prompt IDs"):
        execute(tmp_path, invalid_echo)
    assert servers.p_requests == 1
