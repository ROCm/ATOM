"""CPU HTTP-boundary tests for push concurrency and lease accounting."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github/scripts/atomesh/pd_nixl_push_survey.py"
)


def load_client():
    spec = importlib.util.spec_from_file_location("push_survey", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Servers:
    def __init__(self):
        self.calls = []
        self.values = {}
        self.pending = {}
        self.pumps = 0
        for role in ("P", "D"):
            self.values[role] = {
                'vllm:request_success_total{finished_reason="length"}': 0,
                'vllm:request_success_total{finished_reason="error"}': 0,
                'vllm:request_success_total{finished_reason="abort"}': 0,
                "vllm:num_preemptions_total": 0,
                "vllm:nixl_num_failed_transfers_total": 0,
                "vllm:nixl_num_failed_notifications_total": 0,
                "vllm:nixl_num_kv_expired_reqs_total": 0,
            }
            for source in ("local_compute", "local_cache_hit", "external_kv_transfer"):
                self.values[role][
                    f'vllm:prompt_tokens_by_source_total{{source="{source}"}}'
                ] = 0
        # Exercise lazy transfer series: initial absence must remain unknown.

    async def __call__(self, request):
        role = "P" if request.url.host == "prefill.test" else "D"
        body = json.loads(request.content) if request.content else None
        request_id = request.headers.get("X-Request-Id", "")
        self.calls.append((role, request.url.path, request_id, body))
        if request.url.path == "/tokenize":
            assert "Record" in body["prompt"]
            return httpx.Response(200, json={"tokens": list(range(1100))})
        if request.url.path == "/metrics":
            return httpx.Response(
                200, text="\n".join(f"{k} {v}" for k, v in self.values[role].items())
            )
        assert request.url.path == "/v1/completions"
        assert body["return_token_ids"] is True
        n = len(body["prompt"])
        params = body.get("kv_transfer_params")
        paired = params is not None and "orphan" not in request_id
        if paired:
            event = self.pending.setdefault(request_id, asyncio.Event())
            if role == "P":
                # This can only finish if D is launched before awaiting P.
                await asyncio.wait_for(event.wait(), timeout=1)
            else:
                assert "remote_block_ids" not in params
                assert params["remote_engine_id"] == "push-P-test"
                assert params["remote_host"] == "10.0.0.1"
                assert params["remote_port"] == 15559
                assert params["remote_request_id"] == request_id
                assert params["tp_size"] == params["pp_size"] == 1
                event.set()
            for metrics in self.values.values():
                metrics.setdefault("vllm:nixl_bytes_transferred_sum", 0)
                metrics.setdefault("vllm:nixl_bytes_transferred_count", 0)
            if role == "P":
                self.values[role]["vllm:nixl_bytes_transferred_sum"] += 132120576
                self.values[role]["vllm:nixl_bytes_transferred_count"] += 1
        if "pump" in request_id:
            assert role == "P" and params is None and n == 1
            self.pumps += 1
            if self.pumps == 2:
                self.values["P"]["vllm:nixl_num_kv_expired_reqs_total"] += 1
        self.values[role]['vllm:request_success_total{finished_reason="length"}'] += 1
        source = "local_compute" if role == "P" else "external_kv_transfer"
        self.values[role][
            f'vllm:prompt_tokens_by_source_total{{source="{source}"}}'
        ] += n
        count = body["max_tokens"]
        return httpx.Response(
            200,
            json={
                "id": "cmpl-" + request_id,
                "choices": [
                    {
                        "finish_reason": "length",
                        "prompt_token_ids": body["prompt"],
                        "token_ids": list(range(42, 42 + count)),
                        "text": "first" if count == 1 else "first continued output",
                    }
                ],
                "usage": {"prompt_tokens": n, "completion_tokens": count},
            },
        )


def execute(tmp_path, handler):
    module = load_client()
    args = SimpleNamespace(
        prefill="http://prefill.test",
        decode="http://decode.test",
        model="Qwen3-0.6B",
        output=tmp_path,
        prefill_engine_id="push-P-test",
        prefill_kv_host="10.0.0.1",
        prefill_side_channel_port=15559,
    )

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await module.run(
                args,
                client,
                poll_interval=0,
                settle_polls=1,
                max_polls=3,
                pump_interval=0,
                max_pumps=3,
            )

    return asyncio.run(go())


def test_concurrent_push_then_expiry_then_fresh_pair_with_raw_unknowns(tmp_path):
    servers = Servers()
    result = execute(tmp_path, servers)
    assert result["status"] == "PENDING_REVIEW"
    assert result["lifecycle_log_audit"] == "PENDING_INDEPENDENT_ID_MATCH"
    assert [p["name"] for p in result["phases"]] == [
        "direct_reference",
        "warmup_EXCLUDED",
        "healthy_before",
        "missing_consumer",
        "healthy_after",
    ]
    orphan = result["phases"][3]
    assert len(orphan["pumps"]) == 2
    assert orphan["consumer_request_submitted"] is False
    assert orphan["deltas"]["P"]["vllm:nixl_num_kv_expired_reqs_total"] == 1
    assert orphan["deltas"]["P"]["vllm:nixl_bytes_transferred_sum"] == 0
    assert (
        orphan["deltas"]["D"]['vllm:request_success_total{finished_reason="length"}']
        == 0
    )
    assert result["phases"][1]["deltas"]["P"]["vllm:nixl_bytes_transferred_sum"] is None
    for phase in (result["phases"][2], result["phases"][4]):
        assert phase["accounting_checked"] is True
        assert phase["deltas"]["P"]["vllm:nixl_bytes_transferred_count"] == 1
        assert phase["deltas"]["D"]["vllm:nixl_bytes_transferred_sum"] == 0
    assert result["phases"][2]["request_id"] != result["phases"][4]["request_id"]
    assert not any(
        role == "D" and rid == orphan["request_id"] for role, _, rid, _ in servers.calls
    )
    assert not any("reset" in path for _, path, _, _ in servers.calls)
    assert (
        json.loads((tmp_path / "result.json").read_text())["status"] == "PENDING_REVIEW"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "ids",
        "text",
        "prompt_bool",
        "degenerate",
        "missing_bytes",
        "failure",
        "reset",
        "http",
        "no_expiry",
        "orphan_write",
        "extra_expiry",
    ],
)
def test_stops_on_bad_response_or_incomplete_measured_accounting(tmp_path, fault):
    servers = Servers()

    async def faulty(request):
        response = await servers(request)
        rid = request.headers.get("X-Request-Id", "")
        if request.url.path == "/v1/completions":
            if fault == "http" and "healthy_before" in rid:
                return httpx.Response(500, text="worker failed")
            if (
                fault in ("ids", "text")
                and "healthy_before" in rid
                and request.url.host == "decode.test"
            ):
                data = response.json()
                if fault == "ids":
                    data["choices"][0]["token_ids"][-1] += 1
                else:
                    data["choices"][0]["text"] += " changed"
                return httpx.Response(200, json=data)
            if fault in ("prompt_bool", "degenerate") and "direct_reference" in rid:
                data = response.json()
                if fault == "prompt_bool":
                    data["choices"][0]["prompt_token_ids"][0] = False
                else:
                    data["choices"][0]["token_ids"] = [198] * 32
                return httpx.Response(200, json=data)
        if request.url.path == "/metrics":
            healthy = any("healthy_before" in call[2] for call in servers.calls)
            if fault == "missing_bytes" and healthy:
                return httpx.Response(
                    200,
                    text="\n".join(
                        line
                        for line in response.text.splitlines()
                        if "nixl_bytes_transferred" not in line
                    ),
                )
            if fault == "failure" and healthy:
                return httpx.Response(
                    200,
                    text=response.text.replace(
                        "vllm:nixl_num_failed_notifications_total 0",
                        "vllm:nixl_num_failed_notifications_total 1",
                    ),
                )
            if fault == "reset" and healthy:
                return httpx.Response(
                    200,
                    text=response.text.replace(
                        "vllm:num_preemptions_total 0", "vllm:num_preemptions_total -1"
                    ),
                )
            if fault == "no_expiry":
                return httpx.Response(
                    200,
                    text=response.text.replace(
                        "vllm:nixl_num_kv_expired_reqs_total 1",
                        "vllm:nixl_num_kv_expired_reqs_total 0",
                    ),
                )
            if fault == "extra_expiry" and servers.pumps >= 2:
                return httpx.Response(
                    200,
                    text=response.text.replace(
                        "vllm:nixl_num_kv_expired_reqs_total 1",
                        "vllm:nixl_num_kv_expired_reqs_total 2",
                    ),
                )
            if (
                fault == "orphan_write"
                and servers.pumps
                and request.url.host == "prefill.test"
            ):
                return httpx.Response(
                    200,
                    text=response.text.replace(
                        "vllm:nixl_bytes_transferred_count 2",
                        "vllm:nixl_bytes_transferred_count 3",
                    ),
                )
        return response

    with pytest.raises(RuntimeError):
        execute(tmp_path, faulty)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "FAIL"
    assert "failure_metrics" in result
    assert not any("healthy_after" in call[2] for call in servers.calls)
    assert servers.pumps <= 3


def test_cancellation_retains_failure_artifact(tmp_path):
    servers = Servers()

    async def cancelled(request):
        if request.url.path == "/v1/completions":
            raise asyncio.CancelledError()
        return await servers(request)

    with pytest.raises(asyncio.CancelledError):
        execute(tmp_path, cancelled)
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "FAIL"


def test_missing_measured_baseline_stops_before_first_healthy_pair(tmp_path):
    servers = Servers()

    async def missing(request):
        response = await servers(request)
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text="\n".join(
                    line
                    for line in response.text.splitlines()
                    if "nixl_bytes_transferred" not in line
                ),
            )
        return response

    with pytest.raises(RuntimeError, match="baseline"):
        execute(tmp_path, missing)
    assert not any("healthy_before" in call[2] for call in servers.calls)


@pytest.mark.parametrize("fault", ["producer_zero", "consumer_positive"])
def test_known_warmup_telemetry_cannot_be_excluded(tmp_path, fault):
    servers = Servers()
    for values in servers.values.values():
        values["vllm:nixl_bytes_transferred_sum"] = 0
        values["vllm:nixl_bytes_transferred_count"] = 0

    async def faulty(request):
        response = await servers(request)
        if request.url.path == "/metrics" and any(
            "warmup_EXCLUDED" in call[2] for call in servers.calls
        ):
            lines = []
            for line in response.text.splitlines():
                if line.startswith("vllm:nixl_bytes_transferred"):
                    if fault == "producer_zero" and request.url.host == "prefill.test":
                        line = line.rsplit(" ", 1)[0] + " 0"
                    elif (
                        fault == "consumer_positive"
                        and request.url.host == "decode.test"
                    ):
                        line = line.rsplit(" ", 1)[0] + " 1"
                lines.append(line)
            return httpx.Response(200, text="\n".join(lines))
        return response

    with pytest.raises(RuntimeError):
        execute(tmp_path, faulty)
    assert not any("healthy_before" in call[2] for call in servers.calls)
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "FAIL"
