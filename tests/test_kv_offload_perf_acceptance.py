"""CPU-only tests. Fake byte streams/session objects; no sockets or model imports.

HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest discover -s <this-directory> -p 'test_kv_offload_perf_acceptance.py' -v
"""

import asyncio
import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "tools" / "kv_offload_perf_acceptance.py"
SPEC = importlib.util.spec_from_file_location("kv_offload_perf_acceptance", TOOL)
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)


def event(value, newline="\n"):
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ("data: " + data + newline + newline).encode("utf-8")


def choice(delta=None, finish=None):
    return {
        "model": "DeepSeek-V4.1-Flash",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }


def usage(tokens=64, prompt=100):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": tokens,
        "total_tokens": prompt + tokens,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def good_chunks(tokens=64, prompt=100):
    # Two content events can represent 64 tokens. Character/event count is not
    # a token counter. A role delta and a usage-only tail MUST NOT affect TPOT.
    return [
        b": keepalive\n\n",
        event(choice({"role": "assistant", "content": ""})),
        event(choice({"content": "several tokens bundled here"})),
        b"",
        event(choice({"content": " and another multi-token event"})),
        event(choice(finish="length")),
        event({"choices": [], "usage": usage(tokens, prompt)}),
        event("[DONE]"),
    ]


class FakeClock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return float(self.value)


class FakeContent:
    def __init__(self, chunks, body=b""):
        self.chunks, self.body = chunks, body

    async def iter_any(self):
        for chunk in self.chunks:
            await asyncio.sleep(0)
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk

    async def read(self, n=-1):
        return self.body if n < 0 else self.body[:n]


class FakeResponse:
    def __init__(
        self, chunks=(), status=200, content_type="text/event-stream", body=b""
    ):
        self.status = status
        self.headers = {"Content-Type": content_type}
        self.content = FakeContent(chunks, body)
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def read(self):
        return self.body


class FakeSession:
    def __init__(self, response=None, arm="on", bad_measured=None):
        self.response, self.arm, self.bad_measured = response, arm, bad_measured
        self.calls, self.gets = [], []
        self.measured_number = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, url, data, **kwargs):
        body = json.loads(data)
        self.calls.append((url, body, kwargs))
        if self.response is not None:
            return self.response
        if body["max_tokens"] == 64:
            self.measured_number += 1
            if self.measured_number == self.bad_measured:
                return FakeResponse(status=503, body=b"fake unavailable")
        return FakeResponse(good_chunks(body["max_tokens"]))

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        after = len(self.gets) == 2
        ext = 60 * perf.REQUESTS if after and self.arm == "on" else 0
        text = metric_text(
            eq=100 * perf.REQUESTS if after else 0,
            eh=ext,
            lq=100 * perf.REQUESTS if after else 0,
            lh=0,
            external=self.arm == "on",
        )
        return FakeResponse(content_type="text/plain", body=text.encode())


def metric_text(
    eq=0,
    eh=0,
    lq=0,
    lh=0,
    external=True,
    actual_external=None,
    actual_local=None,
    actual_total=None,
):
    values = {"local_queries": lq, "local_hits": lh}
    if external:
        values.update(external_queries=eq, external_hits=eh)
    actual_external = eh if actual_external is None else actual_external
    actual_local = lh if actual_local is None else actual_local
    actual_total = (
        max(lq, actual_external + actual_local)
        if actual_total is None
        else actual_total
    )
    sources = {
        "external_kv_transfer": actual_external,
        "local_cache_hit": actual_local,
        "local_compute": actual_total - actual_external - actual_local,
    }
    return (
        "# fake Prometheus counters\n"
        + "\n".join(
            f'{perf.METRICS[key]}{{model_name="m",engine="0"}} {value}'
            for key, value in values.items()
        )
        + "\n"
        + "\n".join(
            f'{perf.SOURCE_METRIC}{{model_name="m",engine="0",source="{source}"}} {value}'
            for source, value in sources.items()
        )
        + "\n"
    )


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = perf.build_manifest(
            "graph-dspark-mp", 1, "DeepSeek-V4.1-Flash", namespace="cpu-only"
        )

    def test_fixed_recipe_unique_tails_and_short_prime(self):
        manifest = self.manifest
        self.assertEqual(len(manifest["documents"]), 12)
        self.assertEqual(len(manifest["prime"]), 12)
        self.assertEqual(len(manifest["measured"]), 240)
        self.assertEqual(len(set(manifest["documents"])), 12)
        for text in manifest["documents"]:
            self.assertEqual(text.count(perf.WORDS), 500)
        all_requests = manifest["prime"] + manifest["measured"]
        self.assertEqual(len({item["tail"] for item in all_requests}), 252)
        for request in all_requests:
            body = perf.payload(manifest, request)
            self.assertEqual(
                body["max_tokens"], 1 if request["phase"] == "prime" else 64
            )
            self.assertTrue(body["stream_options"]["include_usage"])
            self.assertTrue(body["ignore_eos"])
            self.assertEqual(body["seed"], 0)
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(
                perf.sha256(perf.canonical(body)), request["payload_sha256"]
            )

    def test_fresh_round_namespace_and_same_pair_payload(self):
        same = perf.build_manifest(
            "graph-dspark-mp", 1, "DeepSeek-V4.1-Flash", namespace="cpu-only"
        )
        fresh = perf.build_manifest(
            "graph-dspark-mp", 2, "DeepSeek-V4.1-Flash", namespace="cpu-only"
        )
        self.assertEqual(same, self.manifest)
        self.assertNotEqual(fresh["document_namespace"], same["document_namespace"])
        self.assertTrue(set(fresh["documents"]).isdisjoint(same["documents"]))
        self.assertNotEqual(
            perf.sha256(perf.canonical(same)), perf.sha256(perf.canonical(fresh))
        )

    def test_default_uuid_is_new(self):
        left = perf.build_manifest("x", 1, "model", 16)
        right = perf.build_manifest("x", 1, "model", 16)
        self.assertNotEqual(left["document_namespace"], right["document_namespace"])
        self.assertEqual(left["concurrency"], 16)

    def test_tampered_manifest_and_overwrite_fail(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "manifest.json"
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                args = [
                    "manifest",
                    "--case",
                    "cpu",
                    "--round",
                    "1",
                    "--namespace",
                    "unit",
                    "--output",
                    str(path),
                ]
                self.assertEqual(perf.main(args), 0)
                self.assertNotEqual(perf.main(args), 0)
            loaded, raw, digest = perf.load_manifest(path)
            self.assertEqual(digest, perf.sha256(raw))
            loaded["measured"][0]["tail"] = "tampered"
            path.write_bytes(perf.canonical(loaded))
            with self.assertRaises(perf.AcceptanceError):
                perf.load_manifest(path)

    def test_invalid_case_round_and_concurrency(self):
        for args in (
            ("../bad", 1, "m", 24),
            ("ok", 0, "m", 24),
            ("ok", 1, "m", 0),
            ("ok", 1, "m", 241),
        ):
            with self.subTest(args=args), self.assertRaises(perf.AcceptanceError):
                perf.build_manifest(*args)


class SSETests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = perf.build_manifest(
            "sse", 1, "DeepSeek-V4.1-Flash", namespace="cpu-only"
        )

    async def consume(self, chunks=None, **kwargs):
        records = []
        response = FakeResponse(good_chunks() if chunks is None else chunks, **kwargs)
        result = await perf.stream_request(
            FakeSession(response),
            "http://fake.invalid/v1/chat/completions",
            self.manifest,
            self.manifest["measured"][0],
            records.append,
            FakeClock(),
        )
        self.assertEqual(records, [result])
        return result

    async def test_usage_tail_empty_chunk_and_multi_token_event_timing(self):
        result = await self.consume()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["usage"], usage())
        self.assertEqual(result["content_event_count"], 2)
        self.assertEqual(result["ttft_s"], 3)
        self.assertAlmostEqual(result["tpot_s"], (5 - 3) / 63)
        self.assertEqual(result["last_content_s"], 5)
        self.assertEqual(result["e2e_s"], 9)
        events = [item for item in result["events"] if "content_characters" in item]
        self.assertTrue(all(item["token_count"] is None for item in events))
        self.assertNotEqual(result["tpot_s"], 2)  # not inter-chunk gap

    async def test_bytewise_utf8_crlf_and_multiline_sse(self):
        raw = (
            b"\xef\xbb\xbf: comment\r\n\r\n"
            + event(choice({"content": "你好"}), "\r\n")
            + event(choice(finish="length"), "\r\n")
        )
        raw += (
            b'data: {"choices": [],\r\ndata: "usage": '
            + json.dumps(usage()).encode()
            + b"}\r\n\r\n"
        )
        raw += event("[DONE]", "\r\n")
        result = await self.consume([bytes([byte]) for byte in raw])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["content_event_count"], 1)
        self.assertEqual(result["tpot_s"], 0)

    async def test_reasoning_content_counts_as_output(self):
        chunks = good_chunks()
        chunks[2] = event(choice({"reasoning_content": "thinking"}))
        result = await self.consume(chunks)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["first_content_s"], 3)

    async def test_missing_done(self):
        result = await self.consume(good_chunks()[:-1])
        self.assertEqual(result["status"], "failed")
        self.assertIn("without [DONE]", result["error"])

    async def test_missing_usage(self):
        result = await self.consume(good_chunks()[:-2] + [event("[DONE]")])
        self.assertEqual(result["status"], "failed")
        self.assertIn("usage", result["error"])

    async def test_wrong_token_count_and_inconsistent_total(self):
        for tokens in (0, 1, 63, 65):
            with self.subTest(tokens=tokens):
                result = await self.consume(good_chunks(tokens))
                self.assertEqual(result["status"], "failed")
                self.assertIn("completion_tokens", result["error"])
        chunks = good_chunks()
        bad = usage()
        bad["total_tokens"] += 1
        chunks[-2] = event({"usage": bad, "choices": []})
        result = await self.consume(chunks)
        self.assertIn("does not equal", result["error"])

    async def test_http_failure_and_wrong_content_type(self):
        result = await self.consume([], status=503, body=b"not ready")
        self.assertEqual(result["status"], "failed")
        self.assertIn("HTTP 503: not ready", result["error"])
        result = await self.consume(content_type="application/json")
        self.assertEqual(result["status"], "failed")
        self.assertIn("Content-Type", result["error"])

    async def test_truncated_json_error_event_and_timeout(self):
        for chunks in (
            [b"data: {broken}\n\n"],
            [event({"error": {"message": "bad"}})],
            [b"event: error\ndata: failed\n\n"],
            [asyncio.TimeoutError("deadline")],
        ):
            with self.subTest(chunks=chunks):
                result = await self.consume(chunks)
                self.assertEqual(result["status"], "failed")

    async def test_cancelled_request_is_recorded_and_propagated(self):
        records = []
        with self.assertRaises(asyncio.CancelledError):
            await perf.stream_request(
                FakeSession(FakeResponse([asyncio.CancelledError()])),
                "http://fake.invalid",
                self.manifest,
                self.manifest["measured"][0],
                records.append,
                FakeClock(),
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "cancelled")
        with self.assertRaises(perf.AcceptanceError):
            perf.summarize_performance(records)

    async def test_missing_finish_and_no_content_fail(self):
        chunks = good_chunks()
        del chunks[5]
        self.assertIn("finish_reason", (await self.consume(chunks))["error"])
        chunks = [
            event(choice(finish="length")),
            event({"usage": usage(), "choices": []}),
            event("[DONE]"),
        ]
        self.assertIn("no nonempty", (await self.consume(chunks))["error"])

    async def test_conflicting_usage_and_post_done_event_fail(self):
        chunks = good_chunks()[:-1] + [
            event({"usage": usage(prompt=101), "choices": []}),
            event("[DONE]"),
        ]
        self.assertIn("conflicting usage", (await self.consume(chunks))["error"])
        chunks = good_chunks()[:-1] + [
            event("[DONE]") + event(choice({"content": "late"}))
        ]
        self.assertIn("after [DONE]", (await self.consume(chunks))["error"])

    async def test_post_done_rejected_independently_of_tcp_chunks(self):
        raw = b"".join(good_chunks()) + event(choice({"content": "late"}))
        for chunks in (
            [raw],
            good_chunks() + [event(choice({"content": "late"}))],
            [bytes([value]) for value in raw],
        ):
            with self.subTest(chunks=len(chunks)):
                result = await self.consume(chunks)
                self.assertEqual(result["status"], "failed")
                self.assertIn("after [DONE]", result["error"])

    async def test_content_after_finish_rejected_but_same_event_allowed(self):
        chunks = good_chunks()
        chunks.insert(-2, event(choice({"content": "late"})))
        self.assertIn(
            "after terminal finish_reason", (await self.consume(chunks))["error"]
        )
        chunks = [
            event(choice({"content": "last token"}, "length")),
            event({"usage": usage(), "choices": []}),
            event("[DONE]"),
        ]
        self.assertEqual((await self.consume(chunks))["status"], "passed")

    async def test_reasoning_field_only_has_content_timing(self):
        chunks = [
            event(choice({"reasoning": "thinking"})),
            event(choice({"reasoning": "more thinking"}, "length")),
            event({"usage": usage(), "choices": []}),
            event("[DONE]"),
        ]
        result = await self.consume(chunks)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["content_event_count"], 2)
        self.assertEqual(result["ttft_s"], 1)
        self.assertAlmostEqual(result["tpot_s"], 1 / 63)

    async def test_prime_control_token_needs_no_visible_content(self):
        chunks = [
            event(choice(finish="length")),
            event({"usage": usage(1), "choices": []}),
            event("[DONE]"),
        ]
        result = await perf.stream_request(
            FakeSession(FakeResponse(chunks)),
            "http://fake.invalid",
            self.manifest,
            self.manifest["prime"][0],
            lambda record: None,
            FakeClock(),
        )
        self.assertEqual(result["status"], "passed")
        self.assertIsNone(result["ttft_s"])
        self.assertIsNone(result["tpot_s"])
        self.assertIsNone(result["first_content_s"])

    async def test_incomplete_trailing_frame_and_invalid_utf8_fail(self):
        for suffix in (b"data: unfinished", b"data: unfinished\n", b"\xff"):
            with self.subTest(suffix=suffix):
                self.assertEqual(
                    (await self.consume(good_chunks() + [suffix]))["status"], "failed"
                )

    async def test_response_byte_and_event_limits_fail_closed(self):
        with mock.patch.object(perf, "MAX_RESPONSE_BYTES", 8):
            result = await self.consume()
            self.assertIn("byte limit", result["error"])
        with mock.patch.object(perf, "MAX_RESPONSE_EVENTS", 2):
            result = await self.consume()
            self.assertIn("event limit", result["error"])
            self.assertEqual(len(result["events"]), 2)

    async def test_prime_is_one_token_tpot_unavailable(self):
        records = []
        result = await perf.stream_request(
            FakeSession(FakeResponse(good_chunks(1))),
            "http://fake.invalid",
            self.manifest,
            self.manifest["prime"][0],
            records.append,
            FakeClock(),
        )
        self.assertEqual(result["status"], "passed")
        self.assertIsNone(result["tpot_s"])


class MetricsTests(unittest.TestCase):
    def test_multiple_labels_sum_and_timestamp_not_value(self):
        text = (
            "# TYPE vllm:external_prefix_cache_hits_total counter\n"
            'vllm:external_prefix_cache_hits_total{rank="0",model_name="a b"} 12 1700000000\n'
            'vllm:external_prefix_cache_hits_total{model_name="a b",rank="1"} 18 1700000001\n'
            'vllm:external_prefix_cache_hits_total_created{rank="0"} 999\n'
        )
        parsed = perf.parse_metrics(text)
        self.assertEqual(sum(parsed["external_hits"].values()), 30)
        self.assertEqual(len(parsed["external_hits"]), 2)
        self.assertEqual(parsed["local_hits"], {})

    def test_labels_order_is_not_series_identity_and_escape(self):
        before = metric_text().replace(
            'model_name="m",engine="0"', 'model_name="a\\"b",engine="0"'
        )
        after = metric_text(eq=80, eh=60, lq=100).replace(
            'model_name="m",engine="0"', 'engine="0",model_name="a\\"b"'
        )
        result = perf.assess_supply(before, after, "on", 100)
        self.assertTrue(result["gates_passed"])
        self.assertEqual(result["tier_fraction_of_actual_prompt"], 0.6)
        self.assertEqual(result["recompute_fraction_of_actual_prompt"], 0.4)

    def test_on_gate_strictly_over_half_actual_prompt(self):
        before = metric_text()
        at_boundary = perf.assess_supply(
            before, metric_text(eq=60, eh=50, lq=100), "on", 100
        )
        self.assertFalse(at_boundary["gates_passed"])
        passed = perf.assess_supply(
            before, metric_text(eq=60, eh=51, lq=100), "on", 100
        )
        self.assertTrue(passed["gates_passed"])
        self.assertEqual(passed["tier_fraction_of_actual_prompt"], 0.51)  # NOT 51/60

    def test_missing_external_off_allowed_unknown_not_zero(self):
        result = perf.assess_supply(
            metric_text(external=False), metric_text(lq=100, external=False), "off", 100
        )
        self.assertTrue(result["gates_passed"])
        self.assertTrue(result["external_metrics_absent"])
        self.assertIsNone(result["counter_deltas"]["external_hits"])
        self.assertIsNone(result["tier_fraction_of_actual_prompt"])
        self.assertIsNone(result["recompute_fraction_of_actual_prompt"])
        self.assertEqual(result["local_fraction_of_actual_prompt"], 0)
        self.assertIn("NOT zero", result["external_note"])

    def test_missing_external_on_or_local_off_fails(self):
        with self.assertRaises(perf.AcceptanceError):
            perf.assess_supply(
                metric_text(external=False), metric_text(external=False), "on", 100
            )
        with self.assertRaises(perf.AcceptanceError):
            perf.assess_supply("", "", "off", 100)
        with self.assertRaises(perf.AcceptanceError):
            perf.assess_supply(metric_text(), metric_text(external=False), "off", 100)
        partial = metric_text() + "\n"
        partial = "\n".join(
            line
            for line in partial.splitlines()
            if "external_prefix_cache_queries" not in line
        )
        with self.assertRaises(perf.AcceptanceError):
            perf.assess_supply(partial, partial, "off", 100)

    def test_off_nonzero_local_or_external_fails(self):
        for after in (metric_text(lq=100, lh=1), metric_text(eq=100, eh=1, lq=100)):
            self.assertFalse(
                perf.assess_supply(metric_text(), after, "off", 100)["gates_passed"]
            )

    def test_fractions_cannot_exceed_one_and_hits_cannot_exceed_queries(self):
        for after in (
            metric_text(eq=110, eh=101, lq=100),
            metric_text(eq=90, eh=80, lq=100, lh=30),
            metric_text(eq=20, eh=60, lq=100),
        ):
            self.assertFalse(
                perf.assess_supply(metric_text(), after, "on", 100)["gates_passed"]
            )

    def test_missing_metric_counter_reset_label_churn_and_nan_fail(self):
        before = metric_text(eq=10, eh=5, lq=10)
        for after in (
            metric_text(eq=9, eh=6, lq=11),
            metric_text(eq=20, eh=12, lq=20).replace('engine="0"', 'engine="1"'),
            metric_text().replace("} 0", "} NaN", 1),
        ):
            with self.subTest(after=after), self.assertRaises(perf.AcceptanceError):
                perf.metric_deltas(before, after, "on")
        with self.assertRaises(perf.AcceptanceError):
            perf.parse_metrics(
                metric_text()
                + '\nvllm:prefix_cache_hits_total{model_name="m",engine="0"} 0'
            )

    def test_multilabel_individual_reset_not_hidden_by_sum(self):
        before = (
            metric_text()
            + '\nvllm:external_prefix_cache_hits_total{model_name="m",engine="1"} 5'
        )
        after = (
            metric_text(eq=100, eh=20)
            + '\nvllm:external_prefix_cache_hits_total{model_name="m",engine="1"} 4'
        )
        with self.assertRaises(perf.AcceptanceError):
            perf.metric_deltas(before, after, "on")

    def test_on_zero_local_hbm_passes(self):
        result = perf.assess_supply(
            metric_text(),
            metric_text(eq=100, eh=60, lq=100, lh=0, actual_local=0),
            "on",
            100,
        )
        self.assertTrue(result["gates_passed"])
        self.assertEqual(result["counter_deltas"]["local_hits"], 0)
        self.assertEqual(result["counter_deltas"]["source_local"], 0)

    def test_on_local_lookup_hits_alone_fail(self):
        # All original gates pass: external lookup/transfer=60%, source sum=100.
        # A positive local lookup delta still invalidates the no-HBM recipe.
        result = perf.assess_supply(
            metric_text(),
            metric_text(eq=100, eh=60, lq=100, lh=1, actual_local=0),
            "on",
            100,
        )
        self.assertFalse(result["gates_passed"])
        self.assertEqual(
            result["errors"],
            [
                "ON requires local HBM lookup hits=0 and actual local_cache_hit tokens=0 for this recipe"
            ],
        )

    def test_on_actual_local_source_alone_fails(self):
        # Lookup zero is not enough: actual local reuse independently fails.
        result = perf.assess_supply(
            metric_text(),
            metric_text(eq=100, eh=60, lq=100, lh=0, actual_local=1),
            "on",
            100,
        )
        self.assertFalse(result["gates_passed"])
        self.assertEqual(
            result["errors"],
            [
                "ON requires local HBM lookup hits=0 and actual local_cache_hit tokens=0 for this recipe"
            ],
        )

    def test_on_local_lookup_and_source_both_nonzero_fail(self):
        result = perf.assess_supply(
            metric_text(),
            metric_text(eq=100, eh=60, lq=100, lh=10, actual_local=10),
            "on",
            100,
        )
        self.assertFalse(result["gates_passed"])
        self.assertEqual(result["tier_fraction_of_actual_prompt"], 0.6)
        self.assertEqual(result["actual_transfer_fraction_of_prompt"], 0.6)
        self.assertEqual(result["actual_source_token_sum"], 100)
        self.assertEqual(
            result["errors"],
            [
                "ON requires local HBM lookup hits=0 and actual local_cache_hit tokens=0 for this recipe"
            ],
        )

    def test_actual_retrieval_required_not_lookup_hits(self):
        result = perf.assess_supply(
            metric_text(),
            metric_text(eq=100, eh=80, lq=100, actual_external=0),
            "on",
            100,
        )
        self.assertFalse(result["gates_passed"])
        self.assertEqual(result["tier_fraction_of_actual_prompt"], 0.8)
        self.assertEqual(result["actual_transfer_fraction_of_prompt"], 0)
        self.assertTrue(any("lookup hits alone" in error for error in result["errors"]))

    def test_missing_source_metric_and_accounting_mismatch_fail(self):
        no_sources = "\n".join(
            line
            for line in metric_text().splitlines()
            if perf.SOURCE_METRIC not in line
        )
        with self.assertRaises(perf.AcceptanceError):
            perf.assess_supply(no_sources, no_sources, "off", 100)
        result = perf.assess_supply(
            metric_text(),
            metric_text(eq=100, eh=60, lq=100, actual_total=101),
            "on",
            100,
        )
        self.assertFalse(result["gates_passed"])
        self.assertTrue(any("deltas sum" in error for error in result["errors"]))

    def test_source_multi_label_sum_and_off_real_transfer_fail(self):
        extra = '\nvllm:prompt_tokens_by_source_total{source="external_kv_transfer",engine="1"} 20'
        parsed = perf.parse_metrics(metric_text(eq=100, eh=60, lq=100) + extra)
        self.assertEqual(sum(parsed["source_external"].values()), 80)
        result = perf.assess_supply(
            metric_text(external=False),
            metric_text(lq=100, external=False, actual_external=1),
            "off",
            100,
        )
        self.assertFalse(result["gates_passed"])


class SummaryTests(unittest.TestCase):
    def records(self):
        return [
            {
                "request_id": f"measured-{i:06d}",
                "status": "passed",
                "usage": usage(),
                "started_monotonic_s": 10 + i,
                "completed_monotonic_s": 12 + i,
                "ttft_s": 0.1 + i / 1000,
                "tpot_s": 0.01,
                "e2e_s": 2,
                "content_event_count": 2,
            }
            for i in range(240)
        ]

    def test_window_throughput_percentiles_and_complete_count(self):
        records = self.records()
        summary = perf.summarize_performance(records)
        self.assertEqual(summary["window_s"], 241)
        self.assertEqual(summary["completion_tokens"], 240 * 64)
        self.assertAlmostEqual(summary["total_tok_per_s"], (240 * 164) / 241)
        self.assertAlmostEqual(summary["output_tok_per_s"], (240 * 64) / 241)
        self.assertEqual(
            summary["latency_ms"]["tpot"], {"p50": 10, "p90": 10, "p99": 10}
        )
        self.assertAlmostEqual(summary["latency_ms"]["ttft"]["p50"], 219.5)
        for bad in (records[:-1], records + records[:1]):
            with self.assertRaises(perf.AcceptanceError):
                perf.summarize_performance(bad)
        records[0]["status"] = "failed"
        with self.assertRaises(perf.AcceptanceError):
            perf.summarize_performance(records)

    def pair(self):
        records = self.records()
        on = {
            "status": "passed",
            "arm": "on",
            "validator_version": perf.VALIDATOR_VERSION,
            "errors": [],
            "manifest_sha256": "same",
            "case": "cpu",
            "round": 1,
            "concurrency": 24,
            "document_namespace": "same",
            "window_definition": perf.WINDOW,
            "prompt_tokens_by_request": {
                r["request_id"]: r["usage"]["prompt_tokens"] for r in records
            },
            "performance": perf.summarize_performance(records),
            "supply": perf.assess_supply(
                metric_text(), metric_text(eq=24000, eh=14400, lq=24000), "on", 24000
            ),
        }
        off = copy.deepcopy(on)
        off["arm"] = "off"
        off["supply"] = perf.assess_supply(
            metric_text(), metric_text(lq=24000), "off", 24000
        )
        return on, off

    def test_pair_requires_same_sha_and_actual_prompt_counts(self):
        on, off = self.pair()
        self.assertEqual(
            perf.compare_summaries(on, off)["on_over_off"]["output_tok_per_s"], 1
        )
        for key, value in (
            ("manifest_sha256", "other"),
            ("status", "failed"),
            ("prompt_tokens_by_request", {"r": 101}),
            ("round", 2),
        ):
            bad = copy.deepcopy(off)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(perf.AcceptanceError):
                perf.compare_summaries(on, bad)

    def test_legacy_or_forged_pass_cannot_bypass_current_supply_gate(self):
        on, off = self.pair()
        for version in (None, 1):
            bad = copy.deepcopy(on)
            bad["validator_version"] = version
            with (
                self.subTest(version=version),
                self.assertRaisesRegex(perf.AcceptanceError, "validator version"),
            ):
                perf.compare_summaries(bad, off)
        bad = copy.deepcopy(on)
        supply = perf.assess_supply(
            metric_text(),
            metric_text(eq=24000, eh=14400, lq=24000, lh=240),
            "on",
            24000,
        )
        self.assertFalse(supply["gates_passed"])
        supply["gates_passed"], supply["errors"] = True, []
        bad["supply"] = supply
        with self.assertRaisesRegex(perf.AcceptanceError, "current validation"):
            perf.compare_summaries(bad, off)

    def test_comparison_rejects_missing_and_nonfinite_counter_deltas(self):
        on, off = self.pair()
        for key in perf.ALL_METRICS:
            for invalid in (None, -1, float("nan"), float("inf")):
                bad = copy.deepcopy(on)
                bad["supply"]["counter_deltas"][key] = invalid
                with (
                    self.subTest(key=key, invalid=invalid),
                    self.assertRaises(perf.AcceptanceError),
                ):
                    perf.compare_summaries(bad, off)

    def test_comparison_rechecks_derived_fields_and_request_denominator(self):
        on, off = self.pair()
        bad = copy.deepcopy(on)
        bad["supply"]["actual_transfer_fraction_of_prompt"] = 0.99
        with self.assertRaises(perf.AcceptanceError):
            perf.compare_summaries(bad, off)
        on["prompt_tokens_by_request"] = off["prompt_tokens_by_request"] = {"r": 100}
        with self.assertRaisesRegex(perf.AcceptanceError, "240 measured request IDs"):
            perf.compare_summaries(on, off)


class FakeEndToEndTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = perf.build_manifest(
            "fake-integration", 1, "DeepSeek-V4.1-Flash", 16, "cpu-only"
        )

    async def run_fake(self, directory, arm, fake, extra_args=()):
        path = Path(directory) / "manifest.json"
        if not path.exists():
            path.write_bytes(perf.canonical(self.manifest) + b"\n")
        args = perf.make_parser().parse_args(
            [
                "run",
                "--manifest",
                str(path),
                "--arm",
                arm,
                "--base-url",
                "http://fake.invalid:8123",
                "--output-root",
                str(Path(directory) / "results"),
                *extra_args,
            ]
        )
        # The entire aiohttp factory is replaced. No real sockets are opened.
        with (
            mock.patch("aiohttp.ClientSession", return_value=fake),
            mock.patch("aiohttp.TCPConnector", return_value=object()),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = await perf.run_arm(args)
        root = (
            Path(directory)
            / "results"
            / "fake-integration"
            / "round-0001"
            / "conc-16"
            / arm
        )
        return result, root, json.loads((root / "summary.json").read_text())

    async def test_fake_full_pair_serial_prime_fixed_count_raw_metrics_and_sha(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            on_session, off_session = FakeSession(arm="on"), FakeSession(arm="off")
            on_rc, on_root, on = await self.run_fake(directory, "on", on_session)
            off_rc, off_root, off = await self.run_fake(directory, "off", off_session)
            self.assertEqual((on_rc, off_rc), (0, 0))
            self.assertEqual(perf.compare_summaries(on, off)["status"], "passed")
            self.assertEqual(on["counts"]["recorded_prime"], 12)
            self.assertEqual(on["counts"]["passed_measured"], 240)
            self.assertEqual(on["performance"]["completion_tokens"], 15360)
            self.assertEqual(on["supply"]["tier_fraction_of_actual_prompt"], 0.6)
            self.assertIsNone(off["supply"]["tier_fraction_of_actual_prompt"])
            self.assertEqual(on["manifest_sha256"], off["manifest_sha256"])
            self.assertEqual(
                on_root.joinpath("manifest.json").read_bytes(),
                off_root.joinpath("manifest.json").read_bytes(),
            )
            self.assertEqual(
                [body for _, body, _ in on_session.calls],
                [body for _, body, _ in off_session.calls],
            )
            for session, root in ((on_session, on_root), (off_session, off_root)):
                self.assertEqual(len(session.calls), 252)
                self.assertEqual(
                    [body["max_tokens"] for _, body, _ in session.calls],
                    [1] * 12 + [64] * 240,
                )
                self.assertEqual(len(session.gets), 2)
                rows = [
                    json.loads(line)
                    for line in root.joinpath("requests.jsonl").read_text().splitlines()
                ]
                prime = [row for row in rows if row["phase"] == "prime"]
                self.assertTrue(
                    all(
                        prime[i]["completed_monotonic_s"]
                        <= prime[i + 1]["started_monotonic_s"]
                        for i in range(11)
                    )
                )
                self.assertLessEqual(
                    prime[-1]["completed_monotonic_s"],
                    (
                        on["performance"]["first_dispatch_monotonic_s"]
                        if session is on_session
                        else off["performance"]["first_dispatch_monotonic_s"]
                    ),
                )
                for suffix in ("before", "after"):
                    raw = root.joinpath(f"metrics_{suffix}.prom").read_bytes()
                    metadata = json.loads(
                        root.joinpath(f"metrics_{suffix}.json").read_text()
                    )
                    self.assertEqual(perf.sha256(raw), metadata["sha256"])
                self.assertTrue(
                    all("prompt_tokens_details" in row["usage"] for row in rows)
                )

    async def test_on_hbm_reuse_nonzero_exit_even_with_valid_external_supply(self):
        class HBMHitSession(FakeSession):
            def get(self, url, **kwargs):
                self.gets.append((url, kwargs))
                after = len(self.gets) == 2
                text = metric_text(
                    eq=100 * perf.REQUESTS if after else 0,
                    eh=60 * perf.REQUESTS if after else 0,
                    lq=100 * perf.REQUESTS if after else 0,
                    lh=perf.REQUESTS if after else 0,
                )
                return FakeResponse(content_type="text/plain", body=text.encode())

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            rc, _root, summary = await self.run_fake(directory, "on", HBMHitSession())
            self.assertNotEqual(rc, 0)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["counts"]["passed_measured"], 240)
            self.assertEqual(
                summary["supply"]["actual_transfer_fraction_of_prompt"], 0.6
            )
            self.assertFalse(summary["supply"]["gates_passed"])
            self.assertTrue(
                any("ON requires local HBM" in error for error in summary["errors"])
            )

    async def test_one_http_failure_nonzero_no_partial_performance(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            rc, root, summary = await self.run_fake(
                directory, "on", FakeSession(bad_measured=3)
            )
            self.assertNotEqual(rc, 0)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["counts"]["recorded_measured"], 240)
            self.assertEqual(summary["counts"]["failed_measured"], 1)
            self.assertIsNone(summary["performance"])
            self.assertTrue(root.joinpath("metrics_after.prom").is_file())

    async def test_round_deadline_cancels_records_and_preserves_after_metrics(self):
        class HungContent(FakeContent):
            async def iter_any(self):
                await asyncio.Future()
                yield b""  # pragma: no cover

        response = FakeResponse()
        response.content = HungContent([])
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            rc, root, summary = await self.run_fake(
                directory,
                "on",
                FakeSession(response=response),
                ["--round-timeout", "0.05", "--metrics-timeout", "0.01"],
            )
            self.assertNotEqual(rc, 0)
            self.assertTrue(any("deadline" in error for error in summary["errors"]))
            rows = [
                json.loads(line)
                for line in root.joinpath("requests.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[0]["status"], "cancelled")
            self.assertTrue(root.joinpath("metrics_after.prom").exists())

    async def test_final_completion_after_work_deadline_cannot_pass(self):
        now = [100.0]

        async def timer_delayed_by_ready_task(awaitable, timeout):
            result = await awaitable
            now[0] = 100.8
            return result

        with (
            tempfile.TemporaryDirectory(dir=HERE) as directory,
            mock.patch.object(perf, "time", mock.Mock(perf_counter=lambda: now[0])),
            mock.patch.object(
                perf.asyncio, "wait_for", side_effect=timer_delayed_by_ready_task
            ),
        ):
            rc, root, summary = await self.run_fake(
                directory,
                "on",
                FakeSession(),
                ["--round-timeout", "1", "--metrics-timeout", "0.25"],
            )
            self.assertNotEqual(rc, 0)
            self.assertEqual(summary["counts"]["passed_measured"], 240)
            self.assertTrue(any("work budget" in e for e in summary["errors"]))
            self.assertTrue(root.joinpath("metrics_after.prom").exists())

    async def test_final_scrape_crossing_round_deadline_cannot_pass(self):
        now = [100.0]
        original_scrape = perf.scrape_metrics

        async def delayed_scrape(session, url, output_dir, name, timeout):
            result = await original_scrape(session, url, output_dir, name, timeout)
            if name == "metrics_after":
                now[0] = 101.1
            return result

        with (
            tempfile.TemporaryDirectory(dir=HERE) as directory,
            mock.patch.object(perf, "time", mock.Mock(perf_counter=lambda: now[0])),
            mock.patch.object(perf, "scrape_metrics", side_effect=delayed_scrape),
        ):
            rc, _root, summary = await self.run_fake(
                directory,
                "on",
                FakeSession(),
                ["--round-timeout", "1", "--metrics-timeout", "0.25"],
            )
            self.assertNotEqual(rc, 0)
            self.assertEqual(summary["counts"]["passed_measured"], 240)
            self.assertTrue(
                any("including final metrics" in e for e in summary["errors"])
            )

    async def test_on_missing_metrics_refuses_before_window(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            session = FakeSession(arm="off")
            rc, root, summary = await self.run_fake(directory, "on", session)
            self.assertNotEqual(rc, 0)
            self.assertEqual(len(session.calls), 12)
            self.assertEqual(summary["counts"]["recorded_measured"], 0)
            self.assertTrue(root.joinpath("metrics_before.prom").exists())
            self.assertTrue(root.joinpath("metrics_after.prom").exists())


if __name__ == "__main__":
    unittest.main()
