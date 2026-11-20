"""Behavior checks for the isolated #3501 diagnostic, without ATOM or a GPU."""

import copy
import importlib.util
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / ".github/scripts/kda_stage1_probe.py"
SPEC = importlib.util.spec_from_file_location("kda_stage1_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def token(probability=0.8, token_id=1, count=32):
    return {
        "token": token_id,
        "logprob": math.log(probability),
        "top": {1: math.log(probability), 2: math.log(0.01)},
        "token_ids": [token_id] * count,
        "logprobs": [math.log(probability)] * count,
        "num_tokens_output": count,
        "finish_reason": "max_tokens",
    }


def stage(arm, phase="cold"):
    if phase == "cold":
        widths = [512, 512] if arm == "baseline" else [1024]
        cursor = 0
    else:
        cursor, widths = 512, [512]
    cached = cursor
    events = []
    for width in widths:
        events.append(
            {"kind": "prefill", "done": cursor, "new": width, "sequence": "req"}
        )
        cursor += width
    if phase != "cold":
        op = {"units": [1, 4, 7], "slot": 2, "layout": "kda", "bytes": 64}
        events.append({"kind": "queued_restore", "operation": op})
        events += [
            {
                "kind": "restore_complete",
                "rank": rank,
                "replayssm": True,
                "operations": [copy.deepcopy(op)],
            }
            for rank in range(8)
        ]
    return {
        "phase": phase,
        "cached_delta": cached,
        "offload_delta": 0,
        "prompt_sha256": "same",
        "first": token(),
        "events": events,
    }


def capture(arm):
    return {
        "complete": True,
        "arm": arm,
        "atom_tree": probe.TREES[arm],
        "source": probe.BASE if arm == "baseline" else probe.CANDIDATE,
        "args": probe.ARGS,
        "env": probe.ENV,
        "runtime": {"torch": "same"},
        "resolved": {"tensor_parallel_size": 8},
        "inputs_sha256": "same",
        "stages": [stage(arm, phase) for phase in probe.PHASES],
    }


class ProbeChecks(unittest.TestCase):
    def test_cache_pin_and_missing_shard_fail_before_loading(self):
        with (
            TemporaryDirectory(dir=".") as directory,
            patch.dict(probe.REVISIONS, clear=True),
        ):
            root = Path(directory)
            probe.REVISIONS[str(root)] = "expected"
            for name, value in {
                ".hf-revision": "expected",
                ".atom_download_complete": "",
                "config.json": "{}",
                "weights.safetensors": "x",
            }.items():
                (root / name).write_text(value)
            self.assertIn(str(root), probe.check_models())
            (root / ".hf-revision").write_text("different")
            with self.assertRaisesRegex(RuntimeError, "wrong cached revision"):
                probe.check_models()
            (root / ".hf-revision").write_text("expected")
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map":{"weight":"missing.safetensors"}}'
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete model shards"):
                probe.check_models()

    def test_native_logprob_record_is_complete_and_finite(self):
        row = token(token_id=0)
        record = [row["logprob"], 1]
        for index in range(20):
            record.extend((index, row["logprob"] - index))
        self.assertEqual(len(probe.first_token(row, [(0, record)])["top"]), 20)
        with self.assertRaises(probe.CoverageError):
            probe.first_token(row, [(0, record[:12])])
        for streamed in ([], [(1, record)], [(0, [-0.3, *record[1:]])]):
            with self.assertRaises(probe.CoverageError):
                probe.first_token(row, streamed)
        record[3] = math.nan
        with self.assertRaises(FloatingPointError):
            probe.first_token(row, [(0, record)])

    def test_real_log_shape(self):
        row = probe.parse_step(
            "Scheduled prefill batch: 1 reqs, 512 new tokens (done: [0], new: [512]), req_ids: (12,)"
        )
        self.assertEqual(row, {"done": 0, "new": 512, "sequence": "12"})

    def test_mixed_requests_are_not_a_single_prompt_witness(self):
        with self.assertRaises(probe.CoverageError):
            probe.parse_step(
                "Scheduled prefill batch: 2 reqs, 1024 new tokens (done: [0, 0], new: [512, 512]), req_ids: (12, 13)"
            )

    def test_numerical_decisions(self):
        self.assertEqual(probe.compare_tokens(token(), token())["status"], "pass")
        self.assertEqual(probe.compare_tokens(token(), token(0.75))["status"], "fail")
        self.assertEqual(
            probe.compare_tokens(token(), token(token_id=2))["status"], "inconclusive"
        )
        bad = token()
        bad["top"][1] = math.nan
        self.assertEqual(probe.compare_tokens(token(), bad)["status"], "fail")

    def test_unseen_top_token_uses_an_upper_bound(self):
        left = token()
        right = token()
        left["top"] = {1: math.log(0.8), 2: math.log(0.015)}
        right["top"] = {1: math.log(0.8), 3: math.log(0.01)}
        result = probe.compare_tokens(left, right)
        self.assertLess(result["max_top20_union_probability_delta_lower"], 0.01)
        self.assertGreater(result["max_top20_union_probability_delta_upper"], 0.01)
        self.assertEqual(result["status"], "inconclusive")

    def test_no_vacuous_or_wrong_source_pass(self):
        for mutate in (
            lambda data: data["stages"].clear(),
            lambda data: data.update(source="wrong"),
            lambda data: data.update(complete=False),
            lambda data: data["stages"][1]["first"].update(finish_reason="aborted"),
            lambda data: data["stages"][1]["first"].update(
                token_ids=[1], logprobs=[-0.2], num_tokens_output=1
            ),
        ):
            base, candidate = capture("baseline"), capture("candidate")
            mutate(candidate)
            with self.assertRaises(probe.CoverageError):
                probe.compare(base, candidate)

    def test_step_and_restore_controls(self):
        for mutate in (
            lambda data: data.update(cached_delta=0),
            lambda data: data.update(offload_delta=512),
            lambda data: data["events"].pop(),
            lambda data: data["events"][2].update(replayssm=False),
            lambda data: data["events"][2]["operations"][0].update(units=[0]),
            lambda data: data["events"][0].update(done=0),
            lambda data: data["first"]["token_ids"].pop(),
            lambda data: data["first"]["logprobs"].pop(),
        ):
            data = stage("candidate", phase="warm")
            mutate(data)
            with self.assertRaises(probe.CoverageError):
                probe.check_trace("candidate", data)
        with self.assertRaises(probe.CoverageError):
            probe.check_trace("candidate", stage("baseline"))

    def test_all_declared_cases_and_both_arm_controls(self):
        base, candidate = capture("baseline"), capture("candidate")
        self.assertEqual(probe.compare(base, candidate)["status"], "pass")
        candidate["stages"][0]["first"] = token(0.7)
        self.assertEqual(probe.compare(base, candidate)["status"], "fail")
        candidate = capture("candidate")
        candidate["args"] = {"offload": True}
        with self.assertRaises(probe.CoverageError):
            probe.compare(base, candidate)

    def test_resume_checks_the_entire_continuation(self):
        for index in (0, 17, 31):
            base = capture("baseline")
            base["stages"][1]["first"]["token_ids"][index] = 2
            self.assertEqual(probe.compare(base)["status"], "inconclusive")
            base = capture("baseline")
            base["stages"][1]["first"]["logprobs"][index] = math.log(0.7)
            self.assertEqual(probe.compare(base)["status"], "fail")
        base["stages"][1]["first"]["logprobs"][31] = math.nan
        with self.assertRaises(FloatingPointError):
            probe.compare(base)

    def test_only_a_stable_baseline_allows_the_next_arm(self):
        for status, error in (
            ("fail", FloatingPointError),
            ("inconclusive", probe.CoverageError),
        ):
            with self.assertRaises(error):
                probe.require_stable_baseline("baseline", {"status": status})
        probe.require_stable_baseline("baseline", {"status": "pass"})
        probe.require_stable_baseline("candidate", {"status": "fail"})


if __name__ == "__main__":
    unittest.main()
