"""Behavior checks for the isolated #3501 diagnostic, without ATOM or a GPU."""

import builtins
import copy
import hashlib
import importlib.util
import json
import math
import os
import runpy
import sys
import time
import types
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
        "top": {i: math.log(probability if i == 1 else 0.001) for i in range(1, 21)},
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
        "args": dict(probe.ARGS),
        "draft_cache": {
            "alias": probe.DRAFT,
            "resolved_path": probe.DRAFT,
            "expected_revision": probe.REVISIONS[probe.DRAFT],
            "digests": {n: h for n, (_size, h) in probe.DRAFT_COMPONENTS.items()},
        },
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

    def test_sampling_construction_matches_the_native_api(self):
        native = runpy.run_path(str(PATH.parents[2] / "atom/sampling_params.py"))
        params = probe.make_sampling_params(native["SamplingParams"])
        self.assertEqual(
            (
                params.temperature,
                params.max_tokens,
                params.ignore_eos,
                params.logprobs,
                params.top_logprobs,
            ),
            (0, probe.OUTPUT_TOKENS, True, True, 20),
        )

    def test_draft_identity_and_effective_path_must_match_between_arms(self):
        for mutate in (
            lambda d: d["draft_cache"].update(expected_revision="wrong"),
            lambda d: d["draft_cache"].update(alias="/unlisted"),
            lambda d: d["draft_cache"]["digests"].update(
                {"model.safetensors": "wrong"}
            ),
            lambda d: d["args"].update(draft_model="/unvalidated"),
            lambda d: d["draft_cache"].update(provenance="different"),
        ):
            with self.subTest(mutate=mutate):
                base, candidate = capture("baseline"), capture("candidate")
                mutate(candidate)
                with self.assertRaises(probe.CoverageError):
                    probe.compare(base, candidate)

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
        self.assertEqual(probe.compare_responses(token(), token())["status"], "pass")
        self.assertEqual(
            probe.compare_responses(token(), token(0.75))["status"], "fail"
        )
        self.assertEqual(
            probe.compare_responses(token(), token(token_id=2))["status"],
            "inconclusive",
        )
        bad = token()
        bad["top"][1] = math.nan
        self.assertEqual(probe.compare_responses(token(), bad)["status"], "fail")

    def test_unseen_top_token_uses_an_upper_bound(self):
        left = token()
        right = token()
        left["top"] = {1: math.log(0.8), 2: math.log(0.015)}
        right["top"] = {1: math.log(0.8), 3: math.log(0.01)}
        result = probe.compare_responses(left, right)
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
        for arm, phases, probability in (
            ("baseline", (1,), 0.7),
            ("candidate", (0, 1), 0.1),
        ):
            for index in (0, 17, 31):
                for field, value, expected in (
                    ("token_ids", 2, "inconclusive"),
                    ("logprobs", math.log(probability), "fail"),
                ):
                    base, candidate = capture("baseline"), capture("candidate")
                    changed = base if arm == "baseline" else candidate
                    for phase in phases:
                        changed["stages"][phase]["first"][field][index] = value
                    self.assertEqual(probe.compare(base, candidate)["status"], expected)
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


class DraftCacheChecks(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.legacy, self.canonical = self.root / "legacy", self.root / "canonical"
        self.config, self.weight = b'{"draft":true}\n', b"verified draft weight"
        self.components = {
            "config.json": (
                len(self.config),
                hashlib.sha1(
                    f"blob {len(self.config)}\0".encode() + self.config
                ).hexdigest(),
            ),
            "model.safetensors": (
                len(self.weight),
                hashlib.sha256(self.weight).hexdigest(),
            ),
        }
        for change in (
            patch.object(
                probe, "DRAFT_ALIASES", (str(self.legacy), str(self.canonical))
            ),
            patch.dict(probe.DRAFT_COMPONENTS, self.components, clear=True),
            patch.dict(
                probe.REVISIONS, {probe.DRAFT: probe.REVISIONS[probe.DRAFT]}, clear=True
            ),
        ):
            change.start()
            self.addCleanup(change.stop)

    def cache(
        self,
        root=None,
        origin=None,
        marker=False,
        metadata=True,
        config=None,
        weight=None,
    ):
        root = root or self.canonical
        root.mkdir(exist_ok=True)
        (root / "config.json").write_bytes(self.config if config is None else config)
        (root / "model.safetensors").write_bytes(
            self.weight if weight is None else weight
        )
        if metadata:
            folder = root / ".cache/huggingface/download"
            folder.mkdir(parents=True, exist_ok=True)
            for name, (_size, digest) in self.components.items():
                (folder / (name + ".metadata")).write_text(
                    f"{origin or probe.REVISIONS[probe.DRAFT]}\n{digest}\n{time.time()}\n"
                )
        if marker:
            (root / ".hf-revision").write_text(probe.REVISIONS[probe.DRAFT])
            (root / ".atom_download_complete").write_text("existing marker\n")
        return root

    def manifest(self, proof):
        tokens = list(range(probe.PROMPT_TOKENS))
        data = {
            "revisions": dict(probe.REVISIONS),
            "model_files": {probe.DRAFT: proof},
            "tokens": tokens,
            "sha256": probe.token_hash(tokens),
        }
        path = self.root / "inputs.json"
        path.write_text(json.dumps(data))
        return path

    def test_standard_paths_preserve_origin_and_do_not_write_markers(self):
        for root, origin in (
            (self.canonical, probe.DRAFT_ORIGINS[0]),
            (self.canonical, probe.DRAFT_ORIGINS[1]),
            (self.legacy, probe.DRAFT_ORIGINS[0]),
        ):
            with self.subTest(root=root, origin=origin):
                self.cache(root, origin=origin)
                before = {
                    str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()
                }
                proof = probe.check_draft_cache()
                self.assertEqual(proof["alias"], str(root))
                self.assertEqual(proof["resolved_path"], str(root.resolve()))
                self.assertEqual(
                    proof["expected_revision"], probe.REVISIONS[probe.DRAFT]
                )
                self.assertEqual(set(proof["origin_revisions"].values()), {origin})
                self.assertEqual(proof["route"], "hf-runtime-components")
                self.assertEqual(
                    proof["digests"], {n: h for n, (_s, h) in self.components.items()}
                )
                self.assertEqual(probe.check_draft_cache(proof), proof)
                self.assertEqual(
                    before,
                    {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()},
                )
                self.assertFalse((root / ".hf-revision").exists())

    def test_marker_route_is_audited_and_cannot_bless_wrong_bytes(self):
        self.cache(marker=True, metadata=False)
        proof = probe.check_draft_cache()
        self.assertEqual(proof["route"], "pinned-marker")
        self.assertEqual(
            proof["provenance"][".hf-revision"]["raw"], probe.REVISIONS[probe.DRAFT]
        )
        (self.canonical / "model.safetensors").write_bytes(b"x" * len(self.weight))
        with self.assertRaisesRegex(RuntimeError, "content hash"):
            probe.check_draft_cache()

    def test_same_sized_wrong_config_or_weight_refuses(self):
        for changes in (
            {"config": b"x" * len(self.config)},
            {"weight": b"x" * len(self.weight)},
        ):
            with self.subTest(changes=changes):
                self.cache(**changes)
                with self.assertRaisesRegex(RuntimeError, "content hash"):
                    probe.check_draft_cache()

    def test_wrong_size_missing_and_extra_payload_refuse(self):
        self.cache(weight=b"short")
        with self.assertRaisesRegex(RuntimeError, "size"):
            probe.check_draft_cache()
        (self.canonical / "model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            probe.check_draft_cache()
        for name in ("other.safetensors", "model.safetensors.index.json", "config.py"):
            self.cache()
            extra = self.canonical / name
            extra.write_text("unexpected")
            with self.assertRaisesRegex(RuntimeError, "unexpected file"):
                probe.check_draft_cache()
            extra.unlink()

    def test_wrong_revision_and_download_provenance_refuse(self):
        self.cache(origin="f" * 40)
        with self.assertRaisesRegex(RuntimeError, "download provenance"):
            probe.check_draft_cache()
        self.cache(marker=True)
        (self.canonical / ".hf-revision").write_text(probe.DRAFT_ORIGINS[1])
        with self.assertRaisesRegex(RuntimeError, "cached draft revision"):
            probe.check_draft_cache()

    def test_malformed_stale_and_wrong_etag_metadata_refuse(self):
        self.cache()
        path = self.canonical / ".cache/huggingface/download/model.safetensors.metadata"
        original = path.read_text().splitlines()
        for text in (
            "missing fields\n",
            "x" * 4097,
            "\n".join([original[0], "wrong", original[2]]),
            "\n".join([original[0], original[1], "0"]),
            "\n".join([original[0], original[1], "nan"]),
        ):
            with self.subTest(text=text[:30]):
                path.write_text(text)
                with self.assertRaises(RuntimeError):
                    probe.check_draft_cache()

    def test_invalid_preferred_alias_does_not_fall_through(self):
        self.cache()
        self.cache(self.legacy, weight=b"x" * len(self.weight))
        with self.assertRaisesRegex(RuntimeError, "content hash"):
            probe.check_draft_cache()

    def test_replacement_with_preserved_mtime_invalidates_preflight(self):
        self.cache()
        proof = probe.check_draft_cache()
        weight = self.canonical / "model.safetensors"
        stamp = weight.stat()
        replacement = self.root / "replacement"
        replacement.write_bytes(b"x" * len(self.weight))
        os.utime(replacement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        os.replace(replacement, weight)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            probe.check_draft_cache(proof)

    def test_file_identity_change_during_hash_refuses(self):
        self.cache()
        weight = self.canonical / "model.safetensors"
        make_digest = hashlib.sha256

        class MutatingDigest:
            def __init__(self):
                self.inner = make_digest()

            def update(self, chunk):
                self.inner.update(chunk)
                stamp = weight.stat()
                os.utime(
                    weight, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2_000_000_000)
                )

            def hexdigest(self):
                return self.inner.hexdigest()

        with (
            patch.object(probe.hashlib, "sha256", MutatingDigest),
            self.assertRaisesRegex(RuntimeError, "changed while hashing"),
        ):
            probe.check_draft_cache()

    def test_preflight_hash_time_is_bounded(self):
        self.cache()
        with (
            patch.object(probe.time, "monotonic", side_effect=[0, 91]),
            self.assertRaisesRegex(RuntimeError, "preflight time limit"),
        ):
            probe.check_draft_cache()

    def test_capture_uses_the_validated_path_and_refuses_before_importing_model(self):
        self.cache(origin=probe.DRAFT_ORIGINS[1])
        proof = probe.check_draft_cache()
        path = self.manifest(proof)
        data, args = probe.capture_inputs(path)
        self.assertEqual(
            args, dict(probe.ARGS, draft_model=str(self.canonical.resolve()))
        )
        self.assertEqual(data["model_files"][probe.DRAFT], proof)
        import_module = builtins.__import__

        def refuse_model_import(name, *args, **kwargs):
            if name in {"aiter", "torch", "triton", "atom"}:
                self.fail("model import reached before draft proof refusal")
            return import_module(name, *args, **kwargs)

        for field in ("alias", "resolved_path", "digests"):
            forged = copy.deepcopy(proof)
            forged[field] = "/unvalidated" if field != "digests" else {}
            path = self.manifest(forged)
            with self.subTest(field=field):
                with (
                    patch.object(builtins, "__import__", refuse_model_import),
                    self.assertRaisesRegex(RuntimeError, "identity changed"),
                ):
                    probe.capture("baseline", path, self.root / "never.json")
                self.assertFalse((self.root / "never.json").exists())
        self.manifest(proof)
        self.cache(self.legacy)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            probe.capture_inputs(path)

    def test_capture_passes_only_validated_args_to_engine_admission(self):
        self.cache()
        proof = probe.check_draft_cache()
        path = self.manifest(proof)
        modules = {
            name: types.ModuleType(name)
            for name in (
                "aiter",
                "torch",
                "triton",
                "atom",
                "atom.model_engine",
                "atom.model_engine.arg_utils",
            )
        }
        modules["aiter"].__file__ = "cpu-boundary/aiter"
        modules["torch"].__version__ = "cpu-boundary"
        modules["torch"].version = types.SimpleNamespace(hip=None)
        modules["triton"].__version__ = "cpu-boundary"
        modules["atom"].__file__ = "cpu-boundary/atom"
        modules["atom"].SamplingParams = object

        class AdmissionReached(Exception):
            pass

        class EngineArgs:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def create_engine(self):
                raise AdmissionReached(self.kwargs)

        modules["atom.model_engine.arg_utils"].EngineArgs = EngineArgs
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, dict(probe.ENV, STAGE1_EVENTS=str(self.root))),
            patch.object(
                probe.importlib.metadata, "version", return_value="0.1.1+gd17ee8dfe"
            ),
            self.assertRaises(AdmissionReached) as reached,
        ):
            probe.capture("baseline", path, self.root / "capture.json")
        self.assertEqual(
            reached.exception.args[0],
            dict(probe.ARGS, draft_model=str(self.canonical.resolve())),
        )
        recorded = json.loads((self.root / "capture.json").read_text())
        self.assertEqual(recorded["draft_cache"], proof)
        self.assertFalse(recorded["complete"])


if __name__ == "__main__":
    unittest.main()
