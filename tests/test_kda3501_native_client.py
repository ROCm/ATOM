"""The fixed native client retains every raw record and closes on refusal."""

import ast
import contextlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

CLIENT = (
    Path(__file__).parents[1] / ".github/scripts/kda3501_native_repeat/native_client.py"
)
spec = importlib.util.spec_from_file_location("native_client", CLIENT)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
DATA = json.loads(CLIENT.with_name("inputs.json").read_text())
RECORD = (-0.1, 1) + sum(((i, -0.1 - i) for i in range(20)), ())


def native_engine_args(engine):
    """Real native constructor/kwargs/creation methods; downstream providers are stubs."""
    source = Path(__file__).parents[1] / "atom/model_engine/arg_utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EngineArgs"
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, (ast.AnnAssign, ast.Assign))
        or isinstance(node, ast.FunctionDef)
        and node.name in ("_get_engine_kwargs", "parallel_config", "create_engine")
    ]

    def create(model, tokenizer=None, **kwargs):
        engine.constructor_model, engine.constructor_kwargs = model, kwargs
        engine.config.speculative_config = kwargs["speculative_config"]
        return engine

    engine.draft_config = Mock(
        side_effect=AssertionError("unexpected draft config construction")
    )
    config = SimpleNamespace(from_dict=lambda value: value)
    namespace = {
        "__name__": __name__,
        "dataclass": dataclass,
        "fields": fields,
        "DP_LB_DEFAULT": "least_requests",
        "CompilationConfig": SimpleNamespace,
        "CUDAGraphMode": {"FULL": "FULL"},
        "SpeculativeConfig": engine.draft_config,
        "ParallelConfig": SimpleNamespace,
        "DSparkConfig": config,
        "DCPConfig": config,
        "EPLBConfig": config,
        "LLMEngine": create,
        "logger": Mock(),
        "envs": SimpleNamespace(
            ATOM_KV_OFFLOAD=None, ATOM_KV_OFFLOAD_EXTRA_CONFIG=None
        ),
        "kv_offload_connector_config": lambda mode, extra: None,
        "compose_kv_offload_config": lambda base, offload: base,
    }
    parse_sizes = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_size_list"
    )
    exec(  # noqa: S102 -- native constructors without importing GPU dependencies
        compile(
            ast.Module(body=[parse_sizes, cls], type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )
    return namespace["EngineArgs"]


class NativeClientTest(unittest.TestCase):
    def test_collects_chunks_and_refuses_bad_records(self):
        records = []
        for first, last in ((0, 7), (7, 18), (18, 32)):
            sample = SimpleNamespace(
                output_tokens=list(range(first, last)),
                logprobs=[RECORD] * (last - first),
            )
            client.receive(sample, records, [])
        self.assertEqual([row["token"] for row in records], list(range(32)))
        for tokens, rows in (
            ([1], []),
            ([1], [(math.nan, *RECORD[1:])]),
            ([1], [(-0.1, 1) + (1, -0.5) * 20]),
            ([32], [RECORD]),
        ):
            with (
                self.subTest(tokens=tokens, rows=rows),
                self.assertRaises(AssertionError),
            ):
                client.receive(
                    SimpleNamespace(output_tokens=tokens, logprobs=rows),
                    records.copy(),
                    [],
                )
        client.receive(SimpleNamespace(output_tokens=[], logprobs=None), [], [])

    def test_saves_exclusively_and_bounds_before_open(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            client.save(path, "one.json", {"value": 1})
            with self.assertRaises(FileExistsError):
                client.save(path, "one.json", {})
            with self.assertRaises(AssertionError):
                client.save(path, "large.json", {"value": "x" * 1024**2})
            self.assertFalse((path / "large.json").exists())

    def run_client(self, folder, bad_cache=False, nonfinite=None, bad_mtp=None):
        root = Path(folder)
        proof = {
            "tokens": DATA["tokens"],
            "revisions": DATA["revisions"],
            "model_files": {
                DATA["engine_args"]["draft_model"]: {
                    "resolved_path": DATA["engine_args"]["draft_model"]
                }
            },
        }
        (root / "proof.json").write_text(json.dumps(proof))
        engine = SimpleNamespace(
            config=SimpleNamespace(
                kv_transfer_config={},
                speculative_config=SimpleNamespace(synthetic_acceptance_rates=None),
            )
        )
        engine.submissions, engine.cached, engine.finished, engine.closed = (
            [],
            0,
            True,
            False,
        )
        engine.callback_errors = []
        engine.get_cache_statistics = lambda **kw: {
            "cached_tokens": engine.cached,
            "offload_tokens": 0,
            "checkpoints_kept": int(bool(engine.submissions)),
            "chunks_cut_for_end": 0,
        }
        engine.get_mtp_statistics = lambda **kw: {
            "enabled": False,
            "total_draft_tokens": 0,
            "total_accepted_tokens": 0,
            "distribution": {},
            **(bad_mtp or {}),
        }
        engine.is_finished = lambda: engine.finished
        engine.close = lambda: setattr(engine, "closed", True)
        engine.core_mgr = SimpleNamespace(
            reset_dp_router=lambda: None,
            offline_backpressure_batch=contextlib.nullcontext,
            is_alive=lambda: True,
            is_rest=lambda: False,
        )
        engine.io_processor = SimpleNamespace(postprocess=lambda outputs: outputs)

        def add_request(prompts, params, stream_callback):
            engine.submissions.append((prompts, params))
            engine.callback, engine.finished = stream_callback, False

        def step():
            ids = list(range(32))
            scores, records = [-0.1] * 32, [RECORD] * 32
            if nonfinite is not None:
                scores[20] = nonfinite
                records[20] = (nonfinite, *RECORD[1:])
            if len(engine.submissions) > 1:
                ids[20] = 100
                engine.cached += 0 if bad_cache else 512
            # Native CoreManager logs callback errors and still delivers the SDK row.
            try:
                engine.callback(SimpleNamespace(output_tokens=ids, logprobs=records))
            except AssertionError as error:
                engine.callback_errors.append(repr(error))
            engine.finished = True
            return {
                0: {
                    "token_ids": ids,
                    "logprobs": scores,
                    "num_tokens_input": 1024,
                    "num_tokens_output": 32,
                    "finish_reason": "max_tokens",
                }
            }

        engine.add_request, engine.step = add_request, step
        atom = ModuleType("atom")
        atom.__file__ = str(root / "source/atom/__init__.py")
        atom.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        modules = {
            "atom": atom,
            "atom.model_engine": ModuleType("atom.model_engine"),
            "torch": SimpleNamespace(
                __version__=DATA["runtime"]["torch"],
                version=SimpleNamespace(hip=DATA["runtime"]["hip"]),
            ),
            "triton": SimpleNamespace(__version__=DATA["runtime"]["triton"]),
            "atom.model_engine.arg_utils": SimpleNamespace(
                EngineArgs=native_engine_args(engine)
            ),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(
                os.environ,
                {**DATA["environment"], "PR3501_SOURCE": str(root / "source")},
            ),
            patch.object(
                sys, "argv", [str(CLIENT), str(root / "out"), str(root / "proof.json")]
            ),
            patch.object(
                client.importlib.metadata,
                "version",
                return_value=DATA["runtime"]["aiter"],
            ),
        ):
            if nonfinite is not None:
                with self.assertRaises((AssertionError, ValueError)):
                    client.main()
            elif bad_cache or bad_mtp:
                with self.assertRaises(AssertionError):
                    client.main()
            else:
                try:
                    client.main()
                except AttributeError as error:
                    self.fail(f"zero-token native config was not handled: {error}")
        return engine

    def test_three_requests_preserve_divergence_and_fresh_params(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = self.run_client(folder)
            self.assertEqual(DATA["engine_args"]["num_speculative_tokens"], 0)
            self.assertIsNone(engine.constructor_kwargs["speculative_config"])
            self.assertNotIn("draft_model", engine.constructor_kwargs)
            engine.draft_config.assert_not_called()
            self.assertTrue(engine.closed)
            self.assertEqual(len(engine.submissions), 3)
            self.assertEqual(len({id(params) for _, params in engine.submissions}), 3)
            self.assertTrue(
                all(prompts == [DATA["tokens"]] for prompts, _ in engine.submissions)
            )
            result = json.loads((Path(folder) / "out/collection.json").read_text())
            self.assertFalse(result["C0_W0_same_ids"])
            self.assertTrue(result["W0_W1_same_ids"])
            self.assertEqual(result["qualification"], "not_assessed")

    def test_active_or_nonzero_mtp_stats_are_retained_and_refused(self):
        for bad in (
            {"enabled": True},
            {"total_draft_tokens": 1},
            {"total_accepted_tokens": 1},
            {"distribution": {"0": 1}},
        ):
            with self.subTest(stats=bad), tempfile.TemporaryDirectory() as folder:
                engine = self.run_client(folder, bad_mtp=bad)
                self.assertTrue(engine.closed)
                self.assertEqual(len(engine.submissions), 1)
                output = Path(folder) / "out"
                saved = json.loads((output / "C0.json").read_text())
                for key, value in bad.items():
                    self.assertEqual(saved["before"]["mtp"][key], value)
                self.assertTrue((output / "failure.json").exists())
                self.assertFalse((output / "collection.json").exists())

    def test_coverage_failure_preserves_phase_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = self.run_client(folder, bad_cache=True)
            self.assertTrue(engine.closed)
            self.assertEqual(len(engine.submissions), 2)
            output = Path(folder) / "out"
            self.assertEqual(
                json.loads((output / "failure.json").read_text())["phase"], "W0"
            )
            self.assertTrue((output / "W0.json").exists())
            self.assertFalse((output / "collection.json").exists())

    def test_swallowed_nonfinite_callback_retains_failed_raw_values(self):
        for value, label in (
            (math.nan, "NaN"),
            (math.inf, "Infinity"),
            (-math.inf, "-Infinity"),
        ):
            with self.subTest(value=label), tempfile.TemporaryDirectory() as folder:
                engine = self.run_client(folder, nonfinite=value)
                self.assertTrue(engine.closed)
                self.assertEqual(len(engine.submissions), 1)
                self.assertEqual(len(engine.callback_errors), 1)
                output = Path(folder) / "out"
                for name in ("C0.json", "failure.json"):
                    self.assertTrue((output / name).exists(), name)
                    saved = json.loads((output / name).read_text())
                    self.assertEqual(
                        saved["outputs"][0]["logprobs"][20], {"nonfinite": label}
                    )
                    rejected = saved["rejected_callbacks"]
                    self.assertEqual(len(rejected), 1)
                    self.assertEqual(rejected[0]["token"], 20)
                    self.assertEqual(rejected[0]["record"][0], {"nonfinite": label})
                    self.assertEqual(len(rejected[0]["record"]), 42)
                    if name == "failure.json":
                        self.assertIn("non-finite native callback", saved["error"])
                self.assertFalse((output / "collection.json").exists())
                self.assertFalse((output / "W0.json").exists())

    def test_rejected_callback_retention_stops_at_the_first_native_row(self):
        rejected = []
        for token in range(64):
            sample = SimpleNamespace(
                output_tokens=[token], logprobs=[(math.nan, *RECORD[1:])]
            )
            with self.assertRaises(AssertionError):
                client.receive(sample, [], rejected)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["token"], 0)
        self.assertEqual(len(rejected[0]["record"]), 42)

    def test_candidate_origin_rejects_boundary_cut_or_missing_checkpoint(self):
        for bad in (
            None,
            "boundary_cut",
            "missing_checkpoint",
            "cut_counter",
            "duplicate_phase",
        ):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as folder:
                self.run_client(folder)
                output = Path(folder) / "out"
                lines = []
                for index, phase in enumerate(("C0", "W0", "W1")):
                    lines.append(f"PR3501 phase {phase} start")
                    spans = [(0, 1024)] if phase == "C0" else [(512, 512)]
                    if phase == "C0" and bad == "boundary_cut":
                        spans = [(0, 512), (512, 512)]
                    for start, length in spans:
                        lines.append(
                            f"[atom 00:00:00] Scheduled prefill batch: 1 reqs, {length} new tokens (done: [{start}], new: [{length}]), req_ids: ({index},)"
                        )
                if bad == "duplicate_phase":
                    lines.append("PR3501 phase W1 start")
                log = Path(folder) / "candidate.log"
                log.write_text("\n".join(lines) + "\n")
                if bad in ("missing_checkpoint", "cut_counter"):
                    path = output / "C0.json"
                    data = json.loads(path.read_text())
                    key = (
                        "checkpoints_kept"
                        if bad == "missing_checkpoint"
                        else "chunks_cut_for_end"
                    )
                    data["after"]["cache"][key] = (
                        0 if bad == "missing_checkpoint" else 1
                    )
                    path.write_text(json.dumps(data))
                with patch.object(sys, "path", [str(CLIENT.parents[1]), *sys.path]):
                    if bad:
                        with self.assertRaises(AssertionError):
                            client.verify_candidate_origin(output, log)
                    else:
                        client.verify_candidate_origin(output, log)
                        self.assertTrue((output / "prefill-origin.json").exists())
                self.assertTrue((output / "C0.json").exists())
                self.assertTrue(log.exists())


class ObservationPostcheckTest(unittest.TestCase):
    def check_records(self, change=None):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            phases = ("C0", "W0", "W1")
            origin = {
                "prefill_steps": {
                    p: [{"sequence": str(i)}] for i, p in enumerate(phases)
                }
            }
            (directory / "prefill-origin.json").write_text(json.dumps(origin))
            rows = []
            for i, phase in enumerate(phases):
                for rank in range(8):
                    row = {
                        "rank": rank,
                        "event": ("store", "restore_1", "restore_2")[i],
                        "request_id": str(i),
                        "prefill": i > 0,
                        "cached": 512 if i else 1024,
                        "scheduled": 512 if i else 1,
                        "unit_ids": list(range(1, 30)),
                        "layout_id": "kda-paged-state-v1:test:tp=8:spec=0:carry=all",
                        "image_bytes": 56_171_520,
                        "op_bytes": 56_171_520,
                        "slot": 3,
                        "page_sha256": f"{rank:064x}",
                    }
                    if i:
                        row.update(
                            identity_matches=True,
                            page_unchanged=True,
                            restored_matches_incoming=True,
                            restored_matches_reference=True,
                            restored_sha256=row["page_sha256"],
                            controls_expected=True,
                            controls={
                                "read_slot": 3,
                                "write_slot": 3,
                                "has_initial_state": True,
                                "query_start_loc": [0, 512],
                                "replayssm_cursor": -1,
                                "position_span": [512, 1023, 512],
                                "num_prefills": 1,
                                "num_actual_tokens": 512,
                            },
                        )
                    rows.append((phase, row))
            if change:
                change(rows)
            lines = []
            for phase in phases:
                lines.append(f"PR3501 phase {phase} start")
                lines.extend(
                    "PR3501_CHECKPOINT " + json.dumps(row, separators=(",", ":"))
                    for p, row in rows
                    if p == phase
                )
            path = directory / "candidate.log"
            path.write_text("\n".join(lines) + "\n")
            client.verify_checkpoint_observations(directory, path)
            return json.loads((directory / "checkpoint-observations.json").read_text())

    def test_complete_coverage_retains_unequal_state_observations(self):
        valid = self.check_records()
        self.assertTrue(valid["all_pages_unchanged"])
        self.assertLessEqual(valid["logged_bytes"], 65536)

        def unequal(rows):
            row = rows[15][1]
            row.update(
                page_unchanged=False,
                restored_matches_reference=False,
                page_sha256="f" * 64,
                restored_sha256="f" * 64,
            )

        changed = self.check_records(unequal)
        self.assertFalse(changed["all_pages_unchanged"])
        self.assertFalse(changed["all_restored_match_reference"])
        self.assertTrue(changed["all_restored_match_incoming"])

    def test_controls_are_independently_checked_and_false_is_retained(self):
        def wrong(rows):
            rows[-1][1]["controls"]["has_initial_state"] = False
            rows[-1][1]["controls_expected"] = False

        self.assertFalse(self.check_records(wrong)["all_controls_expected"])

        def false_claim(rows):
            rows[-1][1]["controls"]["has_initial_state"] = False

        with self.assertRaises(AssertionError):
            self.check_records(false_claim)

    def test_incomplete_duplicate_wrong_identity_and_oversized_records_refuse(self):
        changes = (
            lambda rows: rows.pop(),
            lambda rows: rows.append(rows[-1]),
            lambda rows: rows[-1][1]["unit_ids"].reverse(),
            lambda rows: rows[-1][1].update(padding="x" * 2048),
            lambda rows: rows[-1][1].update(image_bytes=1),
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(AssertionError):
                self.check_records(change)


if __name__ == "__main__":
    unittest.main()
