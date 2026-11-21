"""The fixed native client retains every raw record and closes on refusal."""

import contextlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

CLIENT = (
    Path(__file__).parents[1] / ".github/scripts/kda3501_native_repeat/native_client.py"
)
spec = importlib.util.spec_from_file_location("native_client", CLIENT)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
DATA = json.loads(CLIENT.with_name("inputs.json").read_text())
RECORD = (-0.1, 1) + sum(((i, -0.1 - i) for i in range(20)), ())


class NativeClientTest(unittest.TestCase):
    def test_collects_chunks_and_refuses_bad_records(self):
        records = []
        for first, last in ((0, 7), (7, 18), (18, 32)):
            sample = SimpleNamespace(
                output_tokens=list(range(first, last)),
                logprobs=[RECORD] * (last - first),
            )
            client.receive(sample, records)
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
                    SimpleNamespace(output_tokens=tokens, logprobs=rows), records.copy()
                )
        client.receive(SimpleNamespace(output_tokens=[], logprobs=None), [])

    def test_saves_exclusively_and_bounds_before_open(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            client.save(path, "one.json", {"value": 1})
            with self.assertRaises(FileExistsError):
                client.save(path, "one.json", {})
            with self.assertRaises(AssertionError):
                client.save(path, "large.json", {"value": "x" * 1024**2})
            self.assertFalse((path / "large.json").exists())

    def run_client(self, folder, bad_cache=False):
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
        engine.get_cache_statistics = lambda **kw: {
            "cached_tokens": engine.cached,
            "offload_tokens": 0,
        }
        engine.get_mtp_statistics = lambda **kw: {"requests": len(engine.submissions)}
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
            if len(engine.submissions) > 1:
                ids[20] = 100
                engine.cached += 0 if bad_cache else 512
            engine.callback(SimpleNamespace(output_tokens=ids, logprobs=[RECORD] * 32))
            engine.finished = True
            return {
                0: {
                    "token_ids": ids,
                    "logprobs": [-0.1] * 32,
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
                EngineArgs=lambda **kwargs: SimpleNamespace(
                    create_engine=lambda: engine
                )
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
            if bad_cache:
                with self.assertRaises(AssertionError):
                    client.main()
            else:
                client.main()
        return engine

    def test_three_requests_preserve_divergence_and_fresh_params(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = self.run_client(folder)
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


if __name__ == "__main__":
    unittest.main()
