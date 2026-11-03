"""Decode routing of the M3.1 MooncakeConnector proxy: first-seen round-robin, then sticky."""
import contextlib
import importlib.util
import io
import itertools
import unittest
from pathlib import Path
from types import SimpleNamespace

PROXY = Path(__file__).resolve().parents[3] / "benchmark/m31-mooncake-rdma/mooncake_connector_proxy.py"
spec = importlib.util.spec_from_file_location("m31_mc_proxy", PROXY)
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


def fake_app(num_decoders=2):
    return SimpleNamespace(state=SimpleNamespace(
        decode_clients=[f"d{i}" for i in range(num_decoders)],
        decode_iterator=itertools.cycle(range(num_decoders)),
        session_decoder={},
        warned_no_session=False,
    ))


class DecodeRoutingTest(unittest.TestCase):
    def route(self, app, session_id):
        return proxy.get_next_client(app, "decode", session_id)

    def test_new_sessions_alternate_and_repeats_stick(self):
        app = fake_app()
        self.assertEqual([self.route(app, f"s{i}") for i in range(4)], ["d0", "d1", "d0", "d1"])
        self.assertEqual([self.route(app, s) for s in ("s1", "s0", "s3", "s1")], ["d1", "d0", "d1", "d1"])
        # Repeats must not consume the round-robin slot of the next new session.
        self.assertEqual(self.route(app, "s4"), "d0")
        self.assertEqual(app.state.session_decoder, {"s0": 0, "s1": 1, "s2": 0, "s3": 1, "s4": 0})

    def test_live_sessions_split_evenly(self):
        app = fake_app()
        sessions = [f"conv-{i}" for i in range(24)]
        for turn in range(5):
            picks = [self.route(app, s) for s in sessions]
            self.assertEqual(picks.count("d0"), 12, f"turn {turn}")

    def test_missing_header_round_robins_and_warns_once(self):
        app = fake_app()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            picks = [self.route(app, None) for _ in range(3)] + [self.route(app, "")]
        self.assertEqual(picks, ["d0", "d1", "d0", "d1"])
        self.assertEqual(out.getvalue().count("WARNING"), 1)
        self.assertEqual(app.state.session_decoder, {})

    def test_handler_routes_on_correlation_header(self):
        self.assertIn('"decode", request.headers.get("x-correlation-id")', PROXY.read_text())


if __name__ == "__main__":
    unittest.main()
