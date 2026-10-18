"""Check how the C48 launcher wires the Prefill LMCache MP sidecar port."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_c48_launch import render

GAP = Path(__file__).resolve().parents[3] / "benchmark/c48-model-gap"
ADDR = ("/models/moonshotai/Kimi-K3", "10.0.0.1", "10.0.0.2", 6100, 7000, "/tmp/traces")


def transfer(argv):
    return json.loads(argv[argv.index("--kv-transfer-config") + 1])


class LaunchTest(unittest.TestCase):
    def test_lmcache_port_fills_prefill_connector(self):
        config = json.loads((GAP / "V-QD-LMC.json").read_text())
        argv, _ = render(config, "prefill", *ADDR, lmcache_port=6655)
        lmc = transfer(argv)["kv_connector_extra_config"]["connectors"][1]
        self.assertEqual(lmc["kv_connector"], "LMCacheMPConnector")
        self.assertEqual(lmc["kv_connector_extra_config"]["lmcache.mp.port"], 6655)
        with self.assertRaises(ValueError):
            render(config, "prefill", *ADDR)
        self.assertNotIn("LMCache", json.dumps(transfer(render(config, "decode", *ADDR)[0])))

    def test_simple_cpu_case_ignores_lmcache_port(self):
        config = json.loads((GAP / "V-QD.json").read_text())
        self.assertEqual(render(config, "prefill", *ADDR)[0], render(config, "prefill", *ADDR, lmcache_port=6655)[0])


if __name__ == "__main__":
    unittest.main()
