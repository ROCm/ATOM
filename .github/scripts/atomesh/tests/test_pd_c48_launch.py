"""Check how the C48 launcher wires the Prefill LMCache MP sidecar port."""

import json
import sys
import types
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

    def test_mooncake_cases_change_only_transfer_and_offload(self):
        sources = {"DG5-PNT-SWB-C48-MC": "DG5-PNT", "DG5-PNT-SWB-C56-MC": "DG5-PNT-SWB-C56-CPU26",
                   "DG5-PNT-SWB-C88-MC": "DG5-PNT-SWB-C88-CPU26"}
        prefix = "V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-"
        cases = sorted(GAP.glob(prefix + "*-MC.json"))
        self.assertEqual(len(cases), 8)
        for path in cases:
            suffix = path.stem.removeprefix(prefix)
            mc = json.loads(path.read_text())
            src = json.loads((GAP / (prefix + sources.get(suffix, suffix.removesuffix("-MC")) + ".json")).read_text())
            store = "store" in mc["mooncake"]
            for role in ("prefill", "decode"):
                mc_argv, mc_env = render(mc, role, *ADDR)
                src_argv, src_env = render(src, role, *ADDR)
                i = mc_argv.index("--kv-transfer-config") + 1
                self.assertEqual(mc_argv[:i] + mc_argv[i + 1:], src_argv[:i] + src_argv[i + 1:])
                added = {"MC_GID_INDEX": "1", "MC_MAX_CONCURRENT_REG_MR": "1",
                         "VLLM_MOONCAKE_CONNECTOR_TIMEOUT": "600", "MC_MAX_MR_SIZE": str(256 << 20),
                         "C48_MOONCAKE_HCA_PER_GPU": "1"}
                if store and role == "prefill":
                    added.update(GLIBC_TUNABLES="glibc.malloc.hugetlb=1", MC_MAX_MR_SIZE=str(4 << 20))
                self.assertEqual({k: v for k, v in mc_env.items() if k not in src_env}, added)
                self.assertEqual({k: v for k, v in src_env.items()}, {k: mc_env[k] for k in src_env})
                t = transfer(mc_argv)
                names = [c["kv_connector"] for c in t["kv_connector_extra_config"].get("connectors", [t])]
                expected = (["MooncakeConnector", "MooncakeStoreConnector"] if store else ["MooncakeConnector"]
                            ) if role == "prefill" else ["MooncakeConnector"]
                self.assertEqual(names, expected)
            if store:
                cpu = transfer(render(src, "prefill", *ADDR)[0])["kv_connector_extra_config"]["connectors"][1]
                self.assertEqual(mc["mooncake"]["store"]["global_segment_size"] * 8,
                                 cpu["kv_connector_extra_config"]["cpu_bytes_to_use"])
                self.assertEqual(mc["mooncake"]["store"]["protocol"], "rdma")
                self.assertEqual(mc["mooncake"]["store"]["local_buffer_size"], 64 << 20)

    def test_mooncake_hca_follows_physical_gpu(self):
        sys.modules.setdefault("vllm.v1.worker.gpu_worker", types.SimpleNamespace(Worker=object))
        from pd_c48_worker import mooncake_hca
        self.assertEqual(mooncake_hca(3, {}), "rdma3")
        self.assertEqual(mooncake_hca(1, {"HIP_VISIBLE_DEVICES": "4,5,6,7"}), "rdma5")
        self.assertEqual(mooncake_hca(0, {"ROCR_VISIBLE_DEVICES": "6", "HIP_VISIBLE_DEVICES": "0"}), "rdma6")

    def test_nic_limits_keeps_mr_lines(self):
        sys.modules.setdefault("vllm.v1.worker.gpu_worker", types.SimpleNamespace(Worker=object))
        import subprocess
        from unittest import mock
        from pd_c48_worker import nic_limits
        out = "hca_id:\trdma3\n\tfw_ver:\t\t1.2\n\tmax_mr_size:\t\t0x380000000\n\tpage_size_cap:\t\t0x40201000\n" \
              "\tmax_qp:\t\t4096\n\tmax_mr:\t\t\t131072\n\tmax_mr_size_extra:\t\t1\n"
        with mock.patch.object(subprocess, "run", return_value=types.SimpleNamespace(stdout=out)):
            self.assertEqual(nic_limits("rdma3"),
                             "fw_ver: 1.2 max_mr_size: 0x380000000 page_size_cap: 0x40201000 max_mr: 131072")
        with mock.patch.object(subprocess, "run", side_effect=FileNotFoundError("ibv_devinfo")):
            self.assertTrue(nic_limits("rdma3").startswith("unavailable"))

    def test_simple_cpu_case_ignores_lmcache_port(self):
        config = json.loads((GAP / "V-QD.json").read_text())
        self.assertEqual(render(config, "prefill", *ADDR)[0], render(config, "prefill", *ADDR, lmcache_port=6655)[0])


if __name__ == "__main__":
    unittest.main()
