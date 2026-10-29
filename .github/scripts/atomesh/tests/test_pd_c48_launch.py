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
    def test_eval_removes_synthetic_acceptance_without_changing_benchmark(self):
        config = json.loads((GAP / "V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5-PNT-SWB-C32-PP8-MC.json").read_text())
        before = json.dumps(config)
        for role in ("prefill", "decode"):
            argv, env = render(config, role, *ADDR, phase="eval")
            spec = json.loads(argv[argv.index("--speculative-config") + 1])
            self.assertEqual(spec["rejection_sample_method"], "standard")
            self.assertNotIn("synthetic_acceptance_length", spec)
            self.assertNotIn("synthetic_acceptance_rates", spec)
            self.assertEqual(env["C48_DIAG_AL"], "")
            argv, env = render(config, role, *ADDR, phase="benchmark")
            spec = json.loads(argv[argv.index("--speculative-config") + 1])
            self.assertEqual(spec["rejection_sample_method"], "synthetic")
            self.assertEqual(spec["synthetic_acceptance_length"], 3.36)
        self.assertEqual(json.dumps(config), before)

    def test_eval_manifest_rejects_forced_acceptance(self):
        sys.modules.setdefault("vllm.v1.worker.gpu_worker", types.SimpleNamespace(Worker=object))
        from pd_c48_worker import inspect_speculative
        spec = types.SimpleNamespace(rejection_sample_method="standard", num_speculative_tokens=4,
                                     synthetic_acceptance_rates=None, synthetic_acceptance_length=None)
        self.assertEqual(inspect_speculative(spec, 4, None)["rejection_sample_method"], "standard")
        spec.rejection_sample_method = "synthetic"
        with self.assertRaises(ValueError):
            inspect_speculative(spec, 4, None)

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
        cases = sorted(p for p in GAP.glob(prefix + "*-MC.json")
                       if "-PP8-" not in p.stem and "-OPUS-" not in p.stem)
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
                         "C48_MOONCAKE_HCA_PER_GPU": "1", "MC_IB_TC": "104"}
                if store and role == "prefill":
                    added.update(GLIBC_TUNABLES="glibc.malloc.hugetlb=1", MC_MAX_MR_SIZE=str(1 << 30),
                                 C48_MOONCAKE_STORE_COMPACT="1")
                if role == "decode":
                    added.update(C48_MOONCAKE_DIVERGENT_HITS="1")
                self.assertEqual({k: v for k, v in mc_env.items() if k not in src_env}, added)
                self.assertEqual({k: v for k, v in src_env.items()}, {k: mc_env[k] for k in src_env})
                t = transfer(mc_argv)
                names = [c["kv_connector"] for c in t["kv_connector_extra_config"].get("connectors", [t])]
                expected = (["MooncakeConnector", "MooncakeStoreConnector"] if store else ["MooncakeConnector"]
                            ) if role == "prefill" else ["MooncakeConnector"]
                self.assertEqual(names, expected)
            if store:
                cpu = transfer(render(src, "prefill", *ADDR)[0])["kv_connector_extra_config"]["connectors"][1]
                segment = mc["mooncake"]["store"]["global_segment_size"]
                self.assertEqual(segment, cpu["kv_connector_extra_config"]["cpu_bytes_to_use"] // 8 >> 21 << 21)
                self.assertEqual(mc["mooncake"]["store"]["protocol"], "rdma")
                self.assertEqual(mc["mooncake"]["store"]["local_buffer_size"], 64 << 20)

    def test_mooncake_hca_follows_physical_gpu(self):
        sys.modules.setdefault("vllm.v1.worker.gpu_worker", types.SimpleNamespace(Worker=object))
        from pd_c48_worker import mooncake_hca
        self.assertEqual(mooncake_hca(3, {}), "rdma3")
        self.assertEqual(mooncake_hca(1, {"HIP_VISIBLE_DEVICES": "4,5,6,7"}), "rdma5")
        self.assertEqual(mooncake_hca(0, {"ROCR_VISIBLE_DEVICES": "6", "HIP_VISIBLE_DEVICES": "0"}), "rdma6")

    def test_pp_and_opus_candidates_preserve_unrelated_recipe_settings(self):
        prefix = "V-QD-CPUREF3-FP8ATTN-FQ3-PQ-LDFP4-INPAD-WARM-DG5-PNT-SWB-"
        for concurrency, variant in ((32, "PP8"), (72, "OPUS"), (88, "OPUS")):
            base = json.loads((GAP / f"{prefix}C{concurrency}-MC.json").read_text())
            candidate = json.loads((GAP / f"{prefix}C{concurrency}-{variant}-MC.json").read_text())
            candidate["name"] = base["name"]
            if variant == "PP8":
                for role in ("prefill", "decode"):
                    self.assertEqual(candidate["roles"][role]["env"].pop("C48_MOONCAKE_DCP_TRANSFER"), "1")
                self.assertEqual(candidate["roles"]["prefill"]["env"].pop("VLLM_PP_LAYER_PARTITION"),
                                 "12,12,12,12,12,12,12,9")
                args = candidate["roles"]["prefill"]["args"]
                connector = args["kv-transfer-config"]["kv_connector_extra_config"]["connectors"][0]
                self.assertEqual(connector["kv_connector_extra_config"].pop("dcp_staging_bytes"), 64 << 20)
                self.assertEqual(args.pop("pipeline-parallel-size"), 8)
                for key in ("tensor-parallel-size", "decode-context-parallel-size"):
                    self.assertEqual(args[key], 1)
                    args[key] = base["roles"]["prefill"]["args"][key]
            else:
                self.assertEqual(candidate["roles"]["decode"]["env"].pop("AITER_MLA_USE_OPUS"), "1")
            self.assertEqual(candidate, base)

    def test_nic_limits_degrades_without_verbs(self):
        sys.modules.setdefault("vllm.v1.worker.gpu_worker", types.SimpleNamespace(Worker=object))
        import ctypes
        from unittest import mock
        from pd_c48_worker import nic_limits
        with mock.patch.object(ctypes, "CDLL", side_effect=OSError("libibverbs.so.1")):
            self.assertTrue(nic_limits("rdma3").startswith("unavailable"))

    def test_store_memory_compacts_each_node_once(self):
        sys.modules.setdefault("vllm.v1.worker.gpu_worker", types.SimpleNamespace(Worker=object))
        import os
        import subprocess
        import tempfile
        from unittest import mock
        import pd_c48_worker
        worker = types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(model="/m/a/K3"),
                                              speculative_config=types.SimpleNamespace(model="/m/b/D")),
            parallel_config=types.SimpleNamespace(world_size=8))
        with tempfile.TemporaryDirectory() as tmp:
            sysfs, gib = Path(tmp, "node"), 1 << 30
            for n, free in ((0, 900), (1, 200)):
                d = sysfs / f"node{n}"
                d.mkdir(parents=True)
                (d / "cpulist").write_text("0-1\n")
                (d / "meminfo").write_text(f"Node {n} MemFree: {free * gib >> 10} kB\nNode {n} Active(file): "
                                           f"{100 * gib >> 10} kB\nNode {n} Inactive(file): 0 kB\nNode {n} HugePages_Total: 0\n")
            (sysfs / "node2").mkdir()
            (sysfs / "node2" / "cpulist").write_text("\n")
            Path(tmp, "store.json").write_text(json.dumps({"global_segment_size": 100 * gib}))
            env = {"C48_DIAG_DIR": str(Path(tmp, "diag")), "MOONCAKE_CONFIG_PATH": str(Path(tmp, "store.json"))}
            real_path = pd_c48_worker.Path
            fake_path = lambda p, *a: real_path(str(p).replace("/sys/devices/system/node", str(sysfs)), *a)
            dropped = []
            sys.modules["drop_page_cache"] = types.SimpleNamespace(drop_page_cache=dropped.append)
            with mock.patch.dict(os.environ, env), mock.patch.object(pd_c48_worker, "Path", fake_path), \
                    mock.patch.object(subprocess, "run") as run:
                pd_c48_worker.prepare_store_memory(worker)
                pd_c48_worker.prepare_store_memory(worker)
            del sys.modules["drop_page_cache"]
        self.assertEqual(dropped, ["/m"])
        self.assertEqual([(c.args[0][2], c.args[0][-1]) for c in run.call_args_list], [("0", "400"), ("1", "236")])

    def test_mooncake_divergent_hits_only_on_decode(self):
        from unittest import mock

        class Base:
            supports_divergent_local_hybrid_hits = property(lambda self: False)

        class MooncakeConnector(Base):
            pass

        mods = {"vllm.v1.simple_kv_offload": types.ModuleType("m"),
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector":
                    types.SimpleNamespace(MooncakeConnector=MooncakeConnector)}
        mods["vllm.v1.simple_kv_offload"].manager = types.SimpleNamespace()
        with mock.patch.dict(sys.modules, mods):
            sys.modules.pop("pd_c48_scheduler", None)
            from pd_c48_scheduler import enable_mooncake_divergent_hits
            config = lambda role: types.SimpleNamespace(kv_transfer_config=types.SimpleNamespace(kv_role=role))
            with self.assertRaises(RuntimeError):
                enable_mooncake_divergent_hits(types.SimpleNamespace(connector=Base()), config("kv_consumer"))
            with self.assertRaises(RuntimeError):
                enable_mooncake_divergent_hits(types.SimpleNamespace(connector=MooncakeConnector()),
                                               config("kv_producer"))
            self.assertFalse(MooncakeConnector().supports_divergent_local_hybrid_hits)
            enable_mooncake_divergent_hits(types.SimpleNamespace(connector=MooncakeConnector()), config("kv_consumer"))
            self.assertTrue(MooncakeConnector().supports_divergent_local_hybrid_hits)
            self.assertFalse(Base().supports_divergent_local_hybrid_hits)
        sys.modules.pop("pd_c48_scheduler", None)

    def test_simple_cpu_case_ignores_lmcache_port(self):
        config = json.loads((GAP / "V-QD.json").read_text())
        self.assertEqual(render(config, "prefill", *ADDR)[0], render(config, "prefill", *ADDR, lmcache_port=6655)[0])


if __name__ == "__main__":
    unittest.main()
