# SPDX-License-Identifier: MIT
"""CPU-only regression checks for dashboard metrics and local previews."""

import copy
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / ".github/dashboard/agentic.html").read_text(encoding="utf-8")
NODE = shutil.which("node")


def javascript_function(name):
    start = HTML.index(f"function {name}(")
    end = HTML.index("\nfunction ", start + 1)
    return HTML[start:end]


@unittest.skipUnless(NODE, "Node.js is required for dashboard JavaScript checks")
class ExplorerMetricsTests(unittest.TestCase):
    def run_js(self, script):
        result = subprocess.run(
            [NODE], input=script, capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_inline_javascript_syntax(self):
        scripts = re.findall(r"<script>(.*?)</script>", HTML, flags=re.DOTALL)
        script = "const vm = require('node:vm');\n"
        script += (
            "for (const source of " + json.dumps(scripts) + ") new vm.Script(source);"
        )
        # Windows limits command-line length; feed this larger check over stdin.
        result = subprocess.run(
            [NODE], input=script, capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_p90_does_not_substitute_legacy_median(self):
        start = HTML.index("const ATOMESH_INTERACTIVITY_METRICS =")
        end = HTML.index("\nfunction atomeshInteractivityMetric", start)
        self.run_js(
            "const assert = require('node:assert/strict');\n"
            + HTML[start:end]
            + javascript_function("explorerMetricValue")
            + """
const [normalized, itl] = ATOMESH_INTERACTIVITY_METRICS;
const legacy = {interactivity: 250, interactivityMethod: 'median_tpot'};
assert.equal(explorerMetricValue(legacy, normalized), null);
assert.equal(explorerMetricValue(legacy, itl), null);
const measured = {...legacy, interactivityMethod: 'p90_e2e_normalized', interactivityP90Itl: 180};
assert.equal(explorerMetricValue(measured, normalized), 250);
assert.equal(explorerMetricValue(measured, itl), 180);
for (const value of [null, undefined, NaN, Infinity, 0, -1]) {
  assert.equal(explorerMetricValue({...measured, interactivity: value}, normalized), null);
  assert.equal(explorerMetricValue({...measured, interactivityP90Itl: value}, itl), null);
}
"""
        )

    def test_frontier_maximizes_both_axes_and_handles_ties(self):
        self.run_js(
            "const assert = require('node:assert/strict');\n"
            + javascript_function("atomeshParetoFrontier")
            + """
const points = [
  {x: 5, y: 2}, {x: 1, y: 7}, {x: 3, y: 4},
  {x: 3, y: 3}, {x: 2, y: 4}, {x: 3, y: 4}, {x: 1, y: 1},
];
const before = JSON.stringify(points);
assert.deepEqual(atomeshParetoFrontier(points), [{x: 1, y: 7}, {x: 3, y: 4}, {x: 5, y: 2}]);
assert.equal(JSON.stringify(points), before);
assert.deepEqual(atomeshParetoFrontier([]), []);
"""
        )

    def test_modes_and_variants_keep_independent_frontiers(self):
        self.run_js(
            "const assert = require('node:assert/strict');\n"
            + "\n".join(
                javascript_function(name)
                for name in [
                    "explorerDeployment",
                    "explorerSeriesKey",
                    "atomeshParetoFrontier",
                    "explorerFrontiers",
                ]
            )
            + """
const row = {model:'Example', hardware:'mi355x', sourceLabel:'ATOM CI'};
const pd = {...row, disaggregated:true, isMultinode:false};
const agg = {...row, disaggregated:false, prefillWorkers:1, decodeWorkers:1};
assert.equal(explorerDeployment(pd), 'pd');
assert.equal(explorerDeployment(agg), 'aggregated');
assert.equal(explorerDeployment({...row,isMultinode:false}), 'unknown');
assert.equal(explorerDeployment({...row,configLabel:'legacy_agg_tp8'}), 'aggregated');
assert.equal(explorerDeployment({...row,prefillWorkers:1,decodeWorkers:2}), 'pd');
const points = [{x:100,y:100,row:pd}, {x:10,y:10,row:agg},
                {x:5,y:5,row:agg}, {x:1,y:1,row:{...agg,model:'Example-Eagle3'}}];
assert.deepEqual(explorerFrontiers(points).flat().map(p=>p.x), [100,10,1]);
"""
        )

    def test_throughput_per_gpu_drives_frontier_without_double_normalization(self):
        self.run_js(
            "const assert = require('node:assert/strict');\n"
            + javascript_function("explorerThroughputValue")
            + javascript_function("atomeshParetoFrontier")
            + """
assert.equal(explorerThroughputValue({totalTputPerGpu:100.125,totalTput:800,totalGpu:8}),100.125);
assert.equal(explorerThroughputValue({totalTput:800,totalGpu:8}),100);
for (const count of [null,undefined,0,-1,1.5,NaN,Infinity]) {
  assert.equal(explorerThroughputValue({totalTput:800,totalGpu:count}),null);
}
assert.equal(explorerThroughputValue({totalTputPerGpu:0,totalTput:800,totalGpu:8}),null);
assert.equal(explorerThroughputValue({totalTput:Infinity,totalGpu:8}),null);
const rows = [{x:80,totalTput:1600,totalGpu:32},{x:90,totalTput:1200,totalGpu:8}];
assert.equal(atomeshParetoFrontier(rows.map(row=>({x:row.x,y:row.totalTput}))).length,2);
assert.deepEqual(atomeshParetoFrontier(rows.map(row=>({x:row.x,y:explorerThroughputValue(row)}))),[{x:90,y:150}]);
"""
        )

    def test_ci_points_preserve_recipes_and_use_latest_publication(self):
        functions = [
            "parseAtomeshNumber",
            "atomeshInteractivityMethod",
            "atomeshHardwareKey",
            "parseAtomeshPerfPointExtra",
            "atomeshPointNumber",
            "atomeshPointRow",
            "atomeshMergedModelName",
            "isAgenticTraceRow",
            "explorerWorkloadKey",
            "explorerDeployment",
            "buildExplorerBenchmarkRows",
            "explorerThroughputValue",
            "atomeshParetoFrontier",
        ]
        self.run_js(
            "const assert = require('node:assert/strict');\n"
            "const ATOMESH_DATA = {backend: 'ATOMesh', hardware: 'MI355X'};\n"
            + "\n".join(javascript_function(name) for name in functions)
            + """
const base = {source: 'ATOMesh', model: 'GLM-5.2', hardware: 'mi355x',
  precision: 'fp4', benchmark_kind: 'aiperf_agentic', public_dataset: 'agentic-trace',
  concurrency: 56, total_gpu: 8, interactivity_method: 'p90_e2e_normalized', interactivity: 30};
const bench = point => ({extra: 'Run: https://github.com/ROCm/ATOM/actions/runs/1 | perf_point=' + encodeURIComponent(JSON.stringify({...base, ...point}))});
const older = {date: 100, benches: [bench({config_label: 'recipe-a', total_tput: 900, interactivity_p90_itl: 100})]};
const newer = {date: 200, benches: [
  bench({config_label: 'recipe-a', total_tput: 200, tput_per_gpu: 25, interactivity_p90_itl: 60}),
  bench({config_label: 'recipe-b', total_tput: 100, interactivity_p90_itl: 90}),
  bench({config_label: 'recipe-b', total_tput: 100, interactivity_p90_itl: 90}),
  bench({config_label: 'recipe-a', precision: 'fp8', total_tput: 150, interactivity_p90_itl: 70}),
  bench({config_label: 'recipe-a', public_dataset: 'other-trace', total_tput: 180, interactivity_p90_itl: 75}),
]};
newer.benches.push(bench({benchmark_kind:'serving', public_dataset:'', config_label:'standard', total_tput:999999}));
const data = {entries: {CI: [older, newer]}};
const before = JSON.stringify(data);
const rows = buildExplorerBenchmarkRows(data);
assert.equal(JSON.stringify(data), before);
assert.equal(rows.length, 4);
const cohort = rows.filter(row => row.precision === 'fp4' && row.publicDataset === 'agentic-trace');
assert.deepEqual(cohort.map(row => row.totalTput), [200, 100]);
assert.deepEqual(cohort.map(explorerThroughputValue), [25, 12.5]);
assert(cohort.every(row => row.date === 200 && row.sourceLabel === 'ATOM CI'));
assert(cohort.every(row => row.runUrl === 'https://github.com/ROCm/ATOM/actions/runs/1'));
// Same concurrency: the slower-throughput recipe is still a valid frontier point.
assert.equal(atomeshParetoFrontier(cohort.map(row => ({x: row.interactivityP90Itl, y: explorerThroughputValue(row)}))).length, 2);
"""
        )


class ExplorerPreviewTests(unittest.TestCase):
    def test_preview_keeps_public_data_and_safely_embeds_local_points(self):
        spec = importlib.util.spec_from_file_location(
            "preview", ROOT / "tools/preview_benchmark_dashboard.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in module.ASSETS:
                (output / name).write_text("// cached public asset", encoding="utf-8")
            local = output / "local.json"
            local.write_text(
                json.dumps(
                    {"points": [{"model": "model </script><script>alert(1)</script>"}]}
                ),
                encoding="utf-8",
            )
            fragment = module.prepare(output, [local])
            generated = (output / "agentic.html").read_text(encoding="utf-8")
            self.assertIn("explorerModel=", fragment)
            self.assertIn("window.ATOM_LOCAL_BENCHMARK", generated)
            self.assertNotIn("model </script><script>", generated)
            self.assertIn(r"model \u003c/script>", generated)
            self.assertNotIn(
                "window.ATOM_LOCAL_BENCHMARK",
                (output / "index.html").read_text(encoding="utf-8"),
            )
            self.assertEqual((output / "data.js").read_text(), "// cached public asset")
            self.assertNotIn('src="https://cdnjs.cloudflare.com', generated)

    def test_default_preview_uses_ci_without_sample_injection(self):
        spec = importlib.util.spec_from_file_location(
            "preview", ROOT / "tools/preview_benchmark_dashboard.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in module.ASSETS:
                (output / name).write_text("// cached public asset", encoding="utf-8")
            fragment = module.prepare(output)
            generated = (output / "agentic.html").read_text(encoding="utf-8")
            self.assertIn("explorerSource=ATOM+CI", fragment)
            self.assertIn('<script src="data.js"></script>', generated)
            self.assertNotIn("window.ATOM_LOCAL_BENCHMARK =", generated)


class SweepImportTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "preview", ROOT / "tools/preview_benchmark_dashboard.py"
        )
        self.preview = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.preview)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.file = Path(directory.name) / "agg_bmk.json"
        self.run = "https://github.com/ROCm/ATOM/actions/runs/123456789"
        self.record = {
            "model": "test/Example-FP4",
            "framework": "atom-disagg",
            "hw": "cluster:mi355x-test",
            "precision": "fp4",
            "conc": 16,
            "num_gpus": 8,
            "prefill_num_workers": 1,
            "decode_num_workers": 1,
            "recipe_fingerprint": "recipe-a",
            "run_attempt": 2,
            "dataset": {"loader": "test-trace"},
            "request_metrics": {
                "latency": {
                    "intvty": {"p90": 83.12345},
                    "e2e_norm_intvty": {"p90": 41.67891},
                    "itl": {"p90": 0.012},
                },
                "throughput": {
                    "total": {"tokens_per_second": 123456.789},
                    "input": {"tokens_per_second": 120000},
                    "output": {"tokens_per_second": 3456.789},
                    "per_gpu": {"total_tput_tps": 15432.098625},
                },
            },
        }

    def write_records(self, records):
        self.file.write_text(json.dumps(records), encoding="utf-8")

    def test_sweep_preserves_measured_values_recipe_and_run_attempt(self):
        other = {**self.record, "recipe_fingerprint": "recipe-b"}
        self.write_records([self.record, other])
        first, second = self.preview.load_sweep(self.file, self.run)
        self.assertEqual(first["interactivity_p90_itl"], 83.12345)
        self.assertEqual(first["interactivity"], 41.67891)
        self.assertEqual(first["total_tput"], 123456.789)
        self.assertEqual(first["tput_per_gpu"], 15432.098625)
        self.assertEqual(first["hardware"], "mi355x")
        self.assertEqual(first["public_dataset"], "test-trace")
        self.assertEqual(first["run_url"], self.run + "/attempts/2")
        self.assertIn("attempt 2", first["source_label"])
        self.assertEqual(first["prefill_workers"], 1)
        self.assertNotEqual(first["config_label"], second["config_label"])

    def test_sweep_preserves_pr_and_serving_mode_independently_of_placement(self):
        self.record.update(
            disagg=True,
            is_multinode=False,
            num_nodes=1,
            pr_number=123,
            head_sha="a" * 40,
            sweep_status="partial",
        )
        self.write_records([self.record])
        point = self.preview.load_sweep(self.file, self.run)[0]
        self.assertTrue(point["disaggregated"])
        self.assertFalse(point["is_multinode"])
        self.assertEqual(point["num_nodes"], 1)
        self.assertEqual(point["pr_url"], "https://github.com/ROCm/ATOM/pull/123")
        self.assertIn("PR #123", point["source_label"])
        self.assertEqual(point["head_sha"], "a" * 40)
        self.assertEqual(point["source_status"], "Partial sweep")
        self.record["disagg"] = False
        self.write_records([self.record])
        self.assertFalse(
            self.preview.load_sweep(self.file, self.run)[0]["disaggregated"]
        )

    def test_missing_p90_is_not_replaced_with_mean_or_rounded_latency(self):
        for missing in (None, {"mean": 100}, {"p50": 100}, {"p90": False}):
            with self.subTest(missing=missing):
                record = copy.deepcopy(self.record)
                record["request_metrics"]["latency"]["intvty"] = missing
                record["request_metrics"]["latency"]["e2e_norm_intvty"] = missing
                self.write_records([record])
                point = self.preview.load_sweep(self.file, self.run)[0]
                self.assertIsNone(point["interactivity_p90_itl"])
                self.assertIsNone(point["interactivity"])
                self.assertEqual(point["total_tput"], 123456.789)

    def test_retry_assembly_is_one_source_with_original_point_attempts(self):
        records = [
            {
                **self.record,
                "conc": 16 * attempt,
                "run_attempt": attempt,
                "sweep_run_attempt": 2,
                "run_url": self.run + f"/attempts/{attempt}",
            }
            for attempt in (1, 2)
        ]
        self.write_records(records)
        for supplied_url in (None, self.run, self.run + "/attempts/2"):
            points = self.preview.load_sweep(self.file, supplied_url)
            self.assertEqual(len({point["source_label"] for point in points}), 1)
            self.assertTrue(points[0]["source_url"].endswith("/attempts/2"))
            self.assertTrue(points[0]["run_url"].endswith("/attempts/1"))
            self.assertTrue(points[1]["run_url"].endswith("/attempts/2"))
        for attempt in (0, 1, "2", True):
            self.write_records([{**records[1], "sweep_run_attempt": attempt}])
            with self.assertRaises(ValueError):
                self.preview.load_sweep(self.file)

    def test_sweep_requires_valid_consistent_run_provenance(self):
        self.write_records([self.record])
        for url in (None, "javascript:alert(1)", self.run + "/attempts/3"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.preview.load_sweep(self.file, url)
        self.record["run_url"] = self.run
        self.write_records([self.record])
        self.assertEqual(
            self.preview.load_sweep(self.file)[0]["run_url"], self.run + "/attempts/2"
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.preview.load_sweep(self.file, self.run + "0")

    def test_mixed_engine_sweep_does_not_mislabel_non_atom_points(self):
        other = {**self.record, "framework": "sglang"}
        self.write_records([self.record, other])
        self.assertEqual(len(self.preview.load_sweep(self.file, self.run)), 1)
        for payload in ({"points": []}, [1]):
            self.write_records(payload)
            with self.assertRaises(TypeError):
                self.preview.load_sweep(self.file, self.run)

    def test_sweep_preview_selects_imported_source_without_changing_ci_data(self):
        self.write_records([self.record])
        output = self.file.parent / "site"
        output.mkdir()
        for name in self.preview.ASSETS:
            (output / name).write_text("// original asset", encoding="utf-8")
        fragment = self.preview.prepare(
            output, [], sweep_file=self.file, run_url=self.run
        )
        self.assertIn("explorerSource=Sweep", fragment)
        self.assertIn("hardware=mi355x", fragment)
        self.assertIn(
            self.run + "/attempts/2",
            (output / "agentic.html").read_text(encoding="utf-8"),
        )
        self.assertEqual((output / "data.js").read_text(), "// original asset")


if __name__ == "__main__":
    unittest.main()
