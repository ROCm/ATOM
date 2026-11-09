# SPDX-License-Identifier: MIT
"""CPU-only regression checks for dashboard metrics and local previews."""

import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / ".github/dashboard/index.html").read_text(encoding="utf-8")
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
            "buildExplorerBenchmarkRows",
            "atomeshParetoFrontier",
        ]
        self.run_js(
            "const assert = require('node:assert/strict');\n"
            "const ATOMESH_DATA = {backend: 'ATOMesh', hardware: 'MI355X'};\n"
            + "\n".join(javascript_function(name) for name in functions)
            + """
const base = {source: 'ATOMesh', model: 'GLM-5.2', hardware: 'mi355x',
  precision: 'fp4', benchmark_kind: 'aiperf_agentic', public_dataset: 'agentic-trace',
  concurrency: 56, interactivity_method: 'p90_e2e_normalized', interactivity: 30};
const bench = point => ({extra: 'Run: https://github.com/ROCm/ATOM/actions/runs/1 | perf_point=' + encodeURIComponent(JSON.stringify({...base, ...point}))});
const older = {date: 100, benches: [bench({config_label: 'recipe-a', total_tput: 900, interactivity_p90_itl: 100})]};
const newer = {date: 200, benches: [
  bench({config_label: 'recipe-a', total_tput: 200, interactivity_p90_itl: 60}),
  bench({config_label: 'recipe-b', total_tput: 100, interactivity_p90_itl: 90}),
  bench({config_label: 'recipe-b', total_tput: 100, interactivity_p90_itl: 90}),
  bench({config_label: 'recipe-a', precision: 'fp8', total_tput: 150, interactivity_p90_itl: 70}),
  bench({config_label: 'recipe-a', public_dataset: 'other-trace', total_tput: 180, interactivity_p90_itl: 75}),
]};
const rows = buildExplorerBenchmarkRows({entries: {CI: [older, newer]}});
assert.equal(rows.length, 4);
const cohort = rows.filter(row => row.precision === 'fp4' && row.publicDataset === 'agentic-trace');
assert.deepEqual(cohort.map(row => row.totalTput), [200, 100]);
assert(cohort.every(row => row.date === 200 && row.sourceLabel === 'ATOM CI'));
assert(cohort.every(row => row.runUrl === 'https://github.com/ROCm/ATOM/actions/runs/1'));
// Same concurrency: the slower-throughput recipe is still a valid frontier point.
assert.equal(atomeshParetoFrontier(cohort.map(row => ({x: row.interactivityP90Itl, y: row.totalTput}))).length, 2);
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
            generated = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("mode=Disaggregated", fragment)
            self.assertIn("window.ATOM_LOCAL_BENCHMARK", generated)
            self.assertNotIn("model </script><script>", generated)
            self.assertIn(r"model \u003c/script>", generated)
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
            fragment = module.prepare(output, [])
            generated = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("explorerSource=ATOM+CI", fragment)
            self.assertIn('<script src="data.js"></script>', generated)
            self.assertNotIn("window.ATOM_LOCAL_BENCHMARK =", generated)


if __name__ == "__main__":
    unittest.main()
