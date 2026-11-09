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
FIXTURE = ROOT / ".github/dashboard/demo/inferencex-pr-4045.json"
NODE = shutil.which("node")


def javascript_function(name):
    start = HTML.index(f"function {name}(")
    end = HTML.index("\nfunction ", start + 1)
    return HTML[start:end]


@unittest.skipUnless(NODE, "Node.js is required for dashboard JavaScript checks")
class ExplorerMetricsTests(unittest.TestCase):
    def run_js(self, script):
        result = subprocess.run(
            [NODE, "-e", script], capture_output=True, text=True, check=False
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


class ExplorerPreviewTests(unittest.TestCase):
    def test_fixture_units_and_complete_sweep(self):
        points = json.loads(FIXTURE.read_text(encoding="utf-8"))["points"]
        self.assertEqual(len(points), 9)
        self.assertEqual(
            {(p["config_label"], p["concurrency"]) for p in points},
            {
                ("1P1D", 56),
                ("1P1D", 72),
                ("1P1D", 96),
                ("1P1D", 112),
                ("1P1D", 128),
                ("1P2D", 56),
                ("1P3D", 12),
                ("1P3D", 16),
                ("1P3D", 24),
            },
        )
        for point in points:
            self.assertAlmostEqual(
                point["total_tput"],
                point["input_tput"] + point["output_tput"],
                delta=0.00002,
            )
            self.assertAlmostEqual(
                point["total_tput"],
                point["tput_per_gpu"] * point["total_gpu"],
                delta=0.0001,
            )
            self.assertLessEqual(point["interactivity"], point["interactivity_p90_itl"])
        self.assertEqual(max(p["total_tput"] for p in points), 343950.97409)
        self.assertEqual(max(p["interactivity_p90_itl"] for p in points), 180.81277)

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


if __name__ == "__main__":
    unittest.main()
