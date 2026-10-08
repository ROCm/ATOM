#!/usr/bin/env python3
"""Run the real pinned InferenceX pure parsers, without a database or Bun.

Development check: requires Node >=22.13 and an InferenceX-app source checkout.
The producer has no Node runtime dependency. The independent AgenticViewer
repository owns visualization and is not imported by this compatibility check.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from atom.benchmarks.results.aggregate import COMPATIBILITY_COMMIT
from atom.benchmarks.results.bundle import build_bundle
from atom.benchmarks.results.io import read_json, write_json

# Only TypeScript syntax and module specifiers change. Metric/parser code is
# executed unmodified. Type-only DB imports are erased by Node itself.
NODE = r"""
import { stripTypeScriptTypes } from 'node:module';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
const [src, out, bundle, fixture] = process.argv.slice(2);
const done = new Set();
function compile(relative) {
  if (done.has(relative)) return;
  done.add(relative);
  let code = stripTypeScriptTypes(readFileSync(path.join(src, relative),'utf8'));
  code = code.replace(/(from\s+['"])([^'"]+)(['"])/g, (all, before, spec, after) => {
    if (spec.startsWith('node:')) return all;
    const target = spec === '@semianalysisai/inferencex-constants'
      ? 'packages/constants/src/index.ts'
      : path.join(path.dirname(relative),spec.replace(/\.js$/, '') + '.ts');
    compile(target);
    let dest = path.relative(path.dirname(relative), target).replace(/\.ts$/, '.mjs');
    if (!dest.startsWith('.')) dest = './'+dest;
    return before + dest + after;
  });
  const dest = path.join(out,relative.replace(/\.ts$/,'.mjs'));
  mkdirSync(path.dirname(dest),{recursive:true});
  writeFileSync(dest,code);
}
for (const file of ['benchmark-mapper','full-response-interactivity','skip-tracker']) compile(`packages/db/src/etl/${file}.ts`);
const mod = name => import(pathToFileURL(path.join(out,`packages/db/src/etl/${name}.mjs`)));
const { mapBenchmarkRow } = await mod('benchmark-mapper');
const { fullResponseMetricsFromProfile } = await mod('full-response-interactivity');
const { createSkipTracker } = await mod('skip-tracker');
const exported = JSON.parse(readFileSync(path.join(bundle,'exports/inferencex-v3.json')));
const summary = JSON.parse(readFileSync(path.join(bundle,'summary.json')));
const tracker = createSkipTracker();
const mapped = mapBenchmarkRow(exported,tracker);
assert.ok(mapped,JSON.stringify(tracker.skips));
assert.equal(Object.values(tracker.skips).reduce((a,b)=>a+b,0),0);
assert.equal(tracker.unmappedPrecisions.size,0);
assert.equal(mapped.config.model,'dsv41flash');
assert.equal(mapped.config.hardware,'mi355x');
assert.equal(mapped.config.framework,'atom');
assert.equal(mapped.config.numDecodeGpu,2);
assert.equal(mapped.conc,2);
const close = (actual, expected) => assert.ok(Math.abs(actual-expected) < 1e-6, `${actual} != ${expected}`);
const full = fullResponseMetricsFromProfile(readFileSync(fixture,'utf8'));
for (const [stat, canonical] of [['mean','mean'],['p50','median'],['p90','p90'],['p99','p99']]) {
  close(mapped.metrics[`${canonical}_ttft`],summary.request_metrics.latency.ttft[stat]);
  close(mapped.metrics[`${canonical}_intvty`],summary.request_metrics.latency.full_response_intvty[stat]);
  close(full[`${canonical}_intvty`],summary.request_metrics.latency.full_response_intvty[stat]);
}
close(mapped.metrics.output_tput_per_gpu,summary.request_metrics.throughput.per_gpu.output_tput_tps);
// Normalized E2E is intentionally verified separately: this upstream v3
// flattener does not consume the nested e2e_norm_intvty block.
assert.equal(mapped.metrics.p90_e2e_norm_intvty,undefined);
console.log(JSON.stringify({mapped:true,skips:tracker.skips,config:mapped.config,files:[...done]}));
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--report")
    parser.add_argument(
        "--core-source",
        help="Pinned InferenceX core checkout, for normalized metric cross-check",
    )
    args = parser.parse_args()
    source = Path(args.source).resolve()
    fixtures = Path(__file__).parent / "fixtures"
    lock = read_json(fixtures / "inferencex-source-lock.json")
    for name, expected in lock["sources_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned source hash mismatch: {name}")
    with tempfile.TemporaryDirectory(prefix="atom-inferencex-check-") as tmp:
        tmp = Path(tmp)
        raw = tmp / "aiperf"
        raw.mkdir()
        write_json(raw / "profile_export_aiperf.json", {"submission_valid": True})
        write_json(raw / "server_metrics_export.json", {"fixture": True})
        bundle = tmp / "bundle"
        build_bundle(
            read_json(fixtures / "agentic-config.json"),
            fixtures / "agentic.jsonl",
            bundle,
            record_format="aiperf",
            raw_dir=raw,
        )
        script = tmp / "check.mjs"
        script.write_text(NODE)
        result = subprocess.run(
            [
                "node",
                str(script),
                str(source),
                str(tmp / "compiled"),
                str(bundle),
                str(fixtures / "agentic.jsonl"),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        report = json.loads(result.stdout)
        report["expected_upstream_commit"] = COMPATIBILITY_COMMIT
        report["source_hashes_verified"] = True
        if args.core_source:
            sys.path.insert(0, str(Path(args.core_source).resolve()))
            from infx.results.agentic.request_metrics import compute_latency_stats

            rows = [
                json.loads(line)
                for line in (fixtures / "agentic.jsonl").read_text().splitlines()
            ]
            rows = [
                row
                for row in rows
                if not row.get("error")
                and row["metadata"]["benchmark_phase"] == "profiling"
            ]
            _flat, nested = compute_latency_stats(rows)
            actual = read_json(bundle / "summary.json")["request_metrics"]["latency"][
                "e2e_norm_intvty"
            ]
            for stat in ("mean", "p50", "p75", "p90", "p95"):
                assert abs(actual[stat] - nested["e2e_norm_intvty"][stat]) < 1e-9
            report["normalized_matches_inferencex_core"] = True
            core = Path(args.core_source)
            report["core_source_sha256"] = {
                name: hashlib.sha256((core / name).read_bytes()).hexdigest()
                for name in (
                    "infx/results/agentic/common.py",
                    "infx/results/agentic/request_metrics.py",
                )
            }
        report["sources_sha256"] = {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in report.pop("files")
        }
        report["limitations"] = [
            "CPU fixtures only; no GPU or GitHub artifact round-trip.",
            "SA v3 mapper ignores nested E2E normalized metrics; its trace-derived path is separate.",
        ]
        if args.report:
            write_json(args.report, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
