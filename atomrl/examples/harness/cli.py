# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Command line of examples/run.sh."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

from . import checks, report
from .manifest import ATOMRL_DIR, TIERS, Catalog, Example, ManifestError, load
from .runner import Options, Outcome, Runner, Skip, git_rev
from .site import Site, load_site

COMMANDS = ("list", "show", "run", "suite", "check", "prepare", "compare")
DEFAULT_TIERS = ("t0", "t1", "t2")

EPILOG = """\
examples:
  run.sh list                               every example, its tier and hardware
  run.sh dapo-qwen3-8b-atom-fp8             one example (smoke variant), judged
  run.sh 4                                  the same, by its release-table number
  run.sh suite                              regression: t0 + t1, and t2 on >= 2 hosts
  run.sh suite --tier long --only flowgrpo-qwen-image-lumen
  run.sh prepare 1                          download models and data for example 1
  run.sh compare A/summary.json B/summary.json

tiers: t0 unit tests, t1 one-node smoke, t2 two-node smoke, long full-size runs
(long runs only when asked for). See examples/harness/README.md.
"""


def _common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("environment")
    g.add_argument(
        "--site", help="site file (default $ATOMRL_SITE, then examples/site.yaml)"
    )
    g.add_argument("--data-root", help="override the site's data_root / $DATA_ROOT")
    g.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="KEY=REF",
        help="use REF for image KEY, e.g. release=atomrl:release-20261116-kernels",
    )
    g.add_argument(
        "--atomrl-src",
        default=str(ATOMRL_DIR),
        help="AtomRL checkout mounted into the container (default: this one)",
    )
    g.add_argument(
        "--atom-src",
        help="ATOM checkout mounted over the image's ATOM; results are then not "
        "comparable with the references",
    )


def _run_opts(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("run")
    g.add_argument(
        "--full", action="store_true", help="the full-size variant, not the smoke"
    )
    g.add_argument("--steps", type=int, help="override the number of training steps")
    g.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra config override",
    )
    g.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra container env",
    )
    g.add_argument("--no-check", action="store_true", help="judge by exit code only")
    g.add_argument(
        "--dry-run", action="store_true", help="print the commands, run nothing"
    )
    g.add_argument("--prepare", action="store_true", help="prepare missing data first")
    g.add_argument(
        "--force",
        action="store_true",
        help="run on busy GPUs and remove other containers of the same image",
    )
    g.add_argument(
        "--keep-cache", action="store_true", help="never clear ATOM compile caches"
    )
    g.add_argument(
        "--keep-container",
        action="store_true",
        help="leave the container running afterwards",
    )
    g.add_argument("--verbose", action="store_true", help="stream the whole log")
    g.add_argument("--results", help="results root (default atomrl/output/regression)")
    g.add_argument("--strict", action="store_true", help="count SKIP as a failure")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.sh",
        description="Run AtomRL examples with one command and judge the results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list", help="list the examples")
    s.add_argument("--tier", action="append", choices=TIERS)
    _common(s)
    s = sub.add_parser("show", help="describe one example and print its commands")
    s.add_argument("example")
    s.add_argument("--full", action="store_true")
    _common(s)
    s = sub.add_parser("run", help="run examples by id or alias")
    s.add_argument("examples", nargs="+")
    _common(s)
    _run_opts(s)
    s = sub.add_parser("suite", help="run a regression suite")
    s.add_argument(
        "--tier",
        action="append",
        choices=TIERS,
        help=f"tiers to run (default {', '.join(DEFAULT_TIERS)}; t2 skips on one host)",
    )
    s.add_argument("--with-long", action="store_true", help="add the long tier")
    s.add_argument("--only", default="", help="comma-separated ids or aliases")
    s.add_argument(
        "--skip", default="", help="comma-separated ids or aliases to leave out"
    )
    s.add_argument("--label", default="suite")
    _common(s)
    _run_opts(s)
    s = sub.add_parser("check", help="judge an existing log")
    s.add_argument("example")
    s.add_argument("--log", required=True)
    s.add_argument("--full", action="store_true")
    s.add_argument("--steps", type=int)
    s = sub.add_parser("prepare", help="download models and data for examples")
    s.add_argument("examples", nargs="+")
    _common(s)
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("compare", help="compare two summary.json files")
    s.add_argument("a")
    s.add_argument("b")
    return p


def _site(args) -> Site:
    site = load_site(getattr(args, "site", None))
    if getattr(args, "data_root", None):
        site.data_root = args.data_root
    return site


def _pairs(items: list[str], flag: str) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ManifestError(f"{flag} wants KEY=VALUE, got {item!r}")
        out[key] = value
    return out


def _images(args) -> dict[str, str]:
    return _pairs(getattr(args, "image", []), "--image")


def _options(args, **extra) -> Options:
    opts = Options(
        full=getattr(args, "full", False),
        steps=getattr(args, "steps", None),
        overrides=tuple(getattr(args, "override", [])),
        check=not getattr(args, "no_check", False),
        dry_run=getattr(args, "dry_run", False),
        prepare=getattr(args, "prepare", False),
        force=getattr(args, "force", False),
        keep_cache=getattr(args, "keep_cache", False),
        keep_container=getattr(args, "keep_container", False),
        verbose=getattr(args, "verbose", False),
        atomrl_src=str(Path(getattr(args, "atomrl_src", ATOMRL_DIR)).resolve()),
        atom_src=(
            str(Path(args.atom_src).resolve())
            if getattr(args, "atom_src", None)
            else None
        ),
        images=_images(args),
        env=_pairs(getattr(args, "env", []), "--env"),
    )
    return dataclasses.replace(opts, **extra)


def cmd_list(catalog: Catalog, site: Site, args) -> int:
    rows = [("id", "alias", "tier", "nodes", "image", "status", "title")]
    for ex in sorted(catalog.examples, key=lambda e: TIERS.index(e.tier)):
        if args.tier and ex.tier not in args.tier:
            continue
        rows.append(
            (
                ex.id,
                ",".join(ex.aliases),
                ex.tier,
                f"{ex.nodes}x{ex.gpus_per_node}",
                ex.image,
                ex.status,
                ex.title,
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)) + "  " + r[-1])
    print(
        f"\nsite: {site.source or 'none (this machine only)'}; {len(site.hosts)} host(s); "
        f"data_root {site.data_root or '(unset)'}"
    )
    return 0


def cmd_show(catalog: Catalog, site: Site, args) -> int:
    ex = catalog.get(args.example)
    runner = Runner(catalog, site, _options(args, dry_run=True), None)
    image = runner.image(ex.image)
    v = ex.variant(args.full)
    print(f"{ex.id}  ({', '.join(ex.aliases) or 'no alias'})  {ex.title}")
    print(
        f"  tier {ex.tier}, {ex.nodes} node(s) x {ex.gpus_per_node} GPUs, status {ex.status}"
    )
    print(f"  defined in {ex.source.relative_to(ATOMRL_DIR)}")
    print(f"  image {ex.image} = {image.ref}")
    if ex.requires:
        print(f"  needs under data_root: {', '.join(ex.requires)}")
    if ex.needs:
        print(f"  needs capabilities: {', '.join(ex.needs)}")
    print(
        f"  {'full' if args.full else 'smoke'}: config {v.config or '-'}, steps {v.steps or '-'}"
    )
    for m in v.checks.metrics:
        how = (
            f"ref {m.ref} rtol {m.rtol}"
            if m.rtol is not None
            else f"ref {m.ref} band x{m.band}" if m.band is not None
            else "info" if m.info
            else f"[{m.min}, {m.max}]"
        )  # fmt: skip
        print(f"    step {v.checks.step} {m.name}: {how}")
    if ex.full and not args.full:
        print(f"  full: config {ex.full.config}, steps {ex.full.steps} (use --full)")
    if ex.notes:
        print("  notes: " + ex.notes.replace("\n", "\n         "))
    print()
    runner.run(ex)
    return 0


def _print_outcome(o: Outcome) -> None:
    print(f"== {o.example.id}: {o.status} ({o.duration_s / 60:.1f} min)  {o.reason}")
    if o.verdict:
        for m in o.verdict.metrics:
            print(f"     {m.name:<26} {m.value!s:<12} {m.detail:<36} {m.status}")


def _execute(
    catalog: Catalog, site: Site, args, selected: list[Example], label: str
) -> int:
    opts = _options(args)
    root = Path(
        args.results or site.results_dir or ATOMRL_DIR / "output" / "regression"
    )
    results = (
        None if opts.dry_run else root / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}"
    )
    runner = Runner(catalog, site, opts, results)
    started = time.time()
    meta = {
        "label": label,
        "started": report.stamp(),
        "argv": sys.argv[1:],
        "atomrl_src": opts.atomrl_src,
        "atomrl_rev": git_rev(opts.atomrl_src),
        "atom_src": opts.atom_src,
        "atom_rev": git_rev(opts.atom_src) if opts.atom_src else None,
        "site": str(site.source) if site.source else None,
        "hosts": [f"{h.name}({h.address})" for h in site.hosts],
        "images": {ex.image: runner.image(ex.image).ref for ex in selected},
    }
    outcomes: list[Outcome] = []
    for ex in selected:
        try:
            o = runner.run(ex)
        except KeyboardInterrupt:
            raise
        # CLI isolates one example runner failure and records an ERROR outcome.
        except Exception as exc:  # noqa: BLE001 -- boundary
            o = Outcome(example=ex, status="ERROR", reason=f"{exc!r}")
            traceback.print_exc()
        outcomes.append(o)
        if not opts.dry_run:
            _print_outcome(o)
            meta["duration_min"] = (time.time() - started) / 60
            report.write(results, meta, outcomes)
    if opts.dry_run:
        return 0
    summary = report.write(results, meta, outcomes)
    print()
    print(report.markdown(summary))
    print(f"results: {results}")
    c = report.counts(summary)
    bad = c["FAIL"] + c["ERROR"] + (c["SKIP"] if args.strict else 0)
    if not c["PASS"] + c["XFAIL"] + c["XPASS"]:
        print("nothing ran to completion here; see the SKIP reasons above")
        bad += 1
    if c["XPASS"]:
        print("note: an example with a known_issue passed; remove its known_issue")
    return 1 if bad else 0


def cmd_run(catalog: Catalog, site: Site, args) -> int:
    selected = [catalog.get(name) for name in args.examples]
    label = selected[0].id if len(selected) == 1 else "run"
    return _execute(catalog, site, args, selected, label)


def cmd_suite(catalog: Catalog, site: Site, args) -> int:
    tiers = list(args.tier or DEFAULT_TIERS) + (["long"] if args.with_long else [])
    only = {catalog.get(n).id for n in args.only.split(",") if n}
    skip = {catalog.get(n).id for n in args.skip.split(",") if n}
    selected = [
        ex
        for t in TIERS
        if t in tiers
        for ex in catalog.examples
        if ex.tier == t and (not only or ex.id in only) and ex.id not in skip
    ]
    if not selected:
        print("nothing selected", file=sys.stderr)
        return 2
    print(
        f"suite: {len(selected)} example(s) in tiers {', '.join(tiers)} on {len(site.hosts)} host(s)"
    )
    return _execute(catalog, site, args, selected, args.label)


def cmd_check(catalog: Catalog, site: Site, args) -> int:
    ex = catalog.get(args.example)
    v = ex.variant(args.full)
    text = Path(args.log).read_text(errors="replace")
    verdict = checks.judge(text, v.checks, args.steps or v.steps)
    print(f"{ex.id}: {verdict.status}")
    for m in verdict.metrics:
        print(f"  {m.name:<26} {m.value!s:<12} {m.detail:<36} {m.status}")
    for k, n in verdict.counts.items():
        print(f"  {k:<26} {n}")
    for r in verdict.reasons:
        print(f"  - {r}")
    return 0 if verdict.status == "PASS" else 1


def cmd_prepare(catalog: Catalog, site: Site, args) -> int:
    runner = Runner(catalog, site, _options(args, force=args.force), None)
    rc = 0
    for name in args.examples:
        ex = catalog.get(name)
        image = runner.image(ex.image)
        head = site.head
        try:
            if not runner.missing_data([head], ex) and not args.force:
                print(f"{ex.id}: data already present")
                continue
            runner.prepare_host(head, image, ex)
            runner.run_prepare(head, image, ex)
            left = runner.missing_data([head], ex)
            print(
                f"{ex.id}: {'ready' if not left else 'still missing ' + ', '.join(left)}"
            )
            rc |= bool(left)
        except Skip as exc:
            print(f"{ex.id}: {exc}")
        # Matrix runner reports one example failure before continuing to the next.
        except Exception as exc:  # noqa: BLE001 -- boundary
            print(f"{ex.id}: ERROR {exc}")
            rc = 1
    return rc


def cmd_compare(args) -> int:
    def load_summary(p: str) -> dict:
        path = Path(p)
        return json.loads(
            (path / "summary.json" if path.is_dir() else path).read_text()
        )

    print(report.compare(load_summary(args.a), load_summary(args.b)))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "run")
    args = parser().parse_args(argv)
    try:
        if args.cmd == "compare":
            return cmd_compare(args)
        catalog = load()
        site = _site(args) if args.cmd != "check" else None
        return {
            "list": cmd_list,
            "show": cmd_show,
            "run": cmd_run,
            "suite": cmd_suite,
            "check": cmd_check,
            "prepare": cmd_prepare,
        }[args.cmd](catalog, site, args)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
