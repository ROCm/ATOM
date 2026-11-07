"""Build a small static site from a directory of roofline runs.

Two charts answer two different questions and get two different pages.

The per-operator page (`render_report.py`) asks "how far is this kernel from its
ceiling". This one asks "what regime is the SYSTEM in, and does batching ever
change it" -- one point per concurrency, joined into a curve of throughput
against interactivity. It is the same sweep read a different way: attainable /
FLOPs-per-token is a throughput ceiling, and 1/step-time is an interactivity
ceiling.

Every run's JSON carries its own `identity`, so this file parses no filenames
and reads no model config. Dropping a new JSON into the
runs directory adds a point; a new model adds a curve. Neither needs a code
change, which is the only way "every model, every concurrency" stays tractable.

No JavaScript, same as the operator page: the switchers are CSS radio siblings so
the page opens from a file:// path and screenshots into a slide.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

from render_report import C, build_html, nice_ticks, peaks_from, tick_txt


def load_runs(root: Path) -> list[dict[str, Any]]:
    """Every run under `root`, keyed by what it says it is rather than where it is."""
    runs = []
    for path in sorted(root.rglob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if "identity" not in doc or "rows" not in doc:
            continue  # not one of ours
        doc["_path"] = path
        runs.append(doc)
    return runs


def measured_point(doc: dict[str, Any]) -> dict[str, Any] | None:
    """One serving point: interactivity against throughput, plus its ceiling.

    These are the axes a serving decision is actually made on. One decode step
    emits one token per user, so

        interactivity = 1 / step            tok/s/user
        throughput    = concurrency / step / tp     tok/s/GPU

    and the trade-off between them is the whole curve: more concurrency fills the
    GPU and raises throughput, while each user waits longer behind a bigger batch.

    The roofline's contribution is the CEILING on that point -- what the same step
    would cost if every operator ran at its bound. It is optimistic and the caller
    is told by how much: it covers only the operators that have a cost model, it
    ignores the per-kernel launch floor that most of a small-batch decode step is
    actually sitting on, and it ignores the idle between kernels. The multiple is
    computed per point and printed beside it -- a factor written into a docstring
    is one nothing checks, and this one moves with every formula change. Shown as
    a direction, not a target.
    """
    ident = doc["identity"]
    steps = int(doc.get("sampled_steps") or 0)
    conc = ident.get("concurrency") or 0
    tp = ident.get("tp") or 1
    if not steps or not conc or ident.get("phase") != "decode":
        return None
    rows = doc["rows"]
    busy_us = sum(float(r.get("busy_us") or 0) for r in rows) / steps
    roof_us = (
        sum(
            float(r.get("t_roofline_us") or 0)
            for r in rows
            if r.get("tier") == "roofline"
        )
        / steps
    )
    if busy_us <= 0:
        return None
    wall_us = float(
        ((doc.get("phase_steps") or {}).get("decode") or {}).get("median_us") or busy_us
    )
    priced = sum(
        float(r.get("busy_us") or 0) for r in rows if r.get("tier") == "roofline"
    )
    total = sum(float(r.get("busy_us") or 0) for r in rows) or 1.0

    def point(us: float) -> tuple[float, float]:
        inter = 1e6 / us
        return inter, conc * inter / tp

    inter, thr = point(wall_us)
    c_inter, c_thr = point(roof_us) if roof_us > 0 else (inter, thr)
    return {
        "concurrency": conc,
        "interactivity": inter,
        "throughput": thr,
        "ceiling_interactivity": c_inter,
        "ceiling_throughput": c_thr,
        "step_us": wall_us,
        "roof_us": roof_us,
        # Against WALL, which is what the measured point is drawn at -- the
        # ceiling point is drawn at roof_us, so this multiple is the distance
        # between the two dots on the chart. Dividing by busy instead (what this
        # did) reports a number the picture does not show: busy excludes the idle
        # between kernels, 0.64 ms of a 16.27 ms step at c1.
        "optimism": wall_us / roof_us if roof_us else 0.0,
        "optimism_busy": busy_us / roof_us if roof_us else 0.0,
        "coverage": priced / total,
        "tp": tp,
    }


def series_key(doc: dict[str, Any]) -> tuple[str, str]:
    """A curve is one model on one stack: comparison is framework and platform."""
    ident = doc["identity"]
    return (ident.get("framework") or "?", ident.get("platform") or "?")


def svg_system(
    curves: list[dict[str, Any]],
    width: int = 1280,
    height: int = 560,
) -> str:
    """Throughput against interactivity, one point per concurrency.

    The axes a serving decision is made on: more
    concurrency fills the GPU and lifts throughput, while each user waits longer
    behind a larger batch, and the curve through the points is that trade-off.

    What the roofline adds is the arrow on each point -- where the same step would
    land if every operator ran at its bound. It is not a target: the ceiling covers
    only operators that have a cost model, ignores the per-kernel launch floor
    that most of a small-batch decode step is sitting on, and ignores the idle
    between kernels. Drawn as a direction with its own factor printed,
    which is the honest form of "there is headroom, and here is how much of that
    claim you should believe".
    """
    pad_l, pad_r, pad_t, pad_b = 78, 150, 34, 70
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    pts = [
        (v, w)
        for c in curves
        for p in c["points"]
        for v, w in (
            (p["interactivity"], p["throughput"]),
            (p["ceiling_interactivity"], p["ceiling_throughput"]),
        )
    ]
    if not pts:
        return f'<svg viewBox="0 0 {width} {height}"></svg>'
    x_lo, x_hi = min(a for a, _ in pts) / 1.6, max(a for a, _ in pts) * 1.6
    y_lo, y_hi = min(b for _, b in pts) / 1.6, max(b for _, b in pts) * 1.6

    def px(v: float) -> float:
        v = min(max(v, x_lo), x_hi)
        return pad_l + pw * (math.log10(v) - math.log10(x_lo)) / (
            math.log10(x_hi) - math.log10(x_lo)
        )

    def py(v: float) -> float:
        v = min(max(v, y_lo), y_hi)
        return pad_t + ph * (
            1
            - (math.log10(v) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo))
        )

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    out.append(f'<rect width="{width}" height="{height}" fill="{C["panel"]}" rx="8"/>')
    for d in nice_ticks(y_lo, y_hi):
        y = py(d)
        out.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + pw}" y2="{y:.1f}" '
            f'stroke="{C["border_l"]}"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" fill="{C["t3"]}" font-size="11" '
            f'text-anchor="end" font-family="monospace">{tick_txt(d)}</text>'
        )
    for d in nice_ticks(x_lo, x_hi):
        x = px(d)
        out.append(
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + ph}" '
            f'stroke="{C["border_l"]}"/>'
            f'<text x="{x:.1f}" y="{pad_t + ph + 18}" fill="{C["t3"]}" font-size="11" '
            f'text-anchor="middle" font-family="monospace">{tick_txt(d)}</text>'
        )

    for idx, curve in enumerate(curves):
        colour = [C["green"], C["purple"], C["yellow"], C["red"]][idx % 4]
        pts_m = sorted(curve["points"], key=lambda p: p["concurrency"])

        # Two curves, not a cloud of arrows: what the stack does, and what the
        # roofline says the same steps could do. The distance between them IS the
        # reading, and a per-point connector made it a pile of tick marks instead
        # of a shape.
        #
        # Neither can be extrapolated between points. ATOM picks its operator set
        # by batch size -- deepseek_v4.py:4350 switches to dual-stream MoE below a
        # token threshold, TBO only starts at concurrency 64 -- so a gap in either
        # line is a concurrency nobody ran, not a value to interpolate.
        for kind, xs, ys, dash, alpha in (
            ("roofline", "ceiling_interactivity", "ceiling_throughput", "6 4", 0.75),
            ("measured", "interactivity", "throughput", None, 0.95),
        ):
            if len(pts_m) > 1:
                line = " ".join(f"{px(p[xs]):.1f},{py(p[ys]):.1f}" for p in pts_m)
                out.append(
                    f'<polyline points="{line}" fill="none" stroke="{colour}" '
                    f'stroke-width="{1.8 if dash else 2.4}" opacity="{alpha}"'
                    + (f' stroke-dasharray="{dash}"' if dash else "")
                    + "/>"
                )
            for p in pts_m:
                x, y = px(p[xs]), py(p[ys])
                tip = "\n".join(
                    [
                        (
                            f'{curve["label"]}  concurrency {p["concurrency"]}'
                            f'  (tp={p["tp"]})  --  {kind}'
                        ),
                        (
                            f"  {p[xs]:,.1f} tok/s/user   {p[ys]:,.1f} tok/s/GPU"
                            f'   step {(p["roof_us"] if kind == "roofline" else p["step_us"]) / 1000:,.2f} ms'
                        ),
                        (
                            f'  this ceiling is {p["optimism"]:,.1f}x optimistic: it '
                            f'prices only the {p["coverage"]:.0%} of the step that has '
                            f"a cost model, ignores the per-kernel launch floor "
                            f"most of a small batch actually sits on, and ignores "
                            f"the idle between kernels "
                            f'({p["optimism_busy"]:,.1f}x against GPU-busy alone)'
                            if kind == "roofline"
                            else f'  {p["coverage"]:.0%} of this step is in operators '
                            f"that have a cost model at all"
                        ),
                    ]
                )
                out.append(
                    f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" '
                    f'r="{4.5 if dash else 6}" '
                    f'fill="{"none" if dash else colour}" fill-opacity="0.6" '
                    f'stroke="{colour}" stroke-width="{1.6 if dash else 1.8}"'
                    + (' stroke-dasharray="1.5 1.5"' if dash else "")
                    + f"><title>{html.escape(tip)}</title></circle>"
                )
                if kind == "measured":
                    out.append(
                        f'<text x="{x:.1f}" y="{y + 19:.1f}" fill="{C["t3"]}" '
                        f'font-size="9.5" text-anchor="middle" '
                        f'font-family="monospace">c{p["concurrency"]}</text>'
                    )

    out.append(
        f'<text x="{pad_l + pw / 2:.1f}" y="{height - 12}" fill="{C["t3"]}" '
        f'font-size="12" text-anchor="middle">'
        f"interactivity (tok/s/user) &mdash; faster for one user &rarr;</text>"
    )
    out.append(
        f'<text x="16" y="{pad_t + ph / 2:.1f}" fill="{C["t3"]}" font-size="12" '
        f'text-anchor="middle" transform="rotate(-90 16 {pad_t + ph / 2:.1f})">'
        f"throughput (tok/s/GPU) &mdash; more GPU used &rarr;</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


def page(models: dict[str, list[dict[str, Any]]], title: str) -> str:
    """One model at a time, switched by CSS radio; curves within it are stacks.

    A model picker rather than every model on one chart, because arithmetic
    intensity is a property of an architecture: two models at the same intensity
    are not doing the same thing, and one further right is not better. What IS
    comparable is a model against itself on another framework or platform, which
    is what the curves within a page are.
    """
    ids = {m: f"m{i}" for i, m in enumerate(sorted(models))}
    tabs = "".join(
        f'<input type="radio" name="model" id="{ids[m]}" class="tabin"'
        f'{" checked" if i == 0 else ""}>'
        for i, m in enumerate(sorted(models))
    )
    strip = "".join(
        f'<label class="tab" for="{ids[m]}">{html.escape(m)}</label>'
        for m in sorted(models)
    )
    panels = []
    for model, curves in sorted(models.items()):
        # One row per concurrency, not per phase: both phases of a run live on the
        # same operator page, so two rows pointing at one file was two ways of
        # saying the same thing.
        by_conc: dict[Any, dict[str, Any]] = {}
        for curve in curves:
            for run in curve["runs"]:
                ident = run["identity"]
                slot = by_conc.setdefault(
                    (curve["label"], ident.get("concurrency")),
                    {"href": run["_href"], "phases": {}},
                )
                steps = int(run.get("sampled_steps") or 0) or 1
                busy = sum(float(r.get("busy_us") or 0) for r in run["rows"]) / steps
                priced = sum(
                    float(r.get("busy_us") or 0)
                    for r in run["rows"]
                    if r.get("tier") == "roofline"
                )
                total = sum(float(r.get("busy_us") or 0) for r in run["rows"]) or 1.0
                slot["phases"][ident.get("phase")] = {
                    "step_ms": busy / 1000,
                    "coverage": priced / total,
                }
                if ident.get("phase") == "decode":
                    slot["serving"] = measured_point(run)

        rows = []
        for (label, conc), slot in sorted(
            by_conc.items(), key=lambda kv: kv[0][1] or 0
        ):
            ph = slot["phases"]
            m = slot.get("serving") or {}
            # Mirrors the two curves: measured and roofline side by side, so the
            # table reads the way the chart does. Prefill sits in its own column
            # because it is not on these axes -- there is no per-user token stream
            # during a prefill, its metric is time to first token.
            cov = ph.get("decode", {}).get("coverage")
            pre = ph.get("prefill", {}).get("step_ms")
            cells = [
                f"<td>{html.escape(label)}</td>",
                f"<td class=n>{conc}</td>",
                f'<td class=n>{m.get("step_us", 0) / 1000:,.2f}</td>',
                f'<td class=n>{m.get("interactivity", 0):,.0f}</td>',
                f'<td class=n>{m.get("throughput", 0):,.1f}</td>',
                f'<td class="n gsep">{m.get("roof_us", 0) / 1000:,.2f}</td>',
                f'<td class=n>{m.get("ceiling_interactivity", 0):,.0f}</td>',
                f'<td class=n>{m.get("ceiling_throughput", 0):,.1f}</td>',
                f'<td class="n gsep">{m.get("optimism", 0):,.1f}&thinsp;&times;</td>',
                (
                    f"<td class=n>{cov:.0%}</td>"
                    if cov is not None
                    else "<td>&middot;</td>"
                ),
                (
                    f"<td class=n>{pre:,.0f}</td>"
                    if pre is not None
                    else "<td>&middot;</td>"
                ),
                f'<td><a href="{html.escape(slot["href"])}">operators &rarr;</a></td>',
            ]
            rows.append("<tr>" + "".join(cells) + "</tr>")

        panels.append(
            # A direct sibling of the radio inputs, never nested: `#id:checked ~
            # .mpanel` only matches siblings, and wrapping these in a card made
            # every panel stay display:none -- a page that rendered blank without
            # erroring, which is the worst way for a layout bug to fail.
            f'<div class="mpanel card" id="p-{ids[model]}">'
            f'<div class="ct">{html.escape(model)} &mdash; system roofline'
            f'<span class="lbl m">measured</span> per concurrency &middot;'
            f'<span class="lbl s">attainable</span> for that same run</div>'
            f"{svg_system(curves)}"
            f'<div class="ct" style="margin-top:16px">runs</div>'
            f"<table><tr><th></th><th></th>"
            f'<th class="grp" colspan=3>measured</th>'
            f'<th class="grp gsep" colspan=3>if every operator hit its roofline</th>'
            f'<th class="grp gsep" colspan=3>how far to trust that</th><th></th></tr>'
            f"<tr><th>stack</th><th>c</th>"
            f"<th>step ms</th><th>tok/s/user</th><th>tok/s/GPU</th>"
            f"<th class=gsep>step ms</th><th>tok/s/user</th><th>tok/s/GPU</th>"
            f"<th class=gsep>ceiling &times;</th><th>coverage</th>"
            f"<th>prefill ms</th><th></th></tr>" + "".join(rows) + "</table></div>"
        )

    css = f"""*{{box-sizing:border-box}}
body{{margin:0;background:{C["bg"]};color:{C["t1"]};
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.wrap{{max-width:1360px;margin:0 auto;padding:26px 20px 60px}}
h1{{font-size:21px;margin:0 0 4px}}
.sub{{color:{C["t3"]};font-weight:400;font-size:14px}}
.card{{background:{C["panel"]};border:1px solid {C["border"]};border-radius:10px;
padding:16px;margin:16px 0}}
.ct{{font-size:13px;font-weight:600;margin-bottom:10px}}
.lbl{{font-size:10px;padding:1px 6px;border-radius:4px;margin:0 4px;font-weight:500}}
.lbl.s{{background:{C["purple"]};color:#1a0f2e}}
.lbl.m{{background:{C["ui"]};color:#0b1220}}
.tabin{{display:none}}
.tab{{display:inline-block;padding:6px 13px;margin-right:5px;border-radius:7px;
border:1px solid {C["border"]};color:{C["t2"]};font-size:12.5px;cursor:pointer}}
.mpanel{{display:none}}
{" ".join(f"#{i}:checked~.tabbar label[for={i}]{{background:{C['brand']};color:#14060f;border-color:{C['brand']}}}" for i in ids.values())}
{" ".join(f"#{i}:checked~.mpanel#p-{i}{{display:block}}" for i in ids.values())}
table{{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:6px}}
th{{text-align:left;color:{C["t3"]};font-weight:500;font-size:11px;
text-transform:uppercase;padding:6px 9px;border-bottom:1px solid {C["border"]}}}
td{{padding:7px 9px;border-bottom:1px solid {C["border_l"]};white-space:nowrap}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
a{{color:{C["brand"]}}}
circle.pt{{cursor:help}}
circle.pt:hover{{fill-opacity:0.95}}
.note{{color:{C["t3"]};font-size:11.5px;line-height:1.6}}
th.grp{{color:{C["t2"]};font-size:10px;letter-spacing:.04em;text-align:center}}
.gsep{{border-left:1px solid {C["border"]}}}
th.grp{{color:{C["t2"]};font-size:10px;letter-spacing:.04em}}
.gsep{{border-left:1px solid {C["border"]}}}"""

    return f"""<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{css}</style>
<div class="wrap">
<header><h1>System roofline <span class="sub">{html.escape(title)}</span></h1>
<p class="note">Each point is one measured concurrency. The dashed curve is the
same steps priced at their rooflines &mdash; a direction, not a target: it covers
only the operators that have a cost model and ignores the per-kernel launch floor,
so the <b>ceiling &times;</b> column says how optimistic it is. Gaps are
concurrencies nobody traced; neither curve can be interpolated, since ATOM picks
its operator set by batch size.</p></header>
{tabs}<div class="tabbar">{strip}</div>
{"".join(panels)}
</div>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "runs", help="directory of roofline run JSONs (searched recursively)"
    )
    ap.add_argument("-o", "--output", default="site", help="output directory")
    ap.add_argument("--title", default="ATOM", help="header subtitle")
    args = ap.parse_args()

    root, out = Path(args.runs), Path(args.output)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no roofline run JSONs under {root}")
    out.mkdir(parents=True, exist_ok=True)

    models: dict[str, list[dict[str, Any]]] = {}
    for doc in runs:
        ident = doc["identity"]
        model = ident.get("model") or "unknown"
        key = series_key(doc)
        bucket = models.setdefault(model, [])
        curve = next((c for c in bucket if c["key"] == key), None)
        if curve is None:
            curve = {
                "key": key,
                "label": f"{key[0]} / {key[1]}",
                "peaks": peaks_from(doc),
                "points": [],
                "runs": [],
            }
            bucket.append(curve)
        # One page per (model, stack, concurrency) holding BOTH phases -- that is
        # what the operator page already is, so prefill and decode of the same run
        # point at the same file rather than each getting a half-empty one.
        doc["_href"] = f"{model}/c{ident.get('concurrency')}.html"
        curve["runs"].append(doc)
        m = measured_point(doc)
        if m:
            m["concurrency"] = ident.get("concurrency") or 0
            m["phase"] = ident.get("phase")
            curve["points"].append(m)

    # The per-operator pages, built in-process. Grouped by concurrency because a
    # page shows a prefill and a decode side by side; a concurrency with only one
    # of the two still gets a page, with the missing half left honestly blank.
    pages = 0
    written: set[Path] = set()
    for model, curves in models.items():
        (out / model).mkdir(parents=True, exist_ok=True)
        by_conc: dict[Any, dict[str, dict[str, Any]]] = {}
        for curve in curves:
            for doc in curve["runs"]:
                ident = doc["identity"]
                by_conc.setdefault(ident.get("concurrency"), {})[
                    ident.get("phase") or "?"
                ] = doc
        for conc, phases in sorted(by_conc.items(), key=lambda kv: kv[0] or 0):
            pre, dec = phases.get("prefill"), phases.get("decode")
            if not pre and not dec:
                continue
            head = pre or dec
            title = f"{model} · c{conc}"
            html_text = build_html(head, dec, title=title)
            target = out / model / f"c{conc}.html"
            target.write_text(html_text, encoding="utf-8")
            written.add(target.resolve())
            pages += 1

    (out / "index.html").write_text(page(models, args.title), encoding="utf-8")
    written.add((out / "index.html").resolve())

    # Remove pages this build did not write, and nothing else. Wiping the output
    # directory would be one mistyped -o away from deleting someone's work; leaving
    # them would keep a page for a concurrency whose run is gone, showing stale
    # numbers under a live index -- the same failure this tool spends its time
    # warning about, one level up.
    stale = [q for q in out.rglob("*.html") if q.resolve() not in written]
    for q in stale:
        q.unlink()
    for d in sorted(out.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    if stale:
        print(f"removed        : {len(stale)} page(s) with no run behind them")
    print(
        f"index          : {out / 'index.html'}  ({len(runs)} runs, {len(models)} models)"
    )
    for model, curves in models.items():
        n = sum(len(c["runs"]) for c in curves)
        print(f"  {model:28s} {len(curves)} stack(s), {n} run(s)")


if __name__ == "__main__":
    main()
