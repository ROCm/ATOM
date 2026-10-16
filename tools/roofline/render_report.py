#!/usr/bin/env python3
"""Render roofline.json into a self-contained HTML report.

Self-contained on purpose: inline SVG and inline CSS, no CDN, no JavaScript. It
opens offline and screenshots cleanly, which is what a slide actually needs.
Colours follow ATOM's benchmark dashboard tokens so the two read as one system.

    python3 tools/roofline/render_report.py roofline.json -o report.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

# ATOM benchmark-dashboard tokens (dark).
C = {
    "bg": "#0b0c10",
    "panel": "#141519",
    "panel2": "#1c1d24",
    "border": "#303140",
    "border_l": "#262733",
    "t1": "#eef0f6",
    "t2": "#bfc1d0",
    "t3": "#7e8094",
    "brand": "#d86ecc",
    "ui": "#8899b8",
    "green": "#6dbf80",
    "red": "#f25c4e",
    "orange": "#eda06a",
    "purple": "#c49af5",
    "yellow": "#ddb040",
}
STAGE_COLOR = {
    "attn": "#8899b8",
    "moe": "#6dbf80",
    "indexer": "#c49af5",
    "mhc": "#eda06a",
    "comm": "#f25c4e",
    "other": "#7e8094",
}
# Fallback only. The real values arrive in the summary JSON, written by the tool
# from the same peaks file it priced against -- see peaks_from(). A second copy of
# a constant is a second thing to forget: this list still said fp4 = 20133 long
# after the yaml was corrected to 10066, so the chart drew a ceiling twice as high
# as the one its own dots had been measured against.
CEILINGS = [
    ("fp4", 10066, "#c49af5"),
    ("fp6", 10066, "#b07fd8"),
    ("fp8", 5033, "#6dbf80"),
    ("bf16", 2516, "#8899b8"),
    ("fp64", 79, "#d98f5a"),
    # Listed so a run that really does compute in fp32 gets a line. Nothing in
    # DSV4-Pro does: the operators that READ fp32 still COMPUTE in bf16, so no
    # fp32 line appears for them however they are coloured.
    ("fp32", 157, "#eda06a"),
]
# dtype -> the ceiling it is priced against, so dots and lines share a palette on
# purpose. A dot's colour tells you which horizontal line is its ceiling.
# Keyed on a canonical short name, because two sources feed this: a marker states
# torch's spelling ("float8_e4m3fn") while a config-derived formula states the
# short one ("fp8"). Keying on torch's alone sent 84% of the points -- every
# formula-priced one -- to a grey fallback that is not even in the legend, under a
# caption promising "dot colour = the ceiling it is measured against".
CEILING_COLOR = {
    "fp8": "#6dbf80",
    "bf16": "#8899b8",
    "fp16": "#8899b8",
    "fp4": "#c49af5",
    "fp32": "#eda06a",
}
DTYPE_ALIAS = {
    "float8_e4m3fn": "fp8",
    "float8_e5m2": "fp8",
    "f8_fnuz": "fp8",
    "bfloat16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "float4_e2m1fn_x2": "fp4",
    "float32": "fp32",
    "float": "fp32",
}


def dtype_key(name: object) -> str:
    """One spelling for a dtype, whatever the source called it."""
    text = str(name or "").replace("torch.", "")
    return DTYPE_ALIAS.get(text, text)


def ceiling_key(row: dict, ceilings: list[tuple[str, int, str]]) -> str:
    """Which ceiling this operator was actually priced against.

    Colouring by the memory dtype looked equivalent and is not. An expert GEMM
    stores fp4 and COMPUTES fp8, so it is measured against matrix_fp8 while its
    dtype_a says fp4; the compressor reads fp32 and computes bf16. Two dozen dots
    were therefore drawn in the colour of a line they are not measured against,
    under a caption promising the opposite -- and fp32 earned a legend entry for a
    ceiling nothing is priced against, so no such line could ever be drawn.

    `ceiling_tflops` is what price() actually divided by, so match on it.
    """
    try:
        peak = int(float(row.get("ceiling_tflops") or 0))
    except (TypeError, ValueError):
        peak = 0
    for label, value, _ in ceilings:
        if value == peak:
            return label
    return dtype_key(row.get("dtype_a"))


HBM_TFLOPS_PER_AI = 8.0  # 8000 GB/s -> achieved TFLOP/s = AI * 8; overridden per run


def peaks_from(doc: dict) -> tuple[list[tuple[str, int, str]], float, str]:
    """Ceilings, HBM slope and its label, taken from the run that produced them."""
    peaks = doc.get("peaks") or {}
    tf = peaks.get("compute_tflops") or {}
    bw = float(peaks.get("mem_bw_gbps") or 8000)
    ceilings = [
        (label, int(tf[f"matrix_{label}"]), colour)
        for label, _, colour in CEILINGS
        if f"matrix_{label}" in tf
    ] or CEILINGS
    # Coincident roofs (MXFP6 == MXFP4 on CDNA4) would draw two lines on top of
    # each other and two legend rows for one line. Keep the first, name both.
    merged: list[tuple[str, int, str]] = []
    for label, value, colour in ceilings:
        same = next((i for i, c in enumerate(merged) if c[1] == value), None)
        if same is None:
            merged.append((label, value, colour))
        else:
            prev = merged[same]
            merged[same] = (f"{prev[0]}/{label}", value, prev[2])
    ceilings = merged
    return ceilings, bw / 1000.0, f"{bw / 1000:g} TB/s"


def achieved_tflops(row: dict) -> float | None:
    """Read the value price() already computed; never recompute it here.

    A second copy of this arithmetic is how a per-firing time meets a total FLOP
    count: the throughput then scales with the firing count, points land above the
    compute ceiling, py() clamps them along with the HBM line, and the chart
    renders as a flat bar -- three unrelated-looking symptoms from one duplicated
    formula. There is exactly one place that computes it, and this is not it.
    """
    try:
        value = float(row["achieved_tflops"])
        return value if value > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def nice_ticks(lo: float, hi: float) -> list[float]:
    """1-2-5 decade ticks covering [lo, hi] on a log axis.

    The hardcoded list this replaces (100, 300, 1000, ...) had no entry anywhere
    inside decode's range -- AI 0.4 to 4.4, achieved 0.2 to 6.6 TFLOP/s -- so that
    chart rendered with no gridlines and no tick labels at all, which is a large
    part of why it read as wrong.
    """
    import math as _m

    if lo <= 0 or hi <= lo:
        return []
    out: list[float] = []
    exp = _m.floor(_m.log10(lo))
    while True:
        base = 10.0**exp
        for mult in (1.0, 2.0, 5.0):
            value = base * mult
            if value > hi:
                return out
            if value >= lo:
                out.append(value)
        exp += 1
        if exp > 12:
            return out


def tick_txt(value: float) -> str:
    if value >= 1:
        return f"{value:,.0f}"
    return f"{value:g}"


def _t(us: float) -> str:
    """A duration, in the unit that still shows digits."""
    return f"{us / 1000:,.3f} ms" if us >= 10 else f"{us:,.2f} us"


def svg_roofline(
    rows: list[dict],
    width: int = 1360,
    height: int = 560,
    peaks_bw: str = "8 TB/s",
    latency_floor: float = 0.0,
    ceilings_all: list[tuple[str, int, str]] | None = None,
    slope: float = HBM_TFLOPS_PER_AI,
) -> str:
    """Log-log roofline: HBM slope, one horizontal ceiling per dtype, one dot per op."""
    pad_l, pad_r, pad_t, pad_b = 74, 150, 28, 74
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    # Fit the axes to the data, not to the tallest ceiling. Forcing the y axis up to
    # the fp4 peak (20133) when nothing in the run is fp4 squeezes every point into a
    # narrow band and the chart stops discriminating -- the whole job of a roofline is
    # to show who sits where, and that needs the points to spread.
    # Two kinds of operator do not belong on an HBM roofline and were being drawn
    # on it anyway:
    #   * FLOPs = 0 (permutes, top-k selection, collectives). Arithmetic intensity
    #     is zero, log10(0) is undefined, and px() silently CLAMPED them to the left
    #     edge -- an x position that is not theirs. In prefill that was 36.7% of
    #     measured time plotted at a fabricated coordinate.
    #   * collectives, whose ceiling is the interconnect, not HBM. The diagonal on
    #     this chart is AI x HBM bandwidth; a link-bound operator measured against
    #     it is being compared to the wrong wall entirely.
    def plottable(r: dict) -> bool:
        if r.get("tier") != "roofline" or r.get("bw_kind") == "interconnect":
            return False
        ai = r.get("flop_per_byte")
        return ai not in ("", None) and float(ai) > 0

    excluded = [r for r in rows if r.get("tier") == "roofline" and not plottable(r)]
    rows = [r for r in rows if plottable(r) or r.get("tier") != "roofline"]
    pts_all = [
        (float(r["flop_per_byte"]), achieved_tflops(r)) for r in rows if plottable(r)
    ]
    pts_all = [(a, t) for a, t in pts_all if t]
    # Draw every published roof, not only the ones this run happens to use. The
    # hardware offers them all, and "fp4 sits 2x above the fp8 roof this operator
    # is stuck under" is a reading the chart should deliver -- filtering to
    # in-use precisions makes that invisible. Which roof BINDS an operator is a
    # separate question, carried by the dot's colour.
    ceilings_src = ceilings_all or CEILINGS
    ceilings = list(ceilings_src)
    # Which roofs actually bind something here. Used only to draw the others as
    # dimmed references -- every published roof is still shown.
    priced_against = {
        ceiling_key(r, ceilings_src) for r in rows if r.get("tier") == "roofline"
    }
    if pts_all:
        ai_lo = min(a for a, _ in pts_all) / 2.2
        ai_hi = max(a for a, _ in pts_all) * 2.2
        tf_lo = min(t for _, t in pts_all) / 2.2
        # The whole roof has to fit, bend included. Fitting the axis to the data
        # instead pushes the flat ceilings off-chart and leaves decode with a lone
        # diagonal -- correct arithmetic, but no longer a roofline, since the
        # picture a reader needs is slope, ridge and flat together. So the range is
        # stretched to clear the ridges and decode's points land in the lower-left
        # corner. That IS the finding, drawn to scale: at AI 1-2 they sit ~300x
        # left of the bf16 ridge, nowhere near any matrix ceiling. Ticks are
        # derived rather than listed, or they miss decode's decades entirely.
        cap = max(c[1] for c in ceilings)
        ridge_hi = max(c[1] for c in ceilings) / slope
        ai_hi = max(ai_hi, ridge_hi * 1.7)
        tf_hi = max(max(t for _, t in pts_all) * 2.2, cap * 1.2)
    else:
        ai_lo, ai_hi, tf_lo, tf_hi = 50.0, 6000.0, 100.0, 30000.0
    x_lo, x_hi = ai_lo, ai_hi
    y_lo, y_hi = tf_lo, tf_hi

    def px(ai: float) -> float:
        ai = min(max(ai, x_lo), x_hi)
        return pad_l + pw * (math.log10(ai) - math.log10(x_lo)) / (
            math.log10(x_hi) - math.log10(x_lo)
        )

    def py(tf: float) -> float:
        tf = min(max(tf, y_lo), y_hi)
        return pad_t + ph * (
            1
            - (math.log10(tf) - math.log10(y_lo))
            / (math.log10(y_hi) - math.log10(y_lo))
        )

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    out.append(f'<rect width="{width}" height="{height}" fill="{C["panel"]}" rx="8"/>')

    for decade in nice_ticks(y_lo, y_hi):
        if True:
            y = py(decade)
            out.append(
                f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + pw}" y2="{y:.1f}" '
                f'stroke="{C["border_l"]}" stroke-width="1"/>'
            )
            out.append(
                f'<text x="{pad_l - 8}" y="{y + 4:.1f}" fill="{C["t3"]}" font-size="11" '
                f'text-anchor="end" font-family="monospace">{tick_txt(decade)}</text>'
            )
    for decade in nice_ticks(x_lo, x_hi):
        x = px(decade)
        out.append(
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + ph}" '
            f'stroke="{C["border_l"]}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{pad_t + ph + 18}" fill="{C["t3"]}" font-size="11" '
            f'text-anchor="middle" font-family="monospace">{tick_txt(decade)}</text>'
        )

    # HBM slope: achieved = AI * 8 TFLOP/s. Label position and angle are derived
    # from the drawn segment, not hardcoded -- a fixed anchor at AI=160 falls
    # outside the axis entirely once decode compresses the range to 0.4-4.4, and
    # the line ends up as the one unlabelled thing on the chart.
    import math as _math

    # The two segments are ONE roof. The slope binds left of the ridge and the flat
    # peak binds right of it, so the slope is clipped at the highest ridge actually
    # drawn: continuing it past that point draws a "memory ceiling" through a region
    # where compute already binds lower, which is above the real roof and not a
    # constraint on anything. When every ridge is off-chart to the right (decode
    # lives at AI 1-2) the clip is inert and the slope spans the axis -- correctly,
    # because there the slope IS the entire roof.
    x0 = x_lo
    ridges_drawn = [tf / slope for _, tf, _ in ceilings]
    x1 = min(x_hi, y_hi / slope, max(ridges_drawn) if ridges_drawn else x_hi)
    if x1 > x0:
        ax, ay = px(x0), py(x0 * slope)
        bx, by = px(x1), py(x1 * slope)
        out.append(
            f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
            f'stroke="{C["ui"]}" stroke-width="2"/>'
        )
        angle = _math.degrees(_math.atan2(by - ay, bx - ax))
        lx, ly = ax + (bx - ax) * 0.3, ay + (by - ay) * 0.3 - 7
        out.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{C["t3"]}" font-size="10.5" '
            f'font-family="monospace" transform="rotate({angle:.1f} {lx:.1f} {ly:.1f})">'
            f"HBM {peaks_bw} \u2014 bandwidth roof (P = AI \u00d7 BW)</text>"
        )

    # A roofline is min(peak, AI x BW), so each dtype's roof is a diagonal that
    # bends flat at its ridge -- NOT a horizontal line spanning the whole axis. Drawn
    # the wrong way, the flat fp8 line sits 400x above every decode point and implies
    # a ceiling those kernels are nowhere near; their actual bound is the diagonal.
    # Clipping each flat segment to AI >= ridge makes the binding constraint the only
    # thing drawn, and when a whole run lives left of the ridge its flat part simply
    # never appears -- which is itself the finding.
    for label, tflops, colour in ceilings:
        ridge = tflops / slope
        if ridge > x_hi:
            out.append(
                f'<text x="{pad_l + pw + 8}" y="{py(min(tflops, y_hi)) + 4:.1f}" '
                f'fill="{colour}" font-size="10.5" font-family="monospace" '
                f'opacity="0.55">matrix_{label} ridge {ridge:,.0f} \u2014 off-chart</text>'
            )
            continue
        y = py(tflops)
        x_start = px(max(ridge, x_lo))
        # A roof nothing is priced against is a reference, not a constraint on
        # anything in this run: dashed and dimmed so it reads that way.
        used = label in priced_against
        out.append(
            f'<line x1="{x_start:.1f}" y1="{y:.1f}" x2="{pad_l + pw}" y2="{y:.1f}" '
            f'stroke="{colour}" stroke-width="{1.8 if used else 1.2}" '
            f'opacity="{0.85 if used else 0.4}"'
            + ("" if used else ' stroke-dasharray="6 4"')
            + "/>"
        )
        out.append(
            f'<text x="{pad_l + pw + 8}" y="{y + 4:.1f}" fill="{colour}" font-size="11" '
            f'font-family="monospace">matrix_{label} {tflops:,}</text>'
        )
        if x_lo <= ridge <= x_hi:
            out.append(
                f'<circle cx="{px(ridge):.1f}" cy="{y:.1f}" r="3.5" fill="{colour}"/>'
            )
            out.append(
                f'<text x="{px(ridge):.1f}" y="{y - 9:.1f}" fill="{colour}" font-size="9.5" '
                f'text-anchor="middle" font-family="monospace">ridge {ridge:,.0f}</text>'
            )

    # Latency-bound points are drawn hollow. Their distance from the roof is a true
    # measurement of a ceiling that is not binding them, so a solid dot invites the
    # exact wrong reading -- "far below the roof, so bandwidth is being wasted" --
    # when the operator is too small to reach either ceiling and the fix is batching.
    DOT_R = 5.0
    pts = [(r, achieved_tflops(r)) for r in rows if plottable(r)]
    pts = [(r, t) for r, t in pts if t]

    # Merge coincident points. 17 of decode's 41 operators sit on 7 shared
    # coordinates -- four fp8 GEMMs really do have the same intensity and the same
    # achieved throughput -- so drawing them on top of each other hid three
    # operators per stack and made the chart look sparser than the data is.
    # Jitter was the alternative and it lies about position, which on a roofline is
    # the only thing a dot means. One dot, a count, and every member in the tooltip.
    #
    # Uniform size, too. Encoding recoverable time as area hides neighbours
    # exactly where the chart is densest, and magnitude is already ranked in the
    # table below. Position is what this chart is for.
    groups: dict[tuple[int, int], list[tuple[dict, float]]] = {}
    for row, tf in pts:
        ai = float(row["flop_per_byte"])
        key = (round(px(ai) / 3.0), round(py(tf) / 3.0))
        groups.setdefault(key, []).append((row, tf))

    def tip_for(row: dict, tf: float) -> list[str]:
        shape = (
            f'{row.get("M")}x{row.get("N")}x{row.get("K")}'
            if row.get("shape_provenance") != "config"
            else f'{row.get("formula")}()  [cost from the model config]'
        )
        meas = float(row.get("busy_us") or 0)
        eff = float(row.get("efficiency") or 0)
        lines = [
            f'{row.get("op", "")}',
            f'  kernel     {row.get("kernel") or row.get("kernel_full", "")}',
            f"  shape      {shape}",
            f'  dtype      {row.get("dtype_a") or "?"} -> {row.get("dtype_o") or "?"}'
            + (
                "  (asserted by the cost model)"
                if row.get("dtype_provenance") == "asserted"
                else "  (from the marker)"
            ),
            f'  FLOP/byte  {row.get("flop_per_byte")}   bound: {row.get("bound")}'
            + (f'  (regime: {row.get("regime")})' if row.get("regime") else ""),
            f'  t_roofline {_t(float(row.get("t_roofline_us") or 0))}'
            f"   measured {_t(meas)}"
            + (f"   -> {1 / eff:,.1f}x the floor" if eff > 0 else ""),
            f'  achieved   {tf:,.1f} TFLOP/s   ({float(row.get("achieved_gbps") or 0):,.0f} GB/s)',
            f'  recoverable {float(row.get("recoverable_us") or 0) / 1000:,.3f} ms',
        ]
        if row.get("regime") == "latency":
            lines.append("  LATENCY-BOUND: roofline time is under the fastest kernel")
            lines.append("  observed in this phase, so neither ceiling binds it.")
        if row.get("formula_overestimates"):
            lines.append(f'  FLAGGED: {row.get("overestimate_reason") or ""}')
        return lines

    for members in groups.values():
        row, tf = members[0]
        x, y = px(float(row["flop_per_byte"])), py(tf)
        colour = CEILING_COLOR.get(ceiling_key(row, ceilings_all or CEILINGS), C["t3"])
        lat = all(m[0].get("regime") == "latency" for m in members)
        # Built outside the f-string: a backslash escape inside one is Python 3.12+
        # syntax and ATOM targets >= 3.10, so the module would not even import.
        dash = ' stroke-dasharray="2 2"' if lat else ""
        # A point above the roof needs saying so on the chart, not only in the
        # table. These are real: a working set inside the 256 MB cache reaches
        # 13 TB/s against an 8 TB/s HBM peak, so the roof drawn here is simply not
        # their ceiling. Unmarked, they read as the chart being broken.
        above = any(m[0].get("formula_overestimates") for m in members)
        body = []
        for i, (mrow, mtf) in enumerate(
            sorted(members, key=lambda m: -(float(m[0].get("recoverable_us") or 0)))
        ):
            if i:
                body.append("")
            body.extend(tip_for(mrow, mtf))
        head = (
            [f"{len(members)} operators at this coordinate", ""]
            if len(members) > 1
            else []
        )
        out.append(
            f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="{DOT_R}" '
            f'fill="{"none" if lat else colour}" fill-opacity="0.55" '
            f'stroke="{colour}" stroke-width="{1.6 if lat else 1.2}" ' + dash + ">"
            f"<title>{html.escape(chr(10).join(head + body))}</title></circle>"
        )
        if above:
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{DOT_R + 3.5}" fill="none" '
                f'stroke="{C["red"]}" stroke-width="1.4" stroke-dasharray="1.5 2"/>'
            )
        if len(members) > 1:
            out.append(
                f'<text x="{x:.1f}" y="{y + 3:.1f}" fill="{C["t1"]}" font-size="8" '
                f'text-anchor="middle" pointer-events="none">{len(members)}</text>'
            )

    # Legend inside the SVG, not beside it. The chart is meant to be screenshotted
    # into a slide, and a legend that lives in a sibling <div> does not travel with
    # it. The area above the roof is guaranteed empty -- nothing can exceed the
    # ceiling it is measured against -- so the top-left corner is always free.
    lx, ly0 = pad_l + 14, pad_t + 16
    seen_dtypes = sorted(
        {ceiling_key(r, ceilings_all or CEILINGS) for r, _ in pts} & set(CEILING_COLOR)
    )
    out.append(
        f'<text x="{lx}" y="{ly0}" fill="{C["t2"]}" font-size="10.5" '
        f'font-weight="600">colour = the ceiling it is measured against</text>'
    )
    for i, key in enumerate(seen_dtypes):
        yy = ly0 + 15 + i * 14
        out.append(
            f'<circle cx="{lx + 5}" cy="{yy - 3.5:.0f}" r="4.5" '
            f'fill="{CEILING_COLOR[key]}" fill-opacity="0.55" '
            f'stroke="{CEILING_COLOR[key]}"/>'
        )
        peak = next((t for lab, t, _ in (ceilings_all or CEILINGS) if lab == key), None)
        tail = f" &mdash; matrix_{key} {peak:,}" if peak else ""
        out.append(
            f'<text x="{lx + 16}" y="{yy}" fill="{C["t3"]}" font-size="10.5" '
            f'font-family="monospace">{key}{tail}</text>'
        )

    # No size legend: dots are uniform now. What does need saying is that a dot
    # can stand for several operators -- at decode 17 of 41 share 7 coordinates.
    sy = ly0 + 15 + len(seen_dtypes) * 14 + 18
    out.append(
        f'<text x="{lx}" y="{sy}" fill="{C["t2"]}" font-size="10.5">'
        f"a numbered dot is several operators at the same coordinate &mdash; "
        f"hover it for all of them</text>"
    )

    n_above = sum(
        1 for m in groups.values() if any(x[0].get("formula_overestimates") for x in m)
    )
    if n_above:
        out.append(
            f'<text x="{lx}" y="{sy + 15:.0f}" fill="{C["red"]}" font-size="10.5">'
            f"a red ring is a point ABOVE its own roof ({n_above} here): the working "
            f"set fits the 256 MB cache, so HBM was never its ceiling. Real, and "
            f"kept out of every total.</text>"
        )

    if excluded:
        # Below the axis, not across the top of the plot: a full-width line at
        # pad_t+14 ran straight through the ceiling labels and the roof itself.
        ex_us = sum(float(r.get("busy_us") or 0) for r in excluded)
        comm_n = sum(1 for r in excluded if r.get("bw_kind") == "interconnect")
        out.append(
            f'<text x="{pad_l}" y="{height - 30}" fill="{C["orange"]}" font-size="10.5">'
            f"not on this chart: {comm_n} link-bound + {len(excluded) - comm_n} with "
            f"no FLOPs = {ex_us / 1000:,.0f} ms. Zero intensity has no position on a "
            f"log axis, and a link-bound operator has no business under an HBM roof."
            f"</text>"
        )

    out.append(
        f'<text x="{pad_l + pw / 2:.1f}" y="{height - 12}" fill="{C["t3"]}" font-size="12" '
        f'text-anchor="middle">arithmetic intensity (FLOP/byte)</text>'
    )
    out.append(
        f'<text x="16" y="{pad_t + ph / 2:.1f}" fill="{C["t3"]}" font-size="12" '
        f'text-anchor="middle" transform="rotate(-90 16 {pad_t + ph / 2:.1f})">'
        f"achieved TFLOP/s</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


CONFIG_RULES = [
    ("platform", r"^(mi\d{3}x|b\d{3}|h\d{3})$", str.upper),
    ("framework", r"^(atom|sglang|trtllm|vllm)$", str.upper),
    ("model", r"^(dsv\d+(?:-pro)?)$", str.upper),
    ("quant", r"^dsv\d+-(fp4|fp8|mxfp4|bf16)$", str.lower),
    ("TP", r"^tp(\d+)$", str),
    ("EP", r"^ep(\d+)$", str),
    ("dp-attn", r"^dp(on|off)$", str),
    ("concurrency", r"^c(\d+)$", str),
    ("scenario", r"^(summarize|chat|agentic|code)$", str),
    ("run", r"^(run\d{8})$", str),
]


def parse_config(trace_name: str) -> dict[str, str]:
    """Pull the run coordinate out of the trace filename, as an addressable dict.

    Matched against whole `_`-separated tokens, NOT with \\b: underscore is a word
    character, so a boundary-anchored pattern never fires inside `..._mi355x_...`
    and silently matches nothing at all.
    """
    import re as _re

    tokens = _re.split(r"[_.]", trace_name.lower())
    found: dict[str, str] = {}
    for label, pattern, cast in CONFIG_RULES:
        for token in tokens:
            match = _re.match(pattern, token)
            if match and label not in found:
                found[label] = cast(match.group(1))
                break
    return found


def schedule_model(
    cfg: dict[str, str], isl: int, osl: int, budget: int
) -> dict[str, Any]:
    """Derive a wave from a configuration, never from a worked example.

    batch/rank   = concurrency / TP  (dp-attn on)  or  concurrency (off)
    prompts/step = budget / ISL      when ISL fits, else the prompt is chunked
    prefill steps = ceil(batch / prompts-per-step)
    decode steps  = OSL, one per output token
    """
    import math

    tp = int(cfg.get("TP") or 1)
    concurrency = int(cfg.get("concurrency") or 1)
    dp_on = (cfg.get("dp-attn") or "off") == "on"
    batch = max(1, concurrency // tp) if dp_on else concurrency
    if isl <= budget:
        per_step = max(1, budget // isl)
        steps = math.ceil(batch / per_step)
    else:
        chunks = math.ceil(isl / budget)
        steps = batch * chunks
    return {
        "batch": batch,
        "prefill_steps": steps,
        "decode_steps": osl,
        "assume": (
            f"concurrency {concurrency}, TP {tp}, dp-attn {'on' if dp_on else 'off'} "
            f"\u2192 batch {batch}/rank, ISL {isl}, OSL {osl}, budget {budget} "
            f"\u2192 {steps} prefill + {osl} decode steps"
        ),
    }


def config_chips(trace_name: str, extra: list[tuple[str, str]]) -> str:
    """Render the run coordinate as a compact table rather than baking it into the
    title: a title should name the subject, and a coordinate belongs somewhere you
    can see and eventually change."""
    found = list(parse_config(trace_name).items()) + list(extra)
    half = (len(found) + 1) // 2
    columns = [found[:half], found[half:]]
    rows_html = ""
    for i in range(half):
        cells = ""
        for column in columns:
            if i < len(column):
                key, value = column[i]
                cells += (
                    f'<td class="ck">{html.escape(key)}</td>'
                    f'<td class="cv">{html.escape(str(value))}</td>'
                )
            else:
                cells += "<td></td><td></td>"
        rows_html += f"<tr>{cells}</tr>"
    return f'<table class="cfg">{rows_html}</table>'


def project(rows: list[dict], family: str | None) -> list[dict]:
    """Project every operator onto one layer family.

    efficiency / FLOP-per-byte / bound / ceiling are properties of the SHAPE, and
    the shape is the same in every family, so they carry over untouched. Only the
    time columns are family-specific, and recoverable follows from that family's
    GEMM time times the same (1 - efficiency).
    """
    if family is None:
        return rows
    out = []
    for row in rows:
        busy = float(row.get(f"{family}_us") or 0)
        if busy <= 0:
            continue
        clone = dict(row)
        gemm = float(row.get(f"{family}_gemm_us") or 0)
        clone["busy_us"] = round(busy, 3)
        clone["gemm_us"] = round(gemm, 3)
        eff = row.get("efficiency")
        clone["recoverable_us"] = (
            round(gemm * (1.0 - min(float(eff), 1.0)), 3)
            if eff not in ("", None) and gemm > 0
            else 0.0
        )
        out.append(clone)
    # Each family runs its own sequence. Sorting the family view by the merged
    # `seq` shows CSA's execution order under an HCA heading, which would put
    # HCA's core attention -- 3rd in an HCA layer -- near the bottom.
    key = f"seq_{family}"
    if any(r.get(key) not in ("", None) for r in out):
        out.sort(key=lambda r: (r.get(key) if r.get(key) not in ("", None) else 10**6))
    return out


def step_facts(doc: dict, phase: str, model: dict) -> dict[str, Any] | None:
    """Everything Chart 1b needs about ONE typical step of `phase`.

    The operator table aggregates several steps, so every per-step number here is
    a division by the step count that table actually covered -- not by the run's
    total step count, which is larger. Keeping that divisor in the dict rather
    than in the caller is what stops the two from drifting apart.
    """
    st = (doc.get("phase_steps") or {}).get(phase)
    n = int(doc.get("sampled_steps") or 0)
    if not st or not n:
        return None
    busy = float(doc["measured_total_us"]) / n
    tier = float(doc["roofline_tier_us"]) / n
    roof = float(doc["t_roofline_sum_us"]) / n

    # The step, decomposed into four terms that ADD UP to the wall clock. A single
    # measured/floor ratio says how far the step is from its ceiling and nothing
    # about why, which at decode is the whole question: the ratio is ~9x and the
    # ceiling is not what sets it.
    #
    #   work      what the bytes and FLOPs need                   = roof
    #   issue     the per-kernel floor, where work does not cover it
    #   stall     achieved rate, dependency stalls, wave ramp-up
    #   idle      wall minus the sum of kernel durations
    #
    # `issue` uses the run's own measured floor rather than a constant, so a
    # faster stack or a bigger batch moves it without anything being edited here.
    lf = float(doc.get("latency_floor_us") or 0.0)
    covered = 0.0  # per-step time if every kernel took max(its floor, the issue floor)
    for r in doc.get("rows") or []:
        fires = int(r.get("n_firings") or 0)
        fl = r.get("t_roofline_us")
        if not fires or not isinstance(fl, (int, float)):
            continue
        covered += fires * max(float(fl) / fires, lf)
    covered /= n
    issue = max(0.0, covered - roof)
    stall = max(0.0, busy - covered)
    return {
        "phase": phase,
        "steps_measured": int(st["steps"]),
        "steps_model": int(
            model["prefill_steps"] if phase == "prefill" else model["decode_steps"]
        ),
        "wall_us": float(st["median_us"]),
        "busy_us": busy,
        "tier_us": tier,  # measured time of the ops a roofline can price
        "gap_us": busy - tier,  # measured time of everything else, carried as a floor
        "roof_us": roof,  # what the roofline says those priced ops could take
        "issue_us": issue,  # per-kernel floor not covered by the work itself
        "stall_us": stall,  # measured minus max(work, issue floor)
        "idle_us": max(0.0, float(st["median_us"]) - busy),  # between kernels
        "warmup": bool(st.get("warmup_suspect")),
        "sampled": n,
    }


def svg_timeline(
    steps: dict[str, Any], comp: dict[str, list[tuple[str, float]]], width: int = 1360
) -> str:
    """Chart 1 -- one strip, to scale: how the run splits into prefill and decode.

    This chart answers exactly one question -- how often does each phase run and
    how much of the wall clock does it own. Operator composition and per-step
    roofline are deliberately not layered on top of it: three messages in one
    picture read as none, so composition is a caption and the roofline comparison
    is a table of its own.

    ATOM never mixes a prefill and a decode batch (`scheduler.py`), so the strip
    is a real timeline of consecutive waves, not a stacked average.
    """
    pre, dec = steps.get("prefill"), steps.get("decode")
    e2e = (pre["total_us"] if pre else 0) + (dec["total_us"] if dec else 0)
    height, pad = 172, 24
    bar_w = width - 2 * pad
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    out.append(f'<rect width="{width}" height="{height}" fill="{C["panel"]}" rx="8"/>')
    if not e2e:
        out.append("</svg>")
        return "\n".join(out)

    segs = [
        ("prefill", pre, C["orange"]),
        ("decode", dec, C["green"]),
    ]
    x, y = float(pad), 46.0
    MIN_W = 3.0
    for name, st, colour in segs:
        if not st:
            continue
        share = st["total_us"] / e2e
        w = max(MIN_W, bar_w * share)
        out.append(
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="46" fill="{colour}" '
            f'fill-opacity="0.55" stroke="{colour}"/>'
        )
        inside = w > 150
        tx = x + w / 2 if inside else x + w + 8
        anchor = "middle" if inside else "start"
        out.append(
            f'<text x="{tx:.1f}" y="{y + 20}" fill="{C["t1"]}" font-size="12.5" '
            f'font-weight="600" text-anchor="{anchor}">{name} &#183; {share:.1%}</text>'
        )
        out.append(
            f'<text x="{tx:.1f}" y="{y + 37}" fill="{C["t2"]}" font-size="11" '
            f'text-anchor="{anchor}" font-family="monospace">'
            f'{st["steps"]} steps &#215; {st["median_us"] / 1000:,.1f} ms = '
            f'{st["total_us"] / 1e6:.2f} s</text>'
        )
        x += w

    out.append(
        f'<text x="{pad}" y="{y - 14}" fill="{C["t1"]}" font-size="13" font-weight="600">'
        f"end to end {e2e / 1e6:.1f} s &#8212; drawn to scale</text>"
    )

    lines = [
        (
            "duration = steps<sub>prefill</sub> &#215; t<sub>prefill</sub> + "
            "steps<sub>decode</sub> &#215; t<sub>decode</sub>"
        )
    ]
    for name, _, _ in segs:
        parts = comp.get(name) or []
        if parts:
            txt = " &#183; ".join(f"{k} {v:.0%}" for k, v in parts[:6])
            lines.append(f"{name} GPU time: {txt}")
    body = "<br/>".join(lines)
    out.append(
        f'<foreignObject x="{pad}" y="{y + 58}" width="{bar_w}" height="60">'
        f'<div xmlns="http://www.w3.org/1999/xhtml" style="color:{C["t3"]};'
        f'font:11.5px/1.6 -apple-system,sans-serif">{body}</div></foreignObject>'
    )
    out.append("</svg>")
    return "\n".join(out)


def step_table(facts: list[dict]) -> str:
    """Table 1 -- one typical step: what the roofline says, what the clock said.

    A table rather than bars: the two phases differ by ~100x per step, so any
    shared axis flattens one of them and any split axis invites reading two
    different scales as one. The numbers are the message, and there are eight of
    them.
    """
    if not facts:
        return '<p class="meta">no per-step data</p>'
    head = [
        ("phase", "which wave"),
        ("steps<br>measured", "step count the phase labels report for this run"),
        ("steps<br>model", "step count derived from the configuration alone"),
        ("wall<br>ms/step", "median step duration from the phase labels"),
        ("GPU busy<br>ms/step", "sum of kernel time in one step"),
        (
            "priced ops<br>measured ms",
            "measured time of the operators a roofline can price",
        ),
        (
            "priced ops<br>roofline ms",
            "t_roofline = max(t_compute, t_memory), summed over those same operators",
        ),
        ("above<br>floor", "measured / roofline for the priced operators"),
        (
            "un-priced<br>ms/step",
            "no closed-form cost; this report says nothing about it",
        ),
        (
            "work<br>ms",
            "what the bytes and FLOPs need -- the same roofline ms, as a share of the step",
        ),
        (
            "issue<br>ms",
            "the per-kernel floor measured in this run, where the work does not cover it",
        ),
        (
            "stall<br>ms",
            "achieved rate, dependency stalls, wave ramp-up: measured minus max(work, issue)",
        ),
        ("idle<br>ms", "wall clock minus the sum of kernel durations"),
    ]
    rows_html = []
    for f in facts:
        ratio = f["tier_us"] / f["roof_us"] if f["roof_us"] else 0
        warn = ' <span class="warnpill">warm-up</span>' if f["warmup"] else ""
        rows_html.append(
            "<tr>"
            f'<td><b>{f["phase"]}</b>{warn}</td>'
            f'<td class="n">{f["steps_measured"]:,}</td>'
            f'<td class="n">{f["steps_model"]:,}</td>'
            f'<td class="n">{f["wall_us"] / 1000:,.2f}</td>'
            f'<td class="n">{f["busy_us"] / 1000:,.2f}</td>'
            f'<td class="n">{f["tier_us"] / 1000:,.2f}</td>'
            f'<td class="n" style="color:{C["purple"]}">{f["roof_us"] / 1000:,.2f}</td>'
            f'<td class="n" style="color:{C["orange"]}">{ratio:.2f}&#215;</td>'
            f'<td class="n">{f["gap_us"] / 1000:,.2f} '
            f'<span style="color:{C["t3"]}">({f["gap_us"] / f["busy_us"]:.0%})</span></td>'
            + "".join(
                f'<td class="n{" gsep" if k == "roof_us" else ""}">'
                f"{f[k] / 1000:,.2f} "
                f'<span style="color:{C["t3"]}">({f[k] / f["wall_us"]:.0%})</span></td>'
                for k in ("roof_us", "issue_us", "stall_us", "idle_us")
            )
            + "</tr>"
        )
    ths = "".join(f'<th title="{html.escape(t)}">{h}</th>' for h, t in head)
    note = (
        "<b>work + issue + stall + idle = wall</b>, so the four right-hand columns "
        "say where a step goes rather than only how far it is from its ceiling. "
        "<b>issue</b> is the per-kernel floor THIS run measured, not a constant: "
        "an operator whose entire roofline time is below it is bound by neither "
        "ceiling, and at small batch most of them are. "
        "The two left-hand blocks are not two views of the same thing. "
        "<b>priced ops</b> compares like with like: the same operators, measured and "
        "modelled, so the ratio is a real efficiency. <b>un-priced</b> is the rest of "
        "the step, and this report has no floor for it \u2014 so there is deliberately no "
        "\u201croofline time for a whole step\u201d column. Adding one would price the "
        "un-priced part at zero."
    )
    return (
        '<table class="steps"><thead><tr>'
        + ths
        + "</tr></thead><tbody>"
        + "".join(rows_html)
        + f'</tbody></table><p class="meta" style="margin-top:10px">{note}</p>'
    )


def fmt_ms(us: float) -> str:
    """A duration in a table cell, in the unit that still shows digits.

    Bare milliseconds printed "0.00" for 26 of this run's operators -- every one
    that costs under five microseconds -- which is the same zero-looking number
    the bandwidth and tooltip units already had to be fixed for, in a fourth
    quantity. A cell that reads 0.00 says "no time"; these operators take real
    time, just not milliseconds' worth.
    """
    if us >= 10:
        return f"{us / 1000:,.2f}"
    return f'{us:,.1f}<span class="dt" style="display:inline"> &micro;s</span>'


def fmt_ai(row: dict) -> str:
    """Arithmetic intensity, or an em dash when the operator does no arithmetic.

    A permute, a memset and a collective have FLOPs = 0, so their intensity is
    zero by construction rather than by measurement. Printed as "0" it reads as
    "extremely low intensity" -- a judgement about a kernel that is not doing the
    thing being judged. They are already excluded from the chart for the same
    reason: zero has no position on a log axis.
    """
    ai = row.get("flop_per_byte")
    if ai in ("", None):
        return ""
    if not float(ai):
        return (
            '<span class="absent" title="this operator performs no arithmetic, so '
            'intensity is zero by construction, not by measurement">&mdash;</span>'
        )
    return str(ai)


def sci(value: float) -> str:
    return f"{value:.3g}"


def formula_hint(row: dict) -> dict[str, str]:
    """Native `title=` tooltips carrying the actual substitution, not a symbol.

    The formula has to be visible or the chart is unauditable -- a reader who
    cannot check the arithmetic either over-trusts it or dismisses it. Native
    title attributes keep that promise with no JavaScript, so it survives being
    opened from a file:// path and screenshotted.
    """
    if row.get("tier") != "roofline":
        return {}
    flops, byts = float(row.get("flops") or 0), float(row.get("bytes") or 0)
    peak = float(row.get("ceiling_tflops") or 0)
    tc, tm = float(row.get("t_compute_us") or 0), float(row.get("t_memory_us") or 0)
    gemm = float(row.get("gemm_us") or 0)
    tr = float(row.get("t_roofline_us") or 0)
    m, n, k = row.get("M"), row.get("N"), row.get("K")
    if row.get("shape_provenance") == "config":
        # A config-derived row has no M/N/K, so the GEMM derivation below would
        # render as "FLOPs = 2*M*N*K (M=None N= K=)". Give it its own derivation:
        # which named function ran, what was substituted into it, and the same two
        # divisions every other row gets. The numbers are read back from the row --
        # the tool substituted them at computation time and wrote them down.
        name = str(row.get("formula") or "")
        note = str(row.get("formula_note") or "")
        inputs = str(row.get("formula_inputs") or "")
        peak_key = str(row.get("formula_peak_key") or "")
        inter = row.get("bw_kind") == "interconnect"
        bw_name = "xGMI per-link" if inter else "HBM"
        bw_txt = "76.8e9" if inter else "8000e9"
        head = (
            f"{name}()  --  cost from the model config, not from the marker\n"
            f"the ATOM annotation names this operator but carries no M/N/K, so its\n"
            f"FLOPs and bytes come from a closed form in formulas.py\n"
            f"\n  substituted:  {inputs}\n"
        )
        if note:
            head += f"  model:        {note}\n"
        # The operator's OWN algebra, recorded when it was evaluated. Not a
        # template: a fused attention kernel shows its pass count and its KV-entry
        # selection, a collective shows a message size, an expert GEMM shows its
        # active-expert weight load. Printing 2*M*N*K over all of them would be
        # wrong for two thirds of this table.
        head += "\n" + str(row.get("flops_expr") or "")
        head += "\n" + str(row.get("bytes_expr") or "")
        head += f"\n\n  per firing; summed over {row.get('n_firings', '?')} firings"
        derivation = (
            f"t_roofline = max(t_compute, t_memory) = {tr:.1f} us\n"
            f"t_compute  = FLOPs / {peak_key} peak = {sci(flops)} / {peak:g}e12 = {tc:.1f} us\n"
            f"t_memory   = Bytes / {bw_name} = {sci(byts)} / {bw_txt} = {tm:.1f} us"
        )
        return {
            "fpb": head,
            "bound": f"{derivation}\n\nwhichever is larger binds; here it is "
            f"{'compute' if tc >= tm else 'memory'}",
            "roof": f"{head}\n\n{derivation}",
            "wall": f"measured / t_roofline = {gemm:.1f} / {tr:.1f} = "
            f"{row.get('efficiency')}\n{head}",
            "bw": f"effective bandwidth = Bytes / measured = {sci(byts)} / {gemm:.1f}us\n"
            f"= {float(row.get('achieved_gbps') or 0):,.0f} GB/s",
            "rec": f"recoverable = measured * (1 - efficiency) = "
            f"{float(row.get('recoverable_us') or 0):.1f} us",
        }
    return {
        "fpb": (
            f"FLOP/byte = FLOPs / Bytes = {sci(flops)} / {sci(byts)} = "
            f"{row.get('flop_per_byte')}\n"
            f"FLOPs = 2*M*N*K  (M={m} N={n} K={k}, summed over every firing)\n"
            f"Bytes = M*K*sz(in) + K*N*sz(weight) + M*N*sz(out)"
        ),
        "bound": (
            f"bound = whichever ceiling binds first\n"
            f"t_compute {tc:.1f} us  vs  t_memory {tm:.1f} us"
        ),
        "roof": (
            f"t_roofline = max(t_compute, t_memory) = {tr:.1f} us\n"
            f"t_compute = FLOPs / peak = {sci(flops)} / {peak:g}e12 = {tc:.1f} us\n"
            f"t_memory  = Bytes / HBM  = {sci(byts)} / 8000e9 = {tm:.1f} us\n"
            f"peak is the theoretical matrix peak, no derate -> gap >= 1 by design"
        ),
        "wall": (
            f"memory-bound: effective bandwidth = Bytes / time = {sci(byts)} / "
            f"{gemm:.1f}us = {float(row.get('achieved_gbps') or 0) / 1000:.2f} TB/s "
            f"against an 8 TB/s peak\n"
            if row.get("bound") == "memory"
            else f"compute-bound: achieved = FLOPs / time = {sci(flops)} / {gemm:.1f}us"
            f" = {row.get('achieved_tflops')} TFLOP/s against a {peak:g} TFLOP/s peak\n"
        )
        + "the percentage is the distance to that ceiling -- identical to efficiency,\n"
        + "just expressed in the unit of the wall this kernel actually hits",
        "bw": (
            f"effective bandwidth = Bytes / measured GEMM time = {sci(byts)} / "
            f"{gemm:.1f}us = {float(row.get('achieved_gbps') or 0):,.0f} GB/s\n"
            f"= {float(row.get('bw_util') or 0):.1%} of the 8000 GB/s HBM peak\n"
            f"for a compute-bound kernel this is well under 100% by design: it is\n"
            f"moving few bytes per FLOP, which is why compute binds first"
        ),
        "eff": (
            f"efficiency = t_roofline / measured GEMM time = {tr:.1f} / {gemm:.1f}"
            f" = {row.get('efficiency')}\n"
            f"GEMM time excludes comm and quant kernels inside the same marker"
        ),
        "rec": (
            f"recoverable = GEMM time * (1 - efficiency) = {gemm:.1f} * "
            f"(1 - {row.get('efficiency')}) = {float(row.get('recoverable_us') or 0):.1f} us\n"
            f"the part that could in principle be reclaimed if it hit the ceiling"
        ),
    }


def no_ceiling_reason(row: dict, plain: bool = False) -> str:
    """Why this operator has no roofline value -- and whether that is fixable.

    Four different situations were all rendering as the same blank cell, which is
    how "no ceiling" gets misread as "nothing to do here". Only one of the four is
    a property of the operator; the rest are properties of what the annotation
    happens to record, and those are the ones worth chasing.
    """
    kclass = row.get("kclass", "")
    kernel = str(row.get("kernel") or "")
    if plain:
        import re as _re

        return _re.sub(r"<[^>]+>", "", no_ceiling_reason(row)).replace("&mdash;", "--")
    if kclass == "comm":
        return (
            "collective &mdash; cost is set by the interconnect, not the chip; "
            "belongs to an \u03b1-\u03b2 network model, not this roofline"
        )
    if kclass == "gemm":
        return "<b>marker carries no shape</b> &mdash; a GEMM we could price if M/N/K were annotated"
    if kclass in ("attn", "moe") or "fused" in kernel or "gluon" in kernel:
        return "fused DSL kernel &mdash; the marker names the module, not the internal shapes"
    return (
        "elementwise / layout / sort &mdash; cost is not a function of a GEMM "
        "shape, so no closed form exists to compare against"
    )


def table(rows: list[dict], family: str | None = None, n_steps: int = 0) -> str:
    """Every operator, in the order the model executes them inside a layer.

    One table, execution order, blanks where there is no roofline. Splitting the
    priced operators into their own table did make it narrower, but it broke the
    thing the order is for: you read down the list to follow a layer, and an
    operator missing from the middle of that sequence is worse than an operator
    with empty cells. The blanks are also the honest picture of coverage -- most
    of a decode step is currently blank, and that is the finding, not a defect to
    tidy away.

    Columns are grouped because theory and measurement have different shelf lives:
    FLOP/byte, bound and t_roofline follow from shape and hardware and hold until
    the model or chip changes; the time columns are one build's snapshot.

    An empty roofline cell means UNKNOWN, not zero. Hover it for the reason.
    """
    ranked = sorted(rows, key=lambda r: -(r.get("recoverable_us") or 0))
    rank = {
        id(r): i + 1 for i, r in enumerate(ranked) if (r.get("recoverable_us") or 0) > 0
    }
    # must stay in lockstep with `tkeys` below (the guard at the end of this
    # function catches drift, but keep them adjacent in intent).
    # In a family view the time column is ONE representative layer, averaged over
    # the steady window -- the typical-layer method. A family total divided
    # by its layer count blends layers that are not identical (1.4% apart on
    # average here, 14.8% at worst), and the blend is exactly what the method
    # exists to avoid. The combined view keeps the totals, where they belong.
    time_cols = (
        [f"typical layer &micro;s<br>(mean of {n_steps} firings)"]
        if family
        else ["csa ms", "hca ms", "hash ms"]
    )
    tkeys = (
        [(f"{family}_typical_us", True)]
        if family
        else [("csa_us", True), ("hca_us", False), ("hash_us", False)]
    )
    groups = [
        ("identity", ["#", "module", "kernel", "shape M&times;N&times;K"]),
        (
            "roofline &mdash; from shape + hardware",
            ["FLOP/byte", "bound", "t_roofline ms"],
        ),
        ("measured &mdash; this build only", time_cols),
        ("derived", ["achieved", "eff. BW GB/s vs peak", "recoverable ms"]),
    ]

    def seq_txt(row: dict, fam: str | None) -> str:
        """Show the position in THIS view's sequence, not the merged one."""
        if fam:
            v = row.get(f"seq_{fam}")
            return "" if v in ("", None) else str(v)
        return str(row.get("seq", ""))

    CLASS_COLOUR = {
        "gemm": C["green"],
        "comm": C["red"],
        "quant": C["yellow"],
        "attn": C["ui"],
        "moe": C["purple"],
        "norm": C["orange"],
        "copy": C["t3"],
    }
    top = "".join(
        f'<th class="grp g{n}" colspan="{len(cols)}">{label}</th>'
        for n, (label, cols) in enumerate(groups)
    )
    sub = "".join(
        f'<th class="{"gsep" if k == 0 and n else ""}">{c}</th>'
        for n, (_, cols) in enumerate(groups)
        for k, c in enumerate(cols)
    )
    cells = [f"<tr>{top}</tr>", f"<tr>{sub}</tr>"]
    for row in rows:
        gap_tier = row.get("tier") != "roofline"
        # The row a layer opens on. Drawn as a rule rather than a separate header
        # row so it cannot drift out of sync with the ordering that produced it.
        boundary = bool(row.get("layer_boundary"))
        colour = CLASS_COLOUR.get(row.get("kclass", ""), C["t3"])
        kclass = html.escape(row.get("kclass", ""))
        full = html.escape(row.get("kernel_full") or "")
        ident = (
            f'<td class="seq">{seq_txt(row, family)}</td>'
            f'<td class="op">{html.escape(row["op"])}</td>'
            f'<td class="k" style="border-left:3px solid {colour}" '
            f'title="{kclass} &#183; {full}">{html.escape(row.get("kernel", ""))}</td>'
        )

        tcells = ""
        for n, (key, bold) in enumerate(tkeys):
            value = float(row.get(key) or 0)
            klass = "n gsep" if n == 0 else "n"
            if bold:
                klass += " b"
            if row.get("kclass") == "comm" and value > 1000:
                klass += " hot"
            # us in a family view, ms in the combined one: a typical layer's
            # operator runs for single-digit microseconds, and printing 0.01 ms
            # throws away the two digits that distinguish them.
            if value <= 0:
                text = '<span class="absent">\u00b7</span>'
            elif family:
                text = fmt_ms(value)
            else:
                text = fmt_ms(value)
            tcells += f'<td class="{klass}">{text}</td>'

        if gap_tier:
            # One tooltip, on every blank cell of the row, saying which of the four
            # situations this is. A bare blank reads as "zero" or as an oversight;
            # both readings are wrong and one of the four is actually fixable.
            why = f' title="{html.escape(no_ceiling_reason(row, plain=True))}"'
            blank = f'<td class="blank"{why}></td>'
            sep = f'<td class="blank gsep"{why}></td>'
            cells.append(
                f'<tr class="gaprow{" bnd" if boundary else ""}">'
                + ident
                + blank  # shape
                + sep
                + blank * 2  # FLOP/byte, bound, t_roofline
                + tcells
                + sep
                + blank * 2  # achieved, eff. BW, recoverable
                + "</tr>"
            )
            continue

        eff = float(row.get("efficiency") or 0)
        # A latency-bound operator's efficiency is a true number about a ceiling
        # that is not binding it, so printing it as a red 0.00 sends the reader to
        # bandwidth when the fix is batching or fusion. Name the regime instead.
        latency = row.get("regime") == "latency"
        flagged = bool(row.get("formula_overestimates"))
        link = row.get("bound") == "link"
        cfg_note = row.get("config_note")
        if flagged:
            # Measured faster than its own floor. Shown, never quoted: the row
            # stays in execution order so the sequence is not silently gapped, but
            # it is marked so nobody reads it as headroom.
            bound_txt = (
                f'<span style="color:{C["red"]}" title="'
                f'{html.escape(str(row.get("overestimate_reason") or ""))}">'
                f"floor &gt; measured</span>"
            )
            eff_col = C["red"]
        elif latency:
            bound_txt = f'<span style="color:{C["purple"]}">latency</span>'
            eff_col = C["t3"]
        elif link:
            # Its wall is xGMI, not HBM. Saying "memory" sends the reader to
            # bandwidth when the lever is the collective or the topology.
            bound_txt = (
                f'<span style="color:{C["red"]}" title="bound by the interconnect '
                f'(xGMI 76.8 GB/s per link), not by HBM">link</span>'
            )
            if cfg_note:
                # Not a warning about the number -- the number is fine. A warning
                # that a software default, not the hardware, put this operator on
                # the path it took.
                bound_txt += (
                    f' <span style="color:{C["orange"]};cursor:help" '
                    f'title="{html.escape(cfg_note)}">&#9881;</span>'
                )
            eff_col = (
                C["red"] if eff < 0.35 else C["yellow"] if eff < 0.7 else C["green"]
            )
        else:
            bound_txt = html.escape(str(row.get("bound") or ""))
            eff_col = (
                C["red"] if eff < 0.35 else C["yellow"] if eff < 0.7 else C["green"]
            )
        if row.get("shape_provenance") == "config":
            # No M/N/K to show: this operator's cost came from the model config via
            # a named formula, not from the annotation. Show which formula, so the
            # number is traceable to a function rather than looking like a guess.
            shape = (
                f'<span style="color:{C["purple"]}">'
                f'{html.escape(str(row.get("formula") or ""))}()</span>'
            )
        else:
            shape = "&times;".join(str(row.get(k) or "?") for k in ("M", "N", "K"))
        dtype = f'{row.get("dtype_a") or "?"}\u2192{row.get("dtype_o") or "?"}'
        # A marker STATES a dtype; a formula ASSERTS one from source reading. Mark
        # which, because the two are not the same kind of fact -- the indexer
        # carried an fp4 ceiling taken from the paper while its kernel said fp8.
        asserted = row.get("dtype_provenance") == "asserted"
        dmark = f' <span style="color:{C["orange"]}">&deg;</span>' if asserted else ""
        dtip = (
            ' title="dtype asserted by the cost model from source reading, not read '
            'from the marker -- the marker carries no shape for this operator"'
            if asserted
            else ' title="dtype read from the ATOM marker"'
        )
        hints = formula_hint(row)

        def tip(key: str, _h=hints) -> str:
            text = _h.get(key)
            return f' title="{html.escape(text)}"' if text else ""

        # Report a point in the units of the ceiling that binds it: bandwidth for a
        # memory-bound kernel, throughput for a compute-bound one. Same distance
        # from the roofline either way, but only one of the two is judgeable at a
        # glance, and which one depends on the operator.
        # Report the unit of the wall this operator actually hits. `or 0` on a
        # blank threw a permute and a collective into the throughput branch and
        # printed "0 TF/s" -- the exact reading the blank exists to prevent.
        tf = row.get("achieved_tflops")
        gbps = float(row.get("achieved_gbps") or 0)

        def bw_txt(value: float, unit: str = "") -> str:
            """TB/s once it is worth a decimal there, GB/s below. Dividing a
            1.2 GB/s kernel by 1000 printed "0.00 TB/s" -- the same zero-looking
            number the blank-vs-zero rule exists to prevent, in a new unit."""
            tail = f" {unit}" if unit else ""
            # Three decode operators really do move 10 KB in 2.6 ms -- they are
            # latency-bound, so their bandwidth is genuinely near zero. "0.0 GB/s"
            # states that as nothing rather than as 3.7 MB/s, which is the number.
            if value >= 100:
                return f"{value / 1000:,.2f} TB/s{tail}"
            if value >= 1:
                return f"{value:,.1f} GB/s{tail}"
            return f"{value * 1000:,.1f} MB/s{tail}"

        if link:
            achieved = bw_txt(gbps, "xGMI")
        elif tf in ("", None) or row.get("bound") == "memory":
            achieved = bw_txt(gbps)
        else:
            achieved = f"{float(tf):,.0f} TF/s"
        achieved += (
            f' <span class="pc" style="color:{C["t3"]}">n/a</span>'
            if latency
            else f' <span class="pc" style="color:{eff_col}">{eff:.0%}</span>'
        )
        # Over 100% of peak is not a good score, it is an impossibility: either the
        # bytes are overcounted or the traffic never reached the memory it is being
        # priced against. Printed plain, "139% HBM" reads as an achievement.
        util = float(row.get("bw_util") or 0)
        wall = "xGMI" if link else "HBM"
        bw = f"{gbps:,.0f}" + (
            f' <span class="pc" style="color:{C["red"]}" title="above the peak '
            f"this operator is priced against -- impossible for real {wall} "
            f"traffic, so either the byte count is too high or the bytes never "
            f'reached {wall}">{util:.0%} {wall} &#9888;</span>'
            if util > 1.0
            else f' <span class="pc">{util:.0%} {wall}</span>'
        )
        pos = rank.get(id(row))
        badge = f' <span class="top">#{pos}</span>' if pos and pos <= 5 else ""
        cells.append(
            ('<tr class="bnd">' if boundary else "<tr>")
            + ident
            + f'<td class="d">{shape}<span class="dt"{dtip}>{html.escape(dtype)}'
            f"{dmark}</span></td>"
            f'<td class="n gsep"{tip("fpb")}>{fmt_ai(row)}</td>'
            f'<td{tip("bound")}>{bound_txt}</td>'
            f'<td class="n"{tip("roof")}>{fmt_ms(float(row.get("t_roofline_us") or 0))}</td>'
            + tcells
            + f'<td class="n gsep"{tip("wall")}>{achieved}</td>'
            f'<td class="n"{tip("bw")}>{bw}</td>'
            f'<td class="n rec"{tip("rec")}>'
            f'{fmt_ms(float(row.get("recoverable_us") or 0))}{badge}</td>'
            "</tr>"
        )

    want = sum(len(cols) for _, cols in groups)
    for row_html in cells[2:]:
        got = row_html.count("<td")
        if got != want:
            import re as _re

            seen = [
                _re.sub(r"<[^>]+>", " ", c).strip()[:22]
                for c in _re.findall(r"<td[^>]*>(.*?)</td>", row_html, _re.DOTALL)
            ]
            raise AssertionError(
                f"table column mismatch: header declares {want}, row emits {got}.\n"
                f"  header: {[c for _, cols in groups for c in cols]}\n"
                f"  cells : {seen}"
            )
    return "<table>" + "".join(cells) + "</table>"


FAMILY_TABS = [(None, "All layers"), ("csa", "CSA (29)"), ("hca", "HCA (29)")]


def phase_panel(
    phase: str,
    rows: list[dict],
    chart_no: int,
    latency_floor: float = 0.0,
    n_steps: int = 0,
    typical: dict[str, Any] | None = None,
    ceilings_all: list[tuple[str, int, str]] | None = None,
    slope: float = HBM_TFLOPS_PER_AI,
    bw_label: str = "8 TB/s",
) -> str:
    """One phase, with a layer-family tab strip inside it.

    Pure CSS tabs (radio + :checked ~ sibling). No JavaScript on purpose: the page
    has to open from a file:// path with no network and screenshot cleanly into a
    slide, and a tab that needs a script fails both.
    """
    if not rows:
        return (
            f'<div class="ct">Chart {chart_no} &mdash; {phase} &middot; pending</div>'
            f'<div class="pending"><b>{phase.title()} not available yet.</b><br>'
            "Decode replays a CUDA graph, so markers do not fire and a run trace "
            "carries no decode operators. They live in the per-bs capture trace, and "
            "bridging marker&nbsp;&rarr;&nbsp;kernel needs that file's kernel-launch "
            'events.<br><code>zcat &lt;capture&gt;.json.gz | grep -c \'"cat": '
            '"cuda_runtime"\'</code> must be &gt; 0 &mdash; the capture we have '
            "reports 0.</div>"
        )
    blocks = []
    for idx, (family, label) in enumerate(FAMILY_TABS):
        rid = f"{phase}-{family or 'all'}"
        checked = " checked" if idx == 0 else ""
        blocks.append(
            f'<input type="radio" name="fam-{phase}" id="t-{rid}" class="tabin"{checked}>'
        )
    strip = "".join(
        f'<label class="tab" for="t-{phase}-{family or "all"}">{label}</label>'
        for family, label in FAMILY_TABS
    )
    panels = []
    for family, label in FAMILY_TABS:
        view = project(rows, family)
        shaped = sum(1 for r in view if r.get("tier") == "roofline")
        busy = sum(float(r.get("busy_us") or 0) for r in view)
        comm = sum(float(r.get("comm_us") or 0) for r in view)
        panels.append(
            f'<div class="fpanel" id="p-{phase}-{family or "all"}">'
            f'<div class="ct">Chart {chart_no} &mdash; {phase} &middot; {label}'
            + (
                f" &nbsp;·&nbsp; typical layer <b>{(typical or {}).get(family)}</b>"
                if family and (typical or {}).get(family) not in (None, "")
                else ""
            )
            + f" &nbsp;·&nbsp; {len(view)} operators, {shaped} with a shape"
            f" &nbsp;·&nbsp; {busy / 1000:.0f} ms busy, {comm / 1000:.0f} ms comm"
            f'<span class="lbl r">roofline</span> ceilings &amp; position &middot;'
            f'<span class="lbl m">measured</span> dot area</div>'
            f'<div class="fx">t_roofline = max( FLOPs / peak , Bytes / bandwidth )'
            f" &nbsp;·&nbsp; ridge = peak / HBM"
            f" &nbsp;&mdash;&nbsp; <b>FLOPs and Bytes differ per operator</b>: a GEMM"
            f" from its shape (2&middot;M&middot;N&middot;K), a fused attention from its"
            f" pass count and KV-entry selection, an expert GEMM from its active-expert"
            f" weight load, a collective from its message size. Hover any underlined"
            f" cell for that operator's own expression, with the numbers"
            f" substituted.</div>"
            f"{svg_roofline(view, peaks_bw=bw_label, latency_floor=latency_floor, ceilings_all=ceilings_all, slope=slope)}"
            f'<div class="legend">'
            '<span class="lg note">The key is drawn inside the chart, so it '
            "survives being screenshotted into a slide. Ceilings are theoretical "
            "peaks with no derate, so measured &ge; roofline by construction "
            "&mdash; hover any dot to see which operator it is.</span></div>"
            f'<div class="ct" style="margin-top:18px">Operators, in execution order'
            f' &nbsp;·&nbsp; <span class="lbl r">roofline</span> and'
            f' <span class="lbl m">measured</span> are separate column blocks</div>'
            f"{table(view, family, n_steps)}</div>"
        )
    return "".join(blocks) + f'<div class="tabbar sub">{strip}</div>' + "".join(panels)


def kpi(label: str, value: str, sub: str = "", blank: bool = False) -> str:
    """A KPI that cannot be computed is left blank with its reason, not filled in.

    Forcing a number into a slot the data does not support is worse than an empty
    slot: the empty one is obviously missing, the forced one gets quoted.
    """
    cls = "kpi blank" if blank else "kpi"
    val = (
        '<div class="kv dash">&mdash;</div>'
        if blank
        else f'<div class="kv">{value}</div>'
    )
    return f'<div class="{cls}">{val}<div class="kl">{label}</div><div class="ks">{sub}</div></div>'


def fmt_s(us: float) -> str:
    return f"{us / 1e6:.1f}" if us >= 1e6 else f"{us / 1000:.0f} ms"


def build_html(
    data: dict[str, Any],
    ddoc: dict[str, Any] | None = None,
    title: str = "DeepSeek-V4-Pro",
    token_budget: int = 16384,
    osl: int = 1024,
    isl: int | None = None,
) -> str:
    """The whole per-operator page, from two already-loaded run documents.

    Split out of main() so the site generator can build a page without shelling
    out to this script: one process, one import, and the page is a string the
    caller decides where to put.
    """
    return _build(data, ddoc or {}, title, token_budget, osl, isl)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("summary_json", help="prefill roofline.json")
    ap.add_argument(
        "--decode", default=None, help="decode roofline.json, when it exists"
    )
    ap.add_argument("-o", "--out", default="report.html")
    ap.add_argument(
        "--token-budget", type=int, default=16384, help="scheduler token budget/step"
    )
    ap.add_argument(
        "--osl", type=int, default=1024, help="output length = decode steps"
    )
    ap.add_argument(
        "--isl", type=int, default=None, help="prompt length; default = max M seen"
    )
    ap.add_argument(
        "--title", default="DeepSeek-V4-Pro", help="subject only, no config"
    )
    args = ap.parse_args()
    data = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    ddoc = (
        json.loads(Path(args.decode).read_text(encoding="utf-8")) if args.decode else {}
    )
    Path(args.out).write_text(
        _build(data, ddoc, args.title, args.token_budget, args.osl, args.isl),
        encoding="utf-8",
    )
    print(f"written: {args.out}")


def _build(
    data: dict[str, Any],
    ddoc: dict[str, Any],
    title: str,
    token_budget: int,
    osl: int,
    isl_arg: int | None,
) -> str:
    rows = data["rows"]
    total, priced = data["measured_total_us"], data["roofline_tier_us"]
    gemm_priced = float(data.get("gemm_priced_us") or 0)
    reach = float(data.get("ceiling_reach") or 0)
    steps_stat = data.get("phase_steps") or {}
    pre, dec = steps_stat.get("prefill"), steps_stat.get("decode")
    e2e = (pre["total_us"] if pre else 0) + (dec["total_us"] if dec else 0)

    # Read from the run, not typed in. The tool derives CSA/HCA from operator
    # content and hash from the config, so a model change moves this string on its
    # own instead of quietly contradicting the table beside it.
    lc = data.get("layer_counts") or {}
    obs = lc.get("observed_per_family") or {}
    layers_txt = (
        " + ".join(
            f"{obs[f]} {f.upper()}" for f in ("hash", "hca", "csa") if obs.get(f)
        )
        or "unknown"
    )
    cfg = parse_config(Path(data["run_trace"]).name)
    m_values = [
        int(str(r["M"]).split("\u2013")[-1])
        for r in rows
        if str(r.get("M") or "").strip()
    ]
    isl = isl_arg or (max(m_values) if m_values else token_budget)
    model = schedule_model(cfg, isl, osl, token_budget)

    decode_rows: list[dict] = ddoc.get("rows", []) if ddoc else []

    def composition(rs: list[dict]) -> list[tuple[str, float]]:
        by: dict[str, float] = {}
        for r in rs:
            by[r.get("kclass", "other")] = by.get(
                r.get("kclass", "other"), 0.0
            ) + float(r.get("busy_us") or 0)
        tot = sum(by.values())
        return (
            sorted(((k, v / tot) for k, v in by.items() if v), key=lambda kv: -kv[1])
            if tot
            else []
        )

    comp = {"prefill": composition(rows), "decode": composition(decode_rows)}

    def _typ(rs: list[dict]) -> dict[str, Any]:
        """Which layer each family is represented by, read off any row."""
        r = rs[0] if rs else {}
        return {f: r.get(f"{f}_typical_layer") for f in ("csa", "hca", "hash")}

    typ_pre, typ_dec = _typ(rows), _typ(decode_rows)
    lat_floor = float(ddoc.get("latency_floor_us") or 0) if ddoc else 0.0
    lat_share = 0.0
    if decode_rows:
        dbusy = sum(float(r.get("busy_us") or 0) for r in decode_rows) or 1.0
        lat_share = (
            sum(
                float(r.get("busy_us") or 0)
                for r in decode_rows
                if r.get("regime") == "latency"
            )
            / dbusy
        )

    facts = [
        f
        for f in (
            step_facts(data, data["phase"], model),
            step_facts(ddoc, ddoc.get("phase", ""), model) if ddoc else None,
        )
        if f
    ]

    body = f"""<div class="wrap">
<header>
  <h1>Operator roofline <span class="sub">{html.escape(title)}</span></h1>
  {config_chips(Path(data["run_trace"]).name, [("ISL", f"{isl} tok (measured)"), ("OSL", f"{osl} tok"), ("budget", f"{token_budget} tok/step"), ("layers", layers_txt)])}
  <div class="meta">{html.escape(Path(data["run_trace"]).name)}</div>
</header>

<div class="kpis">
  {kpi("measured &mdash; one prefill step"
       + (" <span class='warnpill'>warm-up</span>" if pre and pre.get("warmup_suspect") else ""),
       f"{pre['median_us'] / 1000:.1f} ms" if pre else "",
       (f"n={pre['steps']} &mdash; warm-up NOT excluded" if pre.get("warmup_suspect")
        else f"median of {pre['steps']} &middot; max {pre['max_us'] / 1000:.0f} ms (warm-up) excluded")
       if pre else "",
       blank=not pre)}
  {kpi("measured &mdash; one decode step",
       f"{dec['median_us'] / 1000:.2f} ms" if dec else "",
       f"median of {dec['steps']} &middot; {dec['streams']:.0f} stream records/step regrouped" if dec else "",
       blank=not dec)}
  {kpi("decode &mdash; what actually binds it",
       f"{lat_share:.0%} latency" if lat_share else "",
       f"roofline time under the {lat_floor:.1f} &micro;s floor any kernel costs here "
       f"&mdash; neither ceiling binds; the lever is batching and fusion, not bandwidth"
       if lat_share else "",
       blank=not lat_share)}
  {kpi("measured &mdash; end to end",
       f"{e2e / 1e6:.1f} s" if e2e else "",
       f"{pre['steps']}&times;prefill + {dec['steps']}&times;decode &middot; "
       f"decode is {dec['total_us'] / e2e:.0%}" if e2e and dec else "",
       blank=not e2e)}
</div>
<div class="fx" style="margin:-4px 0 16px">
  <b>duration = steps<sub>prefill</sub> &times; t<sub>prefill</sub> +
  steps<sub>decode</sub> &times; t<sub>decode</sub></b> &nbsp;&mdash;&nbsp; per-step times are
  what compose; an operator total over a whole run multiplies with nothing.
  The roofline KPI stays empty on purpose: a floor over {reach:.0%} of a step is not a floor
  for the step, and a partial sum quoted as one would price the rest at zero. Table 1 gives
  the honest version of that number &mdash; measured against modelled for the priced
  operators only, with the un-priced remainder listed separately rather than folded in.
</div>
<div class="fx" style="margin:-4px 0 16px">
  coverage is not accuracy. <b>{reach:.0%}</b> of measured time sits in GEMM kernels that a
  matrix ceiling can speak to; a further {(priced - gemm_priced) / total:.0%} sits in the same
  markers but in comm/quant kernels, and {1 - data["coverage"]:.0%} is in operators with no
  closed-form cost at all. <b>There is no such thing as &ldquo;how long prefill should
  take&rdquo; here</b> &mdash; a floor for 18% of the time is not a floor for the phase.
</div>

<div class="card">
  <div class="ct">Chart 1 &mdash; where the run&rsquo;s time goes
    <span class="lbl m">measured</span></div>
  {svg_timeline(steps_stat, comp)}
</div>

<div class="card">
  <div class="ct">Table 1 &mdash; one typical step
    <span class="lbl s">roofline</span> vs <span class="lbl m">measured</span></div>
  {step_table(facts)}
</div>

<input type="radio" name="phase" id="ph-prefill" class="tabin" checked>
<input type="radio" name="phase" id="ph-decode" class="tabin">
<div class="tabbar main">
  <label class="tab" for="ph-prefill">Prefill</label>
  <label class="tab" for="ph-decode">Decode</label>
</div>
<div class="ppanel" id="p-prefill"><div class="card">{phase_panel("prefill", rows, 2, float(data.get("latency_floor_us") or 0), int(data.get("sampled_steps") or 0), typ_pre, *peaks_from(data))}</div></div>
<div class="ppanel" id="p-decode"><div class="card">{phase_panel("decode", decode_rows, 3, float(ddoc.get("latency_floor_us") or 0), int(ddoc.get("sampled_steps") or 0), typ_dec, *peaks_from(ddoc))}</div></div>

<div class="card warn">
  <div class="ct">Read this before quoting a number</div>
  <ul>
    <li><b>This is a thermometer, not a microscope.</b> Magnitudes and bound categories are
        meaningful; absolute numbers are not. Industry anchor for a raw roofline is &asymp;22% error.</li>
    <li><b>A marker names a module, not a kernel.</b> <code>attn.wo_b</code> contains quantise +
        GEMM + the tensor-parallel all-reduce, and the all-reduce is <b>74%</b> of it. Only the
        matmul time is measured against the matmul ceiling; comm is its own column.</li>
    <li><b>fp8 GEMMs cluster at 0.22&ndash;0.28 while bf16 reaches 0.59.</b> A band that tight
        across K from 768 to 7168 is a property of the path, not of the kernels. The sparsity
        explanation is dead &mdash; AMD's product page confirms <code>matrix_fp8 = 5033</code> and
        <code>matrix_bf16 = 2516</code> are both <b>dense</b> figures &mdash; which leaves one
        candidate: the trace was captured with <code>--level 0</code>, so these are <b>untuned</b>
        kernels. <b>Settle this before quoting any fp8 efficiency.</b></li>
    <li><b>The collective model is standard-form, not read from AITER.</b> An all-reduce
        is priced as reduce-scatter + all-gather, <code>2&times;message/N</code> per link.
        The one-stage model it replaces was falsified: 103.7&nbsp;MB in 970&nbsp;&micro;s is
        107&nbsp;GB/s over a 76.8&nbsp;GB/s link, and nothing beats its own wire. Two-stage
        is the textbook form though, not something read out of <code>aiter.dist</code>
        &mdash; confirm it there before quoting a comm efficiency.</li>
    <li><b>Prefill here is one step, and it is the compile step.</b> 2000&nbsp;ms wall against
        497&nbsp;ms busy &mdash; 25% occupancy. Every prefill per-step number on this page is
        warm-up, not steady state.</li>
    <li><b>Layer families come from the config, not from the markers.</b> Hash layers run the same
        operator sequence as an ordinary layer, so only <code>num_hash_layers</code> knows there are
        3 of them. Content decides CSA vs HCA (the indexer chain); the config decides where the
        prologue ends.</li>
  </ul>
</div>
</div>"""

    css = f"""*{{box-sizing:border-box}}
body{{margin:0;background:{C["bg"]};color:{C["t1"]};
 font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:1440px;margin:0 auto;padding:24px}}
header{{border-bottom:1px solid {C["border"]};padding-bottom:14px;margin-bottom:20px}}
h1{{margin:0;font-size:21px;font-weight:600;line-height:28px}}
h1 .sub{{color:{C["brand"]};font-weight:400;font-size:15px;margin-left:10px}}
.meta{{color:{C["t3"]};font:11px/1.5 "SF Mono",Menlo,monospace;margin-top:6px;word-break:break-all}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:16px}}
.kpi{{background:{C["panel"]};border:1px solid {C["border"]};border-radius:8px;padding:14px 16px}}
.kpi.blank{{border-style:dashed;background:transparent}}
.kv.dash{{color:{C["t3"]}}}
.pc{{color:{C["t3"]};font-size:10px}}
.warnpill{{background:{C["orange"]};color:#1a0f04;border-radius:4px;padding:0 5px;
 font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}}
.kv{{font:600 22px/1.2 "SF Mono",Menlo,monospace;color:{C["t1"]}}}
.kl{{color:{C["t2"]};font-size:12px;margin-top:4px}}
.ks{{color:{C["t3"]};font-size:11px;margin-top:2px}}
.card{{background:{C["panel"]};border:1px solid {C["border"]};border-radius:8px;
 padding:16px;margin-bottom:16px;overflow-x:auto}}
.ct{{color:{C["t2"]};font-size:12px;margin-bottom:12px;letter-spacing:.03em;text-transform:uppercase}}
.legend{{margin-top:10px;display:flex;flex-wrap:wrap;gap:14px;align-items:center}}
.lg{{color:{C["t2"]};font-size:11.5px;display:flex;align-items:center;gap:6px}}
.lg i{{width:9px;height:9px;border-radius:50%;display:inline-block}}
.lg.note{{color:{C["t3"]};font-style:italic}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th{{text-align:left;color:{C["t3"]};font-weight:500;font-size:11px;text-transform:uppercase;
 letter-spacing:.04em;padding:6px 9px;border-bottom:1px solid {C["border"]};white-space:nowrap}}
td{{padding:6px 9px;border-bottom:1px solid {C["border_l"]};white-space:nowrap}}
tr:hover td{{background:{C["panel2"]}}}
td.op{{font-family:"SF Mono",Menlo,monospace;color:{C["t1"]}}}
td.k{{font-family:"SF Mono",Menlo,monospace;color:{C["t2"]};font-size:10.5px;
 white-space:normal;word-break:break-word;min-width:330px;max-width:620px;
 line-height:1.4;cursor:help}}
td.n,td.d{{font-family:"SF Mono",Menlo,monospace;color:{C["t2"]};text-align:right}}
td.d{{text-align:left;font-size:11px;color:{C["t3"]}}}
td.rec{{color:{C["t1"]};font-weight:600}}
td.seq{{color:{C["t3"]};font-family:"SF Mono",Menlo,monospace;font-size:11px;text-align:right}}
td.b{{color:{C["t1"]}}}
td.hot{{color:{C["red"]}}}
tr.gaprow td{{opacity:.62}}
circle.pt{{cursor:help}}
circle.pt:hover{{fill-opacity:0.9;stroke-width:2.6}}
.absent{{color:{C["border"]}}}
.top{{background:{C["brand"]};color:#14060f;border-radius:4px;padding:1px 6px;
 font:600 10px/1.5 "SF Mono",Menlo,monospace}}
.gap{{color:{C["t3"]};font-style:italic}}
.dt{{display:block;color:{C["t3"]};font-size:10.5px;font-family:monospace}}
tr.bnd td{{border-top:2px solid {C["red"]};padding-top:10px}}
tr.bnd td.op::after{{content:" \\2190 layer opens here";color:{C["red"]};font-size:10px;white-space:nowrap}}
td.blank{{background:repeating-linear-gradient(135deg,transparent,transparent 5px,{C["border_l"]} 5px,{C["border_l"]} 6px)}}
tr.gaprow td.op,tr.gaprow td.k{{color:{C["t3"]}}}
td.why{{white-space:normal;color:{C["t3"]};font-size:11.5px;max-width:340px;line-height:1.45}}
details.unpriced{{margin-top:14px}}
details.unpriced summary{{cursor:pointer;color:{C["t2"]};font-size:12.5px;padding:6px 0}}
table.steps{{font-size:13px}}
table.steps td{{padding:9px 12px}}
table.steps th{{padding:6px 12px;line-height:1.35}}
table.cfg{{border-collapse:collapse;margin:12px 0 2px;font-size:11.5px}}
table.cfg td{{border:none;padding:2px 0}}
td.ck{{color:{C["t3"]};padding-right:10px;text-transform:uppercase;font-size:10px;
 letter-spacing:.04em;white-space:nowrap}}
td.cv{{color:{C["t1"]};font-family:"SF Mono",Menlo,monospace;padding-right:32px;
 white-space:nowrap}}
.lbl{{border-radius:4px;padding:1px 7px;font-size:10px;font-weight:600;margin:0 4px;
 text-transform:uppercase;letter-spacing:.04em}}
.lbl.r{{background:{C["ui"]};color:#0b0c10}}
.lbl.m{{background:{C["orange"]};color:#1a0f04}}
.lbl.s{{background:{C["purple"]};color:#160b22}}
.fx{{color:{C["t3"]};font:11px/1.7 "SF Mono",Menlo,monospace;margin:-4px 0 12px;
 padding:8px 12px;background:{C["panel2"]};border-radius:6px;border:1px solid {C["border_l"]}}}
td[title]{{cursor:help;text-decoration:underline dotted {C["border"]} 1px;
 text-underline-offset:3px}}
th.grp{{font-size:10px;color:{C["t3"]};font-weight:600;letter-spacing:.06em;
 border-bottom:1px solid {C["border"]};padding-bottom:3px}}
th.grp.g1{{color:{C["ui"]}}}
th.grp.g2{{color:{C["orange"]}}}
th.grp.g3{{color:{C["t2"]}}}
th.gsep,td.gsep{{border-left:1px solid {C["border"]}}}
.tabin{{position:absolute;opacity:0;pointer-events:none}}
.tabbar{{display:flex;gap:4px;margin:18px 0 0}}
.tabbar.sub{{margin:0 0 14px}}
.tab{{padding:7px 18px;border:1px solid {C["border"]};border-bottom:none;
 border-radius:7px 7px 0 0;color:{C["t3"]};font-size:12.5px;cursor:pointer;
 background:{C["bg"]};user-select:none}}
.tabbar.sub .tab{{border-radius:6px;border-bottom:1px solid {C["border"]};padding:5px 14px;
 font-size:11.5px}}
.ppanel,.fpanel{{display:none}}
#ph-prefill:checked~#p-prefill,#ph-decode:checked~#p-decode{{display:block}}
#ph-prefill:checked~.tabbar.main label[for="ph-prefill"],
#ph-decode:checked~.tabbar.main label[for="ph-decode"]{{color:{C["t1"]};
 background:{C["panel"]};border-color:{C["brand"]}}}
{"".join(f'#t-{p}-{f}:checked~.tabbar.sub label[for="t-{p}-{f}"]{{color:{C["t1"]};background:{C["panel2"]};border-color:{C["brand"]}}}#t-{p}-{f}:checked~#p-{p}-{f}{{display:block}}' for p in ("prefill","decode") for f in ("all","csa","hca"))}
.pending{{color:{C["t2"]};font-size:13px;line-height:1.7;padding:26px;
 border:1px dashed {C["border"]};border-radius:8px;background:{C["panel2"]}}}
.pending b{{color:{C["orange"]}}}
.card.warn{{border-color:{C["orange"]}}}
.card.warn ul{{margin:0;padding-left:20px;color:{C["t2"]}}}
.card.warn li{{margin-bottom:8px}}
.card.warn b{{color:{C["t1"]}}}
code{{font-family:"SF Mono",Menlo,monospace;background:{C["panel2"]};padding:1px 5px;
 border-radius:3px;font-size:12px;color:{C["t2"]}}}
@media print{{body{{background:#fff}}}}"""

    return (
        f"<!doctype html><html><head><meta charset=utf-8>"
        f"<title>Operator roofline — {html.escape(title)}</title>"
        f"<style>{css}</style></head><body>{body}</body></html>"
    )


if __name__ == "__main__":
    main()
