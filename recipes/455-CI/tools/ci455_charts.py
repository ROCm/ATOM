"""Render the 455-CI charts from the curves and histories ci455_data.py picks.

- MLPerf Endpoint perf: the LLM-Booster#246 Pareto chart, drawn by its
  endpoint_chart.py (vendored unchanged), one curve per 455-CI image.
- 8K/1K perf: the ATOM benchmark dashboard's Throughput vs Latency chart:
  x = Interactive = 1000 / mean TPOT, y = total tok/s per GPU, each image's
  points joined in x order.
- GSM8K: the dashboard's accuracy trend, one panel per model: flexible-extract
  solid, strict-match dashed, the baseline (when set) grey dashed.

Every chart is laid out in CSS px at dpi 100 and saved at 2x (2640 px wide),
in #246's frame: title and subtitle top left, two footnote lines.
"""

import itertools
import math

import endpoint_chart as ec  # sets the Agg backend and registers the fonts
import matplotlib.pyplot as plt
from ci455_data import curve_styles
from matplotlib.ticker import FuncFormatter, MultipleLocator

PX = ec.PX
W = 1320
# ATOM benchmark dashboard, light theme (.github/dashboard/index.html).
DASH_TICK = "#5a5d70"
DASH_LEGEND = "#4a4d5e"
# Opaque equivalents on white of the dashboard's rgba(0,0,0,0.10) grid and
# rgba(136,153,184,0.5) baseline: matplotlib grid lines carry alpha 1, which
# would override an rgba colour's own alpha.
DASH_GRID = "#E6E6E6"
DASH_BASELINE = "#C4CCDC"
FAIL_GREY = "#8A8F98"
ACCURACY_RED = curve_styles(1)[0][0]  # the newest-curve red of the perf charts

SOURCE = "Source: AMD MI455 455-CI runs on ATOM main + aiter main, self-measured"
MOCK_NOTE = "MOCK DATA: layout preview, not measurements."
NICE_STEPS = (
    1,
    2,
    5,
    10,
    20,
    25,
    50,
    100,
    200,
    250,
    500,
    1000,
    2000,
    2500,
    5000,
    10000,
    20000,
    25000,
    50000,
    100000,
)


def mmdd(image_date):
    return image_date[5:]


def gpu_text(n):
    return f"{n} GPU" if n == 1 else f"{n} GPUs"


def failure_note(image_date, failed, concurrencies):
    """In-plot note for a report image whose run failed, wholly or at some
    concurrencies; None when nothing failed."""
    if not failed:
        return None
    if set(failed) >= set(concurrencies):
        what = "run"
    else:
        what = ", ".join(f"c{c}" for c in failed)
    return f"{mmdd(image_date)} image: {what} failed (N/A), see report."


def axis_range(values, from_zero=False, headroom=0.12, max_ticks=8):
    """(lo, hi, step) on whole ticks covering the data plus `headroom` of its
    top, and of its bottom unless the axis starts `from_zero`."""
    lo_t = 0.0 if from_zero else min(values) * (1 - headroom)
    hi_t = max(values) * (1 + headroom)
    for step in NICE_STEPS:
        lo = math.floor(lo_t / step) * step
        hi = math.ceil(hi_t / step) * step
        if (hi - lo) // step <= max_ticks:
            return lo, hi, step
    raise ValueError(f"no tick step fits {values}")


def _subtitle(curves, lead, mock):
    dates = " → ".join(mmdd(c.image_date) for c in curves)
    text = f"{lead} ({curves[-1].gpus}) · 455-CI images {dates}"
    return f"MOCK DATA · {text}" if mock else text


def _note(note, mock):
    if mock:
        return f"{note} {MOCK_NOTE}" if note else MOCK_NOTE
    return note


# ---------------------------------------------------------------- endpoint


def endpoint_series(curves):
    """endpoint_chart series specs, oldest first: red diamonds, newest solid,
    no lead-in stub (the #246 rules for AMD curves)."""
    specs = []
    for curve, (colour, line) in zip(curves, curve_styles(len(curves))):
        day = mmdd(curve.image_date)
        specs.append(
            {
                "key": f"img{curve.image_date}",
                "legend": f"AMD MI455 (ATOM main) · {day} · {gpu_text(curve.gpus)}",
                "label": f"MI455 · {day}",
                "color": colour,
                "marker": "D",
                "size": 11,
                "gpus": curve.gpus,
                "line": line,
                "stub": False,
                "points": [
                    {"c": p.c, "x": p.x, "y_gpu": p.y, "y_system": p.y * curve.gpus}
                    for p in curve.points
                ],
            }
        )
    return specs


def render_endpoint(curves, out, model_name, note=None, mock=False):
    """The #246 per-GPU Pareto chart, one curve per image."""
    series = endpoint_series(curves)
    xs = [p.x for c in curves for p in c.points]
    ys = [p.y for c in curves for p in c.points]
    x0, x1, xstep = axis_range(xs)
    _, y1, ystep = axis_range(ys, from_zero=True, headroom=0.15, max_ticks=10)
    ec.make_chart(
        series,
        {
            "mode": "per_gpu",
            "out": out,
            "title": f"{model_name} · Agentic: Throughput per GPU vs. E2E Average Interactivity",
            "subtitle": _subtitle(
                curves,
                "Online, agentic workload · throughput normalized by GPU count",
                mock,
            ),
            "note": _note(note, mock),
            "xlim": (x0, x1),
            "ylim": (0, y1),
            "xstep": xstep,
            "ystep": ystep,
            "footnotes": [
                f"{SOURCE} with the MLPerf Endpoints client (10-min windows), not MLCommons-verified.",
                "Y = output tok/s ÷ GPUs · X = E2E average interactivity (tok/s per user) · cN = concurrency.",
            ],
            "labels": {},
            "tags": [s["key"] for s in reversed(series)],  # newest tags first
        },
    )


# ------------------------------------------------------- shared page frame


def _figure(height):
    return plt.figure(figsize=(W / 100, height / 100), dpi=100, facecolor="white")


def _header(fig, H, title, subtitle):
    fig.text(
        24 / W,
        1 - 24 / H,
        title,
        ha="left",
        va="top",
        fontsize=18 * PX,
        fontweight=600,
        color=ec.INK,
    )
    fig.text(
        24 / W,
        1 - 52 / H,
        subtitle,
        ha="left",
        va="top",
        fontsize=13 * PX,
        fontweight=500,
        color=ec.INK_MUTED,
    )


def _footer(fig, H, top, left, lines):
    for i, line in enumerate(lines):
        fig.text(
            left / W,
            1 - (top + i * 15) / H,
            line,
            ha="left",
            va="top",
            fontsize=11 * PX,
            color=ec.INK_MUTED,
        )


def _legend(fig, H, top, left, col_w, entries):
    """Dashboard-style legend at the top: one cell per entry, left to right,
    wrapping every `(W - left) // col_w` cells. An entry is (text, colour,
    line or None, marker or None)."""
    per_row = max(1, int((W - left - 24) // col_w))
    for i, (text, colour, line, marker) in enumerate(entries):
        x = left + (i % per_row) * col_w
        y = top + (i // per_row) * 22
        sample = 32
        if line:
            kw = {} if line == "solid" else {"dashes": (6 * PX, 4 * PX)}
            fig.lines.append(
                plt.Line2D(
                    [x / W, (x + sample) / W],
                    [1 - y / H] * 2,
                    transform=fig.transFigure,
                    color=colour,
                    lw=2.5 * PX,
                    **kw,
                )
            )
        if marker:
            fig.lines.append(
                plt.Line2D(
                    [(x + sample / 2) / W],
                    [1 - y / H],
                    transform=fig.transFigure,
                    ls="none",
                    marker=marker,
                    ms=8 * PX,
                    mfc=colour,
                    mec=colour if marker == "x" else "white",
                    mew=(2 if marker == "x" else 1.2) * PX,
                )
            )
        fig.text(
            (x + sample + 8) / W,
            1 - y / H,
            text,
            ha="left",
            va="center",
            fontsize=12 * PX,
            fontweight=600,
            color=DASH_LEGEND,
        )
    return top + ((len(entries) - 1) // per_row) * 22


def _dashboard_axes(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(True, color=DASH_GRID, lw=1 * PX, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6, labelsize=12 * PX, labelcolor=DASH_TICK)


def _in_plot_note(ax, text):
    if text:
        ax.text(
            0.99,
            0.985,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12 * PX,
            fontweight=500,
            color=ec.NAVY,
            alpha=0.6,
            zorder=1,
        )


# -------------------------------------------------------------------- 8k1k

# Spots tried for a cN tag, in order: (dx px, dy px, ha, va). The dashboard
# draws the tag up and to the right of the point, so that spot comes first.
TAG_SPOTS = (
    (6, 6, "left", "bottom"),
    (6, -6, "left", "top"),
    (-6, 6, "right", "bottom"),
    (-6, -6, "right", "top"),
    (0, 10, "center", "bottom"),
    (0, -10, "center", "top"),
)


def _tag_points(fig, ax, curves, marker_px):
    """Tag every point with cN at the first spot clear of markers, lines,
    earlier tags and the plot edge (the least-overlap spot when none is
    clear). A point within ec.SAME_SPOT px of an already tagged point of the
    same concurrency shares its tag. Newest curve first."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    frame = ax.get_window_extent(renderer)
    half = marker_px / 2 + 2
    boxes = []
    for curve in curves:
        pts = [to_px((p.x, p.y)) for p in curve.points]
        boxes += [(x - half, y - half, x + half, y + half) for x, y in pts]
        for (x0, y0), (x1, y1) in itertools.pairwise(pts):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) // 4))
            for k in range(n + 1):
                x, y = x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n
                boxes.append((x - 2.5, y - 2.5, x + 2.5, y + 2.5))

    def cost(b):
        inside = (
            frame.x0 + 2 <= b[0]
            and b[2] <= frame.x1 - 2
            and frame.y0 + 2 <= b[1]
            and b[3] <= frame.y1 - 2
        )
        overlap = sum(
            max(0, min(b[2], o[2]) - max(b[0], o[0]))
            * max(0, min(b[3], o[3]) - max(b[1], o[1]))
            for o in boxes
        )
        return overlap + (0 if inside else 1e6)

    tagged = []
    for curve in reversed(curves):
        for p in curve.points:
            px, py = to_px((p.x, p.y))
            if any(
                c == p.c
                and abs(px - tx) <= ec.SAME_SPOT
                and abs(py - ty) <= ec.SAME_SPOT
                for c, tx, ty in tagged
            ):
                continue
            tagged.append((p.c, px, py))
            best = None
            for dx, dy, ha, va in TAG_SPOTS:
                t = ax.annotate(
                    f"c{p.c}",
                    (p.x, p.y),
                    xytext=(dx * PX, dy * PX),
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                    fontsize=11 * PX,
                    fontweight=500,
                    color=ec.CONC_INK,
                    zorder=6,
                )
                b = ec.text_box(t, renderer)
                c = cost(b)
                if best is None or c < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (c, t, b)
                else:
                    t.remove()
                if c == 0:
                    break
            boxes.append(best[2])


def render_8k1k(curves, out, model_name, note=None, mock=False):
    """The dashboard's Throughput vs Latency chart, one line per image."""
    left, right, top, plot_h = 100, W - 32, 128, 560
    styles = curve_styles(len(curves))
    entries = [
        (
            f"AMD MI455 (ATOM main) · {mmdd(c.image_date)} · {gpu_text(c.gpus)}",
            colour,
            line,
            "o",
        )
        for c, (colour, line) in zip(curves, styles)
    ]
    # legend under the subtitle, then the plot
    probe_rows = 1 + (len(entries) - 1) // max(1, int((W - left - 24) // 290))
    top += (probe_rows - 1) * 22
    bottom = top + plot_h
    foot_top = bottom + 74
    H = foot_top + 2 * 15 + 24
    fig = _figure(H)
    _header(
        fig,
        H,
        f"{model_name} · 8K/1K: Total Tput per GPU vs. Interactive",
        _subtitle(
            curves, "ISL 8K / OSL 1K · total throughput normalized by GPU count", mock
        ),
    )
    _legend(fig, H, 92, left, 290, entries)

    ax = fig.add_axes([left / W, 1 - bottom / H, (right - left) / W, plot_h / H])
    _dashboard_axes(ax)
    xs = [p.x for c in curves for p in c.points]
    ys = [p.y for c in curves for p in c.points]
    x0, x1, xstep = axis_range(xs, headroom=0.05)  # the dashboard's 5% grace
    y0, y1, ystep = axis_range(ys, headroom=0.05)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.xaxis.set_major_locator(MultipleLocator(xstep))
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel(
        "Interactive (tok/s/user)",
        fontsize=13 * PX,
        fontweight=600,
        color=DASH_TICK,
        labelpad=10,
    )
    ax.set_ylabel(
        "Total Tput (tok/s/gpu)",
        fontsize=13 * PX,
        fontweight=600,
        color=DASH_TICK,
        labelpad=10,
    )

    marker_px = 10
    for curve, (colour, line) in zip(curves, styles):
        px_x = [p.x for p in curve.points]
        px_y = [p.y for p in curve.points]
        kw = {} if line == "solid" else {"dashes": (8 * PX, 5 * PX)}
        ax.plot(
            px_x,
            px_y,
            color=colour,
            lw=(4 if line == "solid" else 3) * PX,
            zorder=2,
            **kw,
        )
        ax.plot(
            px_x,
            px_y,
            ls="none",
            marker="o",
            ms=marker_px * PX,
            mfc=colour,
            mec="white",
            mew=1.5 * PX,
            zorder=4,
        )
    _in_plot_note(ax, _note(note, mock))
    _tag_points(fig, ax, curves, marker_px)
    _footer(
        fig,
        H,
        foot_top,
        left,
        [
            f"{SOURCE} with ATOM benchmark_serving (random dataset, ISL 8192 / OSL 1024).",
            "Y = (input + output) tok/s ÷ GPUs · X = 1000 ÷ mean TPOT (ms) · cN = concurrency.",
        ],
    )
    fig.savefig(out, dpi=ec.DPI_OUT, facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- accuracy


def _percent_step(span):
    return 0.02 if span <= 0.12 else 0.05 if span <= 0.35 else 0.1


def render_accuracy(histories, out, scenario_label, baselines=None, mock=False):
    """GSM8K by image, the dashboard's accuracy trend: one panel per model,
    two per row. `histories` is [(model name, [AccuracyPoint])], `baselines`
    maps a model name to a reference score drawn as a grey dashed line."""
    baselines = baselines or {}
    left, gap, col_w, panel_h, row_h, top = 90, 80, 560, 250, 330, 140
    rows = math.ceil(len(histories) / 2)
    foot_top = top + rows * row_h - row_h + panel_h + 64
    H = foot_top + 2 * 15 + 24
    fig = _figure(H)
    days = sorted({h.image_date for _, hist in histories for h in hist})
    span = f"{mmdd(days[0])} → {mmdd(days[-1])}" if days else "no data yet"
    subtitle = f"455-CI · MI455 (gfx1250) · lm_eval gsm8k, 3-shot · images {span}"
    _header(
        fig,
        H,
        f"{scenario_label} models · GSM8K accuracy by image",
        f"MOCK DATA · {subtitle}" if mock else subtitle,
    )
    if mock:
        fig.text(
            (W - 32) / W,
            1 - 24 / H,
            MOCK_NOTE,
            ha="right",
            va="top",
            fontsize=12 * PX,
            fontweight=500,
            color=ec.NAVY,
            alpha=0.6,
        )
    _legend(
        fig,
        H,
        92,
        left,
        290,
        [
            ("flexible-extract (reported)", ACCURACY_RED, "solid", "o"),
            ("strict-match", ACCURACY_RED, "dashed", None),
            ("baseline (when set)", DASH_BASELINE, "dashed", None),
            ("run failed (N/A)", FAIL_GREY, None, "x"),
        ],
    )

    for i, (name, hist) in enumerate(histories):
        x = left + (i % 2) * (col_w + gap)
        y = top + (i // 2) * row_h
        ax = fig.add_axes([x / W, 1 - (y + panel_h) / H, col_w / W, panel_h / H])
        _dashboard_axes(ax)
        fig.text(
            x / W,
            1 - (y - 10) / H,
            name,
            ha="left",
            va="bottom",
            fontsize=13 * PX,
            fontweight=600,
            color=ec.INK,
        )
        if not hist:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.5,
                0.5,
                "no data yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12 * PX,
                color=DASH_TICK,
            )
            continue
        ok = [(k, h) for k, h in enumerate(hist) if h.flexible is not None]
        values = [h.flexible for _, h in ok] + [
            h.strict for _, h in ok if h.strict is not None
        ]
        if name in baselines:
            values.append(baselines[name])
        ymin = max(0.0, math.floor(((min(values) if values else 0.9) - 0.03) * 20) / 20)
        ax.set_ylim(ymin, 1.0)
        ax.set_xlim(
            -0.5, len(hist) - 0.5 + 0.6
        )  # room right of the newest point for its label
        ax.set_xticks(range(len(hist)))
        ax.set_xticklabels([mmdd(h.image_date) for h in hist])
        ax.yaxis.set_major_locator(MultipleLocator(_percent_step(1.0 - ymin)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))

        if name in baselines:
            ax.axhline(
                baselines[name],
                color=DASH_BASELINE,
                lw=1.2 * PX,
                dashes=(5 * PX, 4 * PX),
                zorder=1,
            )
            # the value goes on the panel's title row, clear of the data
            fig.text(
                (x + col_w) / W,
                1 - (y - 10) / H,
                f"baseline {baselines[name]:.4f}",
                ha="right",
                va="bottom",
                fontsize=11 * PX,
                color=DASH_TICK,
            )
        strict = [(k, h.strict) for k, h in ok if h.strict is not None]
        if strict:
            ax.plot(
                [k for k, _ in strict],
                [v for _, v in strict],
                color=ACCURACY_RED,
                lw=1.5 * PX,
                dashes=(4 * PX, 3 * PX),
                marker="o",
                ms=6 * PX,
                mfc="white",
                mec=ACCURACY_RED,
                mew=1.2 * PX,
                zorder=2,
            )
        ax.plot(
            [k for k, _ in ok],
            [h.flexible for _, h in ok],
            color=ACCURACY_RED,
            lw=2 * PX,
            marker="o",
            ms=8 * PX,
            mfc=ACCURACY_RED,
            mec="white",
            mew=1.2 * PX,
            zorder=3,
        )
        for k, h in enumerate(hist):
            if h.flexible is None:
                ax.plot(
                    k,
                    ymin,
                    ls="none",
                    marker="x",
                    ms=9 * PX,
                    mew=2 * PX,
                    color=FAIL_GREY,
                    clip_on=False,
                    zorder=5,
                )
                ax.annotate(
                    "N/A",
                    (k, ymin),
                    xytext=(0, 8 * PX),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10 * PX,
                    color=FAIL_GREY,
                )
        last = hist[-1]
        if last.flexible is not None:
            ax.annotate(
                f"{last.flexible * 100:.1f}%",
                (len(hist) - 1, last.flexible),
                xytext=(8 * PX, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=12 * PX,
                fontweight=600,
                color=ec.INK,
                zorder=6,
            )

    _footer(
        fig,
        H,
        foot_top,
        left,
        [
            f"{SOURCE}: lm_eval gsm8k, 3-shot, against the served model (the ATOM CI setup).",
            "Score = exact_match, flexible-extract (the reported number) · strict-match shown for reference.",
        ],
    )
    fig.savefig(out, dpi=ec.DPI_OUT, facecolor="white")
    plt.close(fig)
    print("wrote", out)
