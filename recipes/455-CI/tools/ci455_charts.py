"""Render the 455-CI trend charts, in the ATOM benchmark dashboard's trend style.

- perf-<scenario>.png: tok/s per GPU by image, one panel per model and
  concurrency, c4 left and c64 right. Two points per run make a Pareto chart a
  bare segment, so the history is shown as a trend instead.
- accuracy-<scenario>.png: GSM8K by image, one panel per model:
  flexible-extract solid, strict-match dashed, the baseline (when set) grey
  dashed.

Charts are laid out in CSS px at dpi 100 and saved at 2x (2640 px wide), in the
LLM-Booster#246 page frame (endpoint_chart.py): title and subtitle top left,
two footnote lines, Instrument Sans.
"""

import math

import endpoint_chart as ec  # sets the Agg backend and registers the fonts
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

PX = ec.PX
W = 1320
# ATOM benchmark dashboard, light theme (.github/dashboard/index.html). The
# grid and baseline are the opaque equivalents on white of its
# rgba(0,0,0,0.10) and rgba(136,153,184,0.5): matplotlib grid lines carry
# alpha 1, which would override an rgba colour's own alpha.
DASH_TICK = "#5a5d70"
DASH_LEGEND = "#4a4d5e"
DASH_GRID = "#E6E6E6"
DASH_BASELINE = "#C4CCDC"
FAIL_GREY = "#8A8F98"
RED = "#9B1C1C"  # the MI455 deep red of the LLM-Booster#246 charts

SOURCE = "Source: AMD MI455 455-CI runs on ATOM main + aiter main, self-measured"
MOCK_NOTE = "EXAMPLE: mock data, not real measurements."
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
LEFT, GAP, COL_W, TOP = 90, 80, 560, 140


def mmdd(image_date):
    return image_date[5:]


def _num(v):
    return f"{v:,.1f}" if v < 100 else f"{v:,.0f}"


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


# ------------------------------------------------------------ page frame


def _figure(height):
    return plt.figure(figsize=(W / 100, height / 100), dpi=100, facecolor="white")


def _header(fig, H, title, subtitle, mock):
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
        f"EXAMPLE (mock data) · {subtitle}" if mock else subtitle,
        ha="left",
        va="top",
        fontsize=13 * PX,
        fontweight=500,
        color=ec.INK_MUTED,
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


def _footer(fig, H, top, lines):
    for i, line in enumerate(lines):
        fig.text(
            LEFT / W,
            1 - (top + i * 15) / H,
            line,
            ha="left",
            va="top",
            fontsize=11 * PX,
            color=ec.INK_MUTED,
        )


def _legend(fig, H, entries, top=92, col_w=290):
    """Dashboard-style legend under the subtitle, one cell per entry: (text,
    colour, line or None, marker or None)."""
    for i, (text, colour, line, marker) in enumerate(entries):
        x, y, sample = LEFT + i * col_w, top, 32
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


def _layout(n, panel_h, row_h):
    rows = math.ceil(n / 2)
    foot_top = TOP + (rows - 1) * row_h + panel_h + 64
    return foot_top, foot_top + 2 * 15 + 24


def _panel(fig, H, i, panel_h, row_h, title, right=""):
    x = LEFT + (i % 2) * (COL_W + GAP)
    y = TOP + (i // 2) * row_h
    ax = fig.add_axes([x / W, 1 - (y + panel_h) / H, COL_W / W, panel_h / H])
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(True, color=DASH_GRID, lw=1 * PX, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6, labelsize=12 * PX, labelcolor=DASH_TICK)
    fig.text(
        x / W,
        1 - (y - 10) / H,
        title,
        ha="left",
        va="bottom",
        fontsize=13 * PX,
        fontweight=600,
        color=ec.INK,
    )
    if right:
        fig.text(
            (x + COL_W) / W,
            1 - (y - 10) / H,
            right,
            ha="right",
            va="bottom",
            fontsize=12 * PX,
            fontweight=600,
            color=DASH_TICK,
        )
    return ax


def _dates(ax, dates, extra_right=0.0):
    ax.set_xlim(-0.5, len(dates) - 0.5 + extra_right)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([mmdd(d) for d in dates])


def _no_data(ax):
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


def _failures(ax, positions, y):
    for k in positions:
        ax.plot(
            k,
            y,
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
            (k, y),
            xytext=(0, 8 * PX),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10 * PX,
            color=FAIL_GREY,
        )


def _save(fig, out):
    fig.savefig(out, dpi=ec.DPI_OUT, facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ perf


def _latest_text(trend):
    """The newest value and its change against the previous success."""
    if not trend:
        return ""
    *before, (_, now) = trend
    if now is None:
        return "N/A"
    prev = next((v for _, v in reversed(before) if v is not None), None)
    return f"{_num(now)} ({(now / prev - 1) * 100:+.1f}%)" if prev else _num(now)


def render_perf_trends(panels, out, scenario_label, mock=False):
    """tok/s per GPU by image. `panels` is [(title, [(image date, value or
    None)])], two per row: c4 left and c64 right for each model."""
    panel_h, row_h = 190, 270
    foot_top, H = _layout(len(panels), panel_h, row_h)
    fig = _figure(H)
    days = sorted({d for _, trend in panels for d, _ in trend})
    span = f"{mmdd(days[0])} → {mmdd(days[-1])}" if days else "no data yet"
    _header(
        fig,
        H,
        f"{scenario_label} · throughput per GPU by image",
        f"455-CI · MI455 (gfx1250) · ATOM main + aiter main · images {span}",
        mock,
    )
    _legend(
        fig,
        H,
        [
            ("tok/s per GPU", RED, "solid", "o"),
            ("run failed (N/A)", FAIL_GREY, None, "x"),
        ],
    )
    for i, (title, trend) in enumerate(panels):
        ax = _panel(fig, H, i, panel_h, row_h, title, _latest_text(trend))
        ok = [(k, v) for k, (_, v) in enumerate(trend) if v is not None]
        if not trend:
            _no_data(ax)
            continue
        _dates(ax, [d for d, _ in trend])
        if ok:
            lo, hi, step = axis_range([v for _, v in ok], headroom=0.1, max_ticks=5)
            ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(MultipleLocator(step))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _num(v)))
            ax.plot(
                [k for k, _ in ok],
                [v for _, v in ok],
                color=RED,
                lw=2 * PX,
                marker="o",
                ms=8 * PX,
                mfc=RED,
                mec="white",
                mew=1.2 * PX,
                zorder=3,
            )
        else:
            lo = 0
            ax.set_ylim(0, 1)
            ax.set_yticks([])
        _failures(ax, [k for k, (_, v) in enumerate(trend) if v is None], lo)
    if scenario_label == "8K/1K":
        how, tokens = (
            "ATOM benchmark_serving (random dataset, ISL 8192 / OSL 1024)",
            "input + output",
        )
    else:
        how, tokens = (
            "the MLPerf Endpoints client (10-min windows), not MLCommons-verified",
            "output",
        )
    _footer(
        fig,
        H,
        foot_top,
        [
            f"{SOURCE} with {how}.",
            f"Y = {tokens} tok/s ÷ GPUs · top right: the newest value and its change against the previous success.",
        ],
    )
    _save(fig, out)


# -------------------------------------------------------------- accuracy


def _percent_step(span):
    return 0.02 if span <= 0.12 else 0.05 if span <= 0.35 else 0.1


def render_accuracy(histories, out, scenario_label, baselines=None, mock=False):
    """GSM8K by image, the dashboard's accuracy trend: one panel per model,
    two per row. `histories` is [(model name, [AccuracyPoint])], `baselines`
    maps a model name to a reference score drawn as a grey dashed line."""
    baselines = baselines or {}
    panel_h, row_h = 250, 330
    foot_top, H = _layout(len(histories), panel_h, row_h)
    fig = _figure(H)
    days = sorted({h.image_date for _, hist in histories for h in hist})
    span = f"{mmdd(days[0])} → {mmdd(days[-1])}" if days else "no data yet"
    _header(
        fig,
        H,
        f"{scenario_label} models · GSM8K accuracy by image",
        f"455-CI · MI455 (gfx1250) · lm_eval gsm8k, 3-shot · images {span}",
        mock,
    )
    _legend(
        fig,
        H,
        [
            ("flexible-extract (reported)", RED, "solid", "o"),
            ("strict-match", RED, "dashed", None),
            ("baseline (when set)", DASH_BASELINE, "dashed", None),
            ("run failed (N/A)", FAIL_GREY, None, "x"),
        ],
    )
    for i, (name, hist) in enumerate(histories):
        base = baselines.get(name)
        ax = _panel(
            fig,
            H,
            i,
            panel_h,
            row_h,
            name,
            f"baseline {base:.4f}" if base is not None else "",
        )
        if not hist:
            _no_data(ax)
            continue
        ok = [(k, h) for k, h in enumerate(hist) if h.flexible is not None]
        values = [h.flexible for _, h in ok] + [
            h.strict for _, h in ok if h.strict is not None
        ]
        if base is not None:
            values.append(base)
        ymin = max(0.0, math.floor(((min(values) if values else 0.9) - 0.03) * 20) / 20)
        ax.set_ylim(ymin, 1.0)
        _dates(
            ax, [h.image_date for h in hist], extra_right=0.6
        )  # room for the newest label
        ax.yaxis.set_major_locator(MultipleLocator(_percent_step(1.0 - ymin)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
        if base is not None:
            ax.axhline(
                base,
                color=DASH_BASELINE,
                lw=1.2 * PX,
                dashes=(5 * PX, 4 * PX),
                zorder=1,
            )
        strict = [(k, h.strict) for k, h in ok if h.strict is not None]
        if strict:
            ax.plot(
                [k for k, _ in strict],
                [v for _, v in strict],
                color=RED,
                lw=1.5 * PX,
                dashes=(4 * PX, 3 * PX),
                marker="o",
                ms=6 * PX,
                mfc="white",
                mec=RED,
                mew=1.2 * PX,
                zorder=2,
            )
        ax.plot(
            [k for k, _ in ok],
            [h.flexible for _, h in ok],
            color=RED,
            lw=2 * PX,
            marker="o",
            ms=8 * PX,
            mfc=RED,
            mec="white",
            mew=1.2 * PX,
            zorder=3,
        )
        _failures(ax, [k for k, h in enumerate(hist) if h.flexible is None], ymin)
        if hist[-1].flexible is not None:
            ax.annotate(
                f"{hist[-1].flexible * 100:.1f}%",
                (len(hist) - 1, hist[-1].flexible),
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
        [
            f"{SOURCE}: lm_eval gsm8k, 3-shot, against the served model (the ATOM CI setup).",
            "Score = exact_match, flexible-extract (the reported number) · strict-match shown for reference.",
        ],
    )
    _save(fig, out)
