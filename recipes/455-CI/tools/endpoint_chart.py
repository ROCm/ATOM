"""Shared renderer for the MLPerf Endpoints agentic charts (K3, M3, ...).

Draws the endpoints.mlcommons.org "Agentic: Throughput vs. E2E Average
Interactivity" chart style: x = E2E average interactivity, y = system output
tok/s (or per GPU), right-then-down step lines, square markers for MLCommons
results and diamonds for AMD runs, every point tagged with its concurrency.

A model's script supplies the data and layout through `render(chart)`; see
plot_k3_endpoint.py for the fields.
"""

import csv
import glob
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))

for ttf in glob.glob(os.path.join(HERE, "fonts", "*.ttf")):
    font_manager.fontManager.addfont(ttf)
FONT = (
    "Instrument Sans"
    if any(f.name == "Instrument Sans" for f in font_manager.fontManager.ttflist)
    else "DejaVu Sans"
)
plt.rcParams.update(
    {
        "font.family": [FONT, "DejaVu Sans"],  # fallback for ■ ◆ glyphs
        "lines.scale_dashes": False,
        "axes.unicode_minus": False,
    }
)

# The figure is laid out in CSS px (dpi 100) and saved at 2x, like the site's
# "Download PNG" export.
PX = 0.72  # points per px at dpi 100
DPI_OUT = 200

# Site constants (endpoints.mlcommons.org chart code / CSS).
PLOT_FILL = "#eef0f2"
GRID = "#dcdfe3"
INK = "#1e293b"
INK_AXIS = "#334155"  # slate-700, axis titles
INK_MUTED = "#64748b"  # slate-500, ticks / footnote
NAVY = "#14304f"
STEP_LEAD = 0.06  # site's lead-in stub, 6% left of the first point
# Step line widths in px (the site draws 2 / 1; the user asked for bolder).
H_LW, V_LW = 3.4, 2.0
# (horizontal, vertical) dash patterns in px, by a series' "line" key; None
# = solid. Without a "line" key a series keeps the site's look: dotted steps,
# dashed risers.
DASHES = {
    None: ((H_LW, 3), (6, 4)),
    "solid": (None, None),
    "dashed": ((8, 5), (6, 4)),
}
# (horizontal, vertical) widths in px by a series' "line" key: the user wants
# solid lines a bit heavier and dashed ones a touch lighter than the site style.
WIDTHS = {
    None: (H_LW, V_LW),
    "solid": (4.0, 2.4),
    "dashed": (3.0, 1.8),
}
LEGEND_SWATCH = 40  # px: line sample in the legend for series with a "line" key


def dash_kw(pattern):
    return {} if pattern is None else {"dashes": (pattern[0] * PX, pattern[1] * PX)}


X_TITLE = "E2E Average Interactivity (Tokens/s per User) · constraint: minimum or equal"
Y_TITLE = {
    "per_gpu": "System Throughput per GPU (Tokens/s per GPU) · higher is better",
    "system": "System Throughput (Tokens/s) · higher is better",
}


def load_series(path, specs, x_from=None):
    """Points per series spec from the CSV at `path`.

    A spec picks its rows by `match` (a substring of the system column). A
    series in `x_from` without measured x takes it from the same-concurrency
    point of the named series, scaled by the throughput ratio:
    x = x_ref * y / y_ref.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for spec in specs:
        pts = [
            {
                "c": int(r["concurrency"]),
                "x": (
                    float(r["e2e_avg_interactivity"])
                    if r["e2e_avg_interactivity"]
                    else None
                ),
                "y_system": float(r["system_tps"]),
                "y_gpu": float(r["system_tps"]) / spec["gpus"],
            }
            for r in rows
            if spec["match"] in r["system"]
        ]
        assert pts, spec["match"]
        out[spec["key"]] = {**spec, "points": pts}
    for key, ref_key in (x_from or {}).items():
        ref = {p["c"]: p for p in out[ref_key]["points"]}
        for p in out[key]["points"]:
            if p["x"] is None:
                r = ref[p["c"]]
                p["x"] = r["x"] * p["y_gpu"] / r["y_gpu"]
    for s in out.values():
        assert all(p["x"] is not None for p in s["points"]), s["key"]
        s["points"].sort(key=lambda p: p["x"])
    return [out[spec["key"]] for spec in specs]


def step_segments(pts, ykey, stub=True):
    """The site's line: right-then-down steps (sorted by x), led in by a short
    stub left of the first point unless `stub` is off."""
    horiz, vert = [], []
    if stub:
        x0, y0 = pts[0]["x"], pts[0][ykey]
        horiz.append(((x0 * (1 - STEP_LEAD), x0), (y0, y0)))
    for prev, cur in itertools.pairwise(pts):
        horiz.append(((prev["x"], cur["x"]), (prev[ykey], prev[ykey])))
        vert.append(((cur["x"], cur["x"]), (prev[ykey], cur[ykey])))
    return horiz, vert


def marker_style(s):
    """Filled marker with a 2px white edge, as on the site."""
    return {
        "marker": s["marker"],
        "ms": s["size"] * PX,
        "mfc": s["color"],
        "mec": "white",
        "mew": 2 * PX,
    }


def draw_series(ax, s, ykey):
    h_dash, v_dash = DASHES[s.get("line")]
    h_lw, v_lw = WIDTHS[s.get("line")]
    horiz, vert = step_segments(s["points"], ykey, s.get("stub", True))
    for xs, ys in vert:
        ax.plot(xs, ys, color=s["color"], lw=v_lw * PX, zorder=2, **dash_kw(v_dash))
    for xs, ys in horiz:
        ax.plot(xs, ys, color=s["color"], lw=h_lw * PX, zorder=2, **dash_kw(h_dash))
    ax.plot(
        [p["x"] for p in s["points"]],
        [p[ykey] for p in s["points"]],
        ls="none",
        zorder=4,
        **marker_style(s),
    )


def si_tick(v, _pos=None):
    """d3 format('.2s') as used on the site: 0.0, 5.0k, 10k, 25k."""
    if abs(v) < 1e-9:
        return "0.0"
    if abs(v) >= 1000:
        k = v / 1000
        return f"{k:.1f}k" if abs(k) < 10 else f"{k:.0f}k"
    return f"{v:.0f}"


def style_axes(ax, xlim, ylim, xstep, ystep):
    ax.set_facecolor(PLOT_FILL)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    xt = [v for v in range(0, int(xlim[1]) + 1, xstep) if xlim[0] <= v <= xlim[1]]
    yt = [v * ystep for v in range(int(ylim[1] // ystep) + 1)]
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(si_tick))
    ax.tick_params(
        length=0,
        pad=8,
        labelsize=12 * PX,
        labelcolor=INK_MUTED,
    )
    ax.grid(True, color=GRID, lw=1 * PX, dashes=(2 * PX, 3 * PX), zorder=0)
    ax.set_axisbelow(True)


def label(
    ax,
    x,
    y,
    text,
    dx=0,
    dy=0,
    ha="left",
    va="center",
    px=12,
    weight=600,
    color=INK_AXIS,
):
    return ax.annotate(
        text,
        (x, y),
        xytext=(dx * PX, dy * PX),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=px * PX,
        fontweight=weight,
        color=color,
        zorder=6,
    )


CONC_INK = "#475569"  # slate-600
# Spots tried for a concurrency tag, in order: (dx px, dy px, ha, va).
# Below first: on the site's right-then-down steps the line leaves a point to
# the right and arrives from above, so below and left are usually free.
CONC_CANDIDATES = (
    (0, -9, "center", "top"),
    (-12, 0, "right", "center"),
    (0, 9, "center", "bottom"),
    (12, 0, "left", "center"),
    (-8, -8, "right", "top"),
    (-8, 8, "right", "bottom"),
    (8, -8, "left", "top"),
    (8, 8, "left", "bottom"),
)


def conc_label(ax, p, ykey, dx, dy, ha, va="center", halo=False):
    t = label(
        ax,
        p["x"],
        p[ykey],
        f"c{p['c']}",
        dx=dx,
        dy=dy,
        ha=ha,
        va=va,
        px=11,
        weight=500,
        color=CONC_INK,
    )
    if halo:  # plot-coloured backing, so a line under a forced tag drops out
        t.set_bbox({"boxstyle": "square,pad=0.12", "fc": PLOT_FILL, "ec": "none"})
    return t


def text_box(t, renderer, trim=2.5):
    """Text extent in px, trimmed to the glyphs: "c" + digits fill ~0.73 em of
    a line box that is ~1.17 em tall."""
    bb = t.get_window_extent(renderer)
    return (bb.x0, bb.y0 + trim, bb.x1, bb.y1 - trim)


SAME_SPOT = 20  # px: same-concurrency points this close share one tag


def place_concurrency_labels(fig, ax, targets, series, ykey, placed, overrides=None):
    """Tag every point of `targets` with its concurrency.

    Each tag takes the first CONC_CANDIDATES spot that clears every marker,
    step line, already-placed text and the plot edge; if no spot is clear it
    takes the one with the least overlap. A point within `SAME_SPOT` px of an
    already-tagged point of the same concurrency (two runs that coincide) is
    covered by that tag and gets none of its own. `overrides` maps
    (series key, concurrency) to a fixed (dx, dy, ha, va) spot for points in
    clusters too dense for the search; those tags are placed first, with a
    halo, and the rest avoid them.
    """
    overrides = overrides or {}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    frame = ax.get_window_extent(renderer)
    pad = 2.5  # half the widest (4px solid) step line, plus a little air
    boxes = []
    for s in series:
        half = s["size"] / 2 + 1 + pad  # marker plus its edge
        for p in s["points"]:
            cx, cy = to_px((p["x"], p[ykey]))
            boxes.append((cx - half, cy - half, cx + half, cy + half))
        horiz, vert = step_segments(s["points"], ykey, s.get("stub", True))
        for xs, ys in horiz + vert:
            (x0, y0), (x1, y1) = to_px((xs[0], ys[0])), to_px((xs[1], ys[1]))
            boxes.append(
                (
                    min(x0, x1) - pad,
                    min(y0, y1) - pad,
                    max(x0, x1) + pad,
                    max(y0, y1) + pad,
                )
            )
    gap = 3  # keep tags off the fixed texts (series names)
    boxes += [
        (b[0] - gap, b[1] - gap, b[2] + gap, b[3] + gap)
        for b in (text_box(t, renderer) for t in placed)
    ]

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

    for s in targets:
        for p in s["points"]:
            spot = overrides.get((s["key"], p["c"]))
            if spot:
                boxes.append(
                    text_box(conc_label(ax, p, ykey, *spot, halo=True), renderer)
                )

    tagged = []  # (concurrency, px x, px y) of the points tagged so far
    for s in targets:
        for p in s["points"]:
            px, py = to_px((p["x"], p[ykey]))
            if any(
                c == p["c"] and abs(px - tx) <= SAME_SPOT and abs(py - ty) <= SAME_SPOT
                for c, tx, ty in tagged
            ):
                continue
            tagged.append((p["c"], px, py))
            if (s["key"], p["c"]) in overrides:
                continue
            best = None
            for dx, dy, ha, va in CONC_CANDIDATES:
                t = conc_label(ax, p, ykey, dx, dy, ha, va)
                b = text_box(t, renderer)
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


def make_chart(series, view):
    """One PNG. `view` fields: mode ("per_gpu" | "system"), out, title,
    subtitle, note (in-plot status line or None), xlim, ylim (optional;
    default 0 to 1.04 x the top point), xstep, ystep, footnotes (lines),
    labels ({series key: [name specs]}), tags (series keys whose points get
    concurrency tags, in placement order), tag_overrides (optional; see
    place_concurrency_labels)."""
    mode = view["mode"]
    ykey = "y_gpu" if mode == "per_gpu" else "y_system"
    by_key = {s["key"]: s for s in series}

    W = 1320
    left, right, top = 100, W - 32, 128
    plot_h = 600
    bottom = top + plot_h
    foot_top = bottom + 74
    foot_lines = view["footnotes"]
    legend_top = foot_top + len(foot_lines) * 15 + 14
    H = legend_top + len(series) * 22 + 24

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")

    def fx(x):
        return x / W

    def fy(y):
        return 1 - y / H

    ax = fig.add_axes([fx(left), fy(bottom), (right - left) / W, plot_h / H])

    ymax = max(p[ykey] for s in series for p in s["points"])
    ylim = view.get("ylim") or (0, ymax * 1.04)
    style_axes(ax, view["xlim"], ylim, view["xstep"], view["ystep"])

    for s in series:
        draw_series(ax, s, ykey)

    # Axis titles (site wording).
    ax.set_xlabel(
        X_TITLE, fontsize=14 * PX, fontweight=600, color=INK_AXIS, labelpad=12
    )
    ax.set_ylabel(
        Y_TITLE[mode], fontsize=14 * PX, fontweight=600, color=INK_AXIS, labelpad=12
    )

    # "Better" corner tag and the in-plot status note, as on the site.
    fig.text(
        fx(right - 8),
        fy(top - 9),
        "Better ↗",
        ha="right",
        va="bottom",
        fontsize=12 * PX,
        fontweight=600,
        color=NAVY,
        alpha=0.6,
    )
    placed = []
    if view.get("note"):
        placed.append(
            ax.text(
                0.99,
                0.985,
                view["note"],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=12 * PX,
                fontweight=500,
                color=NAVY,
                alpha=0.5,
                zorder=1,
            )
        )

    # Title block (export: 600 18px at 24/24).
    fig.text(
        fx(24),
        fy(24),
        view["title"],
        ha="left",
        va="top",
        fontsize=18 * PX,
        fontweight=600,
        color=INK,
    )
    fig.text(
        fx(24),
        fy(52),
        view["subtitle"],
        ha="left",
        va="top",
        fontsize=13 * PX,
        fontweight=500,
        color=INK_MUTED,
    )

    # Direct series names: "at" places one at a data position; "c" hangs it
    # off that point of the series by a px offset. Identity is then not
    # colour-only (red/green fail CVD separation).
    for s in series:
        for spec in view["labels"].get(s["key"], []):
            if "at" in spec:
                x, y = spec["at"]
            else:
                p = next(p for p in s["points"] if p["c"] == spec["c"])
                x, y = p["x"], p[ykey]
            placed.append(
                label(ax, x, y, spec.get("text", s["label"]), **spec.get("kw", {}))
            )
    if view.get("tags"):
        targets = [by_key[k] for k in view["tags"]]
        place_concurrency_labels(
            fig, ax, targets, series, ykey, placed, view.get("tag_overrides")
        )

    # Footnote.
    for i, line in enumerate(foot_lines):
        fig.text(
            fx(left),
            fy(foot_top + i * 15),
            line,
            ha="left",
            va="top",
            fontsize=11 * PX,
            color=INK_MUTED,
        )

    # Legend rows (export style: marker, 8px gap, 13px semibold name); a
    # series with a "line" key also gets a sample of its line under the marker.
    for i, s in enumerate(series):
        yc = legend_top + i * 22 + 11
        if s.get("line"):
            fig.lines.append(
                plt.Line2D(
                    [fx(left), fx(left + LEGEND_SWATCH)],
                    [fy(yc)] * 2,
                    transform=fig.transFigure,
                    color=s["color"],
                    lw=WIDTHS[s["line"]][0] * PX,
                    **dash_kw(DASHES[s["line"]][0]),
                )
            )
            marker_x, text_x = left + LEGEND_SWATCH / 2, left + LEGEND_SWATCH + 8
        else:
            marker_x, text_x = left + 7, left + 22
        fig.lines.append(
            plt.Line2D(
                [fx(marker_x)],
                [fy(yc)],
                transform=fig.transFigure,
                ls="none",
                **marker_style(s),
            )
        )
        fig.text(
            fx(text_x),
            fy(yc),
            s["legend"],
            ha="left",
            va="center",
            fontsize=13 * PX,
            fontweight=600,
            color=INK,
        )

    fig.savefig(view["out"], dpi=DPI_OUT, facecolor="white")
    plt.close(fig)
    print("wrote", view["out"])


def render(chart):
    """`chart` fields: data (CSV path), series (specs), x_from (optional),
    views (make_chart views)."""
    series = load_series(chart["data"], chart["series"], chart.get("x_from"))
    for view in chart["views"]:
        make_chart(series, view)
