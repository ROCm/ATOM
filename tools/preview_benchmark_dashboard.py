#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preview the dashboard locally with published data and optional extra perf points.

Example:
    python tools/preview_benchmark_dashboard.py --refresh

Only reads public dashboard assets. Extra points stay in the preview directory;
this script does not publish data or require GitHub credentials.
"""

import argparse
import json
import tempfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = "https://rocm.github.io/ATOM/benchmark-dashboard/"
ASSETS = {
    name: PUBLIC + name
    for name in (
        "data.js",
        "models_map.js",
        "atomesh_summary.js",
        "atom_logo.png",
        "atom_logo_mini.png",
    )
}
ASSETS.update(
    {
        "chart.umd.min.js": "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js",
        "chartjs-plugin-datalabels.min.js": "https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-datalabels/2.2.0/chartjs-plugin-datalabels.min.js",
    }
)


def prepare(output, points_files, refresh=False):
    output.mkdir(parents=True, exist_ok=True)
    for name, url in ASSETS.items():
        target = output / name
        if refresh or not target.exists():
            print(f"Downloading {name}", flush=True)
            with urlopen(url, timeout=90) as response:
                payload = response.read()
            target.write_bytes(payload)

    points = []
    for file in points_files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("points"), list):
            raise TypeError(f"{file}: expected an object containing a points array")
        points.extend(payload["points"])

    html = (ROOT / ".github/dashboard/index.html").read_text(encoding="utf-8")
    for name, url in ASSETS.items():
        html = html.replace(url, name)
    if points:
        # Escaping '<' keeps JSON text from terminating its script element.
        extra = json.dumps({"points": points}, ensure_ascii=True).replace(
            "<", "\\u003c"
        )
        script = "<script>window.ATOM_LOCAL_BENCHMARK = " + extra + ";</script>\n"
        html = html.replace(
            '<script src="data.js"></script>',
            script + '<script src="data.js"></script>',
        )
    (output / "index.html").write_text(html, encoding="utf-8")
    params = {
        "mode": "Disaggregated",
        "hardware": "mi355x",
        "intvty": "p90_itl",
        "explorerSource": "ATOM CI",
    }
    if points:
        params.update(
            explorerModel=points[0]["model"],
            explorerSource=points[0].get("source_label", "ATOM CI"),
        )
    return urlencode(params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "atom-dashboard-preview" / "site",
    )
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh cached public assets"
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Prepare files without starting a server",
    )
    args = parser.parse_args()
    fragment = prepare(args.output, args.points, args.refresh)
    print(f"Preview: http://127.0.0.1:{args.port}/#{fragment}", flush=True)
    print(f"Files: {args.output.resolve()}", flush=True)
    if not args.no_serve:
        handler = partial(
            SimpleHTTPRequestHandler, directory=str(args.output.resolve())
        )
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
