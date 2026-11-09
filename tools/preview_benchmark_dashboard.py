#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preview the dashboard locally with published data.

Example:
    python tools/preview_benchmark_dashboard.py --refresh

Only reads public dashboard assets. This script does not publish data or require GitHub credentials.
"""

import argparse
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


def prepare(output, refresh=False):
    output.mkdir(parents=True, exist_ok=True)
    for name, url in ASSETS.items():
        target = output / name
        if refresh or not target.exists():
            print(f"Downloading {name}", flush=True)
            with urlopen(url, timeout=90) as response:
                target.write_bytes(response.read())
    for page in ("index.html", "agentic.html"):
        html = (ROOT / ".github/dashboard" / page).read_text(encoding="utf-8")
        for name, url in ASSETS.items():
            html = html.replace(url, name)
        (output / page).write_text(html, encoding="utf-8")
    return urlencode(
        {"hardware": "mi355x", "intvty": "p90_itl", "explorerSource": "ATOM CI"}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    fragment = prepare(args.output, args.refresh)
    print(f"Preview: http://127.0.0.1:{args.port}/agentic.html#{fragment}", flush=True)
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
