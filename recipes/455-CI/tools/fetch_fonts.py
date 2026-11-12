#!/usr/bin/env python3
"""Download Instrument Sans (the endpoints.mlcommons.org font) into ./fonts.

Google Fonts serves TTF instead of WOFF2 to an old user agent. Without these
files endpoint_chart.py falls back to DejaVu Sans, which works but looks
different from the site.
"""

import os
import re
import subprocess

CSS = "https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700"
os.makedirs("fonts", exist_ok=True)
css = subprocess.run(
    ["curl", "-4", "-sS", "-m", "60", "-A", "Mozilla/4.0", CSS],
    capture_output=True,
    text=True,
    check=True,
).stdout
for weight, url in re.findall(
    r"font-weight:\s*(\d+);.*?src:\s*url\(([^)]+)\)", css, re.DOTALL
):
    out = f"fonts/InstrumentSans-{weight}.ttf"
    subprocess.run(["curl", "-4", "-sS", "-m", "60", "-o", out, url], check=True)
    print("wrote", out)
