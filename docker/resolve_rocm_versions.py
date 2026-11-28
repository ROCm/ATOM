#!/usr/bin/env python3
"""Resolve the newest ROCm nightly date that carries a complete, pinned stack.

A nightly line cannot pin its ROCm version: the channel prunes old date-stamped
builds, so a pin that works today 404s next week. But it must not float
everything either -- one nightly date publishes several torch lines side by side
(2.13, 2.14, 2.15, 2.16.0a0 all against the same SDK), so "take the newest of
each package" would quietly carry the image onto a torch alpha.

So only the date floats. The torch trio stays pinned the way the stable line
pins it, and this script picks the newest date on which every required wheel --
SDK, per-arch device payloads, and those exact torch versions -- exists at once,
at or after a floor. The floor is what stops a line from drifting back onto a
build older than a fix it depends on.

Emits one shell assignment on stdout:

    ROCM_SDK_VERSION=10.3.0a20261118
"""

import argparse
import re
import sys
import urllib.request

TIMEOUT = 120


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "atom-rocm-resolver"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read().decode("utf-8", "replace")


def wheels(index_url, package):
    """Wheel filenames a PEP 503 simple index lists for one package."""
    url = f"{index_url.rstrip('/')}/{package}/"
    try:
        html = fetch(url)
    except Exception as exc:  # noqa: BLE001 - reported as a build failure below
        sys.exit(f"resolve_rocm_versions: cannot read {url}: {exc}")
    return re.findall(r">([^<>]+\.whl)<", html)


def dates_for_sdk(index_url, package, series):
    """Dates on which <package> ships a wheel versioned <series>.0a<date>."""
    stem = re.escape(package.replace("-", "_"))
    pat = re.compile(rf"^{stem}-{re.escape(series)}\.0a(\d{{8}})[-.]")
    return {m.group(1) for m in map(pat.match, wheels(index_url, package)) if m}


def dates_for_version(index_url, package, version, series, python_tag):
    """Dates on which <package> ships exactly <version>+rocm<series>.0a<date>.

    Device payload wheels carry no python tag, so the tag filter only applies to
    the wheels that have one.
    """
    stem = re.escape(package.replace("-", "_"))
    pat = re.compile(
        rf"^{stem}-{re.escape(version)}\+rocm{re.escape(series)}\.0a(\d{{8}})"
        rf"(?:-(\S+?))?\.whl$"
    )
    found = set()
    for name in wheels(index_url, package):
        m = pat.match(name)
        if not m:
            continue
        tags = m.group(2) or ""
        if python_tag and tags and python_tag not in tags:
            continue
        found.add(m.group(1))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-url", required=True)
    ap.add_argument("--series", required=True, help="e.g. 10.3")
    ap.add_argument("--min-version", default="", help="floor, e.g. 10.3.0a20261115")
    ap.add_argument("--arches", default="", help="space/semicolon separated gfx list")
    ap.add_argument("--python-tag", default="cp312")
    ap.add_argument("--torch", required=True)
    ap.add_argument("--torchvision", required=True)
    ap.add_argument("--torchaudio", required=True)
    args = ap.parse_args()

    floor = ""
    if args.min_version:
        m = re.search(r"(\d{8})$", args.min_version)
        if not m:
            sys.exit(
                f"resolve_rocm_versions: --min-version {args.min_version!r} has no date suffix"
            )
        floor = m.group(1)

    arches = [a for a in re.split(r"[;\s]+", args.arches) if a]

    per_pkg = {}
    for pkg in ["rocm-sdk-core", "rocm-sdk-libraries", "rocm-sdk-devel"] + [
        f"rocm-sdk-device-{a}" for a in arches
    ]:
        per_pkg[pkg] = dates_for_sdk(args.index_url, pkg, args.series)

    versioned = [
        ("torch", args.torch),
        ("torchvision", args.torchvision),
        ("torchaudio", args.torchaudio),
    ]
    versioned += [(f"amd-torch-device-{a}", args.torch) for a in arches]
    versioned += [(f"amd-torchvision-device-{a}", args.torchvision) for a in arches]
    for pkg, ver in versioned:
        per_pkg[f"{pkg}=={ver}"] = dates_for_version(
            args.index_url, pkg, ver, args.series, args.python_tag
        )

    empty = [p for p, d in per_pkg.items() if not d]
    if empty:
        sys.exit(
            f"resolve_rocm_versions: no {args.series} wheels on {args.index_url} "
            f"for: {', '.join(sorted(empty))}"
        )

    common = set.intersection(*per_pkg.values())
    if floor:
        common = {d for d in common if d >= floor}
    if not common:
        # Name what is holding the line back rather than just "nothing found":
        # on a nightly channel the answer is almost always one lagging package.
        newest = sorted(((max(d), p) for p, d in per_pkg.items()))[:4]
        sys.exit(
            f"resolve_rocm_versions: no {args.series} date >= {floor or '(any)'} carries "
            "every pinned wheel. Oldest-lagging packages: "
            + ", ".join(f"{p} (newest {d})" for d, p in newest)
        )

    print(f"ROCM_SDK_VERSION={args.series}.0a{max(common)}")


if __name__ == "__main__":
    main()
