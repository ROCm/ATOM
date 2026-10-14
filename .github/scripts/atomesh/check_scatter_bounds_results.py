#!/usr/bin/env python3
"""Reject empty, skipped or failed GPU suites even if pytest returned zero."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def check_results(paths):
    for path in paths:
        cases = list(ET.parse(path).getroot().iter("testcase"))
        if not cases or any(
            case.find(tag) is not None
            for case in cases
            for tag in ("failure", "error", "skipped")
        ):
            raise ValueError(
                f"{path}: expected nonempty GPU tests with no skips/failures"
            )
        print(f"{Path(path).name}: {len(cases)} GPU tests passed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: check_scatter_bounds_results.py RESULTS.xml [...]")
    check_results(sys.argv[1:])
