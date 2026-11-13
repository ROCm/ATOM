# SPDX-License-Identifier: MIT
"""Read one MTP acceptance number out of an accuracy result JSON.

The jq fallback for `check_mtp_acceptance.sh`. The nightly host runner has jq
and no python; the PR path runs in a container with python and no jq, and the
gate has to read the same number either way.

Prints the value, or an empty line when it is absent -- which the caller
treats as a failure rather than a pass, since a missing acceptance means the
spec-decode metrics were never captured.
"""

import json
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: read_mtp_metric.py RESULT_JSON overall|position [INDEX]")
        return 2
    path, what = sys.argv[1], sys.argv[2]
    index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle).get("atom_ci_metadata", {})

    if what == "overall":
        value = meta.get("mtp_acceptance_overall")
    else:
        positions = meta.get("mtp_per_pos_acceptance") or []
        value = positions[index] if index < len(positions) else None

    print("" if value is None else value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
