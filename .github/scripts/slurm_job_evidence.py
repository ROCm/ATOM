# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Select bounded diagnostic fields from one scontrol job response.

Never persist raw responses: they may contain command or environment values.
This formatter supplies evidence only; the shell helper owns result parsing.
"""

import argparse
import datetime
import json
import re
import sys


FIELDS = frozenset({
    "JobId", "JobState", "Reason", "ExitCode", "DerivedExitCode", "BatchHost",
    "NodeList", "FailedNode", "NumNodes", "SubmitTime", "EligibleTime",
    "StartTime", "EndTime", "RunTime", "TimeLimit", "Restarts", "BatchFlag",
})


def select_fields(raw, job_id):
    matches = list(re.finditer(r"(?<!\S)([A-Za-z][A-Za-z0-9_]*)=", raw))
    fields = {}
    selected = False
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        key = match.group(1)
        value = raw[match.end():end].strip()
        if key == "JobId":
            selected = value == job_id
        if selected and key in FIELDS:
            value = re.sub(r"[\x00-\x1f\x7f]", " ", value)
            fields[key] = " ".join(value.split())[:512]
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True, type=int)
    args = parser.parse_args()
    fields = select_fields(sys.stdin.read(1024 * 1024), str(args.job_id))
    print(json.dumps({
        "schema_version": 1,
        "job_id": args.job_id,
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "scontrol_show_job",
        "record_found": bool(fields),
        "fields": fields,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
