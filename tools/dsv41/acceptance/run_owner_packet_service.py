"""Full GSM or one fresh service arm of the installed owner packet package."""

import json

from run_owner_serving import OUTPUT


def commit_counts():
    result = {}
    for path in OUTPUT.glob("serving_commit_pid*.json"):
        for key, value in json.loads(path.read_text())["counts"].items():
            result[key] = result.get(key, 0) + value
    return result
