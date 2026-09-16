"""Check a serving API's model list, never a DP worker's metrics endpoint.

For multi-node DP, pass the coordinator's /v1/models URL. Its API starts after
all engine ranks are ready; remote worker HTTP ports expose metrics only.
"""

import argparse
import json
from http.client import HTTPException
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Serving API /v1/models URL")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()

    try:
        with urlopen(args.url, timeout=args.timeout) as response:
            if response.status != 200:
                return 1
            result = json.load(response)
    except (OSError, ValueError, HTTPException):
        return 1

    if not isinstance(result, dict) or result.get("object") != "list":
        return 1
    models = result.get("data")
    if not isinstance(models, list) or not models:
        return 1
    if not all(
        isinstance(model, dict) and isinstance(model.get("id"), str) and model["id"]
        for model in models
    ):
        return 1

    print(json.dumps(result)[:60])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
