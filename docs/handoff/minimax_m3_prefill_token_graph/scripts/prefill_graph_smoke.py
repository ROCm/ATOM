#!/usr/bin/env python3
import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8890/v1/chat/completions")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = "/models/MiniMax-M3-MXFP4"
    prompts = {
        "base": "alpha " * 300,
        "base_extend": "alpha " * 300 + " beta" * 30,
        "other": "gamma " * 100,
    }
    results = {}
    for name, prompt in prompts.items():
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8,
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            args.url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as response:
            data = json.load(response)
        results[name] = {
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage"),
        }
        print(name, results[name])
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
