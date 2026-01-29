#! /usr/bin/env python3

import json
import sys
from pathlib import Path

def escape_md(s: str) -> str:
    return s.replace("|", "\\|")

def to_str(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)

def json_to_md_table(data):
    if isinstance(data, dict):
        rows = [data]
    elif isinstance(data, list):
        if not data or not isinstance(data[0], dict):
            raise ValueError("JSON list must contain objects (dict).")
        rows = data
    else:
        raise ValueError("JSON must be an object or a list of objects.")

    headers = list(rows[0].keys())
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("|" + " | ".join(["---"] * len(headers)) + "|")

    for row in rows:
        vals = [escape_md(to_str(row.get(h))) for h in headers]
        md_lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(md_lines)

def main():
    # Usage: python .github/scripts/summarize.py input.json
    if len(sys.argv) < 2:
        print("Usage: python .github/scripts/summarize.py <input.json>", file=sys.stderr)
        sys.exit(1)

    in_path = Path(sys.argv[1])
    data = json.loads(in_path.read_text(encoding="utf-8"))
    md = json_to_md_table(data)
    print(md)

if __name__ == "__main__":
    main()
