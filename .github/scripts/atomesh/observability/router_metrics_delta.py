"""Router metrics of one benchmark run: the increase of every counter and
histogram series between the /metrics snapshots taken before and after it."""

import argparse
import re
import sys
from pathlib import Path

SCRAPED_AT = "# scraped_at "
SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?\s+(\S+)")
CUMULATIVE = {"counter", "histogram"}


def parse(text: str) -> tuple:
    """Returns the scrape time, the metric types and the samples by (name, labels)."""
    scraped_at, types, samples = None, {}, {}
    for line in text.splitlines():
        if line.startswith(SCRAPED_AT):
            scraped_at = line[len(SCRAPED_AT) :].strip()
        elif line.startswith("# TYPE "):
            _, _, name, kind = line.split(maxsplit=3)
            types[name] = kind.strip()
        elif line and not line.startswith("#"):
            match = SAMPLE.match(line)
            if match:
                samples[match[1], match[2] or ""] = float(match[3])
    return scraped_at, types, samples


def family(name: str, types: dict) -> str:
    for suffix in ("_bucket", "_sum", "_count"):
        if name.endswith(suffix) and types.get(name[: -len(suffix)]) == "histogram":
            return name[: -len(suffix)]
    return name


def number(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def delta(before: str, after: str) -> tuple:
    """Returns the run's increase in Prometheus text format, and whether the
    router restarted between the snapshots."""
    start, before_types, start_samples = parse(before)
    end, types, end_samples = parse(after)
    types = {**before_types, **types}

    def cumulative(key):
        return types.get(family(key[0], types)) in CUMULATIVE

    # Counters and buckets only grow while the router runs, so a lower or
    # missing series means it restarted. Histogram sums can fall.
    reset = any(
        end_samples.get(key, 0.0) < value
        for key, value in start_samples.items()
        if cumulative(key) and not key[0].endswith("_sum")
    )
    lines = [
        f"# router metrics increase from {start or 'unknown'} to {end or 'unknown'}",
        "# counter reset: "
        + ("yes, values count from the router restart" if reset else "no"),
    ]
    typed = set()
    for (name, labels), value in end_samples.items():
        if not cumulative((name, labels)):
            continue
        base = family(name, types)
        if base not in typed:
            typed.add(base)
            lines.append(f"# TYPE {base} {types[base]}")
        if not reset:
            value -= start_samples.get((name, labels), 0.0)
        lines.append(f"{name}{labels} {number(value)}")
    return "\n".join(lines) + "\n", reset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text, reset = delta(args.before.read_text(), args.after.read_text())
    args.output.write_text(text)
    if reset:
        print(
            f"WARNING: the router restarted during the run; {args.output} counts "
            "from the restart",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
