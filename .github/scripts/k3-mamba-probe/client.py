"""Compare isolated local D and real PD at Mamba state boundaries."""

import argparse
import asyncio
import json
import re
import time
import uuid
from pathlib import Path

import aiohttp


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


async def post(session, base, route, body=None):
    async with session.post(f"{base}/{route}", json=body) as response:
        response.raise_for_status()
        return await response.json()


async def metrics(session, base, path):
    async with session.get(f"{base}/metrics") as response:
        response.raise_for_status()
        text = await response.text()
    path.write_text(text)
    totals = {}
    for line in text.split("\n"):
        if line and not line.startswith("#"):
            key, value = line.rsplit(" ", 1)
            name = key.split("{", 1)[0]
            if name.endswith("_total"):
                totals[name] = totals.get(name, 0) + float(value)
    return totals


async def prepare(session, args):
    async def tokenize(text):
        value = await post(
            session,
            args.prefill,
            "tokenize",
            {
                "model": args.model,
                "prompt": text,
                "add_special_tokens": False,
            },
        )
        return value["tokens"]

    filler = await tokenize(
        "The library opens at nine. Books are sorted by author and subject. " * 1000
    )
    lengths = [1536, 1537, 1538, 12288, 12289, 12290]
    cases = []
    for i in range(64):
        start = await tokenize(f"Record {i}. Read the background.\n")
        end = await tokenize(
            f"\nA shop has {i + 12} boxes with 7 pencils in each box. "
            "It sells 5 pencils. How many pencils remain? "
            "Give only the number.\nAnswer:"
        )
        length = lengths[i % len(lengths)]
        needed = length - len(start) - len(end)
        assert needed > 0
        prompt = start + (filler * (needed // len(filler) + 1))[:needed] + end
        assert len(prompt) == length
        cases.append({"case": i, "prompt": prompt, "expected": (i + 12) * 7 - 5})
    with (args.out / "prompts.jsonl").open("w") as stream:
        for case in cases:
            stream.write(json.dumps(case) + "\n")
    return cases


async def cohort(session, args, cases, role, concurrency, experiment):
    label = f"{role}-c{concurrency}"
    base = args.decode if role == "direct" else args.router
    before = await metrics(session, args.decode, args.out / f"{label}.before.metrics")
    sem = asyncio.Semaphore(concurrency)
    rows = []
    output = args.out / f"{label}.jsonl"
    output.write_text("")

    async def one(case):
        salt = f"k3mb-{experiment}-{label}-{case['case']}"
        row = {
            "case": case["case"],
            "role": role,
            "concurrency": concurrency,
            "salt": salt,
            "prompt_len": len(case["prompt"]),
            "expected": case["expected"],
        }
        body = {
            "model": args.model,
            "prompt": case["prompt"],
            "cache_salt": salt,
            "temperature": 0,
            "seed": 1234,
            "max_tokens": 32,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        async with sem:
            started = time.monotonic()
            try:
                value = await post(session, base, "v1/completions", body)
                row["response"] = value
                json.dumps(value, allow_nan=False)
                choice = value["choices"][0]
                row["ok"] = (
                    value["usage"]["prompt_tokens"] == row["prompt_len"]
                    and value["usage"]["completion_tokens"] == 32
                    and type(choice["token_ids"]) is list
                    and len(choice["token_ids"]) == 32
                    and all(
                        type(token) is int and token >= 0
                        for token in choice["token_ids"]
                    )
                    and choice["prompt_token_ids"] == case["prompt"]
                    and choice["finish_reason"] == "length"
                )
                match = re.match(r"\s*(\d+)", choice["text"])
                row["answer_starts_correct"] = (
                    bool(match) and int(match[1]) == case["expected"]
                )
            except (
                aiohttp.ClientError,
                TimeoutError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            row["seconds"] = time.monotonic() - started
            rows.append(row)
            with output.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                json.dumps({k: v for k, v in row.items() if k != "response"}),
                flush=True,
            )

    await asyncio.gather(*(one(case) for case in cases))
    after = await metrics(session, args.decode, args.out / f"{label}.after.metrics")
    hits = "vllm:prefix_cache_hits_total"
    delta = {k: after[k] - before[k] for k in before if k in after}
    assert hits in delta, "Missing local prefix cache metric"
    summary = {
        "requests": len(rows),
        "ok": sum(r["ok"] for r in rows),
        "answers_start_correct": sum(
            r.get("answer_starts_correct", False) for r in rows
        ),
        "metric_delta": delta,
        "cold_local_prefix": delta[hits] == 0,
    }
    dump(args.out / f"{label}.summary.json", summary)
    assert summary["ok"] == len(cases), summary
    assert summary["cold_local_prefix"], summary
    return {r["case"]: r for r in rows}


def check_observations(args, expected):
    observations = {}
    for path in args.logs.glob("decode-*.log"):
        for line in path.open(errors="replace"):
            if "K3_MAMBA_BOUNDARY " not in line:
                continue
            value = json.loads(line.split("K3_MAMBA_BOUNDARY ", 1)[1])
            observations.setdefault(value["salt"], []).append(value)
    checks = []
    for row in expected:
        found = observations.get(row["salt"], [])
        marked = row["role"] == "pd" and row["prompt_len"] in (1537, 12289)
        valid = len(found) == 1
        if valid:
            obs = found[0]
            valid = (
                obs["N"] == row["prompt_len"]
                and obs["restored"] == marked
                and obs["local"] == 0
            )
            if row["role"] == "pd":
                valid = (
                    valid
                    and obs["C"] == obs["external"] == row["prompt_len"] - 1
                    and obs["has_sync"]
                    and obs["Q"] in (1, 5)
                )
            else:
                valid = valid and obs["C"] == obs["external"] == 0
        checks.append(
            {
                "salt": row["salt"],
                "expected_marker": marked,
                "valid": valid,
                "observations": found,
            }
        )
    dump(args.out / "boundary-observations.json", checks)
    assert all(c["valid"] for c in checks), (
        "Boundary coverage/salt/cold-state check failed"
    )
    q5_lengths = {
        c["observations"][0]["N"]
        for c in checks
        if c["expected_marker"] and c["observations"][0]["Q"] == 5
    }
    dump(
        args.out / "coverage-summary.json", {"q5_boundary_lengths": sorted(q5_lengths)}
    )
    assert q5_lengths == {1537, 12289}, (
        "Both state boundaries need actual DSpark Q5 coverage"
    )


async def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    experiment = uuid.uuid4().hex
    timeout = aiohttp.ClientTimeout(total=600)
    expected = []
    comparison = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        cases = await prepare(session, args)
        for concurrency, selected in [(1, cases[:6]), (64, cases)]:
            direct = await cohort(
                session, args, selected, "direct", concurrency, experiment
            )
            pd = await cohort(session, args, selected, "pd", concurrency, experiment)
            expected.extend(direct.values())
            expected.extend(pd.values())
            for index in direct:
                a = direct[index]["response"]["choices"][0]["token_ids"]
                b = pd[index]["response"]["choices"][0]["token_ids"]
                comparison.append(
                    {
                        "case": index,
                        "concurrency": concurrency,
                        "token_ids_equal": a == b,
                        "first_token_equal": a[:1] == b[:1],
                        "direct_answer_correct": direct[index]["answer_starts_correct"],
                        "pd_answer_correct": pd[index]["answer_starts_correct"],
                    }
                )
        dump(args.out / "comparisons.json", comparison)
        check_observations(args, expected)
    print(
        "MAMBA-PROBE completed; comparison is diagnostic, full GSM8K follows",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prefill", "decode", "router", "model"]:
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
