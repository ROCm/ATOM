#!/usr/bin/env python3
"""Long-prompt logprob A/B for Kimi-K3.1 Pro prefill changes.

``run`` cuts prompts from one text file at evenly spaced token offsets,
prefills each (three 16384-token chunks at the default length) with
``tools/k31_prefill_profile.py``'s engine config, less forced acceptance, trace
markers and the profiler, and writes the first greedy token's top-20 logprobs
to JSON. That token comes from the last chunk's logits, so it depends on every
chunk and on nothing decode does.

``compare A.json B.json`` reports per prompt whether the greedy tokens match,
how many top-20 ids the two share, and the largest |delta logprob| and
|delta probability|. It exits 1 when the probability delta exceeds --threshold
anywhere or a logprob is NaN. Probability gates, not logprob: rounding moves
the far tail of the top 20 by whole nats while its probability barely changes.
The default threshold holds only on text the model predicts near-certainly,
such as ``pydoc_data/topics.py``: where two tokens nearly tie, rounding alone
can move their probabilities past it.

To run another tree, launch from its root with ``PYTHONPATH=<tree>``.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

TOP_LOGPROBS = 20


def run(args):
    from k31_prefill_profile import DEFAULT_ARGS, DEFAULT_ENV
    from transformers import AutoTokenizer

    import atom
    from atom import SamplingParams
    from atom.model_engine.arg_utils import EngineArgs

    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)
    # Logprobs refuse forced acceptance.
    dropped = ("spec_decode_acceptance_length", "mark_trace", "torch_profiler_dir")
    engine_args = {k: v for k, v in DEFAULT_ARGS.items() if k not in dropped}

    with open(args.text, "rb") as f:
        text = f.read()
    tokenizer = AutoTokenizer.from_pretrained(
        engine_args["model"], trust_remote_code=True
    )
    ids = tokenizer.encode(text.decode(), add_special_tokens=False)
    span = len(ids) - args.prompt_len
    if span < args.num_prompts - 1:
        sys.exit(f"{args.text} has {len(ids)} tokens, too few for the prompts asked")
    step = span // max(args.num_prompts - 1, 1)
    offsets = [i * step for i in range(args.num_prompts)]
    prompts = [ids[o : o + args.prompt_len] for o in offsets]
    git = ["git", "-C", os.path.dirname(os.path.dirname(atom.__file__))]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if subprocess.run([*git, "diff", "--quiet", "HEAD"], check=False).returncode:
        commit += "-dirty"

    outputs = [None] * len(prompts)

    def keep(i):
        def callback(output):
            if output.logprobs:
                outputs[i] = output

        return callback

    llm = EngineArgs(**engine_args).create_engine()
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
    )
    # LLMEngine.generate with a callback a prompt: only the stream output
    # carries the top logprobs.
    with llm.core_mgr.offline_backpressure_batch():
        llm.add_request(
            prompts, params, stream_callback=[keep(i) for i in range(len(prompts))]
        )
    while not llm.is_finished() and (llm.core_mgr.is_alive() or llm.core_mgr.is_rest()):
        llm.io_processor.postprocess(llm.step())
    llm.close()
    if None in outputs:
        sys.exit("the engine stopped before every prompt had its logprobs")
    result = {
        "text": args.text,
        "text_sha256": hashlib.sha256(text).hexdigest(),
        "commit": commit,
        "prompt_len": args.prompt_len,
        "prompts": [
            {
                "offset": offset,
                "sha256": hashlib.sha256(str(prompt).encode()).hexdigest(),
                "token": output.output_tokens[0],
                **top_of(output.logprobs[0]),
            }
            for offset, prompt, output in zip(offsets, prompts, outputs)
        ],
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"Wrote {args.out} at {commit}")


def top_of(record):
    """The top ids and logprobs of one stream logprob record.

    A record is (logprob, rank, id0, logprob0, id1, logprob1, ...).
    """
    return {"top_ids": list(record[2::2]), "top_logprobs": list(record[3::2])}


def gap(top_a, top_b, token):
    """|delta| of one token's logprob or probability, or the part the lists prove.

    A token missing from a list sits at or below that list's last value.
    """
    if token not in top_b:
        return max(0.0, top_a[token] - min(top_b.values()))
    if token not in top_a:
        return max(0.0, top_b[token] - min(top_a.values()))
    return abs(top_a[token] - top_b[token])


def compare(a, b):
    """Per prompt: (greedy tokens match, shared top ids, max |delta logprob|,
    max |delta probability|); both deltas are inf where a logprob is NaN."""
    if [p["sha256"] for p in a["prompts"]] != [p["sha256"] for p in b["prompts"]]:
        raise ValueError("the two runs were given different prompts")
    rows = []
    for pa, pb in zip(a["prompts"], b["prompts"]):
        lp_a = dict(zip(pa["top_ids"], pa["top_logprobs"]))
        lp_b = dict(zip(pb["top_ids"], pb["top_logprobs"]))
        p_a = {t: math.exp(v) for t, v in lp_a.items()}
        p_b = {t: math.exp(v) for t, v in lp_b.items()}
        ids = lp_a.keys() | lp_b.keys()
        deltas = [max(gap(x, y, t) for t in ids) for x, y in [(lp_a, lp_b), (p_a, p_b)]]
        if any(map(math.isnan, [*lp_a.values(), *lp_b.values()])):
            deltas = [math.inf, math.inf]
        rows.append(
            (pa["token"] == pb["token"], len(lp_a.keys() & lp_b.keys()), *deltas)
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Prefill the prompts, write JSON.")
    run_parser.add_argument("--text", required=True, help="File to cut prompts from.")
    run_parser.add_argument("--out", required=True, help="JSON to write.")
    run_parser.add_argument("--num-prompts", type=int, default=8)
    run_parser.add_argument("--prompt-len", type=int, default=49152)
    compare_parser = sub.add_parser("compare", help="Compare two runs' JSON.")
    compare_parser.add_argument("a")
    compare_parser.add_argument("b")
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Largest |delta probability| that passes (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)

    runs = []
    for path in (args.a, args.b):
        with open(path) as f:
            runs.append(json.load(f))
    rows = compare(*runs)
    print(f"A {runs[0]['commit']}  B {runs[1]['commit']}")
    print("prompt  offset  match  overlap  max|delta logprob|  max|delta p|")
    for i, (prompt, (match, overlap, d_logprob, delta)) in enumerate(
        zip(runs[0]["prompts"], rows)
    ):
        print(
            f"{i:>6}  {prompt['offset']:>6}  {'yes' if match else 'no':>5}  "
            f"{overlap:>4}/{TOP_LOGPROBS}  {d_logprob:>18.2f}  {delta:.4f}"
        )
    failed = [i for i, (*_, delta) in enumerate(rows) if delta > args.threshold]
    print(f"{'FAIL' if failed else 'PASS'} at threshold {args.threshold}: {failed}")
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
