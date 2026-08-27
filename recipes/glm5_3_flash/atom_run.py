"""Run GLM-5.3-Flash under ATOM and diff against the transformers oracle.

Kept separate from `atom.examples.simple_inference` for two reasons: the prompts
there run to thousands of tokens, and v1 of this model is only exact at or below
`index_topk` (2048) tokens; and this one uses the same prompt as `ref_run.py` so
the greedy continuation can be compared directly.

  python -m recipes.glm5_3_flash.atom_run --model /models/GLM-5.3-Flash -tp 4
"""

import argparse
import json
import os

from transformers import AutoTokenizer

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser

PROMPT = "Give three reasons why the sky appears blue."
ORACLE = "/out/ref_top10.json"
ATOM_OUT = "/out/atom_gen.json"


def build_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="GLM-5.3-Flash under ATOM",
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=0,
        help="request N logprobs and diff the first token against the oracle",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.cudagraph_capture_sizes = "[1]"

    llm = EngineArgs.from_cli_args(args).create_engine()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    print(f"\nprompt tokens: {len(ids)}", flush=True)

    if os.path.exists(ORACLE):
        with open(ORACLE) as f:
            oracle = json.load(f)
        same = ids == oracle["input_ids"]
        print(
            f"tokenisation matches transformers oracle: {same}"
            + ("" if same else f"\n  atom={ids}\n  ref ={oracle['input_ids']}"),
            flush=True,
        )
        print(
            "oracle greedy first token: "
            f"{oracle['top10_ids'][0]} {tokenizer.decode(oracle['top10_ids'][:1])!r}",
            flush=True,
        )

    outputs = llm.generate(
        [text],
        SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            logprobs=args.logprobs or None,
        ),
    )
    print("\n=== ATOM completion ===", flush=True)
    print(outputs[0]["text"], flush=True)

    # Hand the generated ids to `score_atom_tokens.py`, which asks the
    # transformers reference how likely it finds each one. Agreeing on a whole
    # sequence is far stronger evidence than agreeing on greedy text, which can
    # fork on a near-tie without anything being wrong.
    if os.path.isdir(os.path.dirname(ATOM_OUT)):
        with open(ATOM_OUT, "w") as f:
            json.dump(
                {
                    "prompt": args.prompt,
                    "prompt_ids": ids,
                    "output_ids": outputs[0].get("token_ids", []),
                    "logprobs": outputs[0].get("logprobs", []),
                    "text": outputs[0]["text"],
                },
                f,
                indent=1,
            )
        print(f"\nwrote {ATOM_OUT}", flush=True)

    if args.logprobs:
        compare_first_token(outputs[0], tokenizer)


def compare_first_token(output, tokenizer) -> None:
    """Diff ATOM's first-token distribution against the transformers oracle.

    Greedy text can diverge on a near-tie without anything being wrong, so the
    distribution is the honest comparison, not the sampled token.
    """
    print(f"\n[logprobs] output keys: {sorted(output.keys())}", flush=True)
    lp = output.get("logprobs")
    if not lp:
        print("[logprobs] engine returned none", flush=True)
        return
    first = lp[0]
    if not isinstance(first, dict):
        print(
            f"[logprobs] engine returned per-token scalars ({type(first).__name__}), "
            f"not a top-k table: first 4 = {lp[:4]}\n"
            f"[logprobs] token_ids = {output.get('token_ids', [])[:4]}",
            flush=True,
        )
        return
    ranked = sorted(first.items(), key=lambda kv: -_val(kv[1]))

    print("\n=== ATOM first-token top-10 ===", flush=True)
    atom_ids = []
    for tok, v in ranked[:10]:
        atom_ids.append(int(tok))
        print(
            f"  {int(tok):>7}  {_val(v):9.4f}  {tokenizer.decode([int(tok)])!r}",
            flush=True,
        )

    if not os.path.exists(ORACLE):
        return
    with open(ORACLE) as f:
        oracle = json.load(f)
    ref_ids = oracle["top10_ids"]
    overlap = len(set(atom_ids) & set(ref_ids))
    print(
        f"\ntop-1 match: {atom_ids[0] == ref_ids[0]}  "
        f"(atom {atom_ids[0]} vs oracle {ref_ids[0]})\n"
        f"top-10 overlap: {overlap}/10",
        flush=True,
    )


def _val(v):
    return v.logprob if hasattr(v, "logprob") else float(v)


if __name__ == "__main__":
    main()
