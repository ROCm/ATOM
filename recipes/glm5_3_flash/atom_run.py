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


def build_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="GLM-5.3-Flash under ATOM",
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", type=str, default=PROMPT)
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
        [text], SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    )
    print("\n=== ATOM completion ===", flush=True)
    print(outputs[0]["text"], flush=True)


if __name__ == "__main__":
    main()
