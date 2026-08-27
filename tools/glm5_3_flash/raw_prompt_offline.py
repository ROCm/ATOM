"""Feed a raw (un-templated) prompt through ATOM's offline engine.

Companion to `atom_run.py`, which applies the chat template. GSM8K is scored on
`/v1/completions` with a few-shot text prompt and no template, so when the served
path misbehaves this isolates whether the fault is in the model or in serving:
same prompt, same weights, single request, no scheduler.

  python -m tools.glm5_3_flash.raw_prompt_offline --model /models/GLM-5.3-Flash -tp 4
"""

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser

FEWSHOT = """Question: Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. #### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: In the beginning, Betty has only 100 / 2 = $50. Betty's grandparents gave her 15 * 2 = $30. This means, Betty needs 100 - 50 - 30 - 15 = $5 more. #### 5

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Answer:"""

EXPECTED = "624"


def main() -> None:
    parser = FlexibleArgumentParser(description="raw-prompt offline generation")
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--max-tokens", type=int, default=192)
    args = parser.parse_args()
    args.cudagraph_capture_sizes = "[1]"

    llm = EngineArgs.from_cli_args(args).create_engine()
    out = llm.generate(
        [FEWSHOT], SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    )
    text = out[0]["text"]
    print("\n=== OFFLINE COMPLETION ===", flush=True)
    print(repr(text), flush=True)
    print(
        f"\ncontains '#### {EXPECTED}': {('#### ' + EXPECTED) in text}"
        f"\ncontains '{EXPECTED}': {EXPECTED in text}",
        flush=True,
    )


if __name__ == "__main__":
    main()
