#!/usr/bin/env python3
"""Profile Kimi-K3.1 Pro bs=4 DSpark decode over long contexts at TP8 + DCP8.

Prefills four random prompts unprofiled, starts the profiler once every request
has its first token, and profiles the rest of their decode at a fixed acceptance
length (all drafts accepted unless --spec-decode-acceptance-length says
otherwise). Graphs are captured for batch sizes 1, 2 and 4 with --mark-trace on,
so each rank's profiler directory also holds the capture traces
``tools/parse_decode_trace.py`` attributes the replayed kernels through.

A compile cached without --mark-trace has no layer markers: clear
``$HOME/.cache/atom`` before the first run, and whenever --mark-trace changes.
"""

import os
import random
import threading
from glob import glob

from k31_prefill_profile import DEFAULT_ARGS, DEFAULT_ENV, parser
from transformers import AutoTokenizer

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs

NUM_PROMPTS = 4
DECODE_ENV = DEFAULT_ENV | {"ATOM_ENABLE_DETAILED_ANNOTATION": "1"}
DECODE_ARGS = DEFAULT_ARGS | {
    "max_num_seqs": NUM_PROMPTS,
    "cudagraph_capture_sizes": "[1, 2, 4]",
    "spec_decode_acceptance_length": None,
}

parser.description = __doc__
parser.epilog = (
    "Defaults (Kimi-K3.1 Pro TP8 + DCP8 serving flags; the acceptance length "
    "defaults to --num-speculative-tokens + 1):\n"
    + "\n".join(f"  --{k.replace('_', '-')} {v}" for k, v in DECODE_ARGS.items())
    + "\nEnvironment defaults (an exported value wins):\n"
    + "\n".join(f"  {k}={v}" for k, v in DECODE_ENV.items())
)
parser.set_defaults(**DECODE_ARGS, prompt_len=161000)
parser.add_argument(
    "--output-len",
    type=int,
    default=320,
    help="Tokens per request: 40 decode steps at 8 tokens a step.",
)


def main():
    args = parser.parse_args()
    if args.spec_decode_acceptance_length is None:
        args.spec_decode_acceptance_length = args.num_speculative_tokens + 1
    for key, value in DECODE_ENV.items():
        os.environ.setdefault(key, value)

    llm = EngineArgs.from_cli_args(args).create_engine()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Kimi's special tokens sit above every ordinary token id.
    vocab = min(tokenizer.all_special_ids)
    rng = random.Random(0)
    prompts = [
        [rng.randrange(vocab) for _ in range(args.prompt_len)]
        for _ in range(NUM_PROMPTS)
    ]
    params = SamplingParams(max_tokens=args.output_len, ignore_eos=True)

    decoding, all_decoding = set(), threading.Event()

    def on_output(output):  # runs on the engine's output thread
        if output.output_tokens:
            decoding.add(output.request_id)
        if len(decoding) == NUM_PROMPTS:
            all_decoding.set()

    llm.add_request(prompts, params, stream_callback=on_output)
    while not all_decoding.wait(timeout=10):
        if not llm.core_mgr.is_healthy():
            raise SystemExit("The engine stopped before every request decoded.")
    llm.start_profile()
    while not llm.is_finished():
        llm.io_processor.postprocess(llm.step())
    llm.stop_profile()
    llm.close()
    rank0 = os.path.join(args.torch_profiler_dir, "rank_0", "*.pt.trace.json.gz")
    print(f"Rank 0 trace: {max(glob(rank0), key=os.path.getmtime)}")


if __name__ == "__main__":
    main()
