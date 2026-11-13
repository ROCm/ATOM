#!/usr/bin/env python3
"""Profile 16384-token Kimi-K3.1 Pro prefill chunks at TP8 + DCP8.

Warms up on one random prompt, then profiles a different one with prefix
caching off and one output token, so rank 0's trace holds one ``prefill[``
forward per 16384-token chunk (ctx=16384/32768/49152 at the default length).
``tools/parse_prefill_trace.py`` turns that trace into the per-layer table.

A compile cached without --mark-trace has no layer markers: clear
``$HOME/.cache/atom`` before the first run, and whenever --mark-trace changes.
"""

import argparse
import os
import random
from glob import glob

from transformers import AutoTokenizer

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser

# The Kimi-K3.1 Pro TP8 + DCP8 serving environment; an exported value wins.
DEFAULT_ENV = {
    "AITER_LOG_LEVEL": "WARNING",
    "AITER_SITUV2_A4W4": "1",
    "AITER_FLYDSL_USE_HERD": "1",
    "AITER_FLYDSL_HERD_MIN_M": "64",
    "ATOM_USE_FLYDSL_FP8_PREFILL_ATTN": "1",
    "ATOM_GDN_SSM_DTYPE": "fp32",
    "ATOM_ENABLE_REPLAYSSM": "1",
    "ATOM_MONO_ENABLE": "0",
    "ATOM_STATE_CHECKPOINT_DEMAND": "0",
    "AITER_REUSE_IDENTICAL_COMM_GROUPS": "0",
    "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD": "0",
    "AITER_CUSTOM_AR_DISABLE_LAMPORT": "1",
    "ATOM_DCP_PREFILL_QREP": "1",
    "ATOM_MLA_ABSORB_BF16": "1",
    "ATOM_DCP_PREFILL_A2A_FP8": "1",
}

# The Kimi-K3.1 Pro TP8 + DCP8 serving flags, except prefix caching (off) and
# the LMCache CPU tier (not started: pass --kv-transfer-config to add it).
DEFAULT_ARGS = {
    "model": "/shared/data/amd_int/models/Kimi-K3.1-Pro",
    "trust_remote_code": True,
    "tensor_parallel_size": 8,
    "decode_context_parallel_size": 8,
    "kv_cache_dtype": "fp8",
    "index_cache_dtype": "fp4",
    "index_cache_layout": "mfma",
    "block_size": 64,
    "enable_prefix_caching": False,
    "max_num_seqs": 16,
    "max_num_batched_tokens": 16384,
    "gpu_memory_utilization": 0.94,
    "state_checkpoint_interval_tokens": -1,
    "level": 3,
    "cudagraph_mode": "FULL",
    "cudagraph_capture_sizes": str(list(range(1, 129))),
    "method": "dspark",
    "draft_model": (
        "/shared/data/amd_int/models/yuhua/Kimi-K3.1-Pro-DSpark-preview-6a2ef20a"
    ),
    "num_speculative_tokens": 7,
    "spec_decode_acceptance_length": 3.69,
    "mark_trace": True,
    "torch_profiler_dir": "./profiler_traces",
}

parser = FlexibleArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=__doc__,
    epilog="Defaults (Kimi-K3.1 Pro TP8 + DCP8 serving flags):\n"
    + "\n".join(f"  --{k.replace('_', '-')} {v}" for k, v in DEFAULT_ARGS.items())
    + "\nEnvironment defaults (an exported value wins):\n"
    + "\n".join(f"  {k}={v}" for k, v in DEFAULT_ENV.items()),
)
EngineArgs.add_cli_args(parser)
parser.set_defaults(**DEFAULT_ARGS)
parser.add_argument(
    "--prompt-len",
    type=int,
    default=49152,
    help="Tokens per prompt; a multiple of --max-num-batched-tokens gives full chunks.",
)


def main():
    args = parser.parse_args()
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)

    llm = EngineArgs.from_cli_args(args).create_engine()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Kimi's special tokens sit above every ordinary token id.
    vocab = min(tokenizer.all_special_ids)
    rng = random.Random(0)
    warmup, profiled = (
        [rng.randrange(vocab) for _ in range(args.prompt_len)] for _ in range(2)
    )
    params = SamplingParams(max_tokens=1, ignore_eos=True)

    llm.generate([warmup], params)
    llm.start_profile()
    llm.generate([profiled], params)
    llm.stop_profile()
    llm.close()
    rank0 = os.path.join(args.torch_profiler_dir, "rank_0", "*.pt.trace.json.gz")
    print(f"Rank 0 trace: {max(glob(rank0), key=os.path.getmtime)}")


if __name__ == "__main__":
    main()
