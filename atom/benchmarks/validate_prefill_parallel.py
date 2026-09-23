# SPDX-License-Identifier: MIT
"""Capture full-vocabulary logits for identical prefill probes across layouts.

This is a validation runner, never a performance measurement. Instrumentation
is installed in spawned workers as well as the parent, and copies only logits
on rank zero. --reference-exchange uses independent PyTorch packing/transport.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser


def install_probe():
    directory = os.environ.get("ATOM_PREFILL_LOGITS_DIR")
    if not directory:
        return
    from atom.model_engine.model_runner import ModelRunner

    if getattr(ModelRunner.postprocess, "_prefill_probe", False):
        return
    original = ModelRunner.postprocess
    counter = 0

    def postprocess(self, batch, logits, *args, **kwargs):
        nonlocal counter
        if (
            torch.distributed.get_rank() == 0
            and not batch.is_dummy_run
            and logits is not None
        ):
            torch.save(
                {
                    "logits": logits[: batch.total_seqs_num].detach().cpu(),
                    "prefill_tokens": batch.total_tokens_num_prefill,
                },
                Path(directory) / f"logits-{counter:04d}.pt",
            )
            counter += 1
        return original(self, batch, logits, *args, **kwargs)

    postprocess._prefill_probe = True
    ModelRunner.postprocess = postprocess
    if os.environ.get("ATOM_PREFILL_REFERENCE_EXCHANGE") == "1":
        from atom.distributed import ulysses_sp as sp

        def exchange(out, source, group):
            torch.distributed.all_to_all_single(out, source, group=group.device_group)

        def pack(source, spec, world):
            peers = []
            for peer in range(world):
                fields = []
                for offset, width, shards in spec:
                    shards = shards or world
                    local = width // shards
                    begin = offset + (peer // (world // shards)) * local
                    fields.append(source[:, begin : begin + local])
                peers.append(torch.cat(fields, dim=1))
            return torch.stack(peers)

        sp.all_to_all_into = exchange
        sp.pack_fields = pack


# multiprocessing.spawn imports this module before calling worker entrypoints.
install_probe()


def main():
    parser = FlexibleArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--lengths", default="17,1025,4097,131072")
    parser.add_argument("--reference-exchange", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-length", type=int, default=1)
    args = parser.parse_args()
    if args.enable_prefix_caching:
        parser.error("pass --no-enable_prefix_caching")
    lengths = [int(value) for value in args.lengths.split(",")]
    if min(lengths + [args.repeats, args.output_length]) < 1:
        parser.error("lengths, repeats and output length must be positive")
    target = Path(args.dump_dir)
    target.mkdir(parents=True, exist_ok=True)
    if list(target.glob("logits-*.pt")):
        parser.error("dump directory already contains logits; use a fresh directory")
    os.environ["ATOM_PREFILL_LOGITS_DIR"] = str(target.resolve())
    os.environ["ATOM_PREFILL_REFERENCE_EXCHANGE"] = str(int(args.reference_exchange))
    install_probe()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rng = random.Random(321)
    words = [
        "river",
        "mountain",
        "library",
        "telescope",
        "apple",
        "copper",
        "violet",
        "ocean",
        "engine",
        "winter",
        "island",
        "notebook",
        "garden",
        "forest",
        "village",
    ]
    text = " ".join(rng.choices(words, k=max(lengths)))
    tokens = tokenizer.encode(text, add_special_tokens=False)
    llm = EngineArgs.from_cli_args(args).create_engine()
    outputs = []
    try:
        for length in lengths:
            prefix = tokenizer.encode(
                "Remember the code: 731942.\n", add_special_tokens=False
            )
            suffix = tokenizer.encode(
                "\nWhat is the code? Answer:", add_special_tokens=False
            )
            if length < len(prefix) + len(suffix):
                prompt = tokens[:length]
            else:
                prompt = prefix + tokens[: length - len(prefix) - len(suffix)] + suffix
            assert len(prompt) == length
            for repeat in range(args.repeats):
                output = llm.generate(
                    [prompt],
                    SamplingParams(
                        temperature=0,
                        max_tokens=args.output_length,
                        ignore_eos=True,
                        logprobs=1,
                    ),
                )
                outputs.append(
                    {
                        "length": length,
                        "repeat": repeat,
                        "prompt_sha256": hashlib.sha256(
                            json.dumps(prompt).encode()
                        ).hexdigest(),
                        "output": output,
                    }
                )
                print(
                    f"VALIDATED_LENGTH {length}, repeat {repeat}: {output}", flush=True
                )
        (target / "outputs.json").write_text(json.dumps(outputs, indent=2))
        (target / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    finally:
        llm.close()


if __name__ == "__main__":
    main()
