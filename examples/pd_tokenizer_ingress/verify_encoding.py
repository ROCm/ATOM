"""CPU-only parity check against the checkpoint's real ATOM chat encoder."""

import argparse
import json

from exact_atom_chat import ExactAtomChatEncoder
from fastokens_final_encode import install_fastokens_final_encode
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", help="Optional JSON array of real chat bodies")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=True
    )
    probe = "Hello world 你好"
    if tokenizer.decode(tokenizer.encode(probe), skip_special_tokens=True) != probe:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    bodies = [
        {"model": args.model, "messages": [{"role": "user", "content": probe}]},
    ]
    for enabled in (False, True):
        bodies.append(
            dict(bodies[0], thinking={"type": "enabled" if enabled else "disabled"})
        )
    tool = {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    bodies.append(dict(bodies[0], tools=[tool], tool_choice="auto"))
    bodies.append(
        {
            "model": args.model,
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"city":"Austin"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            ],
            "tools": [tool],
        }
    )
    bodies.append(dict(bodies[0], prompt_token_ids=[0, 1, 123]))
    if args.requests:
        with open(args.requests, encoding="utf-8") as handle:
            bodies.extend(json.load(handle))
    encoder = ExactAtomChatEncoder(args.model, tokenizer)
    references = [encoder.prepare_body(body) for body in bodies]
    texts = ["", probe, "é é 👩‍💻\n\t", "<think>reason</think>"]
    text_references = {
        (text, add): tokenizer.encode(text, add_special_tokens=add)
        for text in texts
        for add in (False, True)
    }
    metadata = install_fastokens_final_encode(tokenizer)
    encoder = ExactAtomChatEncoder(args.model, tokenizer)
    for index, (body, reference) in enumerate(zip(bodies, references)):
        if encoder.prepare_body(body) != reference:
            raise AssertionError(f"Exact body/token mismatch in case {index}")
    for (text, add), reference in text_references.items():
        if tokenizer.encode(text, add_special_tokens=add) != reference:
            raise AssertionError("Special-token or Unicode encode mismatch")
    for kwargs in (
        {"padding": True},
        {"truncation": True},
        {"max_length": 5},
        {"text_pair": "other"},
    ):
        try:
            tokenizer.encode("test", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Unsupported encode mode accepted: {kwargs}")
    print(
        json.dumps(
            {
                "chat_cases": len(bodies),
                "text_cases": len(text_references),
                "unsupported_modes_rejected": 4,
                "adapter": metadata,
            }
        )
    )


if __name__ == "__main__":
    main()
