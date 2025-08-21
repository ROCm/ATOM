import os
from atom import LLMEngine, SamplingParams
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config of test",
)

parser.add_argument(
    "--kv_cache_dtype",
    choices=["bf16", "fp8"],
    type=str,
    default="bf16",
    help="""KV cache type. Default is 'bf16'.
    e.g.: -kv_cache_dtype fp8""",
)

parser.add_argument(
    "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path."
)


def main():
    args = parser.parse_args()
    model_name_or_path = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    llm = LLMEngine(
        model_name_or_path,
        enforce_eager=False,
        tensor_parallel_size=1,
        kv_cache_dtype=args.kv_cache_dtype,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "1+2+3=?",
        # "2+3+4=?",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    print("This is prompts:", prompts)
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
