# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import requests
import time
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer


def start_profiling(base_url: str):
    """Start profiling on the server"""
    print("Starting profiler...")
    response = requests.post(f"{base_url}/start_profile")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def stop_profiling(base_url: str):
    """Stop profiling and generate trace files"""
    print("Stopping profiler...")
    response = requests.post(f"{base_url}/stop_profile")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def send_completion_request(base_url: str, prompt: str, max_tokens: int):
    """Send a completion request to the server"""
    print(f"Sending completion request with {len(prompt.split())} words, max_tokens={max_tokens}")
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": "test",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
    )
    if response.status_code == 200:
        return response.json()
    return None


def send_chat_request(base_url: str, message: str, max_tokens: int):
    """Send a chat completion request to the server"""
    print(f"Sending chat request with {len(message.split())} words, max_tokens={max_tokens}")
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": message}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
    )
    return response.status_code == 200


def generate_random_prompt(tokenizer: AutoTokenizer, input_length: int) -> str:
    """Generate random prompt with exact token length"""
    vocab_size = tokenizer.vocab_size
    random_token_ids = [random.randint(0, vocab_size - 1) for _ in range(input_length)]
    prompt = tokenizer.decode(random_token_ids, skip_special_tokens=True)
    # Re-encode to verify and truncate to exact length
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)[:input_length]
    prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Profile online inference requests")
    parser.add_argument(
        "--input-length",
        type=int,
        default=128,
        help="Input prompt length in tokens (used with --random-input)",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=32,
        help="Number of output tokens to generate (max_tokens) (default: 32)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--random-input",
        action="store_true",
        help="Use random generated input. Otherwise use a predefined prompt.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name or path for tokenizer (required when using --random-input)",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=1,
        help="Batch size (number of concurrent requests to send)",
    )
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print("=" * 60)
    print("Atom OpenAI Server Profiling Example")
    print("=" * 60)
    
    # Prepare input prompt
    if args.random_input:
        if not args.model:
            print("Error: --model is required when using --random-input")
            return
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        input_prompt = generate_random_prompt(tokenizer, args.input_length)
        print(f"Using randomly generated input with length: {args.input_length} tokens")
    else:
        input_prompt = "hello, who are you?"
        print(f"Using predefined prompt: {input_prompt}")
    
    # Start profiling
    if not start_profiling(base_url):
        print("Failed to start profiling!")
        return

    time.sleep(2)
    
    # Send concurrent requests based on batch size
    print(f"\nSending {args.bs} concurrent request(s)...")
    results = []
    
    if args.bs == 1:
        # Single request
        result = send_completion_request(base_url, input_prompt, args.output_length)
        if result is None:
            print("Warning: Request failed or returned no result")
        results = [result]
    else:
        # Concurrent requests using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.bs) as executor:
            # Submit all requests and track their indices
            future_to_index = {
                executor.submit(send_completion_request, base_url, input_prompt, args.output_length): i + 1
                for i in range(args.bs)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                if result is None:
                    print(f"Warning: Request {index}/{args.bs} failed or returned no result")
                else:
                    print(f"Request {index}/{args.bs} completed")
                results.append(result)
    
    time.sleep(2)
    
    # Stop profiling
    print("\n" + "=" * 60)
    if not stop_profiling(base_url):
        print("Failed to stop profiling!")
        return
    
    # Print output results for non-random input
    if not args.random_input and results:
        print("\n" + "=" * 60)
        print("Generated Output:")
        print("=" * 60)
        for i, result in enumerate(results):
            if result and "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["text"]
                if args.bs > 1:
                    print(f"Output [{i+1}/{args.bs}]: {generated_text}\n")
                else:
                    print(f"Output: {generated_text}\n")
        print("=" * 60)


if __name__ == "__main__":
    main()

