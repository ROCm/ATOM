# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import aiohttp
import time
from typing import List
import random
from transformers import AutoTokenizer


async def send_chat_request(session: aiohttp.ClientSession, url: str, message: str, request_id: int, max_tokens: int = 100):
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "user", "content": message}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "ignore_eos": True
    }
    
    start_time = time.time()
    try:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            latency = time.time() - start_time
            
            if response.status == 200:
                content = result["choices"][0]["message"]["content"]
                usage = result["usage"]
                ttft = usage.get("ttft_s", 0.0)
                tpot = usage.get("tpot_s", 0.0)
                print(f"✓ Request {request_id} completed in {latency:.2f}s")
                print(f"  Prompt: {message[:50]}...")
                print(f"  Response: {content[:100]}...")
                print(f"  Tokens: {usage['prompt_tokens']} input, {usage['completion_tokens']} output")
                print(f"  TTFT: {ttft:.3f}s, TPOT: {tpot:.3f}s")
                return {
                    "success": True, 
                    "latency": latency, 
                    "request_id": request_id,
                    "ttft": ttft,
                    "tpot": tpot,
                    "num_tokens": usage['completion_tokens']
                }
            else:
                print(f"✗ Request {request_id} failed: {response.status}")
                return {"success": False, "latency": latency, "request_id": request_id}
                
    except Exception as e:
        latency = time.time() - start_time
        print(f"✗ Request {request_id} error: {e}")
        return {"success": False, "latency": latency, "request_id": request_id, "error": str(e)}


async def send_completion_request(session: aiohttp.ClientSession, url: str, prompt: str, request_id: int, max_tokens: int = 100):
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "ignore_eos": True
    }
    
    start_time = time.time()
    try:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            latency = time.time() - start_time
            
            if response.status == 200:
                text = result["choices"][0]["text"]
                usage = result["usage"]
                ttft = usage.get("ttft_s", 0.0)
                tpot = usage.get("tpot_s", 0.0)
                print(f"✓ Request {request_id} completed in {latency:.2f}s")
                print(f"  Prompt: {prompt[:50]}...")
                print(f"  Response: {text[:100]}...")
                print(f"  Tokens: {usage['prompt_tokens']} input, {usage['completion_tokens']} output")
                print(f"  TTFT: {ttft:.3f}s, TPOT: {tpot:.3f}s")
                return {
                    "success": True, 
                    "latency": latency, 
                    "request_id": request_id,
                    "ttft": ttft,
                    "tpot": tpot,
                    "num_tokens": usage['completion_tokens']
                }
            else:
                print(f"✗ Request {request_id} failed: {response.status}")
                return {"success": False, "latency": latency, "request_id": request_id}
                
    except Exception as e:
        latency = time.time() - start_time
        print(f"✗ Request {request_id} error: {e}")
        return {"success": False, "latency": latency, "request_id": request_id, "error": str(e)}


def generate_random_prompt(tokenizer: AutoTokenizer, input_length: int) -> str:
    """Generate random prompt with exact token length"""
    vocab_size = tokenizer.vocab_size
    random_token_ids = [random.randint(0, vocab_size - 1) for _ in range(input_length)]
    prompt = tokenizer.decode(random_token_ids, skip_special_tokens=True)
    # Re-encode to verify and truncate to exact length
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)[:input_length]
    prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
    return prompt


async def test_concurrent_requests(base_url: str, tokenizer: AutoTokenizer, num_requests: int = 60, 
                                   request_rate: float = 2.0, concurrency: int = 4, use_chat: bool = True, 
                                   input_tokens: int = 20, output_tokens: int = 100):
    estimated_duration = num_requests / request_rate
    
    print(f"\n{'='*60}")
    print(f"Starting Load Test")
    print(f"Server URL: {base_url}")
    print(f"Request Type: {'Chat Completion' if use_chat else 'Text Completion'}")
    print(f"Total Requests: {num_requests}")
    print(f"Request Rate: {request_rate} req/s")
    print(f"Max Concurrency: {concurrency}")
    print(f"Estimated Duration: ~{estimated_duration:.1f}s")
    print(f"Input Tokens: ~{input_tokens} tokens")
    print(f"Output Tokens: {output_tokens} tokens")
    print(f"{'='*60}\n")
    
    request_interval = 1.0 / request_rate
    
    results = []
    active_tasks = set()
    sent_count = 0
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        url = f"{base_url}/v1/chat/completions" if use_chat else f"{base_url}/v1/completions"
        
        while sent_count < num_requests or active_tasks:
            if sent_count < num_requests and len(active_tasks) < concurrency:
                sent_count += 1
                prompt = generate_random_prompt(tokenizer, input_tokens)
                
                if use_chat:
                    task = asyncio.create_task(
                        send_chat_request(session, url, prompt, sent_count, output_tokens)
                    )
                else:
                    task = asyncio.create_task(
                        send_completion_request(session, url, prompt, sent_count, output_tokens)
                    )
                
                active_tasks.add(task)
                task.add_done_callback(lambda t: active_tasks.discard(t))
                
                if sent_count < num_requests:
                    next_request_time = start_time + sent_count * request_interval
                    sleep_time = max(0, next_request_time - time.time())
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            else:
                if active_tasks:
                    done, active_tasks = await asyncio.wait(
                        active_tasks, 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    results.extend([task.result() for task in done])
    
    total_time = time.time() - start_time
    
    total_requests = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total_requests - successful
    
    # Calculate statistics for successful requests
    successful_results = [r for r in results if r["success"]]
    
    if successful_results:
        avg_latency = sum(r["latency"] for r in results) / total_requests
        min_latency = min(r["latency"] for r in results)
        max_latency = max(r["latency"] for r in results)
        
        latencies = sorted([r["latency"] for r in results])
        p50 = latencies[int(len(latencies) * 0.50)]
        p90 = latencies[int(len(latencies) * 0.90)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        
        # Calculate TTFT and TPOT statistics
        ttfts = [r.get("ttft", 0) for r in successful_results if r.get("ttft", 0) > 0]
        tpots = [r.get("tpot", 0) for r in successful_results if r.get("tpot", 0) > 0]
        
        if ttfts:
            avg_ttft = sum(ttfts) / len(ttfts)
            min_ttft = min(ttfts)
            max_ttft = max(ttfts)
            ttfts_sorted = sorted(ttfts)
            p50_ttft = ttfts_sorted[int(len(ttfts_sorted) * 0.50)]
            p90_ttft = ttfts_sorted[int(len(ttfts_sorted) * 0.90)]
            p95_ttft = ttfts_sorted[int(len(ttfts_sorted) * 0.95)]
            p99_ttft = ttfts_sorted[int(len(ttfts_sorted) * 0.99)]
        else:
            avg_ttft = min_ttft = max_ttft = 0
            p50_ttft = p90_ttft = p95_ttft = p99_ttft = 0
        
        if tpots:
            avg_tpot = sum(tpots) / len(tpots)
            min_tpot = min(tpots)
            max_tpot = max(tpots)
            tpots_sorted = sorted(tpots)
            p50_tpot = tpots_sorted[int(len(tpots_sorted) * 0.50)]
            p90_tpot = tpots_sorted[int(len(tpots_sorted) * 0.90)]
            p95_tpot = tpots_sorted[int(len(tpots_sorted) * 0.95)]
            p99_tpot = tpots_sorted[int(len(tpots_sorted) * 0.99)]
        else:
            avg_tpot = min_tpot = max_tpot = 0
            p50_tpot = p90_tpot = p95_tpot = p99_tpot = 0
    else:
        avg_latency = min_latency = max_latency = 0
        p50 = p90 = p95 = p99 = 0
        avg_ttft = min_ttft = max_ttft = 0
        p50_ttft = p90_ttft = p95_ttft = p99_ttft = 0
        avg_tpot = min_tpot = max_tpot = 0
        p50_tpot = p90_tpot = p95_tpot = p99_tpot = 0
    
    print(f"\n{'='*60}")
    print(f"Test Completed!")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Requests: {total_requests}")
    print(f"Successful: {successful}/{total_requests} ({successful*100/max(total_requests,1):.1f}%)")
    print(f"Failed: {failed}/{total_requests} ({failed*100/max(total_requests,1):.1f}%)")
    print(f"\nLatency Statistics:")
    print(f"  Average: {avg_latency:.2f}s")
    print(f"  Min: {min_latency:.2f}s")
    print(f"  Max: {max_latency:.2f}s")
    print(f"  P50: {p50:.2f}s")
    print(f"  P90: {p90:.2f}s")
    print(f"  P95: {p95:.2f}s")
    print(f"  P99: {p99:.2f}s")
    print(f"\nTTFT (Time To First Token) Statistics:")
    print(f"  Average: {avg_ttft:.3f}s")
    print(f"  Min: {min_ttft:.3f}s")
    print(f"  Max: {max_ttft:.3f}s")
    print(f"  P50: {p50_ttft:.3f}s")
    print(f"  P90: {p90_ttft:.3f}s")
    print(f"  P95: {p95_ttft:.3f}s")
    print(f"  P99: {p99_ttft:.3f}s")
    print(f"\nTPOT (Time Per Output Token) Statistics:")
    print(f"  Average: {avg_tpot:.3f}s")
    print(f"  Min: {min_tpot:.3f}s")
    print(f"  Max: {max_tpot:.3f}s")
    print(f"  P50: {p50_tpot:.3f}s")
    print(f"  P90: {p90_tpot:.3f}s")
    print(f"  P95: {p95_tpot:.3f}s")
    print(f"  P99: {p99_tpot:.3f}s")
    print(f"\nThroughput:")
    print(f"  Actual Request Rate: {total_requests/total_time:.2f} req/s")
    print(f"  Token Throughput: {total_requests*(input_tokens+output_tokens)/total_time:.2f} tokens/s")
    print(f"{'='*60}\n")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI API Server Load Testing")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("--model", type=str, required=True, help="Model name or path for tokenizer")
    parser.add_argument("--num-requests", "-n", type=int, default=60, help="Total number of requests")
    parser.add_argument("--request-rate", "-r", type=float, default=2.0, help="Request rate (requests/second)")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="Max concurrent requests")
    parser.add_argument("--type", choices=["chat", "completion"], default="chat", help="Request type")
    parser.add_argument("--input-tokens", "-i", type=int, default=1024, help="Input token count")
    parser.add_argument("--output-tokens", "-o", type=int, default=1024, help="Output token count (max_tokens)")
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    use_chat = args.type == "chat"
    await test_concurrent_requests(
        base_url=args.url,
        tokenizer=tokenizer,
        num_requests=args.num_requests,
        request_rate=args.request_rate,
        concurrency=args.concurrency,
        use_chat=use_chat,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens
    )


if __name__ == "__main__":
    asyncio.run(main())
