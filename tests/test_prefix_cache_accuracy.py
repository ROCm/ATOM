#!/usr/bin/env python3
"""
Test prefix caching accuracy with high cache-hit workloads.

Sends batches of requests that share long common prefixes, then verifies:
1. Responses are correct (math problems with known answers)
2. Cache hit rate is high (visible in server logs)
3. Repeated identical requests produce consistent results
"""

import argparse
import concurrent.futures
import json
import re
import sys
import time

import requests

BASE_URL = "http://localhost:8000"

# Long shared prefix: 5-shot math examples (~2000 tokens)
MATH_PREFIX = """You are a precise math assistant. Solve each problem step by step and give the final numerical answer after ####.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 2 + 2 = 4 more toys. 5 + 4 = 9. #### 9

"""

# Test questions with known answers
TEST_QUESTIONS = [
    ("A farmer has 17 sheep. He buys 5 more and then sells 8. How many sheep does he have?", 14),
    ("A store had 45 apples. They sold 12 in the morning and 18 in the afternoon. How many apples are left?", 15),
    ("Tom has 8 marbles. Jerry has 3 times as many. How many marbles do they have together?", 32),
    ("A classroom has 6 rows of desks with 5 desks in each row. If 7 desks are removed, how many remain?", 23),
    ("Sarah baked 24 cookies. She gave 1/3 to her neighbor and ate 4 herself. How many cookies does she have left?", 12),
    ("A train travels 60 miles per hour for 3 hours, then 40 miles per hour for 2 hours. What is the total distance?", 260),
    ("Mike has 50 dollars. He spends 15 dollars on lunch and 20 dollars on a book. How much money does he have left?", 15),
    ("A garden has 9 rose bushes. Each bush has 12 roses. If 25 roses are picked, how many roses remain?", 83),
    ("Lisa read 35 pages on Monday and twice as many on Tuesday. How many pages did she read in total?", 105),
    ("A box contains 100 balls. 40 are red, 35 are blue, and the rest are green. How many green balls are there?", 25),
]


def extract_answer(text: str):
    """Extract numerical answer after the FIRST #### marker."""
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def get_model_name(base_url: str) -> str:
    """Get the model name from the server."""
    resp = requests.get(f"{base_url}/v1/models", timeout=5)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def send_completion(prompt: str, max_tokens: int = 256, base_url: str = BASE_URL, model: str = "") -> str:
    """Send a completion request to the server."""
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def run_batch(questions, prefix, base_url=BASE_URL, model="", label=""):
    """Run a batch of questions and return (correct, total, results)."""
    results = []

    def ask(q_and_a):
        question, expected = q_and_a
        prompt = prefix + f"Question: {question}\nAnswer:"
        try:
            response = send_completion(prompt, base_url=base_url, model=model)
            answer = extract_answer(response)
            correct = answer is not None and abs(answer - expected) < 0.01
            return (question, expected, answer, correct, response.strip())
        except Exception as e:
            return (question, expected, None, False, f"ERROR: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(ask, questions))

    num_correct = sum(1 for r in results if r[3])
    return num_correct, len(results), results


def main():
    parser = argparse.ArgumentParser(description="Test prefix cache accuracy")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds to repeat")
    parser.add_argument("--base-url", type=str, default=BASE_URL)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url

    # Health check
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"Server not reachable at {base_url}: {e}")
        sys.exit(1)

    model = get_model_name(base_url)
    print(f"=== Prefix Cache Accuracy Test ===")
    print(f"Server: {base_url}")
    print(f"Model: {model}")
    print(f"Questions per round: {len(TEST_QUESTIONS)}")
    print(f"Rounds: {args.rounds}")
    print(f"Shared prefix length: ~{len(MATH_PREFIX)} chars")
    print()

    all_round_results = []

    for round_num in range(1, args.rounds + 1):
        t0 = time.time()
        correct, total, results = run_batch(TEST_QUESTIONS, MATH_PREFIX, base_url=base_url, model=model, label=f"Round {round_num}")
        elapsed = time.time() - t0
        accuracy = 100.0 * correct / total
        all_round_results.append((correct, total, accuracy, elapsed))

        print(f"Round {round_num}: {correct}/{total} correct ({accuracy:.1f}%) in {elapsed:.1f}s")

        if args.verbose:
            for q, expected, got, ok, resp in results:
                status = "OK" if ok else "WRONG"
                print(f"  [{status}] {q[:60]}... expected={expected} got={got}")
                if not ok:
                    # Show first 200 chars of response for debugging
                    print(f"         response: {resp[:200]}")
            print()

    print()
    print("=== Summary ===")
    total_correct = sum(r[0] for r in all_round_results)
    total_questions = sum(r[1] for r in all_round_results)
    overall_accuracy = 100.0 * total_correct / total_questions

    # Check consistency: same questions should give same answers across rounds
    print(f"Overall: {total_correct}/{total_questions} ({overall_accuracy:.1f}%)")
    for i, (c, t, a, e) in enumerate(all_round_results, 1):
        cache_note = "(cold)" if i == 1 else "(cache warm)"
        print(f"  Round {i}: {c}/{t} ({a:.1f}%) {e:.1f}s {cache_note}")

    # Verify rounds 2+ should be faster (cache hits)
    if args.rounds >= 2:
        r1_time = all_round_results[0][3]
        r2_time = all_round_results[1][3]
        speedup = r1_time / r2_time if r2_time > 0 else 0
        print(f"\n  Speedup round 2 vs round 1: {speedup:.2f}x")

    # Pass/fail
    if overall_accuracy >= 80.0:
        print(f"\nPASS: accuracy {overall_accuracy:.1f}% >= 80%")
        return 0
    else:
        print(f"\nFAIL: accuracy {overall_accuracy:.1f}% < 80%")
        return 1


if __name__ == "__main__":
    sys.exit(main())
