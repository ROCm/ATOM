#! /usr/bin/env python3

import json
import datetime
import sys
from pathlib import Path

def main():
    # Usage: python .github/scripts/summarize.py input.json
    if len(sys.argv) < 2:
        print("Usage: python .github/scripts/summarize.py <result_dir>", file=sys.stderr)
        sys.exit(1)

    summary_header = f'''\
| Date | Backend | Model | Best of | Number of prompts | Request rate | Burstiness | Max concurrency | Duration | Completed | Total input tokens | Total output tokens | Request throughput | Request goodput | Output throughput | Total token throughput | Mean TTFT (ms) | Median TTFT (ms) | Std TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | Std TPOT (ms) | P99 TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | Std ITL (ms) | P99 ITL (ms) | Mean E2EL (ms) | Median E2EL (ms) | Std E2EL (ms) | P99 E2EL (ms) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |\
    '''
    print(summary_header)

    in_path = Path(sys.argv[1])

    files = sorted(in_path.glob("*.json"))
    for json_path in files:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        row = [
            datetime.datetime.strptime(data.get("date", ""), "%Y%m%d-%H%M%S").strftime("%Y-%m-%d %H:%M:%S"),
            "ATOM",
            data.get("model_id", "").split("/")[-1],
            data.get("best_of", ""),
            data.get("num_prompts", ""),
            data.get("request_rate", ""),
            f"{data.get('burstiness', '')}%",
            f"{data.get('max_concurrency', '')}",
            f"{data.get('duration', ''):.2f}",
            data.get("completed", ""),
            f"{data.get('total_input_tokens', ''):.2f}",
            f"{data.get('total_output_tokens', ''):.2f}",
            f"{data.get('request_throughput', ''):.2f}",
            data.get('request_goodput', ''),
            f"{data.get('output_throughput', ''):.2f}",
            f"{data.get('total_token_throughput', ''):.2f}",
            f"{data.get('mean_ttft_ms', ''):.2f}",
            f"{data.get('median_ttft_ms', ''):.2f}",
            f"{data.get('std_ttft_ms', ''):.2f}",
            f"{data.get('p99_ttft_ms', ''):.2f}",
            f"{data.get('mean_tpot_ms', ''):.2f}",
            f"{data.get('median_tpot_ms', ''):.2f}",
            f"{data.get('std_tpot_ms', ''):.2f}",
            f"{data.get('p99_tpot_ms', ''):.2f}",
            f"{data.get('mean_itl_ms', ''):.2f}",
            f"{data.get('median_itl_ms', ''):.2f}",
            f"{data.get('std_itl_ms', ''):.2f}",
            f"{data.get('p99_itl_ms', ''):.2f}",
            f"{data.get('mean_e2el_ms', ''):.2f}",
            f"{data.get('median_e2el_ms', ''):.2f}",
            f"{data.get('std_e2el_ms', ''):.2f}",
            f"{data.get('p99_e2el_ms', ''):.2f}",
        ]
        print("| " + " | ".join(str(x) for x in row) + " |")

if __name__ == "__main__":
    main()