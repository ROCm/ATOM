#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the M3 GSM8K launcher and real API client against a local fake server.

No model, dataset download or GPU is needed. This validates client plumbing,
not model accuracy; the real GSM8K gate still runs on the benchmark node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


def launcher_args(cell: dict, directory: Path, port: int) -> list[str]:
    source = (Path(__file__).parent / "pd_server_atom.sh").read_text()
    run_eval = source[
        source.index("run_eval() {") : source.index("run_benchmark_and_eval() {")
    ]
    accuracy = cell["accuracy"]
    env = dict(os.environ)
    env.update(
        RUN_EVAL="true",
        EVAL_TASK=accuracy["task"],
        EVAL_MODEL_TYPE=accuracy["model_type"],
        EVAL_ENDPOINT=accuracy["endpoint"],
        EVAL_CONCURRENCY=str(accuracy["concurrency"][0]),
        MODEL_PATH=cell["model_path"],
        ROUTER_PORT=str(port),
        TOPOLOGY=cell["topology"],
        RUN_DIR=str(directory),
        PREFLIGHT_ARGV=str(directory / "argv"),
    )
    for field, variable in (
        ("fewshot", "EVAL_FEWSHOT"),
        ("limit", "EVAL_LIMIT"),
        ("batch_size", "EVAL_BATCH_SIZE"),
        ("max_gen_toks", "EVAL_MAX_GEN_TOKS"),
        ("apply_chat_template", "EVAL_APPLY_CHAT_TEMPLATE"),
        ("fewshot_as_multiturn", "EVAL_FEWSHOT_AS_MULTITURN"),
        ("threshold", "EVAL_THRESHOLD"),
    ):
        value = accuracy.get(field)
        env[variable] = "" if value is None else str(value).lower()
    # Stop at the actual lm_eval invocation, before run_eval reads score files.
    script = (
        "set -euo pipefail\n"
        + run_eval
        + """
lm_eval() { printf '%s\\0' "$@" > "$PREFLIGHT_ARGV"; return 97; }
run_eval
"""
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 97:
        raise RuntimeError(
            f"Cannot capture evaluation command: {result.stdout}{result.stderr}"
        )
    return (directory / "argv").read_bytes().decode().rstrip("\0").split("\0")


def check_cell(cell: dict) -> None:
    from importlib.metadata import version

    import yaml
    from lm_eval import __file__ as lm_eval_file
    from lm_eval._cli import HarnessCLI
    from lm_eval.api.instance import Instance
    from lm_eval.config.evaluate_config import EvaluatorConfig
    from lm_eval.models.openai_completions import LocalCompletionsAPI
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    received = []

    def answer(prompt):
        return hashlib.sha256(prompt.encode()).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, payload))
            prompts = payload["prompt"]
            if isinstance(prompts, str):
                prompts = [prompts]
            response = json.dumps(
                {
                    "choices": [
                        {"index": i, "text": answer(p)} for i, p in enumerate(prompts)
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    with tempfile.TemporaryDirectory(prefix="m3-eval-client-") as temp:
        directory = Path(temp)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            argv = launcher_args(cell, directory, server.server_port)
            with patch.object(sys, "argv", ["lm_eval", *argv]):
                config = EvaluatorConfig.from_cli(HarnessCLI().parse_args())
            assert config.model == "local-completions"
            assert config.num_fewshot == cell["accuracy"]["fewshot"]
            assert config.limit == cell["accuracy"]["limit"]
            assert not config.apply_chat_template and not config.fewshot_as_multiturn
            assert config.model_args["tokenized_requests"] is False

            # A local tokenizer exercises real HF initialization without model files.
            tokenizer = Tokenizer(WordLevel({"<unk>": 0, "</s>": 1}, unk_token="<unk>"))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, unk_token="<unk>", eos_token="</s>"
            ).save_pretrained(directory / "tokenizer")
            client = LocalCompletionsAPI(
                **config.model_args, tokenizer=str(directory / "tokenizer"), timeout=10
            )
            task = yaml.safe_load(
                (Path(lm_eval_file).parent / "tasks/gsm8k/gsm8k.yaml").read_text()
            )
            # More than one concurrent wave, and prompts longer than the client's
            # default 2048-token context; raw completions must retain every byte.
            prompts = [f"case {i}: " + "word " * (3000 + i) for i in range(33)]
            requests = [
                Instance(
                    request_type="generate_until",
                    doc={},
                    arguments=(prompt, task["generation_kwargs"]),
                    idx=i,
                )
                for i, prompt in enumerate(prompts)
            ]
            results = client.generate_until(requests, disable_tqdm=True)
            assert results == [answer(p) for p in prompts], (
                "truncated or misordered output"
            )
            sent_prompts = []
            for path, payload in received:
                assert path == "/v1/completions"
                assert payload["max_tokens"] == cell["accuracy"]["max_gen_toks"]
                assert payload["temperature"] == 0
                assert "Question:" in payload["stop"]
                assert payload["model"] == cell["model_path"]
                prompt = payload["prompt"]
                sent_prompts.extend([prompt] if isinstance(prompt, str) else prompt)
            assert sorted(sent_prompts) == sorted(prompts)
            print(
                f"[client-preflight] {cell['name']}: lm-eval {version('lm-eval')}, "
                f"33 loopback requests passed; full prompt, output ordering and "
                f"max_tokens={cell['accuracy']['max_gen_toks']} verified. "
                "No model accuracy was measured."
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    args = parser.parse_args()
    cells = json.loads(args.matrix.read_text())["include"]
    selected = [
        cell
        for cell in cells
        if cell["name"].startswith("minimax-m3-fp4-1p1d-cpp4-") and cell["run_eval"]
    ]
    if not selected:
        print("[client-preflight] No enabled M3 PP4 evaluation; skipping.")
        return
    for cell in selected:
        check_cell(cell)


if __name__ == "__main__":
    main()
