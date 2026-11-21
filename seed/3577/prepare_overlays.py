"""Prepare two reviewable patches from exact Git blobs; never modifies a checkout."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

repo = Path(sys.argv[1]).resolve()
output = Path(__file__).resolve().parent
arms = {"A": "fb1304722a9f79cd16e7383b8b504ccb29851d77",
        "B": "8e354a56986bb6342139b657bb9e45e460171385"}


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new)


def overlay(path, text):
    if path == "atom/model_engine/model_runner.py":
        text = replace_once(text, "            self.profiler.__enter__()\n",
            '            self.profiler.__enter__()\n'
            '            if os.environ.get("ATOM_3577_OBSERVER_DIR"):\n'
            '                from pp3577_observer import begin\n'
            '                begin(self)\n')
        text = replace_once(text,
            '        t0 = time.monotonic()\n        logger.info("Rank %d: stopping profiler...", self.rank)\n',
            '        if os.environ.get("ATOM_3577_OBSERVER_DIR"):\n'
            '            from pp3577_observer import end\n'
            '            end(self)\n'
            '        t0 = time.monotonic()\n        logger.info("Rank %d: stopping profiler...", self.rank)\n')
        text = replace_once(text,
            '            logger.exception("Rank %d: profiler stop failed", self.rank)\n',
            '            logger.exception("Rank %d: profiler stop failed", self.rank)\n            raise\n')
        return replace_once(text,
            '                    for p in (tmp_json_path, output_path):\n'
            '                        if os.path.exists(p):\n'
            '                            os.remove(p)\n',
            '                    for p in (tmp_json_path, output_path):\n'
            '                        if os.path.exists(p):\n'
            '                            os.remove(p)\n'
            '                    raise\n')
    if path == ".github/scripts/atom_test.sh":
        return replace_once(text,
            'if curl -sf "http://localhost:${ATOM_SERVER_PORT}/health"',
            'if curl -sf --connect-timeout 2 --max-time 5 "http://localhost:${ATOM_SERVER_PORT}/health"')
    if path == "atom/benchmarks/benchmark_serving.py":
        text = replace_once(text, "    goodput_config_dict = check_goodput_args(args)\n",
            '    # Validation receipts are written before benchmark timing starts.\n'
            '    import hashlib\n'
            '    from pathlib import Path\n'
            '    receipt = Path(os.environ["ATOM_3577_PROMPTS"])\n'
            '    with receipt.open("x") as stream:\n'
            '        for index, (prompt, length, output_length, media) in enumerate(input_requests):\n'
            '            assert media is None\n'
            '            ids = tokenizer.encode(prompt, add_special_tokens=False)\n'
            '            assert len(ids) == length\n'
            '            stream.write(json.dumps(dict(index=index, prompt=prompt, token_ids=ids,\n'
            '                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),\n'
            '                input_tokens=length, output_tokens=output_length)) + "\\n")\n'
            '    goodput_config_dict = check_goodput_args(args)\n')
        return replace_once(text,
            '        "input_lens": [output.prompt_len for output in outputs],\n',
            '        "input_lens": [output.prompt_len for output in outputs],\n'
            '        "p3577_cached_tokens": [output.cached_tokens for output in outputs],\n')
    if path == "atom/benchmarks/backend_request_func.py":
        text = replace_once(text, '    full_response_duration_s: float | None = None\n',
                            '    full_response_duration_s: float | None = None\n'
                            '    cached_tokens: int | None = None\n')
        first = text.index("async def async_request_openai_completions(")
        last = text.index("\nasync def ", first + 1)
        function = text[first:last]
        function = replace_once(function,
            '                                output.output_tokens = usage.get("completion_tokens")\n',
            '                                output.output_tokens = usage.get("completion_tokens")\n'
            '                                output.prompt_len = usage["prompt_tokens"]\n'
            '                                output.cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]\n')
        function = replace_once(function,
            '                    output.error = response.reason or ""\n',
            '                    output.error = f"HTTP {response.status}: {await response.text()}"\n')
        return text[:first] + function + text[last:]
    if path == "atom/entrypoints/openai/serving_completion.py":
        first = text.index("async def stream_completion_response(")
        last = text.index("\ndef build_completion_response(", first)
        function = text[first:last]
        function = replace_once(function, "    num_tokens_output = 0\n",
                                "    num_tokens_output = 0\n    num_cached_tokens = None\n")
        function = replace_once(function,
            '            chunk_data = await stream_collector.get()\n',
            '            chunk_data = await stream_collector.get()\n'
            '            if "num_cached_tokens" in chunk_data:\n'
            '                num_cached_tokens = max(num_cached_tokens or 0, chunk_data["num_cached_tokens"])\n')
        function = replace_once(function,
            '                usage_chunk = {\n',
            '                assert num_cached_tokens is not None, "missing cache receipt"\n'
            '                usage_chunk = {\n')
        function = replace_once(function,
            '                        "prompt_tokens": num_tokens_input,\n',
            '                        "prompt_tokens": num_tokens_input,\n'
            '                        "prompt_tokens_details": {"cached_tokens": num_cached_tokens},\n')
        return text[:first] + function + text[last:]
    raise AssertionError(path)


paths = [".github/scripts/atom_test.sh", "atom/model_engine/model_runner.py",
         "atom/benchmarks/benchmark_serving.py", "atom/benchmarks/backend_request_func.py",
         "atom/entrypoints/openai/serving_completion.py"]
manifest = {}
for arm, sha in arms.items():
    patch = ""
    entries = []
    for path in paths:
        raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{sha}:{path}"])
        original = raw.decode()
        changed = overlay(path, original)
        patch += "".join(difflib.unified_diff(original.splitlines(keepends=True),
                         changed.splitlines(keepends=True), fromfile="a/" + path, tofile="b/" + path))
        entries.append(dict(path=path, base_sha256=hashlib.sha256(raw).hexdigest(),
                            overlay_sha256=hashlib.sha256(changed.encode()).hexdigest()))
    data = patch.encode()
    (output / f"{arm}-validation.patch").write_bytes(data)
    manifest[arm] = dict(source=sha, patch_sha256=hashlib.sha256(data).hexdigest(), files=entries)
(output / "source-overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
