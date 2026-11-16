"""MMStar against the deployment under test, run by lmms-eval as
.github/scripts/atom_mmstar_ci.sh runs it.

Passes when no request failed and, on a topology with a reference topology,
the score is within 2 x the reference's standard error of the reference's
score. Run by accept.sh in the client container; the helpers here are shared
with the other lmms-eval checks.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

# The text lmms-eval's OpenAI client answers with once a request has failed.
FAILED_MARK = "[LMMS_EVAL_REQUEST_FAILED"


def env(name):
    return os.environ[name]


def stderr(score, n):
    """Standard error of an accuracy over n questions."""
    return math.sqrt(score * (1 - score) / n)


# lmms-eval's OpenAI client builds its SDK client with the SDK's 600 s timeout
# and two silent retries, and sends no sampling parameter but the
# temperature. Run it with a client that waits as long as ATOMesh does, never
# retries, and adds the check's extra request fields.
LMMS_EVAL_MAIN = """
import json, sys
import openai
import lmms_eval.models.simple.openai as backend
extra = json.loads(sys.argv.pop(1))

def client(*args, **kwargs):
    sdk = openai.OpenAI(*args, timeout=14400, max_retries=0, **kwargs)
    create = sdk.chat.completions.create
    sdk.chat.completions.create = lambda **body: create(**extra, **body)
    return sdk

backend.OpenAI = client
from lmms_eval.__main__ import cli_evaluate
cli_evaluate()
"""


def run_lmms_eval(name, tasks, gen_kwargs, concurrency, include_path=None, extra=None):
    """Run lmms-eval on tasks against ENDPOINT, one attempt per request, with
    extra (a dict) added to every request body.

    Returns lmms-eval's results per task and, per task, its sample lines.
    """
    out = Path(env("RUN_DIR")) / name
    model_args = (
        f"model={env('SERVED_MODEL')},base_url={env('ENDPOINT')}/v1,api_key=EMPTY,"
        f"max_retries=1,num_concurrent={concurrency},max_size_in_mb=50"
    )
    args = [
        sys.executable,
        "-c",
        LMMS_EVAL_MAIN,
        json.dumps(extra or {}),
        "--model",
        "openai",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        "1",
        "--process_with_media",
        "--gen_kwargs",
        gen_kwargs,
        "--log_samples",
        "--output_path",
        str(out),
    ]
    if include_path:
        args += ["--include_path", str(include_path)]
    limit = os.environ.get("LIMIT")
    if limit:
        args += ["--limit", limit]
    with open(f"{out}.log", "w") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    results = json.loads(max(out.rglob("*_results.json")).read_text())["results"]
    samples = {
        task: max(out.rglob(f"*_samples_{task}.jsonl")).read_text().splitlines()
        for task in tasks
    }
    return results, samples


def failed_requests(lines):
    return sum(FAILED_MARK in line for line in lines)


def reference():
    """The reference topology's recorded result for this model and check."""
    path = Path(env("RESULTS_DIR"), env("ACCEPT_MODEL"), env("REF_TOPOLOGY"))
    path = path / f"{env('ACCEPT_CHECK')}.json"
    if not path.exists():
        raise RuntimeError(f"no reference at {path}")
    ref = json.loads(path.read_text())
    if not ref["passed"]:
        raise RuntimeError(f"reference {path} did not pass")
    return ref


def finish(result, failures):
    """Record the result, print its JSON line and PASS or FAIL, and exit."""
    result = {
        "model": env("ACCEPT_MODEL"),
        "topology": env("ACCEPT_TOPOLOGY"),
        "check": env("ACCEPT_CHECK"),
        "passed": not failures,
        "reason": "; ".join(failures),
        **result,
        "run_dir": env("RUN_DIR"),
    }
    line = json.dumps(result)
    Path(env("RUN_DIR"), "result.json").write_text(line + "\n")
    if not os.environ.get("LIMIT"):
        Path(env("RESULTS_DIR"), result["model"], result["topology"]).mkdir(
            parents=True, exist_ok=True
        )
        Path(
            env("RESULTS_DIR"),
            result["model"],
            result["topology"],
            f"{result['check']}.json",
        ).write_text(line + "\n")
    print(line)
    print("PASS" if not failures else f"FAIL {result['reason']}")
    sys.exit(0 if not failures else 1)


def run_check(check):
    """Run check(), reporting an exception as a FAIL."""
    try:
        check()
    except Exception as exc:  # noqa: BLE001 - any error is a FAIL
        traceback.print_exc()
        finish({}, [f"{type(exc).__name__}: {exc}"])


def mmstar_task_dir():
    """lmms-eval's MMStar task, renamed, without its HF token requirement
    (Lin-Chen/MMStar is public)."""
    import lmms_eval

    src = Path(lmms_eval.__file__).parent / "tasks" / "mmstar"
    dst = Path(env("RUN_DIR"), "tasks", "mmstar_notoken")
    dst.mkdir(parents=True)
    shutil.copy(src / "utils.py", dst)
    template = (src / "_default_template_yaml").read_text()
    (dst / "_default_template_yaml").write_text(
        template.replace("token: True", "token: False")
    )
    task = (src / "mmstar.yaml").read_text()
    (dst / "mmstar.yaml").write_text(
        task.replace('task: "mmstar"', 'task: "mmstar_notoken"')
    )
    return dst.parent


def main():
    task = "mmstar_notoken"
    ref = reference() if env("REF_TOPOLOGY") else None
    results, samples = run_lmms_eval(
        "mmstar",
        [task],
        "temperature=0,max_new_tokens=16384",
        32,
        mmstar_task_dir(),
    )
    n = len(samples[task])
    score = results[task]["average,none"]
    result = {
        "score": score,
        "stderr": stderr(score, n),
        "n": n,
        "failed_requests": failed_requests(samples[task]),
    }
    failures = []
    if result["failed_requests"]:
        failures.append(f"{result['failed_requests']} failed requests")
    if ref:
        tolerance = 2 * ref["stderr"]
        result["reference"] = {
            "topology": env("REF_TOPOLOGY"),
            "score": ref["score"],
            "stderr": ref["stderr"],
        }
        if abs(score - ref["score"]) > tolerance:
            failures.append(
                f"score {score:.4f} is not within {tolerance:.4f} of "
                f"{env('REF_TOPOLOGY')}'s {ref['score']:.4f}"
            )
    finish(result, failures)


if __name__ == "__main__":
    run_check(main)
