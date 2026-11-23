"""MMStar against the deployment under test, run by lmms-eval as
.github/scripts/atom_mmstar_ci.sh runs it.

Passes when no request failed, the score is at least the model profile's
MMSTAR_THRESHOLD (as .github/scripts/atom_mmstar_ci.sh gates it), and, on a
topology with a reference topology, the score is within 2 x the reference's
standard error of the reference's score. The floor keeps a broken direct run
from passing and becoming the other topologies' reference; a LIMIT run, which
is never recorded as a reference and scores only MMStar's first questions,
skips it. Run by accept.sh in the client container.
"""

import os
import shutil
from pathlib import Path

from _common import (
    env,
    failed_requests,
    finish,
    reference,
    run_check,
    run_lmms_eval,
    stderr,
    within,
)


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
    threshold = float(env("MMSTAR_THRESHOLD"))
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
        "threshold": threshold,
    }
    failures = []
    if result["failed_requests"]:
        failures.append(f"{result['failed_requests']} failed requests")
    if score < threshold and not os.environ.get("LIMIT"):
        failures.append(f"score {score:.4f} is below the threshold {threshold:.4f}")
    if ref:
        tolerance = 2 * ref["stderr"]
        result["reference"] = {
            "topology": env("REF_TOPOLOGY"),
            "score": ref["score"],
            "stderr": ref["stderr"],
        }
        if not within(score, ref["score"], ref["stderr"]):
            failures.append(
                f"score {score:.4f} is not within {tolerance:.4f} of "
                f"{env('REF_TOPOLOGY')}'s {ref['score']:.4f}"
            )
    finish(result, failures)


if __name__ == "__main__":
    run_check(main)
