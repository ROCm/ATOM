"""MMMU-Pro standard (10 options) and vision against the deployment under
test, with the recipe's sampling (temperature 1.0, top-p 0.95, up to 65,536
new tokens), by lmms-eval.

lmms-eval's vision task sends the image with an empty text; this check sends
MMMU-Pro's own vision prompt (direct setting) instead, which asks for a last
line "Answer: X". A vision answer is the letter of the last such line, or,
when there is none, the option letter lmms-eval's multiple-choice extractor
finds in the response; a standard answer is the extractor's letter. lmms-eval's
own vision scorer compares the whole response to the letter.
Passes when no request failed and each subset's score is within 2 x the
reference's standard error of the reference's score. On a topology with a
reference topology the reference is that topology's recorded run; otherwise
(direct) it is the model profile's recipe score (MMMU_PRO_REF_STANDARD,
MMMU_PRO_REF_VISION), with the standard error of that score over this run's
questions.
"""

import json
import re
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

SUBSETS = {
    "standard": "mmmu_pro_standard",
    "vision": "mmmu_pro_vision_prompt",
}
LETTERS = list("ABCDEFGHIJ")

# MMMU-Pro's vision prompt, direct setting (MMMU-Benchmark/MMMU,
# mmmu-pro/prompts.yaml).
VISION_PROMPT = (
    "Answer with the option letter from the given choices directly. The last "
    "line of your response should be of the following format: 'Answer: $LETTER' "
    "(without quotes) where LETTER is one of options."
)
# A whole line "Answer: X", markdown emphasis and parentheses allowed, so that
# "Answer: I think it is (C)" is not read as option I.
ANSWER_LINE = re.compile(r"^\W*Answer:\s*\(?([A-J])\)?\W*$", re.MULTILINE)


def vision_task_dir():
    """lmms-eval's mmmu_pro_vision, renamed, with MMMU-Pro's vision prompt as
    its text."""
    import lmms_eval

    src = Path(lmms_eval.__file__).parent / "tasks" / "mmmu_pro"
    dst = Path(env("RUN_DIR"), "tasks", "mmmu_pro_vision_prompt")
    dst.mkdir(parents=True)
    shutil.copy(src / "_default_template_yaml", dst)
    utils = (src / "utils.py").read_text()
    utils += (
        "\n\ndef vision_prompt(doc, lmms_eval_specific_kwargs=None):\n"
        f"    return {VISION_PROMPT!r}\n"
    )
    (dst / "utils.py").write_text(utils)
    task = (src / "mmmu_pro_vision.yaml").read_text()
    task = re.sub(r"tag:\n(  - .*\n)+", "", task)
    task = task.replace('task: "mmmu_pro_vision"', 'task: "mmmu_pro_vision_prompt"')
    task = task.replace("utils.mmmu_pro_doc_to_text", "utils.vision_prompt")
    (dst / "mmmu_pro_vision_prompt.yaml").write_text(task)
    return dst.parent


def extracted_letter(response):
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    return extract_mcq_answer(response, LETTERS)


def answer_line_letter(response):
    found = ANSWER_LINE.findall(response)
    return found[-1] if found else extracted_letter(response)


def accuracy(lines, letter):
    samples = [json.loads(line) for line in lines]
    correct = 0
    for sample in samples:
        response = sample["filtered_resps"]
        if isinstance(response, list):
            response = response[0]
        correct += letter(response) == sample["target"]
    return correct / len(samples)


SCORERS = {"standard": extracted_letter, "vision": answer_line_letter}


def subset_reference(ref, subset, n):
    """The reference score and standard error for a subset: the reference
    topology's recorded run when there is one, else the recipe's score."""
    if ref:
        return ref[subset]["score"], ref[subset]["stderr"], ref["topology"]
    score = float(env(f"MMMU_PRO_REF_{subset.upper()}"))
    return score, stderr(score, n), "recipe"


def main():
    ref = reference() if env("REF_TOPOLOGY") else None
    _, samples = run_lmms_eval(
        "mmmu_pro",
        list(SUBSETS.values()),
        "temperature=1.0,max_new_tokens=65536",
        64,
        vision_task_dir(),
        extra={"top_p": 0.95},
    )
    result, failures = {}, []
    for subset, task in SUBSETS.items():
        n = len(samples[task])
        score = accuracy(samples[task], SCORERS[subset])
        ref_score, ref_stderr, source = subset_reference(ref, subset, n)
        failed = failed_requests(samples[task])
        result[subset] = {
            "score": score,
            "stderr": stderr(score, n),
            "n": n,
            "failed_requests": failed,
            "reference": {"source": source, "score": ref_score, "stderr": ref_stderr},
        }
        if failed:
            failures.append(f"{subset}: {failed} failed requests")
        if not within(score, ref_score, ref_stderr):
            failures.append(
                f"{subset} {score:.4f} is not within {2 * ref_stderr:.4f} of "
                f"{source}'s {ref_score:.4f}"
            )
    finish(result, failures)


if __name__ == "__main__":
    run_check(main)
