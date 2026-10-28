"""The gate's trigger regex lives inside a `github-script` JS block in the workflow YAML, so a
pure-python copy of it would drift silently the moment someone edits the YAML and not the test.
Instead this extracts the live pattern out of the workflow text and runs it for real, in
Python's `re` engine, against the strings the design depends on:

  - bare `atom-bot review` (the documented form) must fire
  - `@atom-bot review` and `@rocm-atom-bot review` must still fire (habit still works --
    the bare phrase is a substring of both, and that's accepted, not fought)
  - a quoted trigger (`> atom-bot review`) must not fire
  - an ordinary sentence that merely mentions the bot, without the phrase opening the line,
    must not fire -- this is the anchoring that keeps the un-@-mentioned trigger from being
    weaker than intended (prose / quoted logs / code blocks can contain the words).
"""

import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = HERE.parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "atom-review-bot.yml"


def _extract_pattern():
    text = WORKFLOW.read_text(encoding="utf-8")
    m = re.search(r"/\^@\?\(\?:rocm-\)\?atom-bot\\s\+review\\b/i", text)
    assert (
        m
    ), "trigger regex literal not found in atom-review-bot.yml -- did it move or change?"
    # strip the JS /.../i wrapper, keep the body re.IGNORECASE-compatible
    body = m.group(0)[1:-2]
    return re.compile(body, re.IGNORECASE)


def _triggers(pattern, body):
    for raw in body.split("\n"):
        line = re.sub(r"^\s+", "", raw)
        if line.startswith(">"):
            continue
        if pattern.search(line):
            return True
    return False


def test_bare_phrase_triggers():
    assert _triggers(_extract_pattern(), "atom-bot review")


def test_atom_bot_mention_still_triggers():
    assert _triggers(_extract_pattern(), "@atom-bot review")


def test_rocm_atom_bot_mention_still_triggers():
    assert _triggers(_extract_pattern(), "@rocm-atom-bot review")


def test_quoted_trigger_does_not_fire():
    assert not _triggers(_extract_pattern(), "> atom-bot review")


def test_mid_sentence_mention_does_not_fire():
    assert not _triggers(
        _extract_pattern(), "I think atom-bot review is kind of slow these days."
    )
