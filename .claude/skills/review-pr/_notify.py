"""Route a failed review to the owner responsible for that CLASS of failure and post one PR
comment. Single source of truth for the failure-class -> owner map, so run_one.sh (which tags
the failure with `fail <class> ...`) and the workflow (which reports it) cannot drift:
check_prompts.py asserts every class run_one.sh can emit is known here.

Called by the workflow's notify step as `python3 _notify.py <pr>`. Reads `.atom-review-status`
(written by run_one.sh as "<class>\t<message>"). Owners come from owners.txt, next to this file,
so ownership changes without touching code.
"""

import json
import os
import pathlib
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent


def _owners():
    """Read owners.txt: '<class> <login> <description>' per line, '#' comments.

    The map is in version control, not in repo variables: changing who gets
    paged should be an auditable commit, not an invisible settings edit.
    """
    out = {}
    for line in (HERE / "owners.txt").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(None, 2)
        if len(parts) >= 2:
            out[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return out


OWNERS = _owners()


def owner_for(cls):
    """An unknown class routes to the bot owner rather than to nobody."""
    row = OWNERS.get(cls) or OWNERS.get("flow")
    return row[0] if row else "zufayu"


def mark_notified():
    """Tell the report job this PR has been told -- only once a post has actually landed.

    Every early return in main() is a path where nothing reached the PR: no status file, no
    token, or a POST that threw. The report job on hosted infra is the last thing that could
    speak in those cases, and it stays quiet when this flag is set, so setting it anywhere else
    would trade a visible failure for a silent one.
    """
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("notified=true\n")


def main():
    if len(sys.argv) < 2:
        print("usage: _notify.py <pr>")
        return 2
    pr = sys.argv[1]
    path = os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), ".atom-review-status")
    if not os.path.exists(path):
        return 0  # the review succeeded -> nothing to report
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    cls, _, msg = raw.strip().partition("\t")
    if cls not in OWNERS:
        cls = "flow"  # an unknown tag routes to the bot owner rather than nobody
    owner = owner_for(cls)
    domain = OWNERS[cls][1]
    body = (
        f"⚠️ **rocm-atom-bot** — this review could not complete: a problem with **{domain}**, "
        f"not with this PR.\n\n> {msg}\n\n@{owner} — please take a look."
    )
    if cls == "glm":
        body += (
            f"\n\n<sub>Triage: if GLM serves but its output is malformed it may be the ATOM "
            f"framework (@{owner_for('atom')}); if the box, GPU, disk or network is unhealthy "
            f"it may be the machine (@{owner_for('machine')}).</sub>"
        )
    body += "\n\nRe-comment `@rocm-atom-bot review` once it is resolved."

    tok = os.environ.get("ATOM_BOT_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not tok or not repo:
        print("no ATOM_BOT_TOKEN / GITHUB_REPOSITORY -- would post:\n" + body)
        return 0
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues/{pr}/comments",
        data=json.dumps({"body": body}).encode(),
        headers={
            "Authorization": f"token {tok}",
            "Accept": "application/vnd.github+json",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=15)
        mark_notified()
        print(f"reported [{cls}] -> @{owner}")
    except Exception as e:  # noqa: BLE001 - a failed notice must not fail CI
        print(f"notify failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
