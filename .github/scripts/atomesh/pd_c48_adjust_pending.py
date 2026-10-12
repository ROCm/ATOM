"""Move only an unstarted C48 cohort job to the same replacement node pair."""
import argparse
import json
from pathlib import Path
import re
import subprocess

RUN = "38218421252"
NODES = "pit2-p03-g12,pit2-p03-g24"


def fields(raw):
    return dict(re.findall(r"(?:^|\s)([A-Za-z/]+)=([^\s]+)", raw))


def validate(raw, job):
    data = fields(raw)
    assert data.get("JobId") == job, data
    assert re.fullmatch(r"kimi-k3-c48-gap-kimi-k3-vllm-c48-gap-(v0|v-s1|v-s2|v-q)-vllm-38218421252-1", data.get("JobName", "")), data
    assert data.get("JobState") == "PENDING", "Refuse to change a started job"
    assert data.get("NodeList") == "(null)", "Refuse to change an allocation"
    assert data.get("Account") == "amd-frameworks" and data.get("QOS") == "amd-frameworks-qos"
    assert data.get("NumNodes") == "2" and data.get("ReqGPUs") == "16"
    assert data.get("RunTime") == "00:00:00"
    assert data.get("ReqNodeList") in ("pit2-p03-g13,pit2-p03-g25", NODES)
    return data


def adjust(job, out, command):
    out.mkdir(parents=True, exist_ok=True)
    def show(name):
        raw = command("show", "job", job)
        (out / name).write_text(raw)
        return raw
    before = validate(show("before.txt"), job)
    if before["ReqNodeList"] == NODES:
        return {"already_applied": True, "job_id": job, "nodes": NODES}
    held = False
    try:
        held = True
        command("hold", job)
        validate(show("held.txt"), job)
        command("update", f"JobId={job}", f"NodeList={NODES}")
        after = validate(show("updated.txt"), job)
        assert after["ReqNodeList"] == NODES, "Controller did not apply requested nodes"
        return {"updated": True, "job_id": job, "nodes": NODES,
                "account": after["Account"], "qos": after["QOS"], "node_count": 2}
    finally:
        try:
            if held:
                command("release", job)
        finally:
            show("final.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert args.job.isdigit()
    def command(*items):
        proc = subprocess.run(["scontrol", *items], text=True, capture_output=True, timeout=30)
        with (args.out / "commands.jsonl").open("a") as log:
            log.write(json.dumps({"argv": ["scontrol", *items], "returncode": proc.returncode,
                                  "stdout": proc.stdout, "stderr": proc.stderr}) + "\n")
        proc.check_returncode()
        return proc.stdout
    result = adjust(args.job, args.out, command)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
