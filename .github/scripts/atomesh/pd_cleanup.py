"""Remove leftover ATOMesh containers only for explicitly selected prior jobs."""

import argparse
import json
import re
import subprocess


def select_containers(output, job_ids):
    selected = []
    pattern = re.compile(r"atomesh-.+-(\d+)-\d+(?:-benchmark|-eval)?")
    for line in output.splitlines():
        container = json.loads(line)
        match = pattern.fullmatch(container["Names"])
        if match and match[1] in job_ids:
            selected.append(container)
    return selected


def cleanup(job_ids):
    if not job_ids:
        print("[cleanup] No prior job IDs selected; no existing containers removed")
        return
    command = ["docker", "ps", "--format", "{{json .}}"]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=20, check=True
    )
    selected = select_containers(result.stdout, job_ids)
    for container in selected:
        name = container["Names"]
        print(f"[cleanup] Prior job container: {name}", flush=True)
        for action in (["stop", "-t", "10"], ["rm", "-f"]):
            try:
                result = subprocess.run(
                    ["docker", *action, container["ID"]],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if result.returncode:
                    print(f"[cleanup] {action[0]} failed: {result.stderr.strip()}")
            except subprocess.TimeoutExpired:
                print(f"[cleanup] {action[0]} timed out for {name}", flush=True)
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=20, check=True
    )
    remaining = select_containers(result.stdout, job_ids)
    if remaining:
        names = [container["Names"] for container in remaining]
        raise RuntimeError(f"Prior job containers are still running: {names}")
    print(f"[cleanup][OK] No running containers remain for jobs {sorted(job_ids)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-ids", default="")
    parser.add_argument("--current-job-id", required=True)
    args = parser.parse_args()
    job_ids = {part.strip() for part in args.job_ids.split(",") if part.strip()}
    if any(not re.fullmatch(r"[1-9]\d*", job_id) for job_id in job_ids):
        parser.error("cleanup job IDs must be positive integers")
    if args.current_job_id in job_ids:
        parser.error("the current allocation cannot be a cleanup target")
    cleanup(job_ids)


if __name__ == "__main__":
    main()
