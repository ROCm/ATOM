"""Build a plugin CI Slurm cell JSON from workflow matrix inputs."""

import argparse
import json
import os
import re


def slug(value):
    slug_value = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return slug_value or "cell"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True, choices=("vllm", "sglang"))
    parser.add_argument("--matrix-json", required=True)
    parser.add_argument("--id-field", default="display_name")
    parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args()

    matrix = json.loads(args.matrix_json)
    cell_id = slug(str(matrix.get(args.id_field) or matrix.get("model_name", "cell")))

    cell = {
        "id": f"{args.plugin}-{cell_id}",
        "plugin": args.plugin,
        "matrix": matrix,
        "github_repo_url": os.environ.get("GITHUB_REPO_URL", ""),
        "github_commit_sha": os.environ.get("GITHUB_COMMIT_SHA", ""),
        "pr_base_sha": os.environ.get("PR_BASE_SHA", ""),
        "pr_head_sha": os.environ.get("PR_HEAD_SHA", ""),
        "aiter_artifact_id": os.environ.get("AITER_ARTIFACT_ID", ""),
        "skip_docker_login": os.environ.get("SKIP_DOCKER_LOGIN", "0"),
        "atom_base_nightly_image": os.environ.get(
            "ATOM_BASE_NIGHTLY_IMAGE", "rocm/atom-dev:latest"
        ),
        "nightly_plugin_image_tag": os.environ.get("NIGHTLY_PLUGIN_IMAGE_TAG", ""),
        "container_name": os.environ.get("CONTAINER_NAME", f"plugin_ci_{cell_id}"),
        "num_nodes": 1,
        "nodes": [
            node.strip()
            for node in os.environ.get("PLUGIN_CI_SLURM_NODES", "").split(",")
            if node.strip()
        ],
        "runner": {
            "slurm_submit_runner": os.environ.get(
                "PLUGIN_CI_SLURM_SUBMIT_RUNNER",
                "atom-mi355-8gpu-vllm-sgl-ci",
            ),
            "slurm_account": os.environ.get("PLUGIN_CI_SLURM_ACCOUNT", "amd-aifw-dev"),
            "slurm_partition": os.environ.get("PLUGIN_CI_SLURM_PARTITION", "amd-spur"),
            "log_root": os.environ.get(
                "PLUGIN_CI_LOG_ROOT",
                "/home/junyyang/ATOM_PLUGIN_RUNNER/LOG/",
            ),
            "gpus_per_node": int(os.environ.get("PLUGIN_CI_GPUS_PER_NODE", "8")),
            "cpus_per_task": int(os.environ.get("PLUGIN_CI_CPUS_PER_TASK", "114")),
            "time_limit": os.environ.get("PLUGIN_CI_TIME_LIMIT", "03:00:00"),
        },
    }

    payload = json.dumps(cell, separators=(",", ":"))
    print(payload)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if args.github_output and github_output:
        with open(github_output, "a", encoding="utf-8") as output:
            output.write(f"cell_json={payload}\n")


if __name__ == "__main__":
    main()
