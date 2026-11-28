import os
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "download_aiter_wheel.sh"


def _write_fake_curl(bin_dir: Path) -> None:
    curl = bin_dir / "curl"
    curl.write_text(textwrap.dedent(r"""
            #!/usr/bin/env python3
            import json
            import os
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            output = None
            url = None
            skip_next = False
            for index, arg in enumerate(args):
                if skip_next:
                    skip_next = False
                    continue
                if arg in {"-H", "-o", "--connect-timeout", "--max-time"}:
                    if arg == "-o":
                        output = args[index + 1]
                    skip_next = True
                    continue
                if arg.startswith("http://") or arg.startswith("https://"):
                    url = arg

            if not url:
                print(f"fake curl could not find URL in args: {args}", file=sys.stderr)
                sys.exit(2)

            with open(os.environ["CURL_LOG"], "a", encoding="utf-8") as log:
                log.write(url + "\n")

            def write_text(payload):
                if output:
                    Path(output).write_text(payload, encoding="utf-8")
                else:
                    print(payload)

            def write_bytes(payload):
                if not output:
                    sys.stdout.buffer.write(payload)
                    return
                Path(output).write_bytes(payload)

            main_manifest_url = os.environ["S3_MAIN_MANIFEST_URL"]
            if url.startswith(main_manifest_url):
                write_text(json.dumps({
                    "branch": "main",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "commit": "sha-old",
                    "wheels": {
                        "cp312": {
                            "wheel_name": "amd_aiter-0.0.1-cp312-cp312-linux_x86_64.whl",
                            "wheel_url": "https://wheels.example/old.whl",
                        },
                    },
                }))
            elif url.startswith("https://api.github.com/repos/ROCm/aiter/actions/workflows/179476100/runs"):
                write_text(json.dumps({
                    "workflow_runs": [
                        {
                            "id": 123,
                            "head_sha": "sha-new",
                            "created_at": "2026-01-02T00:00:00Z",
                        },
                    ],
                }))
            elif url.startswith("https://api.github.com/repos/ROCm/aiter/actions/runs/123/artifacts"):
                write_text(json.dumps({
                    "artifacts": [
                        {
                            "id": 456,
                            "name": "aiter-whl-manylinux-py3.12",
                            "expired": False,
                            "created_at": "2026-01-02T00:05:00Z",
                        },
                    ],
                }))
            elif url.startswith("https://s3.example/custom/commits/sha-new/latest.json?ts="):
                write_text(json.dumps({
                    "branch": "main",
                    "timestamp": "2026-01-02T00:10:00Z",
                    "commit": "sha-new",
                    "wheels": {
                        "cp312": {
                            "wheel_name": "amd_aiter-0.0.2-cp312-cp312-linux_x86_64.whl",
                            "wheel_url": "https://wheels.example/new.whl",
                        },
                    },
                }))
            elif url == "https://wheels.example/new.whl":
                write_bytes(b"fake wheel")
            else:
                print(f"unexpected curl URL: {url}", file=sys.stderr)
                sys.exit(22)
            """).lstrip())
    curl.chmod(0o755)


def test_stale_main_uses_commit_manifest_without_false_fallback_log(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_curl(bin_dir)

    curl_log = tmp_path / "curl.log"
    output_dir = tmp_path / "aiter-whl"
    main_manifest_url = "https://s3.example/custom/main/latest.json?existing=1"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "ATOM_PYTHON_TAG": "cp312",
        "GITHUB_TOKEN": "fake-token",
        "AITER_WHEEL_DOWNLOAD_MAX_ATTEMPTS": "1",
        "AITER_WHEEL_OUTPUT_DIR": str(output_dir),
        "CURL_LOG": str(curl_log),
        "S3_MAIN_MANIFEST_URL": main_manifest_url,
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    downloaded_wheel = output_dir / "amd_aiter-0.0.2-cp312-cp312-linux_x86_64.whl"
    requested_urls = curl_log.read_text(encoding="utf-8").splitlines()

    assert downloaded_wheel.read_bytes() == b"fake wheel"
    requested_urls_text = "\n".join(requested_urls)
    assert f"{main_manifest_url}&ts=" in requested_urls_text
    assert f"{main_manifest_url}?ts=" not in requested_urls_text
    assert (
        "https://s3.example/custom/commits/sha-new/latest.json?ts="
        in requested_urls_text
    )
    assert "Using wheel from S3 commit-specific manifest" in result.stdout
    assert (
        "No GitHub fallback artifact found while checking manifest freshness"
        not in result.stdout
    )
