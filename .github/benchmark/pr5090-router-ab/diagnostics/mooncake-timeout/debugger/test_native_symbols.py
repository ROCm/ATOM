"""Linux CPU integration: python test_native_symbols.py DEBUGGER_IMAGE."""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

scripts = Path(__file__).resolve().parents[5] / "scripts/atomesh"
sys.path.insert(0, str(scripts))
from pd_native_stacks import capture_stacks


def main():
    image = sys.argv[1]
    job = "gdb-symbol-test-" + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="atomesh-native-symbols-") as temporary:
        root = Path(temporary)
        subprocess.run(
            [
                "g++",
                "-shared",
                "-fPIC",
                "-O1",
                "-fno-omit-frame-pointer",
                "-pthread",
                str(Path(__file__).with_name("engine_fixture.cpp")),
                "-o",
                str(root / "engine.so"),
            ],
            check=True,
        )
        program = (
            'import ctypes,threading,time; lib=ctypes.CDLL("/tmp/engine.so"); '
            "threading.Thread(target=lib.block_in_engine,daemon=True).start(); time.sleep(120)"
        )
        worker = subprocess.check_output(
            [
                "docker",
                "run",
                "-d",
                "--name",
                job,
                "--label",
                "srtctl.job=" + job,
                "--label",
                "srtctl.step=prefill_0_fixture",
                "--network=none",
                "--read-only",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                str(root / "engine.so") + ":/tmp/engine.so:ro",
                "--entrypoint",
                "/usr/bin/python3",
                image,
                "-c",
                program,
            ],
            text=True,
        ).strip()
        try:
            time.sleep(0.2)
            result = capture_stacks(job, 1, image, image, root / "capture")
            text = (root / "capture/stacks.txt").read_text()
            assert "mooncake::WorkerPool::performPostSend" in text
            assert result["mooncake_symbols_present"] and result["frames_present"]
            assert not result["sidecars_remaining"]
            assert re.search(r"(?m)^[0-9a-f]+-[0-9a-f]+ .*/engine[.]so$", text)
            assert "rip" in text
            status = subprocess.check_output(
                ["docker", "exec", worker, "cat", "/proc/1/status"], text=True
            )
            assert "TracerPid:\t0" in status
            print(
                json.dumps(
                    {
                        "native_cpp_symbol_loaded": True,
                        "maps_and_registers": True,
                        "target_alive_and_detached": True,
                        "sidecars_remaining": [],
                    }
                )
            )
        finally:
            subprocess.run(
                ["docker", "rm", "-f", worker], check=True, stdout=subprocess.DEVNULL
            )


if __name__ == "__main__":
    main()
