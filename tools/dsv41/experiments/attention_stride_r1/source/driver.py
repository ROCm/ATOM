"""Reuse the retained deployment, executable cache and single temporary pool."""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r8"))
from common import AITER, CACHE, CANDIDATE, ROOT, restore, setup_pool

sys.path.insert(0, str(CANDIDATE / "tools/dsv41"))
from run_dashboard import clean_environment, load_profile


def main():
    output = Path(os.environ["WINDOW_OUTPUT"])
    setup_pool(output)
    restore(output)
    profile = load_profile()
    env, _ = clean_environment(os.environ, CANDIDATE, AITER, CACHE, profile)
    env.update(
        {
            k: v
            for k, v in profile["owner_execution"]["environment"].items()
            if k.startswith("ATOM_")
        }
    )
    env.update(
        PYTHONPATH=os.pathsep.join(
            map(
                str,
                (
                    HERE,
                    CANDIDATE / "tools/dsv41",
                    HERE.parent,
                    ROOT / "validation/whole_join_20261028",
                    CANDIDATE,
                    AITER,
                ),
            )
        ),
        WINDOW_OUTPUT=str(output),
        WINDOW_CLOCK_LIBRARY=str(
            HERE.parent / "owner_window_cost_r3/output/window_clock.so"
        ),
        ROCR_VISIBLE_DEVICES="0,1,2,3",
        FLYDSL_GPU_ARCH="gfx950",
        GPU_ARCHS="gfx950",
        PYTORCH_ROCM_ARCH="gfx950",
        AMD_COMGR_CACHE="0",
        AMD_COMGR_USE_VFS="1",
        AMD_COMGR_SAVE_TEMPS="0",
        AMD_COMGR_SAVE_LLVM_TEMPS="0",
        NCCL_IB_DISABLE="1",
        NCCL_DEBUG="WARN",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        MAX_JOBS="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        ATOM_LOADER_NUM_THREADS="1",
        TMPDIR="/tmp/owner-service/tmp",
        AITER_META_DIR=str(AITER),
        AITER_LOG_LEVEL="WARNING",
        LD_PRELOAD=str(
            ROOT / "validation/structure_ceiling_20261024/serialize_comgr.so"
        ),
        DSV41_COMGR_LOCK_PATH="/tmp/owner-service/tmp/comgr_action.lock",
        DSV41_COMGR_RECEIPT_DIR=str(output),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(HERE / "probe.py"),
        ],
        env=env,
        check=True,
        timeout=2400,
    )


if __name__ == "__main__":
    main()
