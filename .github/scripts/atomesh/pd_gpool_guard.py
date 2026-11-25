"""Check the current test's whole-node lease before crossing execution phases."""

import getpass
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

GPOOL = os.environ.get("ATOMESH_GPOOL_CLI", "/share_nfs/yajizhan/gpu-pool/bin/gpool")


def validate_lease(status, lease_id, hostname, username, mine):
    lease = status["leases"].get(lease_id)
    if not lease or lease["user"] != username.split("@", 1)[0]:
        raise RuntimeError("The test no longer owns its GPU lease")
    node = hostname.split(".", 1)[0].removeprefix("pit2-p03-")
    if lease["alloc"] != [{"node": node, "gpus": list(range(8))}]:
        raise RuntimeError("The lease must cover GPUs 0-7 on this node")
    if status["flags"][lease_id].get("yield") or f"node {node} no longer held" in mine:
        raise RuntimeError(
            "Lease requests yield or its team hold ended; stop at this phase boundary"
        )
    return lease


def main():
    lease_id = os.environ["GPOOL_LEASE"]
    phase = sys.argv[1]
    mine = subprocess.check_output([GPOOL, "mine"], text=True)
    status = json.loads(subprocess.check_output([GPOOL, "status", "--json"], text=True))
    lease = validate_lease(
        status, lease_id, socket.gethostname(), getpass.getuser(), mine
    )
    touched = subprocess.check_output(
        [GPOOL, "touch", lease_id, "--note", "ATOMesh " + phase], text=True
    )
    if "YIELD" in touched:
        raise RuntimeError("Lease became YIELD; release before the next phase")
    out = Path(os.environ["RUN_DIR"])
    (out / f"gpool-{phase}.json").write_text(
        json.dumps(
            {"lease": lease, "phase": phase, "checked_at": status["now"]}, indent=2
        )
    )
    print(f"[gpool] {lease_id} owns this node's GPUs 0-7; phase={phase}", flush=True)


if __name__ == "__main__":
    main()
