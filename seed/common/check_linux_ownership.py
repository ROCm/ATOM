"""Reviewed CPU-only Linux API check. Disposable helpers exit naturally within 0.4s."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import sys
import time
import traceback

assert len(sys.argv) == 2, 'supply one new receipt path'
output = Path(sys.argv[1])
assert not output.exists() and output.parent.is_dir()
assert sys.platform == 'linux' and len(list(Path('/proc/self/task').iterdir())) == 1
assert all(hasattr(os, n) for n in ('pidfd_open', 'waitid', 'P_PIDFD', 'P_ALL', 'WNOWAIT', 'fork'))
signal.signal(signal.SIGCHLD, signal.SIG_DFL)
assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
deadline = time.monotonic() + 6
fds, records = {}, []
receipt = dict(status='failed', script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               executable=sys.executable, pid_namespace=os.readlink('/proc/self/ns/pid'),
               cpu_affinity=sorted(os.sched_getaffinity(0)), supervisor_pid=os.getpid(), records=records)


def children():
    return [int(x) for x in Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()]


def capture(pid):
    assert pid in children()
    fd = os.pidfd_open(pid)
    try:
        os.waitid(os.P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        assert int(fields[1]) == os.getpid()
        records.append(dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]), sid=int(fields[3]),
                            start_ticks=int(fields[19]), proof='PIDFD WNOWAIT; unreaped kernel direct child'))
        fds[pid] = fd
    except BaseException:
        os.close(fd)
        raise
    return fd


def wait_exit(fd):
    while time.monotonic() < deadline:
        result = os.waitid(os.P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if result is not None:
            return result
        time.sleep(0.01)
    raise TimeoutError('six-second CPU ownership-check deadline')


def reap(pid):
    first = wait_exit(fds[pid])
    second = os.waitid(os.P_PIDFD, fds[pid], os.WEXITED | os.WNOHANG | os.WNOWAIT)
    assert first.si_pid == second.si_pid == pid and first.si_status == second.si_status
    result = os.waitid(os.P_PIDFD, fds[pid], os.WEXITED | os.WNOHANG)
    assert result.si_pid == pid
    os.close(fds.pop(pid))
    return result


try:
    nonchild = os.pidfd_open(os.getpid())
    try:
        try:
            os.waitid(os.P_PIDFD, nonchild, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            raise AssertionError('non-child pidfd accepted by waitid')
        except ChildProcessError:
            receipt['nonchild_rejected'] = True
    finally:
        os.close(nonchild)

    direct = os.fork()
    if direct == 0:
        time.sleep(0.2)
        os._exit(7)
    capture(direct)
    result = reap(direct)
    assert result.si_code == os.CLD_EXITED and result.si_status == 7
    receipt['repeated_wnowait_then_reap'] = True

    read_fd, write_fd = os.pipe()
    parent = os.fork()
    if parent == 0:
        os.close(read_fd)
        try:
            grandchild = os.fork()
            if grandchild == 0:
                os.close(write_fd)
                os.setsid()
                time.sleep(0.4)
                os._exit(0)
            os.write(write_fd, str(grandchild).encode())
        finally:
            os._exit(0)
    os.close(write_fd)
    capture(parent)
    try:
        assert select.select([read_fd], [], [], 2)[0], 'helper PID pipe timeout'
        adopted = int(os.read(read_fd, 32))
    finally:
        os.close(read_fd)
    wait_exit(fds[parent])
    # The dead parent remains unreaped until its adopted helper also has a stable handle.
    capture(adopted)
    assert reap(parent).si_status == 0 and reap(adopted).si_status == 0
    receipt['setsid_helper_adopted_and_reaped'] = True
    assert not children() and not fds
    try:
        os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        raise AssertionError('kernel still reports children')
    except ChildProcessError:
        receipt['kernel_no_children'] = True
    receipt['status'] = 'passed'
except BaseException:
    receipt['error'] = traceback.format_exc()
finally:
    # Only our disposable, short-lived direct/adopted CPU children are drained here.
    try:
        while children() and time.monotonic() < deadline:
            for pid in children():
                if pid not in fds:
                    capture(pid)
            for pid in list(fds):
                reap(pid)
        receipt['children_after'] = children()
        if receipt['children_after']:
            receipt['status'] = 'failed'
    except BaseException:
        receipt.update(status='failed', cleanup_error=traceback.format_exc())
    for fd in fds.values():
        os.close(fd)
    receipt.update(completed_epoch=time.time(), gpu_used=False, compiler_used=False, gpu_lease_acquired=False)
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))
raise SystemExit(0 if receipt['status'] == 'passed' else 1)
