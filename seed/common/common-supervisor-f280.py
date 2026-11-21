"""Two fixed #3618 entrypoints; reviewed inputs are never rewritten."""
import ctypes
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

JOB = Path('/app/agent-team-scratch/pr3618-13b-v1')
LOCK = Path('/app/agent-team-scratch/g24-gpu.lock')
BUNDLE = 'f4fcf71bd84c44f50c7c40ce8621cf610f6ade5c4c34ae32a4f387d4db750ab4'
IMAGE = 'sha256:ea74fa0b05d8c7a062ae235b901cb2b1bc0c169b0a5edd7ca07bfa3b967ca462'
mode = sys.argv[1] if len(sys.argv) == 2 else None
assert mode in ('gpu', 'compile'), 'choose exactly gpu or compile'
state = {'mode': mode, 'supervisor_pid': os.getpid(), 'lease_held': False,
         'job_started_utc': None, 'job_ended_utc': None, 'owned_identities': []}
child = None
handles = {}
stop_signal = None
scan_errors = []
sent_signals = set()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save():
    state['updated_utc'] = utc()
    temporary = JOB / (mode + '-state.json.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(JOB / (mode + '-state.json'))


def command(args, limit=15):
    p = subprocess.run(args, capture_output=True, text=True, timeout=limit)
    return {'args': args, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), ppid=int(fields[1]), pgid=int(fields[2]),
                sid=int(fields[3]), start_ticks=int(fields[19]), state=fields[0])


def direct_child_ids():
    # This single-threaded parent does not reap between reading children and opening their pidfds.
    text = Path(f'/proc/self/task/{os.getpid()}/children').read_text()
    pids = [int(value) for value in text.split()]
    assert len(pids) == len(set(pids)) and all(pid > 0 and pid != os.getpid() for pid in pids)
    return pids


def child_identity(fd, pid):
    # WNOWAIT proves the pidfd refers to our child without allowing its PID to be recycled.
    result = os.waitid(os.P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    assert result is None or result.si_pid == pid
    item = identity(pid)
    assert item['ppid'] == os.getpid(), 'kernel child is not directly parented by supervisor'
    return item


def owned():
    global scan_errors
    scan_errors = []
    for pid in direct_child_ids():
        fd = None
        try:
            if pid in handles:
                current = child_identity(handles[pid]['fd'], pid)
                assert current['start_ticks'] == handles[pid]['identity']['start_ticks'], 'retained child identity changed'
            else:
                fd = os.pidfd_open(pid)
                item = child_identity(fd, pid)
                handles[pid] = {'fd': fd, 'identity': item}
                fd = None
                state['owned_identities'].append(dict(item, ownership='kernel direct child; waitid PIDFD WNOWAIT'))
        except (OSError, AssertionError, ValueError, IndexError) as error:
            scan_errors.append(f'child ownership unresolved for {pid}: {error!r}')
        finally:
            if fd is not None:
                os.close(fd)
    members = [entry['identity'] for entry in handles.values()]
    state['kernel_children_empty'] = False
    if not members:
        try:
            os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            state['kernel_children_empty'] = True
        else:
            scan_errors.append('kernel still has a child; repeat direct-child capture')
    state['scan_errors'] = scan_errors
    state['owned_remaining'] = members
    return members


def reap():
    # Only stable, already proven child handles may be reaped. No Popen.poll/wait competes here.
    for pid, entry in list(handles.items()):
        result = os.waitid(os.P_PIDFD, entry['fd'], os.WEXITED | os.WNOHANG)
        if result is None:
            continue
        assert result.si_pid == pid
        if result.si_code == os.CLD_EXITED:
            code = result.si_status
        else:
            assert result.si_code in (os.CLD_KILLED, os.CLD_DUMPED)
            code = -result.si_status
        # Latch the original result before reporting; a later adopted child may reuse this PID.
        if child is not None and pid == child.pid and child.returncode is None:
            child.returncode = code
            state['entrypoint_result'] = dict(entry['identity'], returncode=code)
        state.setdefault('reaped_children', []).append(dict(entry['identity'], returncode=code, reaped_utc=utc()))
        os.close(entry['fd'])
        del handles[pid]


def signal_owned(members, signum):
    for item in members:
        entry = handles[item['pid']]
        expected = entry['identity']
        assert expected['start_ticks'] == item['start_ticks'], 'unproven signal identity'
        key = (item['pid'], item['start_ticks'], signum)
        if key in sent_signals:
            continue
        try:
            current = child_identity(entry['fd'], item['pid'])
            assert current['start_ticks'] == expected['start_ticks'], 'retained signal identity changed'
            signal.pidfd_send_signal(entry['fd'], signum)
            sent_signals.add(key)
            state.setdefault('cleanup_signals', []).append(dict(pid=item['pid'],
                start_ticks=item['start_ticks'], signal=signum, utc=utc()))
        except ProcessLookupError:
            pass


def report_cleanup():
    # A full/unwritable state volume must not block termination or reaping.
    try:
        save()
    except OSError as error:
        state['cleanup_reporting_error'] = repr(error)
        state['cleanup_reporting_failures'] = state.get('cleanup_reporting_failures', 0) + 1


def cleanup(hard_end):
    members = owned()
    reap()
    members = owned()
    if members:
        state['cleanup_started_utc'] = utc()
        state['status'] = 'cleaning owned descendants'
        report_cleanup()
    term_until = min(time.monotonic() + 9.5, hard_end - 0.5)
    while members and time.monotonic() < term_until:
        signal_owned(members, signal.SIGTERM)
        time.sleep(0.2)
        reap()
        members = owned()
    while members and time.monotonic() < hard_end:
        signal_owned(members, signal.SIGKILL)
        time.sleep(0.1)
        reap()
        members = owned()
    # An unreapable task never authorizes unlocking or a subsequent GPU admission.
    while members or scan_errors:
        state['status'] = 'quarantined: owned cleanup unresolved; lease retained'
        report_cleanup()
        signal_owned(members, signal.SIGKILL)
        time.sleep(1)
        reap()
        members = owned()
    state['cleanup_completed_utc'] = utc()
    report_cleanup()


def finish_cleanup(hard_end):
    # Cleanup/reporting errors must not unwind the lifetime lease context.
    while True:
        try:
            cleanup(hard_end)
            return
        except BaseException:
            state.update(status='quarantined: cleanup exception; lease retained',
                         cleanup_exception=traceback.format_exc())
            report_cleanup()
            time.sleep(1)


def requested_stop(signum, frame):
    global stop_signal
    stop_signal = signum


def preflight():
    assert JOB.is_dir() and not (JOB / (mode + '-state.json')).exists()
    approval = json.loads((JOB / (mode + '-protocol-approval.json')).read_text())
    admission = json.loads((JOB / (mode + '-admission.json')).read_text())
    assert approval['launch_cleared'] is True and approval['mode'] == mode
    assert approval['bundle_sha256'] == BUNDLE and approval['supervisor_sha256'] == sha(__file__)
    assert approval['admission_sha256'] == sha(JOB / (mode + '-admission.json'))
    assert admission['mode'] == mode and 0 <= time.time() - admission['checked_epoch'] <= 120
    assert admission['image_configuration_id'] == approval['image_configuration_id']
    assert admission['python_executable'] == sys.executable
    check_path = JOB / (mode + '-linux-ownership.json')
    assert sha(check_path) == admission['linux_ownership_receipt_sha256']
    check = json.loads(check_path.read_text())
    assert check['script_sha256'] == approval['linux_check_sha256'] == sha(JOB / 'check_linux_ownership.py')
    assert check['status'] == 'passed' and not check['children_after'] and check['kernel_no_children'] is True
    assert check['executable'] == sys.executable and check['pid_namespace'] == os.readlink('/proc/self/ns/pid')
    assert 0 <= time.time() - check['completed_epoch'] <= 120
    assert sha(JOB / 'input/bundle-manifest.json') == BUNDLE
    manifest = json.loads((JOB / 'input/bundle-manifest.json').read_text())
    expected = {'bundle-manifest.json'}
    for entry in manifest:
        p = JOB / 'input' / entry['file']
        assert p.parent == JOB / 'input' and p.is_file() and not p.is_symlink()
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
        expected.add(entry['file'])
    assert {p.name for p in (JOB / 'input').iterdir()} == expected
    cores = admission['cpu_affinity']
    assert len(set(cores)) == len(cores) == 4 and set(cores) <= os.sched_getaffinity(0)
    os.sched_setaffinity(0, cores)
    state.update(supervisor_identity=identity(os.getpid()), protocol_approval=approval,
                 admission=admission, cpu_affinity=sorted(os.sched_getaffinity(0)))
    expected_packages = {'triton': '3.8.0+amd.rocm7.2.0.git7cb7b059'}
    if mode == 'gpu':
        expected_packages['torch'] = '2.10.0+rocm7.2.4.lw.git3d3aa833'
    packages = {p: importlib.metadata.version(p) for p in expected_packages}
    assert packages == expected_packages
    state['distribution_versions'] = packages
    assert all(hasattr(os, name) for name in ('pidfd_open', 'waitid', 'P_PIDFD', 'P_ALL', 'WNOWAIT'))
    assert hasattr(signal, 'pidfd_send_signal')
    assert len(list(Path('/proc/self/task').iterdir())) == 1
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
    state.update(child_subreaper=True, child_wait_policy='single thread; SIGCHLD default; PIDFD WNOWAIT before reap')
    if mode == 'gpu':
        assert admission['image_configuration_id'] == IMAGE
        assert not any((JOB / p).exists() for p in ('sources', 'results', 'job.log'))
        assert admission['ownership_verified'] is True
        assert set(admission['visibility']) == {'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'}
        assert len(set(admission['visibility'].values())) == 1
        assert next(iter(admission['visibility'].values())).isdecimal()
        assert set(admission['identity']) == {'pci', 'uuid'}
        assert Path('/sys/module/amdgpu/version').read_text().strip() == '6.16.13'
        pin = command(['git', '-C', '/app/aiter-test', 'rev-parse', 'HEAD'])
        clean = command(['git', '-C', '/app/aiter-test', 'diff', 'HEAD', '--exit-code', '--quiet'])
        state.update(aiter_sha=pin, aiter_clean=clean)
        assert pin['returncode'] == clean['returncode'] == 0
        assert pin['stdout'].strip() == 'b9e08daca8c45dd758f4fa1fbb7d0da027f399c9'
    else:
        assert not any((JOB / p).exists() for p in ('compile-results', 'compile-tmp', 'compile-job.log'))
        assert sha(admission['compiler_library']['path']) == admission['compiler_library']['sha256']
    return admission


def run(admission, lease):
    global child
    env = {k: os.environ[k] for k in ('PATH', 'LD_LIBRARY_PATH', 'HOME', 'LANG', 'LC_ALL', 'ROCM_PATH') if k in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONPATH='',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MAX_JOBS='1',
               PR3618_BUNDLE_SHA256=BUNDLE)
    env['PATH'] = str(Path(sys.executable).parent) + ':' + env.get('PATH', '')
    if mode == 'gpu':
        explicit = command(['rocm-smi', '--showpids'])
        gpu = command(['rocm-smi', '--showuse', '--showmeminfo', 'vram', '--json'])
        state.update(explicit_kfd=explicit, gpu_admission=gpu)
        save()
        assert explicit['returncode'] == 0 and 'No KFD PIDs currently running' in explicit['stdout']
        assert gpu['returncode'] == 0
        cards = json.loads(gpu['stdout'])
        assert set(cards) == {f'card{i}' for i in range(8)}
        assert all(float(c['GPU use (%)']) == 0 and int(c['VRAM Total Used Memory (B)']) < 512 * 1024**2
                   and int(c['VRAM Total Memory (B)']) - int(c['VRAM Total Used Memory (B)']) >= 3 * 1024**3 for c in cards.values())
        env.update(admission['visibility'], PR3618_JOB=str(JOB),
                   PR3618_EXPECT_PCI=admission['identity']['pci'], PR3618_EXPECT_UUID=admission['identity']['uuid'])
        cap, entry, log = 1200, 'run_gpu.sh', 'job.log'
    else:
        (JOB / 'compile-tmp').mkdir()
        env.update(HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='',
                   TMPDIR=str(JOB / 'compile-tmp'), PR3618_COMPILE_OUTPUT=str(JOB / 'compile-results'))
        cap, entry, log = 920, 'run_compile.sh', 'compile-job.log'
    invocation = ['timeout', '--signal=TERM', '--kill-after=10s', str(cap) + 's', 'bash', str(JOB / 'input' / entry)]
    assert stop_signal is None, 'interrupted before launch'
    assert not owned() and not scan_errors, 'preflight child remains; do not launch another job'
    started = time.monotonic()
    state.update(status='running', invocation=invocation, job_started_utc=utc(), outer_seconds=cap, kill_grace_seconds=10)
    try:
        with (JOB / log).open('xb') as output:
            child = subprocess.Popen(invocation, cwd='/app', env=env, stdout=output, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True,
                                     pass_fds=(lease.fileno(),) if lease else ())
            owned()
            assert not scan_errors and child.pid in handles, 'initial child ownership unproven'
            state['child_identity'] = handles[child.pid]['identity']
            while True:
                owned()
                reap()
                save()
                if child.returncode is not None or stop_signal or scan_errors or time.monotonic() >= started + cap:
                    break
                time.sleep(0.2)
            state.update(returncode=child.returncode, stop_signal=stop_signal,
                         deadline_exceeded=time.monotonic() >= started + cap)
    finally:
        finish_cleanup(min(started + cap + 10, time.monotonic() + 10))
    state.update(returncode=child.returncode, job_ended_utc=utc())
    state['status'] = 'completed' if child.returncode == 0 and not stop_signal and not state.get('deadline_exceeded') and not state.get('identity_conflicts') else 'failed: stop and diagnose'


def main():
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
        signal.signal(sig, requested_stop)
    admission = preflight()
    save()
    if mode == 'gpu':
        with LOCK.open('rb') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state.update(lease_held=True, lease_inode=LOCK.stat().st_ino, lease_acquired_utc=utc())
            save()
            try:
                run(admission, lease)
            finally:
                finish_cleanup(time.monotonic() + 10)
                fcntl.flock(lease, fcntl.LOCK_UN)
                state.update(lease_held=False, lease_released_utc=utc())
                save()
    else:
        run(admission, None)
    save()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        state.update(status='blocked before launch' if child is None else 'failed: supervisor exception',
                     error=str(error), traceback=traceback.format_exc())
        if JOB.is_dir():
            save()
        raise
    else:
        if state['status'] != 'completed':
            raise SystemExit(1)
    finally:
        if JOB.is_dir():
            (JOB / (mode + '-exit.json')).write_text(json.dumps(state, indent=2))
