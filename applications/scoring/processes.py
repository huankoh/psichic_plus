"""POSIX invocation ownership. PID numbers alone never establish ownership."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import threading
import time

import psutil

from .contracts import RunBusy


def process_identity(pid):
    process = psutil.Process(pid)
    return {'pid': pid, 'started': process.create_time(), 'pgid': os.getpgid(pid)}


def alive(identity):
    if identity is None:
        return False
    try:
        process = psutil.Process(identity['pid'])
        return (process.create_time() == identity['started'] and os.getpgid(process.pid) == identity['pgid']
                and process.status() != psutil.STATUS_ZOMBIE)
    except (psutil.NoSuchProcess, ProcessLookupError):
        return False
    except (psutil.AccessDenied, PermissionError):
        return None


def signal_owned(identity, sig):
    if alive(identity) is True:
        psutil.Process(identity['pid']).send_signal(sig)
        return True
    return False


def stop_group(identity, sig=signal.SIGTERM):
    if alive(identity) is True and identity['pid'] == identity['pgid']:
        os.killpg(identity['pgid'], sig)
        return True
    return False


@contextmanager
def exclusive(path):
    path = Path(path)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RunBusy(f'Another local executor owns {path.parent}') from error
        yield
    finally:
        os.close(fd)


def lock_held(path):
    try:
        with exclusive(path):
            return False
    except RunBusy:
        return True


def parent_guard():
    """A child cannot infer before registration and dies with its lifecycle owner."""
    import sys

    if sys.stdin.buffer.readline() != b'GO\n':
        raise SystemExit(125)

    def watch():
        os.read(sys.stdin.fileno(), 1)
        os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch, name='owner-pipe', daemon=True).start()


def wait_owned(process, identity, timeout, interrupted):
    deadline = time.monotonic() + timeout if timeout else None
    terminated = False
    while True:
        # Leave the leader unreaped until its process group is cleared. Its PID cannot
        # be reused while it is our zombie child, including on macOS without waitid.
        try:
            leader = psutil.Process(process.pid)
            exited = leader.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            exited = True
        if exited:
            try:
                if os.getpgid(process.pid) == identity['pgid'] and identity['pid'] == identity['pgid']:
                    os.killpg(identity['pgid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
            return process.wait(), terminated
        if interrupted() or deadline is not None and time.monotonic() >= deadline:
            if not terminated:
                stop_group(identity)
                deadline = time.monotonic() + 5
                terminated = True
            elif time.monotonic() >= deadline:
                stop_group(identity, signal.SIGKILL)
        time.sleep(.05)
