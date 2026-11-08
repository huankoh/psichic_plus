"""A model-group member clears descendants if both model and owner die."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def start_guardian(directory):
    model = psutil.Process(os.getpid())
    identity = {'pid': model.pid, 'started': model.create_time(), 'pgid': os.getpgrp()}
    guardian = subprocess.Popen([sys.executable, '-B', str(Path(__file__)), json.dumps(identity)],
                                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    guard = psutil.Process(guardian.pid)
    record = {'pid': guard.pid, 'started': guard.create_time(), 'pgid': os.getpgid(guard.pid)}
    path = Path(directory) / 'guardian.json'
    with path.with_suffix('.tmp').open('x') as handle:
        json.dump(record, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(path.with_suffix('.tmp'), path)
    guardian.stdin.write(b'GO\n')
    guardian.stdin.close()
    return guardian


def main():
    model = json.loads(sys.argv[1])
    if sys.stdin.buffer.readline() != b'GO\n':
        return 125
    if os.getpgrp() != model['pgid']:
        return 125
    while True:
        try:
            process = psutil.Process(model['pid'])
            if process.create_time() != model['started'] or process.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(.05)
    # This process is still a member, so this group ID cannot belong to a reused PID.
    os.killpg(os.getpgrp(), signal.SIGKILL)


if __name__ == '__main__':
    sys.exit(main())
