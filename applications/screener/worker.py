"""Execute each frozen external invocation once, then export its immutable receipts."""

import os
from pathlib import Path
import shutil
import signal
import sys
import threading

from applications.scoring.contracts import Ownership, Task, loads
from applications.scoring.evidence import atomic_pointer
from applications.scoring.processes import exclusive, process_identity
from applications.scoring.runtime import checksum, verify_inputs
from applications.scoring.worker import own, publish


def execute(map_path, entry_key, output):
    map_path, output = Path(map_path), Path(output)
    if checksum(map_path) != map_path.stem:
        raise ValueError('Dispatch map digest mismatch')
    manifest = loads(map_path.read_bytes())
    directory = Path(manifest['invocation'])
    if checksum(directory / 'request.json') != manifest['request_sha256']:
        raise ValueError('Frozen invocation request changed')
    request = loads((directory / 'request.json').read_bytes())
    tasks = tuple(Task.from_dict(t) for t in request['tasks'])
    if (any(t.ownership is not Ownership.EXTERNAL for t in tasks)
            or [t.entry_key for t in tasks] != [r['key'] for r in manifest['rows']]
            or any(t.run_id != manifest['run_id'] or t.population_digest != manifest['population_digest']
                   or t.engine.value != manifest['engine'] for t in tasks)):
        raise ValueError('External invocation identities differ from native row map')
    task = next(t for t in tasks if t.entry_key == entry_key)
    verify_inputs(directory, request)
    with exclusive(directory / 'external.lock'):
        if (directory / 'completion.json').exists():
            publish(directory)
        elif (directory / 'external-started.json').exists():
            raise ValueError('Invocation already started. Missing completion stays unresolved; coordinator reconciliation is required.')
        else:
            atomic_pointer(directory / 'external-started.json', process_identity(os.getpid()))
            own(directory, registered=True)
        found = []
        for transfer in (directory / 'transfers').iterdir():
            path = transfer / 'receipt.json'
            if path.is_file() and loads(path.read_bytes())['task']['attempt_id'] == task.attempt_id:
                found.append(transfer)
        if len(found) != 1:
            raise ValueError('No unique qualified receipt. Missing runtime evidence remains pending.')
        shutil.copytree(found[0], output / 'receipt')


def main():
    if sys.stdin.buffer.readline() != b'GO\n':
        return 125

    def parent_closed():
        os.read(sys.stdin.fileno(), 1)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=parent_closed, daemon=True).start()
    execute(*sys.argv[1:])
    return 0


if __name__ == '__main__':
    sys.exit(main())
