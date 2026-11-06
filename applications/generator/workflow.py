"""A native process completion barrier followed by recoverable final scoring."""

import csv
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from applications.scoring.config import ConfigurationError, child_environment
from applications.scoring.contracts import canonical, digest, loads
from applications.scoring.evidence import atomic_pointer, fsync_directory, write_file
from applications.scoring.processes import alive, exclusive, process_identity, stop_group, wait_owned
from .products import snapshot


def environment(controls=None):
    env = child_environment(controls or {})
    inherited = os.environ.get('PYTHONPATH')
    if inherited:
        env['PYTHONPATH'] += os.pathsep + inherited
    return env


def preflight(native_args):
    result = subprocess.run([sys.executable, '-B', '-m', 'applications.generator.native', 'preflight'],
                            input=canonical(native_args), text=True, capture_output=True, env=environment(), timeout=120)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end='', flush=True)
    try:
        value = loads(result.stdout)
    except ValueError as error:
        raise ConfigurationError('NATIVE_PREFLIGHT', 'Native preflight did not return a valid response. Inspect stderr.') from error
    if 'error' in value:
        error = value['error']
        raise ConfigurationError(error['code'], error['message'], error['next_action'])
    if result.returncode:
        raise ConfigurationError('NATIVE_PREFLIGHT', 'Native preflight failed. Inspect stderr.')
    return value['preflight']


def create(directory, scoring_args, native_args, native, scoring_contract, generation_environment):
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    request = {'schema_version': 1, 'scoring_args': scoring_args, 'native_args': native_args,
               'cwd': str(Path.cwd()), 'preflight': native, 'scoring_directory': str(directory / 'scoring'),
               'scoring_contract': scoring_contract, 'generation_environment': generation_environment,
               'host': socket.gethostname()}
    write_file(directory / 'workflow.json', canonical(request) + '\n')
    write_file(directory / 'workflow.sha256', digest(request) + '\n')
    generation = directory / 'generation'
    generation.mkdir()
    write_file(generation / 'request.json', canonical(request) + '\n')
    fsync_directory(generation)
    fsync_directory(directory)
    return request


def read(directory):
    request = loads((Path(directory) / 'workflow.json').read_bytes())
    if digest(request) != (Path(directory) / 'workflow.sha256').read_text().strip():
        raise ConfigurationError('WORKFLOW_CHANGED', 'The frozen automatic workflow request changed')
    if request['scoring_directory'] != str(Path(directory).resolve() / 'scoring'):
        raise ConfigurationError('WORKFLOW_MOVED', 'Resume the workflow at its original path on the same host')
    if request['host'] != socket.gethostname():
        raise ConfigurationError('WORKFLOW_HOST', 'Automatic workflow recovery is qualified only on its original POSIX host')
    return request


def generate(directory, request):
    directory = Path(directory)
    generation = directory / 'generation'
    command = [sys.executable, '-B', '-m', 'applications.generator.native', 'generate', str(generation)]
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    process = child = None
    try:
        with exclusive(directory / 'workflow.lock'):
            if Path(request['preflight']['source_run']).exists():
                raise ConfigurationError('NATIVE_OUTPUT_EXISTS', 'Native --save_dir appeared after preflight. Generation was not started.')
            with (generation / 'stdout.log').open('xb') as stdout, (generation / 'stderr.log').open('xb') as stderr:
                try:
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                               cwd=request['cwd'], env=environment(request['generation_environment']), start_new_session=True)
                    child = process_identity(process.pid)
                    atomic_pointer(generation / 'child.json', child)
                    process.stdin.write(b'GO\n')
                    process.stdin.flush()
                    returncode, terminated = wait_owned(process, child, None, lambda: interrupted)
                finally:
                    if process is not None:
                        if process.poll() is None:
                            if child is not None:
                                stop_group(child, signal.SIGKILL)
                            else:
                                process.kill()
                            process.wait()
                        process.stdin.close()
                    for handle in (stdout, stderr):
                        handle.flush()
                        os.fsync(handle.fileno())
            completion = {'schema_version': 1, 'kind': 'observed-native-exit', 'command': command,
                          'returncode': returncode, 'interrupted': interrupted, 'terminated': terminated,
                          'child': child, 'group_retired': True, 'observed_exit_ns': time.time_ns()}
            if completed(completion):
                try:
                    with snapshot(request['preflight']['source_run'], request['scoring_args']['id_column'],
                                  request['scoring_args']['smiles_column']) as products:
                        completion.update(source_sha256=products.sha256, product_rows=products.row_count)
                except (ValueError, OSError, csv.Error) as error:
                    completion['product_error'] = str(error)
            atomic_pointer(generation / 'completion.json', completion)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if interrupted and completed(completion):
        raise KeyboardInterrupt
    return completion


def completed(completion):
    return (completion.get('kind') == 'observed-native-exit' and completion.get('returncode') == 0
            and not completion.get('interrupted') and not completion.get('terminated') and completion.get('group_retired') is True)


def completion(directory):
    directory = Path(directory)
    for name in ('child', 'guardian'):
        path = directory / 'generation' / f'{name}.json'
        if path.exists() and alive(loads(path.read_bytes())) is not False:
            raise ConfigurationError('GENERATION_LIVE', 'Native generation still has a live or uninspectable owned process',
                                     'Wait for or cancel the owning command before resuming final scoring.')
    path = directory / 'generation/completion.json'
    if not path.exists():
        raise ConfigurationError('GENERATION_UNCONFIRMED', 'No successful native process completion was observed. Resume never reruns generation.',
                                 'Inspect generation logs. Start a fresh workflow, or explicitly import a separately confirmed completed run with rank --source-run.')
    value = loads(path.read_bytes())
    if not completed(value):
        raise ConfigurationError('GENERATION_FAILED', 'Native generation failed or was cancelled. Dependent scoring is stopped.',
                                 'Inspect generation logs and start a fresh workflow after correcting the failure.')
    return value
