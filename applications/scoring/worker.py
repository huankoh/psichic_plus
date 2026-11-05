"""Invocation workers own native outputs and receipts, never coordinator state."""

from dataclasses import asdict
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import uuid

from applications.scorer import Failure, Success
from applications.scorer.boltz2.adapter import parse_affinity
from applications.scorer.boltz2.models import Candidate
from .config import child_environment
from .contracts import Engine, Task, canonical, digest, loads
from .evidence import Artifact, Receipt, atomic_pointer, encode_native_parameter, write_file
from .guardian import start_guardian
from .processes import exclusive, parent_guard, process_identity, stop_group, wait_owned
from .runtime import identity, offline_only, verify_files, verify_inputs, verify_loaded


def read_request(directory):
    return loads((Path(directory) / 'request.json').read_bytes())


def native_json(value):
    """Tag non-finite diagnostics and pandas missing values instead of silently nulling them."""
    if hasattr(value, 'item'):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return encode_native_parameter(value)
    if isinstance(value, dict):
        return {k: native_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native_json(v) for v in value]
    if type(value).__name__ in ('NAType', 'NaTType'):
        return {'native_missing': type(value).__name__}
    if hasattr(value, 'tolist'):
        return native_json(value.tolist())
    raise ValueError(f'Unsupported native diagnostic type: {type(value).__name__}')


def infer_psichic(directory):
    parent_guard()
    start_guardian(directory)
    offline_only()
    request = read_request(directory)
    profile = request['profile']
    verify_inputs(directory, request)
    verify_files(profile['source'] + profile['weights'])
    import random
    import numpy as np
    import torch
    from applications.scorer.psichic_plus.predictor import Predictor
    from applications.scorer.psichic_plus.provenance import source_provenance

    options = request['options']
    random.seed(options['seed'])
    np.random.seed(options['seed'])
    torch.manual_seed(options['seed'])
    predictor = Predictor(request['fasta'], options['model_dir'], options['device'],
                          str(Path(options['cache']) / request['tasks'][0]['run_id'] / request['tasks'][0]['invocation_id']), options['batch_size'])
    loaded = verify_loaded(profile)
    atomic_pointer(directory / 'loaded-start.json', {'profile_digest': digest(identity(profile)),
                   'loaded': loaded, 'predictor_source': source_provenance(predictor)})
    tasks = tuple(Task.from_dict(value) for value in request['tasks'])
    inputs = [(t.entry_key, t.fingerprint.submitted_smiles) for t in tasks]
    evidence = predictor.score_with_evidence(inputs, output=tasks[0].fingerprint.output)
    if [item.outcome.token for item in evidence] != [key for key, _ in inputs]:
        raise ValueError('Predictor changed the exact prepared batch identity')
    loaded = verify_loaded(profile)
    atomic_pointer(directory / 'loaded.json', {'profile_digest': digest(identity(profile)),
                   'loaded': loaded, 'predictor_source': source_provenance(predictor)})
    for task, result in zip(tasks, evidence, strict=True):
        if isinstance(result.outcome, Success):
            record = {'status': 'succeeded', 'raw_affinity': result.outcome.raw_affinity,
                      'adaptive_score': result.outcome.adaptive_score,
                      'submitted_smiles': task.fingerprint.submitted_smiles, 'normalized_smiles': None,
                      'distribution': asdict(result.outcome.distribution)}
        elif isinstance(result.outcome, Failure):
            record = {'status': 'failed', 'stage': result.outcome.stage, 'reason': result.outcome.reason}
        else:
            raise ValueError('Unknown Predictor outcome')
        atomic_pointer(directory / 'raw' / f'{task.attempt_id}.json', {
            'record': record, 'raw': {'schema_version': 1, 'task': task.to_dict(), 'native': native_json(result.native)}})


def _link(source, destination):
    os.link(source, destination)


def publish(directory):
    request = read_request(directory)
    if (directory / 'published.json').exists():
        return
    verify_inputs(directory, request)
    completion = loads((directory / 'completion.json').read_bytes())
    loaded_path = directory / 'loaded.json'
    if not loaded_path.is_file():
        return
    loaded = loads(loaded_path.read_bytes())
    if loaded['profile_digest'] != digest(identity(request['profile'])):
        raise ValueError('Completed worker loaded a different runtime')
    verify_files(request['profile']['source'] + request['profile']['weights'])
    mapping = {item['entry_key']: item for item in request['mapping']}
    for task_value in request['tasks']:
        task = Task.from_dict(task_value)
        item = mapping[task.entry_key]
        if item['attempt_id'] != task.attempt_id:
            raise ValueError('Native name mapping does not match the task')
        roles = {'native_input': directory / 'inputs' / item['input'], 'stdout': directory / 'stdout.log',
                 'stderr': directory / 'stderr.log', 'execution': directory / 'completion.json',
                 'loaded_runtime': loaded_path, 'runtime_inventory': directory / 'profile.json',
                 'native_mapping': directory / 'mapping.json'}
        if task.engine is Engine.PSICHIC:
            raw_path = directory / 'raw' / f'{task.attempt_id}.json'
            if not raw_path.exists():
                continue
            row = loads(raw_path.read_bytes())
            record = row['record']
        else:
            native_key = item['native_key']
            native_directory = directory / 'native' / 'boltz_results_inputs' / 'predictions' / native_key
            native_path = native_directory / f'affinity_{native_key}.json'
            extension = 'cif' if request['options']['native']['output_format'] == 'mmcif' else 'pdb'
            structure = native_directory / f'{native_key}_model_0.{extension}'
            try:
                score = parse_affinity(native_path, Candidate(item['native_row'], '', task.fingerprint.submitted_smiles))
                if not structure.is_file() or not structure.stat().st_size:
                    raise ValueError('Missing native structure')
                record = {'status': 'succeeded', 'affinity_pred_value': score.affinity_pred_value,
                          'affinity_probability_binary': score.affinity_probability_binary,
                          'native': score.native, 'submitted_smiles': score.submitted_smiles}
                roles.update(native_affinity=native_path, structure=structure)
                for index, path in enumerate(sorted(native_directory.rglob('*'))):
                    if path.is_file() and path not in (native_path, structure):
                        roles[f'native_artifact_{index:04d}'] = path
            except (ValueError, OSError) as error:
                # A killed or crashed batch does not establish a model failure for missing rows.
                if completion['returncode'] != 0:
                    continue
                record = {'status': 'failed', 'stage': 'boltz_output', 'reason': str(error)}
        transfer = directory / 'transfers' / f'{task.attempt_id}-{uuid.uuid4().hex}'
        transfer.mkdir()
        for role, path in roles.items():
            _link(path, transfer / role)
        write_file(transfer / 'provenance', canonical(asdict(task.fingerprint)) + '\n')
        roles['provenance'] = transfer / 'provenance'
        if task.engine is Engine.PSICHIC:
            write_file(transfer / 'raw_psichic', canonical(row['raw']) + '\n')
            roles['raw_psichic'] = transfer / 'raw_psichic'
        Receipt(task, canonical(record), tuple(Artifact.from_file(transfer, role, role) for role in roles)).publish(transfer)

    atomic_pointer(directory / 'published.json', {'request_digest': digest(request), 'completed': True})


def own(directory, *, salvage=False):
    with exclusive(directory / 'owner.lock'):
        if salvage:
            publish(directory)
            return 0
        atomic_pointer(directory / 'owner.json', process_identity(os.getpid()))
        if sys.stdin.buffer.readline() != b'GO\n':
            return 125
        request = read_request(directory)
        interrupted = False

        def interrupt(signum, frame):
            nonlocal interrupted
            interrupted = True

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        command = request['command']
        env = child_environment(request['controls'], boltz=request['engine'] == 'boltz')
        process = None
        with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
            try:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                           env=env, cwd=directory, start_new_session=True)
                child = process_identity(process.pid)
                atomic_pointer(directory / 'child.json', child)
                process.stdin.write(b'GO\n')
                process.stdin.flush()
                returncode, terminated = wait_owned(process, child, request['timeout'], lambda: interrupted)
            finally:
                if process is not None:
                    if process.poll() is None:
                        stop_group(child, signal.SIGKILL)
                        process.wait()
                    process.stdin.close()
                stdout.flush()
                stderr.flush()
                os.fsync(stdout.fileno())
                os.fsync(stderr.fileno())
        atomic_pointer(directory / 'completion.json', {'schema_version': 1,
                       'invocation_id': request['tasks'][0]['invocation_id'], 'command': command,
                       'returncode': returncode})
        atomic_pointer(directory / 'termination.json', {'interrupted': interrupted, 'timeout_or_signal': terminated})
        publish(directory)
        return 0


def main():
    action, path = sys.argv[1:]
    directory = Path(path).absolute()
    try:
        if action == 'psichic':
            infer_psichic(directory)
            return 0
        return own(directory, salvage=action == 'publish')
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
