"""Public local driver over the durable core. One invocation at a time."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from applications.scorer import Output
from applications.scorer.boltz2.adapter import BoltzRunner
from applications.scorer.boltz2.models import Candidate
from applications.scorer.boltz2.target import Target as NativeTarget
from .config import ConfigurationError, candidates, child_environment, retain_inputs, verify_runtime
from .contracts import Asset, BatchContext, Engine, Fingerprint, Ownership, Resolution, State, canonical, digest, loads
from .evidence import atomic_pointer, fsync_directory, write_file
from .processes import alive, exclusive, lock_held, process_identity, signal_owned
from .run import ScoringRun
from .runtime import checksum, verify_files


def create_run(plan):
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    run = ScoringRun.create(plan.output, candidates(plan.source, plan.id_column, plan.smiles_column),
                            target=plan.target, settings=plan.settings)
    retain_inputs(plan, run)
    local = run.path / 'local'
    local.mkdir()
    (local / 'prepared').mkdir()
    (local / 'invocations').mkdir()
    fsync_directory(local)
    return run


def _profile_assets(profile, name):
    declarations = [{'name': f['name'], 'sha256': f['sha256']} for f in profile[name]]
    assets = (Asset(name + '-inventory-v1', digest(declarations)),)
    if name == 'source' and profile.get('generated_source'):
        assets += (Asset('generated-source-v1', digest(profile['generated_source'])),)
    return assets


def native_options(profile):
    tokens = []
    for name, value in profile['resolved'].items():
        if name in ('out_dir', 'model', 'override', 'use_msa_server') or name.startswith(('msa_', 'api_key_')) or value is None:
            continue
        flag = profile['option_flags'][name]
        if name in profile['flag_options']:
            if value:
                tokens.append(flag)
        else:
            tokens += [flag, str(value)]
    return tokens


def prepare_invocation(run, runtime, engine, rows):
    profile = runtime['profiles'][engine.value]
    verify_files(profile['source'] + profile['weights'])
    options = loads(getattr(run.settings, engine.value + '_options'))
    prepared = run.path / 'local' / 'prepared' / uuid.uuid4().hex
    prepared.mkdir()
    (prepared / 'inputs').mkdir()
    (prepared / 'raw').mkdir()
    (prepared / 'transfers').mkdir()
    target = NativeTarget(runtime['target_document'], runtime['ligand_id'], 'single', run.path / 'inputs', run.path / 'inputs/target.yaml')
    mapping, declarations = [], []
    batch = BatchContext.for_rows(rows, seed=runtime['seed'])
    for index, row in enumerate(rows, 1):
        native = Candidate(index, row.entry.id, row.entry.smiles)
        filename = native.key + '.yaml' if engine is Engine.BOLTZ else row.key + '.json'
        path = prepared / 'inputs' / filename
        if engine is Engine.BOLTZ:
            target.render(runtime['target_document'], native, path)
        else:
            write_file(path, canonical({'entry': row.to_dict(), 'sequence': run.target.sequence}) + '\n')
        fingerprint = Fingerprint(engine, profile['engine_version'], row.entry.smiles, run.target, checksum(path),
                                  _profile_assets(profile, 'weights'), _profile_assets(profile, 'source'),
                                  canonical(profile['environment']), profile['device'], profile['precision'],
                                  'native-evidence-v1', 'local-bounded-v1',
                                  getattr(run.settings, engine.value + '_options'), batch,
                                  run.settings.required_output if engine is Engine.PSICHIC else Output.AFFINITY, 'single', ())
        declarations.append((row.key, fingerprint))
        mapping.append({'entry_key': row.key, 'native_key': native.key, 'native_row': index, 'input': filename})
    tasks = run.prepare(engine, declarations, ownership=Ownership.LOCAL)
    directory = run.path / 'local' / 'invocations' / tasks[0].invocation_id
    for task, item in zip(tasks, mapping, strict=True):
        item['attempt_id'] = task.attempt_id
    command = [sys.executable, '-B', '-m', 'applications.scoring.worker', 'psichic', str(directory)]
    native_command = None
    if engine is Engine.BOLTZ:
        runner = BoltzRunner(profile['executable'], target, directory, options=native_options(profile))
        native_command = runner.prediction_command(directory / 'inputs', directory / 'native')
        command = [profile['executable'], '-B', str(Path(__file__).with_name('runtime.py')), '--execute', str(directory)]
    request = {'schema_version': 1, 'engine': engine.value, 'tasks': [t.to_dict() for t in tasks],
               'options': options, 'profile': profile, 'mapping': mapping, 'command': command,
               'native_command': native_command, 'fasta': str(run.path / 'inputs/target.fasta'),
               'timeout': options.get('timeout'), 'controls': runtime['controls'],
               'target_assets': [{'path': str(run.path / 'inputs' / a['copy']), 'sha256': a['sha256']}
                                 for a in runtime['target_assets']]}
    write_file(prepared / 'request.json', canonical(request) + '\n')
    write_file(prepared / 'profile.json', canonical(profile) + '\n')
    write_file(prepared / 'mapping.json', canonical(mapping) + '\n')
    fsync_directory(prepared)
    os.rename(prepared, directory)
    fsync_directory(directory.parent)
    return directory


def _read_optional(path):
    return loads(path.read_bytes()) if path.exists() else None


def _ingest(run, directory):
    for path in (directory / 'transfers').iterdir():
        if (path / 'receipt.json').exists():
            run.accept(path)


def reconcile(run, runtime):
    """Ingest before liveness checks. Never treat external ownership as local."""
    root = run.path / 'local' / 'invocations'
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        _ingest(run, directory)
        if (directory / 'completion.json').exists() and not (directory / 'published.json').exists() and not lock_held(directory / 'owner.lock'):
            # Receipt recovery remains a worker write, in a separate process.
            with (directory / 'recovery.log').open('ab') as log:
                result = subprocess.run([sys.executable, '-B', '-m', 'applications.scoring.worker', 'publish', str(directory)],
                    stdout=log, stderr=log, env=child_environment(runtime['controls']), timeout=600)
            if result.returncode:
                raise ConfigurationError('RECEIPT_RECOVERY', f'Cannot validate completed artifacts in {directory}',
                                         'Inspect recovery.log. Preserve the run and correct the reported provenance or artifact problem.')
            _ingest(run, directory)
    for after in range(0, run.row_count, 4096):
        diagnostics = tuple(run.diagnostics(after=after, limit=4096))
        for item in diagnostics:
            if item.state not in (State.SUBMITTED, State.UNKNOWN):
                continue
            task = run.task(item.attempt_id)
            if task.ownership is not Ownership.LOCAL:
                continue
            directory = root / task.invocation_id
            if not directory.exists():
                resolution, evidence = Resolution.STOPPED, 'No published invocation. Registration gate could not authorize model execution.'
            else:
                owner = _read_optional(directory / 'owner.json') or _read_optional(directory / 'launch.json')
                child = _read_optional(directory / 'child.json')
                guardian = _read_optional(directory / 'guardian.json')
                states = (alive(owner), alive(child), alive(guardian))
                if lock_held(directory / 'owner.lock') or True in states:
                    resolution, evidence = Resolution.RUNNING, 'Matching process start identity or held invocation lock; no redispatch.'
                elif None in states:
                    resolution, evidence = Resolution.UNKNOWN, 'Process identity cannot be inspected; no redispatch.'
                else:
                    resolution, evidence = Resolution.STOPPED, 'Owner and child start identities are stopped; invocation lock is free.'
            run.reconcile(task.attempt_id, resolution, evidence=evidence)


def launch(directory, runtime, interrupted):
    command = [sys.executable, '-B', '-m', 'applications.scoring.worker', 'owner', str(directory)]
    with (directory / 'owner.log').open('xb') as log:
        owner = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                 env=child_environment(runtime['controls']), start_new_session=True)
        try:
            registered = process_identity(owner.pid)
            atomic_pointer(directory / 'launch.json', registered)
            owner.stdin.write(b'GO\n')
            owner.stdin.flush()
            owner.stdin.close()
            stopping = False
            while owner.poll() is None:
                if interrupted() and not stopping:
                    signal_owned(registered, signal.SIGTERM)
                    stopping = True
                time.sleep(.05)
            return owner.returncode
        finally:
            if not owner.stdin.closed:
                owner.stdin.close()


def execute(run, *, retry_failed=False, engine=None, progress=None, verified_runtime=None):
    """Execute pending local work using frozen settings. Return Report or Waiting."""
    with exclusive(run.path / 'driver.lock'):
        if checksum(run.path / 'local.json') != (run.path / 'local.sha256').read_text().strip():
            raise ConfigurationError('CREATE_NEW_RUN', 'The frozen local runtime settings changed. Create a new run.')
        runtime = loads((run.path / 'local.json').read_bytes())
        if checksum(run.path / 'inputs/candidates.csv') != runtime['candidate_sha256']:
            raise ConfigurationError('FROZEN_INPUT_CHANGED', 'The retained input CSV changed. Preserve the run and create a new run.')
        for asset in runtime['target_assets']:
            if checksum(run.path / 'inputs' / asset['copy']) != asset['sha256']:
                raise ConfigurationError('FROZEN_INPUT_CHANGED', 'A retained target asset changed. Create a new run.')
        if (run.path / 'inputs/target.fasta').read_text() != f'>target\n{run.target.sequence}\n':
            raise ConfigurationError('FROZEN_INPUT_CHANGED', 'The retained sequence changed. Create a new run.')
        if verified_runtime is None:
            verify_runtime(runtime, run.settings)
        elif canonical(runtime) != canonical(verified_runtime):
            raise ConfigurationError('CREATE_NEW_RUN', 'Resolved runtime differs from the retained configuration')
        reconcile(run, runtime)
        diagnostics = run.diagnostics()
        try:
            if any(d.state in (State.SUBMITTED, State.UNKNOWN) and not d.resumable for d in diagnostics):
                diagnostics.close()
                return run.finalize()
        finally:
            diagnostics.close()
        if retry_failed:
            run.retry_failed(engine=engine)
        interrupted = False

        def interrupt(signum, frame):
            nonlocal interrupted
            interrupted = True

        previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            for current_engine in run.settings.engines:
                if current_engine is Engine.BOLTZ and run.settings.pyramid:
                    run.select()
                while not interrupted:
                    rows = run.pending(current_engine, limit=runtime['chunk_size'])
                    if not rows:
                        break
                    directory = prepare_invocation(run, runtime, current_engine, rows)
                    if progress:
                        progress(f'{current_engine.value}: {len(rows)} entries in invocation {directory.name}')
                    launch(directory, runtime, lambda: interrupted)
                    _ingest(run, directory)
                    # Unknown rows are retried only on a later explicit resume, not in a loop.
                    unresolved = any(d.attempt_id in {m['attempt_id'] for m in loads((directory / 'mapping.json').read_bytes())}
                                     and d.state in (State.SUBMITTED, State.UNKNOWN)
                                     for d in run.diagnostics(after=rows[0].ordinal - 1,
                                                              limit=min(4096, rows[-1].ordinal - rows[0].ordinal + 1)))
                    if unresolved or interrupted:
                        interrupted = True
                        break
                if interrupted:
                    break
            return run.finalize()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
