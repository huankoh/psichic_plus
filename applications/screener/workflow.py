"""One coordinator advances native stages after exact receipt completion."""

from dataclasses import replace
import heapq
from pathlib import Path
import shutil
import sys
import tempfile

from applications.scoring.config import ConfigurationError, resolve, verify_runtime
from applications.scoring.contracts import Engine, Resolution, State, canonical, loads
from applications.scoring.evidence import atomic_pointer, write_file
from applications.scoring.executor import create_run
from applications.scoring.processes import alive, exclusive, lock_held
from applications.scoring.runtime import checksum
from .native import copy_tools, ingest, prepare_stage
from .population import read_control, snapshot


def create(args):
    if not 1 <= args.stage_size <= 4096:
        raise ConfigurationError('STAGE_SIZE', 'Use a native stage window of 1..4096 entries')
    source = args.source_run if args.action == 'rank' else args.afvs
    with snapshot(source, args.scenario, completed=args.action == 'rank') as population:
        args.candidates, args.id_column, args.smiles_column = population.path, 'ID', 'SMILES'
        plan = resolve(args)
        base = Path(population.config['docking_scenario_basefolder'])
        if not base.is_absolute():
            base = population.source / 'tools' / base
        target_config = base / args.scenario / 'config.txt'
        if target_config.is_file() and not target_config.read_text().lstrip().startswith('{'):
            native_target = read_control(target_config).get('protein_fasta')
            if native_target:
                fasta = Path(native_target)
                if not fasta.is_absolute():
                    fasta = target_config.parent / fasta
                sequence = ''.join(line.strip() for line in fasta.read_text().splitlines() if not line.startswith('>')).upper()
                if sequence != plan.target.sequence:
                    raise ConfigurationError('TARGET_MISMATCH', 'Final scoring target differs from the selected native scenario')
        plan = replace(plan, runtime={**plan.runtime, 'source': population.evidence['selection']})
        screen = {'schema_version': 1, 'source': population.evidence, 'native_config': population.config,
                  'afvs_python': str(Path(args.afvs_python or sys.executable).resolve(strict=True)),
                  'stage_size': args.stage_size, 'allocation_policy': 'separate-native-workunits',
                  'storage': 'same-host-sharedfs', 'upstream': str(Path(args.afvs).resolve(strict=True))}
        with tempfile.TemporaryDirectory(prefix='afvs-template-') as temporary:
            copy_tools(args.afvs, Path(temporary) / 'template')
            population.verify()
            if args.validate_only:
                return None, plan, screen
            run = create_run(plan, entries=population.entries())
            root = run.path / 'screening'
            root.mkdir()
            (root / 'stages').mkdir()
            (root / 'imports').mkdir()
            (root / 'reconciliations').mkdir()
            (root / 'source').mkdir()
            screen['source_files'] = []
            for name in ('selection', 'config'):
                saved = root / 'source' / name
                shutil.copyfile(population.evidence[name], saved)
                sha = population.evidence[name + '_sha256']
                if checksum(saved) != sha:
                    raise ConfigurationError('SOURCE_CHANGED', 'Native source changed while retaining its original bytes')
                screen['source_files'].append({'copy': 'source/' + name, 'sha256': sha})
            shutil.copytree(Path(temporary) / 'template', root / 'template')
            write_file(root / 'screen.json', canonical(screen) + '\n')
            write_file(root / 'screen.sha256', checksum(root / 'screen.json') + '\n')
            return run, plan, screen


def read(run):
    path = run.path / 'screening/screen.json'
    if checksum(path) != (path.parent / 'screen.sha256').read_text().strip():
        raise ValueError('Frozen native screen configuration changed')
    screen = loads(path.read_bytes())
    for item in screen['source_files']:
        if checksum(path.parent / item['copy']) != item['sha256']:
            raise ValueError('Retained native source changed')
    return screen


def advance(run, *, retry=False, engine=None, verified_runtime=None):
    with exclusive(run.path / 'driver.lock'):
        screen = read(run)
        if checksum(run.path / 'local.json') != (run.path / 'local.sha256').read_text().strip():
            raise ValueError('Frozen runtime configuration changed')
        runtime = loads((run.path / 'local.json').read_bytes())
        if verified_runtime is None:
            verify_runtime(runtime, run.settings)
        elif canonical(runtime) != canonical(verified_runtime):
            raise ValueError('Runtime differs from the preflight contract')
        if checksum(run.path / 'inputs/candidates.csv') != runtime['candidate_sha256']:
            raise ValueError('Frozen candidate snapshot changed')
        for asset in runtime['target_assets']:
            if checksum(run.path / 'inputs' / asset['copy']) != asset['sha256']:
                raise ValueError('Frozen target asset changed')
        if (run.path / 'inputs/target.fasta').read_text() != f'>target\n{run.target.sequence}\n':
            raise ValueError('Frozen target sequence changed')
        transfers = ingest(run)
        if retry:
            run.retry_failed(engine=engine)
        unresolved = run.diagnostics()
        try:
            if any(d.state in (State.SUBMITTED, State.UNKNOWN) and not d.resumable for d in unresolved):
                unresolved.close()
                return run.finalize(), transfers, None
        finally:
            unresolved.close()
        for current in run.settings.engines:
            if current is Engine.BOLTZ:
                if any(d.engine is Engine.PSICHIC and d.state not in (State.SUCCEEDED, State.FAILED) for d in run.diagnostics()):
                    break
                if run.settings.pyramid:
                    run.select()
            if run.pending(current, limit=1):
                stage = prepare_stage(run, runtime, screen, current)
                return run.finalize(), transfers, stage
        return run.finalize(), transfers, None


def reconcile(run, attempt, resolution, evidence, job_id):
    with exclusive(run.path / 'driver.lock'):
        read(run)
        task = run.task(attempt)
        directory = run.path / 'local/invocations' / task.invocation_id
        if resolution == Resolution.STOPPED.value:
            for name in ('owner', 'child', 'guardian', 'external-started'):
                path = directory / (name + '.json')
                if path.exists() and alive(loads(path.read_bytes())) is not False:
                    raise ValueError('A matching worker is live or uninspectable; no redispatch')
            if directory.exists() and (lock_held(directory / 'owner.lock') or lock_held(directory / 'external.lock')):
                raise ValueError('Invocation is still owned; no redispatch')
        record = {'attempt_id': attempt, 'invocation_id': task.invocation_id, 'resolution': resolution,
                  'scheduler_job_id': job_id, 'evidence': evidence}
        import uuid
        atomic_pointer(run.path / 'screening/reconciliations' / (uuid.uuid4().hex + '.json'), record)
        run.reconcile(attempt, resolution, evidence=f'Native scheduler job {job_id}: {evidence}')


def stages(run):
    screen = read(run)
    latest = heapq.nlargest(20, (run.path / 'screening/stages').iterdir(), key=lambda p: p.stat().st_mtime_ns)
    for path in reversed(latest):
        if not (path / 'prepared.json').exists():
            yield {'directory': str(path), 'state': 'preparation-incomplete', 'next_action': 'Inspect native preparation logs; no automatic redispatch.'}
            continue
        value = loads((path / 'prepared.json').read_bytes())
        status = loads((path / 'workflow/status.json').read_bytes())
        units = len(status['workunits'])
        yield {**value, 'directory': str(path), 'workunits': units,
               'submission': [screen['afvs_python'], 'afvs_submit_jobs.py', '1', str(units)],
               'submission_directory': str(path / 'tools'), 'submission_is_explicit': True,
               'scheduler_status': str(path / 'workflow/status.json')}
