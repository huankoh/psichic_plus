"""Pinned native preparation and staged sharedfs receipt import."""

import csv
import hashlib
from importlib.resources import files
import io
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import uuid

from psichic_adaptiveflow.install import REVISION, TOUCHED

from applications.scoring.config import ConfigurationError
from applications.scoring.contracts import Acceptance, Engine, Ownership, canonical, digest, loads
from applications.scoring.evidence import Artifact, Receipt, atomic_pointer, safe_path, write_file
from applications.scoring.executor import prepare_invocation
from applications.scoring.runtime import checksum, identity


def copy_tools(source, destination):
    source, destination = Path(source), Path(destination)
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != REVISION:
        raise ConfigurationError('AFVS_REVISION', f'Use pinned AdaptiveFlow {REVISION}')
    patch = files('psichic_adaptiveflow').joinpath('native.patch').read_bytes()
    with tempfile.TemporaryDirectory(prefix='afvs-patch-check-') as temporary:
        expected = Path(temporary)
        originals = {}
        for name in TOUCHED:
            originals[name] = subprocess.check_output(['git', '-C', str(source), 'show', f'{REVISION}:{name}'])
            target = expected / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(originals[name])
        subprocess.run(['git', 'apply', '-'], input=patch, cwd=expected, check=True, capture_output=True,
                       env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(expected.parent)))
        for name in TOUCHED:
            if (source / name).read_bytes() not in (originals[name], (expected / name).read_bytes()):
                raise ConfigurationError('AFVS_SOURCE', f'Unexpected native patch target changes: {name}')
        for name in ('tools/afvs_prepare_workunits.py', 'tools/afvs_submit_jobs.py'):
            if (source / name).read_bytes() != subprocess.check_output(['git', '-C', str(source), 'show', f'{REVISION}:{name}']):
                raise ConfigurationError('AFVS_SOURCE', f'Pinned native interface changed: {name}')
        shutil.copytree(source / 'tools', destination / 'tools', ignore=shutil.ignore_patterns('__pycache__'))
        for name in TOUCHED:
            shutil.copyfile(expected / name, destination / name)


def control(path, config):
    with path.open('x') as handle:
        for name, value in config.items():
            if isinstance(value, (str, int, float)):
                if '\n' in str(value) or '\r' in str(value):
                    raise ConfigurationError('NATIVE_CONFIG', f'Newline in native setting {name}')
                handle.write(f'{name}={value}\n')
            elif isinstance(value, list):
                separator = ',' if name in ('summary_formats', 'print_attrs_in_summary') else ':'
                handle.write(f'{name}={separator.join(map(str, value))}\n')


def prepare_stage(run, runtime, screen, engine):
    stage = run.path / 'screening/stages' / (engine.value + '-' + uuid.uuid4().hex)
    stage.mkdir()
    shutil.copytree(run.path / 'screening/template/tools', stage / 'tools')
    dispatch, library = stage / 'dispatch', stage / 'library'
    dispatch.mkdir()
    library.mkdir()
    scenario = stage / 'input-files/score'
    scenario.mkdir(parents=True)
    write_file(scenario / 'config.txt', canonical({
        'schema_version': 1, 'runtime_python': sys.executable,
        'runtime_path': str(Path(__file__).resolve().parents[2]),
        'dispatch_root': str(dispatch), 'engine': engine.value}) + '\n')
    selected = stage / 'selected.csv'
    database = sqlite3.connect(stage / 'coverage.sqlite3')
    database.execute('CREATE TABLE expected(collection TEXT,ligand TEXT,attempt TEXT,seen INTEGER DEFAULT 0,PRIMARY KEY(collection,ligand))')
    count = 0
    try:
        with selected.open('x', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['collection_name', 'collection_number', 'ligand'])
            while count < screen['stage_size']:
                rows = run.pending(engine, limit=min(runtime['chunk_size'], screen['stage_size'] - count))
                if not rows:
                    break
                invocation = prepare_invocation(run, runtime, engine, rows, ownership=Ownership.EXTERNAL)
                request = loads((invocation / 'request.json').read_bytes())
                manifest = {'schema_version': 1, 'run_id': run.run_id, 'population_digest': run.population_digest,
                            'engine': engine.value, 'invocation': str(invocation), 'request_sha256': checksum(invocation / 'request.json'),
                            'rows': [row.to_dict() for row in rows]}
                data = (canonical(manifest) + '\n').encode()
                sha = hashlib.sha256(data).hexdigest()
                write_file(dispatch / (sha + '.json'), data.decode())
                name, number = 'SC' + sha, '0000001'
                folder = library / name
                folder.mkdir()
                with tarfile.open(folder / (number + '.tar.gz'), 'x:gz') as archive:
                    for row, task in zip(rows, request['tasks'], strict=True):
                        payload = (row.entry.smiles + '\n').encode()
                        member = tarfile.TarInfo(number + '/' + row.key + '.smi')
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
                        writer.writerow([name, number, row.key])
                        database.execute('INSERT INTO expected(collection,ligand,attempt) VALUES (?,?,?)',
                                         (name + '_' + number, row.key, task['attempt_id']))
                count += len(rows)
        config = dict(screen['native_config'])
        config.update(job_name='screen' + stage.name.split('-')[-1], data_storage_mode='sharedfs', job_storage_mode='sharedfs',
                      data_collection_addressing_mode='metatranche', data_collection_identifier='',
                      collection_folder=str(library), collection_list_file=str(selected), collection_list_type='csv_collection_key_ligand',
                      docking_scenario_names='score', docking_scenario_methods='boltz2' if engine is Engine.BOLTZ else 'psichic_plus',
                      docking_scenario_replicas=1, docking_scenario_batchsizes=1, docking_scenario_basefolder=str(scenario.parent),
                      threads_to_use=1, ligand_library_format='smi', prescreen_mode=0, run_atom_check=0, dynamic_tranche_filtering=0,
                      print_attrs_in_summary='smi,psichic_affinity,boltz_affinity,binder_probability', summary_formats='csv.gz')
        for deprecated, replacement in (('object_store_job_addressing_mode', 'job_addressing_mode'),):
            if deprecated in config and replacement not in config:
                config[replacement] = config[deprecated]
        for deprecated in ('object_store_data_collection_addressing_mode', 'object_store_data_collection_identifier',
                           'job_letter', 'object_store_job_addressing_mode'):
            config.pop(deprecated, None)
        ctrl = stage / 'tools/templates/all.ctrl'
        ctrl.unlink()
        control(ctrl, config)
        for script in ('afvs_prepare_folders.py', 'afvs_prepare_workunits.py'):
            with (stage / (script + '.log')).open('xb') as log:
                result = subprocess.run([screen['afvs_python'], script], cwd=stage / 'tools',
                                        stdout=log, stderr=subprocess.STDOUT, timeout=300,
                                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            if result.returncode:
                raise ConfigurationError('NATIVE_PREPARATION', f'Native preparation failed. Inspect {stage / (script + ".log")}')
        for path in (stage / 'workflow/workunits').glob('*.tar.gz'):
            with tarfile.open(path) as archive:
                packed = loads(archive.extractfile('af_input/config.json').read())
            for subjob, value in packed['subjobs'].items():
                for collection, item in value['collections'].items():
                    for ligand in item['ligands']:
                        found = database.execute('SELECT seen FROM expected WHERE collection=? AND ligand=?', (collection, ligand)).fetchone()
                        if found is None or found[0]:
                            raise ValueError('Native preparation duplicated or changed a selected identity')
                        database.execute('UPDATE expected SET seen=1 WHERE collection=? AND ligand=?', (collection, ligand))
            write_file(stage / (path.stem + '.sha256'), checksum(path) + '\n')
        if database.execute('SELECT 1 FROM expected WHERE seen=0 LIMIT 1').fetchone():
            raise ValueError('Native preparation omitted a selected identity')
        database.commit()
        atomic_pointer(stage / 'prepared.json', {'schema_version': 1, 'entries': count, 'engine': engine.value,
                                                'run_id': run.run_id, 'population_digest': run.population_digest})
    finally:
        database.close()
    return stage


def validate_runtime_receipt(run, receipt, transfer):
    request = loads((run.path / 'local/invocations' / receipt.task.invocation_id / 'request.json').read_bytes())
    roles = {a.role: a for a in receipt.artifacts}
    required = {'loaded_runtime', 'runtime_inventory', 'native_mapping'}
    if receipt.task.engine is Engine.BOLTZ and request['profile'].get('native_generation'):
        required.add('native_builds')
    if not required <= roles.keys():
        raise ValueError('Native receipt is missing required qualified runtime evidence')
    evidence = {}
    for name in required:
        artifact = roles[name]
        if Artifact.from_file(transfer, name, artifact.path) != artifact:
            raise ValueError('Transferred runtime artifact digest/size mismatch')
        evidence[name] = loads((transfer / artifact.path).read_bytes())
    if (evidence['runtime_inventory'] != request['profile'] or evidence['native_mapping'] != request['mapping']
            or evidence['loaded_runtime']['profile_digest'] != digest(identity(request['profile']))
            or not evidence['loaded_runtime'].get('loaded')):
        raise ValueError('Native receipt runtime evidence differs from the prepared qualified invocation')


def ingest(run):
    results = {state.value: 0 for state in Acceptance}
    imported = run.path / 'screening/imports'
    for stage in (run.path / 'screening/stages').iterdir():
        for path in (stage / 'output-files/score/logs').glob('*/*.tar.gz'):
            owned = imported / uuid.uuid4().hex
            owned.mkdir()
            snapshot = owned / 'native.tar.gz'
            shutil.copyfile(path, snapshot)
            retained = imported / checksum(snapshot)
            if retained.exists():
                shutil.rmtree(owned)
            else:
                owned.rename(retained)
            marker = retained / 'imported.json'
            if marker.exists():
                previous = loads(marker.read_bytes())
                for name, count in previous.items():
                    results[Acceptance.IDENTICAL.value if name == Acceptance.ACCEPTED.value else name] += count
                continue
            accepted = accept_archive(run, retained)
            atomic_pointer(marker, accepted)
            for name, count in accepted.items():
                results[name] += count
    return results


def accept_archive(run, retained):
    results = {state.value: 0 for state in Acceptance}
    unpacked = retained / 'files'
    if unpacked.exists():
        shutil.rmtree(unpacked)
    unpacked.mkdir()
    try:
        with tarfile.open(retained / 'native.tar.gz', 'r|gz') as archive:
            for member in archive:
                relative = safe_path(member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError('Native transfer contains a nonregular artifact')
                destination = unpacked / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open('xb') as output:
                    shutil.copyfileobj(source, output)
    except (EOFError, tarfile.ReadError):
        results[Acceptance.INCOMPLETE.value] += 1
        return results
    for path in unpacked.rglob('receipt.json'):
        receipt = Receipt.from_json(path.read_bytes())
        expected = run.task(receipt.task.attempt_id)
        if receipt.task != expected or receipt.task.ownership is not Ownership.EXTERNAL:
            raise ValueError('Transferred receipt differs from its exact external task')
        # A short direct sharedfs copy can contain a header before its artifact bytes arrive.
        if any(not (path.parent / a.path).is_file() or (path.parent / a.path).stat().st_size < a.size for a in receipt.artifacts):
            results[Acceptance.INCOMPLETE.value] += 1
            continue
        validate_runtime_receipt(run, receipt, path.parent)
        results[run.accept(path.parent).value] += 1
    return results
