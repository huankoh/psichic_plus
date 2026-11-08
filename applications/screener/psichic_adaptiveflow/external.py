"""Python 3.10 native boundary. The isolated runtime owns inference and receipts."""

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess

METHODS = ('psichic_plus', 'boltz2')


def configuration(path):
    value = json.loads(Path(path).read_text())
    if set(value) != {'schema_version', 'runtime_python', 'runtime_path', 'dispatch_root', 'engine'} or value['schema_version'] != 1:
        raise ValueError('Unsupported external scoring configuration')
    for key in ('runtime_python', 'runtime_path', 'dispatch_root'):
        if not Path(value[key]).is_absolute():
            raise ValueError(f'{key} must be an explicit absolute worker-visible path')
    if not Path(value['runtime_python']).is_file() or value['engine'] not in ('psichic', 'boltz'):
        raise ValueError('Missing qualified runtime executable or invalid engine')
    return value


def mapping(config, collection):
    match = re.fullmatch(r'SC([0-9a-f]{64})_0000001', collection)
    if not match:
        raise ValueError('Expected a digest-bound generated collection')
    path = Path(config['dispatch_root']) / (match[1] + '.json')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != match[1]:
        raise ValueError('Native row map digest mismatch')
    value = json.loads(data)
    if value['schema_version'] != 1 or value['engine'] != config['engine']:
        raise ValueError('Native row map schema or engine mismatch')
    return path, value


def register(programs):
    from .adapter import complete_batch as legacy_batch

    programs['psichic_plus'] = {'ligands': 'batch', 'native_batch': legacy_batch}
    programs['boltz2'] = {'ligands': 'batch', 'native_batch': complete_batch}


def complete_batch(batch, scenario, scorers, setup, clean, temp_dir):
    config = configuration(scenario['config'])
    for item in batch['items']:
        item['log'].update(scenario_key=item['scenario_key'], replica_index=item['replica_index'], stage='external-worker')
        try:
            setup(item, 'single', temp_dir)
            if item['replica_index'] != 0 or scenario['replicas'] != 1:
                raise ValueError('External scoring requires replica zero only')
            if item['attrs'].get('psichic_input_failure'):
                raise ValueError(item['attrs']['psichic_input_failure'])
            path, manifest = mapping(config, item['collection_key'])
            row = next((r for r in manifest['rows'] if r['key'] == item['ligand_key']), None)
            if row is None or Path(item['ligand_path']).read_text() != row['smiles'] + '\n':
                raise ValueError('Native selected identity or exact SMILES differs from frozen map')
            output = Path(item['output_dir'])
            env = dict(os.environ, PYTHONPATH=config['runtime_path'], PYTHONDONTWRITEBYTECODE='1')
            env.pop('PYTHONHOME', None)
            command = [config['runtime_python'], '-B', '-m', 'applications.screener.worker', str(path),
                       item['ligand_key'], str(output)]
            with (output / 'bridge.stdout').open('xb') as stdout, (output / 'bridge.stderr').open('xb') as stderr:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                           env=env, start_new_session=True)
                (output / 'bridge.json').write_text(json.dumps({'command': command, 'pid': process.pid}))
                try:
                    process.stdin.write(b'GO\n')
                    process.stdin.flush()
                    try:
                        code = process.wait(timeout=item['timeout'])
                    except subprocess.TimeoutExpired:
                        process.send_signal(signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                        raise TimeoutError('Qualified invocation deadline expired; inspect retained invocation stderr')
                finally:
                    process.stdin.close()
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
            if code:
                raise RuntimeError(f'Qualified worker exit {code}; see bridge.stderr and invocation logs')
            receipt = json.loads((output / 'receipt' / 'receipt.json').read_text())
            record = receipt['record']
            if record['status'] != 'succeeded':
                item['log'].update(stage=record['stage'], reason=record['reason'])
            else:
                item['status'] = 'success'
                if config['engine'] == 'psichic':
                    item['score'] = record['adaptive_score']
                    item['attrs']['psichic_affinity'] = record['raw_affinity']
                    item['log']['distribution'] = record['distribution']
                else:
                    item['score'] = record['affinity_pred_value']
                    item['attrs']['boltz_affinity'] = record['affinity_pred_value']
                    item['attrs']['binder_probability'] = record['affinity_probability_binary']
                item['attrs']['smi'] = row['smiles']
        except Exception as error:
            item['log']['reason'] = str(error)
        finally:
            if 'tmp_run_dir' in item:
                clean(item)
            Path(item['log_path']).write_text(json.dumps(dict(item['log'], status=item['status']), allow_nan=False) + '\n')
    batch.update(type='docking_complete', execution_type='single')
