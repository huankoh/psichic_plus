import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess

import pandas as pd
import pytest

pytest.importorskip('pyarrow')

from psichic_plus_adaptive import Failure, Success, UPSTREAM_REVISION
from psichic_plus_adaptive.library import collection_key, convert_library, read_library
from psichic_plus_adaptive.workflow import checked_upstream, local_environment, reconcile, select_primary
from psichic_plus_adaptive import worker

EXAMPLE = Path(__file__).resolve().parents[1] / 'psichic_plus_adaptive' / 'example' / 'library.csv'


def library_file(tmp_path, rows, columns=('collection_id', 'molecule_id', 'tranche', 'smiles')):
    path = tmp_path / 'library.csv'
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def test_identifiers_are_strings_and_tokens_are_globally_unique(tmp_path):
    path = library_file(tmp_path, [('001', '0001', 'one', 'CCO'), ('002', '0001', 'two', 'invalid')])
    molecules = read_library(path)
    assert [(m.collection_id, m.molecule_id) for m in molecules] == [('001', '0001'), ('002', '0001')]
    assert len({m.token for m in molecules}) == 2
    assert all(m.token.isalnum() and len(m.token) == 65 for m in molecules)
    assert read_library(path) == molecules


@pytest.mark.parametrize('rows, message', [
    ([('a', 'm', 'x', 'CCO'), ('a', 'm', 'x', 'CCC')], 'Duplicate'),
    ([('a', 'm', 'x', 'CCO'), ('a', 'n', 'y', 'CCC')], 'multiple tranches'),
    ([('', 'm', 'x', 'CCO')], 'Empty'),
    ([('a', 'm', 'x', 'CCO\nCCC')], 'one line'),
    ([], 'empty'),
])
def test_rejects_duplicate_identities_and_bad_grouping(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        read_library(library_file(tmp_path, rows))


def test_requires_all_csv_columns(tmp_path):
    with pytest.raises(ValueError, match='requires columns'):
        read_library(library_file(tmp_path, [('a', 'm', 'CCO')], ('collection_id', 'molecule_id', 'smiles')))


@pytest.fixture
def upstream():
    source = os.environ.get('AFVS_UPSTREAM')
    if not source:
        pytest.skip('Set AFVS_UPSTREAM to the clean pinned upstream checkout')
    return Path(source)


@pytest.fixture
def patched(tmp_path, upstream):
    pytest.importorskip('boto3')
    pytest.importorskip('IPython')
    return checked_upstream(upstream, tmp_path / 'upstream')


def load_upstream(patched):
    spec = importlib.util.spec_from_file_location('adaptive_test_upstream', patched / 'tools/templates/afvs_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_upstream_unpacks_full_and_sparse_archives(tmp_path, patched):
    import tarfile

    molecules = read_library(EXAMPLE)
    library = tmp_path / 'converted'
    samples = convert_library(molecules, library, 2)
    module = load_upstream(patched)
    for mode, folder in [('all', 'full'), ('prescreen_mode', 'sparse')]:
        observed = set()
        for index, archive in enumerate((library / folder).glob('*/*.tar.gz')):
            temp = tmp_path / f'{folder}-{index}'
            temp.mkdir()
            with tarfile.open(archive) as handle:
                assert any(m.name.endswith('/.listing') for m in handle.getmembers())
            item = {'temp_dir': str(temp), 'local_path': str(archive), 'collection_key': 'base',
                    'collection': {'collection_number': archive.name.removesuffix('.tar.gz'),
                                   'mode': mode, 'prescreen_ligands_per_tranche': 2}}
            previous = Path.cwd()
            try:
                unpacked = module.unpack_item(item)
            finally:
                os.chdir(previous)
            for token, info in unpacked.items():
                molecule = next(m for m in molecules if m.token == token)
                expected_key = collection_key(molecule) if mode == 'prescreen_mode' else 'base'
                assert info['collection_key'] == expected_key
                observed.add(token)
        assert observed == (set(samples) if mode == 'prescreen_mode' else {m.token for m in molecules})
    manifest = pd.read_csv(library / 'manifest.csv', dtype=str, keep_default_na=False)
    assert manifest.collection_id.tolist() == ['001'] * 3 + ['002'] * 3
    assert pd.read_parquet(library / 'collections.parquet').LigandCount.sum() == 6


def test_worker_reuses_predictor_and_completes_partial_batches(tmp_path, patched, monkeypatch):
    from psichic_plus_adaptive import predictor

    calls = []

    class StubPredictor:
        def __init__(self, **config):
            calls.append('init')

        def score(self, inputs):
            calls.append([token for token, _ in inputs])
            return [Failure(token, 'ligand_parse', 'invalid') if smiles == 'invalid' else Success(token, 7., -7.)
                    for token, smiles in inputs]

    monkeypatch.setattr(predictor, 'Predictor', StubPredictor)
    monkeypatch.setattr(worker, '_predictor', None)
    config = tmp_path / 'config.txt'
    config.write_text('{}')
    module = load_upstream(patched)
    scenario = {'program': 'external_batch', 'key': 'psichic'}
    completed = queue.Queue()
    for batch in [[('a', 'CCO'), ('b', 'invalid')], [('c', 'CCC')]]:
        items = []
        for token, smiles in batch:
            path = tmp_path / f'{token}.smi'
            path.write_text(f'{smiles}\t{token}\n')
            output = tmp_path / token
            output.mkdir()
            items.append({'ligand_key': token, 'ligand_path': str(path), 'output_dir': str(output),
                          'config_path': str(config), 'base_collection_key': 'base',
                          'collection_key': 't_c', 'scenario_key': 'psichic', 'attrs': {}})
        module.docking_process_batch(completed, scenario, items, str(tmp_path))
    assert calls == ['init', ['a', 'b'], ['c']]
    first, second = completed.get_nowait(), completed.get_nowait()
    assert first['type'] == second['type'] == 'docking_complete'
    assert [item['status'] for item in first['items']] == ['success', 'failed']
    assert first['items'][0]['score'] == -7.
    assert first['items'][0]['attrs']['raw_affinity'] == 7.
    assert first['items'][1]['log']['stage'] == 'ligand_parse'
    assert [item['ligand_key'] for item in second['items']] == ['c']


def test_predictor_initialization_error_completes_every_item(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, '_predictor', None)
    inputs = []
    for token in ['a', 'b']:
        path = tmp_path / f'{token}.smi'
        path.write_text(f'CCO\t{token}\n')
        inputs.append({'ligand_key': token, 'ligand_path': str(path), 'config_path': str(tmp_path / 'missing'),
                       'attrs': {}, 'log': {}, 'log_path': str(tmp_path / f'{token}.log')})
    worker.score_items(inputs)
    assert [item['status'] for item in inputs] == ['failed', 'failed']
    assert all(item['log']['stage'] == 'predictor_init' for item in inputs)


def prescreen_frame(molecules, affinities):
    return pd.DataFrame([{'token': m.token, 'collection_id': m.collection_id, 'tranche': m.tranche,
                          'status': 'failed' if affinity is None else 'success',
                          'raw_affinity': affinity, 'adaptive_score': None if affinity is None else -affinity}
                         for m, affinity in zip(molecules, affinities)])


def test_real_selector_uses_negative_mean_and_keeps_generated_todo(tmp_path, patched):
    molecules = read_library(EXAMPLE)
    library = tmp_path / 'library'
    convert_library(molecules, library, 2)
    scores = prescreen_frame([molecules[0], molecules[1], molecules[3], molecules[4]], [9., 7., 3., None])
    todo = select_primary(patched, library, scores, 2, tmp_path, local_environment(tmp_path))
    assert todo.read_text() == f'{collection_key(molecules[0])} 3\n'
    selection = json.loads((tmp_path / 'selection.json').read_text())
    assert selection['primary_target'] == 2
    assert selection['selected_molecules'] == 3
    assert 'tranche_ave_score' in (tmp_path / 'commands.log').read_text()


def test_real_selector_excludes_unscored_tranches(tmp_path, patched):
    molecules = read_library(EXAMPLE)
    library = tmp_path / 'library'
    convert_library(molecules, library, 1)
    scores = prescreen_frame([molecules[0], molecules[3]], [None, 3.])
    todo = select_primary(patched, library, scores, 3, tmp_path, local_environment(tmp_path))
    assert todo.read_text() == f'{collection_key(molecules[3])} 3\n'
    assert json.loads((tmp_path / 'selection.json').read_text())['excluded_unscored_collections'] == 1
    with pytest.raises(ValueError, match='unreachable'):
        select_primary(patched, library, scores, 4, tmp_path, {})


@pytest.mark.parametrize('affinities, target, message', [
    ([None, None], 1, 'All prescreen'), ([3., None], 4, 'unreachable'), ([-3., None], 1, 'negative'),
    ([float('inf'), None], 1, 'finite'),
])
def test_selector_rejects_invalid_scores_and_unreachable_target(tmp_path, affinities, target, message):
    molecules = read_library(EXAMPLE)
    library = tmp_path / 'library'
    convert_library(molecules, library, 1)
    scores = prescreen_frame([molecules[0], molecules[3]], affinities)
    with pytest.raises(ValueError, match=message):
        select_primary(tmp_path, library, scores, target, tmp_path, {})
    assert not (tmp_path / 'primary.todo').exists()


def test_incomplete_workflow_exports_missing_tokens_as_failures(tmp_path):
    expected = read_library(EXAMPLE)[:2]
    stage = tmp_path / 'prescreen'
    stage.mkdir()
    with pytest.raises(RuntimeError, match='Incomplete'):
        reconcile(stage, expected, {'1': {'subjobs': {'0': {}}}})
    result = pd.read_csv(stage / 'results.csv', dtype=str, keep_default_na=False)
    assert result.token.tolist() == [m.token for m in expected]
    assert result.status.tolist() == ['failed', 'failed']
    assert result.failure_stage.tolist() == ['completion', 'completion']
    assert (stage / 'incomplete.json').is_file()


def test_reconciliation_joins_queue_success_and_failure_without_losing_ids(tmp_path):
    expected = read_library(EXAMPLE)[:2]
    overview = tmp_path / 'output-files/summary/1/0.json.gz'
    overview.parent.mkdir(parents=True)
    with gzip.open(overview, 'wt') as handle:
        json.dump({'total_dockings': 2, 'failed_downloads': 0, 'skipped_ligands': 0,
                   'failed_list': [{'ligand_key': expected[1].token, 'collection_key': collection_key(expected[1]),
                                    'reason': 'invalid', 'stage': 'ligand_parse'}]}, handle)
    parquet = tmp_path / 'output-files/psichic/parquet/1/0.parquet'
    parquet.parent.mkdir(parents=True)
    pd.DataFrame([{'ligand': expected[0].token, 'collection_key': collection_key(expected[0]),
                   'score_min': -8., 'attr_raw_affinity': 8.}]).to_parquet(parquet)
    result = reconcile(tmp_path, expected, {'1': {'subjobs': {'0': {}}}})
    assert result.status.tolist() == ['success', 'failed']
    assert result.collection_id.tolist() == ['001', '001']
    assert result.molecule_id.tolist() == ['0001', '0002']
    assert result.raw_affinity.iloc[0] == 8.
    with pytest.raises(RuntimeError, match='Incomplete'):
        reconcile(tmp_path, expected, {'1': {'subjobs': {'0': {}}}}, 'Worker exited with 1')
    pd.DataFrame([{'ligand': expected[0].token, 'collection_key': collection_key(expected[0]),
                   'score_min': 8., 'attr_raw_affinity': 8.}]).to_parquet(parquet)
    with pytest.raises(RuntimeError, match='Incomplete'):
        reconcile(tmp_path, expected, {'1': {'subjobs': {'0': {}}}})


def test_checked_patch_rejects_wrong_revision(tmp_path):
    source = tmp_path / 'wrong'
    subprocess.run(['git', 'init', str(source)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test', '-c', 'user.email=test@example.org',
                    'commit', '--allow-empty', '-m', 'Test'], check=True, capture_output=True)
    with pytest.raises(ValueError, match=UPSTREAM_REVISION):
        checked_upstream(source, tmp_path / 'copy')
    assert not (tmp_path / 'copy').exists()


@pytest.mark.parametrize('corruption', ['missing_column', 'duplicate', 'wrong_collection', 'unknown_token'])
def test_reconciliation_rejects_corrupt_predictions(tmp_path, corruption):
    molecule = read_library(EXAMPLE)[0]
    overview = tmp_path / 'output-files/summary/1/0.json.gz'
    overview.parent.mkdir(parents=True)
    with gzip.open(overview, 'wt') as handle:
        json.dump({'total_dockings': 1, 'failed_downloads': 0, 'skipped_ligands': 0, 'failed_list': []}, handle)
    parquet = tmp_path / 'output-files/psichic/parquet/1/0.parquet'
    parquet.parent.mkdir(parents=True)
    row = {'ligand': molecule.token, 'collection_key': collection_key(molecule),
           'score_min': -8., 'attr_raw_affinity': 8.}
    if corruption == 'missing_column':
        del row['attr_raw_affinity']
    elif corruption == 'wrong_collection':
        row['collection_key'] = 'wrong'
    elif corruption == 'unknown_token':
        row['ligand'] = 'unknown'
    pd.DataFrame([row] * (2 if corruption == 'duplicate' else 1)).to_parquet(parquet)
    with pytest.raises(RuntimeError, match='Incomplete'):
        reconcile(tmp_path, [molecule], {'1': {'subjobs': {'0': {}}}})
    assert len(pd.read_csv(tmp_path / 'results.csv')) == 1
