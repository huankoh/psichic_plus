import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import pytest

from psichic_adaptiveflow.adapter import Affinity, Failure, LigandKey, ScoreInput, Scorer, Target, unpack_collection

FAULT_CHILD = Path(__file__).with_name('fault_child.py')


def entries(count=2, smiles='CCO'):
    return [ScoreInput(LigandKey(f'AA_{index}', 'same'), index, Path('unused'), smiles) for index in range(count)]


@pytest.mark.parametrize('mode', ['before-read', 'initialize', 'predict', 'partial', 'eof', 'bad-position', 'nan'])
def test_failed_batch_completes_and_next_batch_recovers(mode, tmp_path):
    target = Target(tmp_path / 'target.fasta', tmp_path, 'cpu')
    scorer = Scorer(target, 1, [sys.executable, str(FAULT_CHILD), mode])
    started = time.monotonic()
    batch = entries(1 if mode == 'nan' else 2, 'C' * 1_000_000 if mode == 'before-read' else 'CCO')
    failed = scorer.score(batch, 0.4)
    assert len(failed) == len(batch)
    assert all(isinstance(result, Failure) and result.reason for result in failed)
    assert time.monotonic() - started < 2.5
    assert scorer.process is None
    scorer.command = [sys.executable, str(FAULT_CHILD), 'success']
    try:
        assert scorer.score(entries(), 5) == [Affinity(5), Affinity(6)]
        pid = scorer.process.pid
        assert scorer.score(entries(), 5) == [Affinity(5), Affinity(6)]
        assert scorer.process.pid == pid
    finally:
        scorer.close()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_hung_child_ignoring_term_is_killed_and_reaped(tmp_path):
    pidfile = tmp_path / 'pid'
    scorer = Scorer(Target(tmp_path, tmp_path, 'cpu'), 1, [sys.executable, str(FAULT_CHILD), 'before-read', str(pidfile)])
    assert isinstance(scorer.score(entries(), 0.5)[0], Failure)
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_parent_death_stops_real_inference_child(tmp_path):
    pidfile = tmp_path / 'pid'
    code = """
import os, subprocess, sys, time
from pathlib import Path
child_code = "from psichic_adaptiveflow._child import watch_parent; import os,sys,time; from pathlib import Path; watch_parent(); Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
child = subprocess.Popen([sys.executable, '-c', child_code, sys.argv[1]], env=dict(os.environ, PSICHIC_AFVS_PARENT_PID=str(os.getpid())), start_new_session=True)
for _ in range(100):
    if Path(sys.argv[1]).exists():
        break
    time.sleep(0.02)
else:
    raise RuntimeError('Child did not become ready')
"""
    parent = subprocess.Popen([sys.executable, '-c', code, str(pidfile)], stderr=subprocess.DEVNULL)
    parent.wait(timeout=5)
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
        if not status or status.startswith('Z'):
            break
        time.sleep(0.1)
    else:
        os.killpg(pid, 9)
        pytest.fail('Inference child outlived parent')


def archive_item(tmp_path, files, mode='all', listing=None):
    archive = tmp_path / 'input.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        for name, content in files + ([('1/.listing', listing)] if listing is not None else []):
            data = content.encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            handle.addfile(member, io.BytesIO(data))
    return {'local_path': str(archive), 'temp_dir': str(tmp_path / 'unpack'), 'collection_key': 'sparse_1', 'collection': {'collection_number': '1', 'mode': mode, 'prescreen_ligands_per_tranche': 1}}


def test_native_sparse_listing_preserves_original_identifiers(tmp_path):
    item = archive_item(tmp_path, [('1/original.smi', 'CCO\n'), ('1/other.smi', 'CCN\n')], 'prescreen_mode', 'AB_0000012,original,0\nAB_0000012,other,1\n')
    result = unpack_collection(item)
    assert list(result) == ['original']
    assert result['original']['collection_key'] == 'AB_0000012'
    assert Path(result['original']['path']).read_text() == 'CCO\n'


@pytest.mark.parametrize('files', [[('1/same.smi', 'CCO'), ('1/same.smi', 'CCN')], [('1/same.smi', 'CCO'), ('1/same.extra.smi', 'CCN')]])
def test_ambiguous_archive_ids_fail_before_overwrite(tmp_path, files):
    item = archive_item(tmp_path, files)
    result = unpack_collection(item)
    assert result['same']['failure'].startswith('Invalid native SMILES archive: Ambiguous duplicate')
    assert not (tmp_path / 'unpack' / '1' / 'same.smi').exists()


def test_archive_cannot_extract_outside_collection(tmp_path):
    item = archive_item(tmp_path, [('1/../../escape.smi', 'CCO')])
    assert all('failure' in ligand for ligand in unpack_collection(item).values())
    assert not (tmp_path / 'escape.smi').exists()


def test_native_stdout_diagnostics_cannot_corrupt_response(tmp_path):
    code = """
from psichic_adaptiveflow._child import protocol
import os, sys
os.write(1, b'native model diagnostics\\n')
sys.stdin.readline()
protocol.write(b'[{"position":0,"affinity":7.5}]\\n')
"""
    scorer = Scorer(Target(tmp_path, tmp_path, 'cpu'), 1, [sys.executable, '-c', code])
    try:
        assert scorer.score(entries(1), 5) == [Affinity(7.5)]
    finally:
        scorer.close()


def test_checked_installer_is_exact_and_refuses_dirty_targets(tmp_path):
    from psichic_adaptiveflow.install import install
    source = os.environ.get('AFVS_TEST_UPSTREAM')
    if not source:
        pytest.skip('Set AFVS_TEST_UPSTREAM to a pinned AFVS checkout')
    checkout = tmp_path / 'afvs'
    subprocess.run(['git', 'clone', '--quiet', '--local', source, str(checkout)], check=True)
    assert install(checkout) == 'applied'
    assert install(checkout) == 'already applied'
    ctrl = checkout / 'tools/templates/all.ctrl'
    ctrl.write_text(ctrl.read_text().replace('docking_scenario_methods=qvina02', 'docking_scenario_methods=psichic_plus'))
    preparation = subprocess.run([sys.executable, 'afvs_prepare_folders.py'], cwd=checkout / 'tools', capture_output=True, text=True)
    assert preparation.returncode != 0
    assert 'requires native per-ligand .smi archives' in preparation.stderr
    assert not (checkout / 'workflow').exists()
    runner = checkout / 'tools/templates/afvs_run.py'
    runner.write_text(runner.read_text() + '\n# unrelated edit\n')
    with pytest.raises(ValueError, match='additional changes'):
        install(checkout)
    subprocess.run(['git', '-C', str(checkout), 'add', 'tools/templates/afvs_run.py'], check=True)
    with pytest.raises(ValueError, match='staged changes'):
        install(checkout)


def test_target_paths_resolve_at_scenario_and_are_rechecked_on_worker(tmp_path, monkeypatch):
    scenario = tmp_path / 'scenario'
    scenario.mkdir()
    (scenario / 'target.fasta').write_text('>protein\nMKTAYIAKQRQISFVKSHFSRQDILD\n')
    model = tmp_path / 'weights'
    model.mkdir()
    for name in ('config.json', 'degree.pt', 'model.pt'):
        (model / name).write_text('fixture')
    config = scenario / 'config.txt'
    config.write_text(f'protein_fasta=target.fasta\nmodel_dir={model}\ndevice=cpu\n')
    monkeypatch.chdir(tmp_path)
    target = Target.read(config)
    assert target.protein_fasta == scenario / 'target.fasta'
    assert target.model_dir == model
    (model / 'model.pt').unlink()
    with pytest.raises(ValueError, match='Missing model file on this machine'):
        Target.read(config)


def test_shutdown_kills_group_when_descendant_ignores_term(tmp_path):
    pidfile = tmp_path / 'descendant'
    scorer = Scorer(Target(tmp_path, tmp_path, 'cpu'), 1, [sys.executable, str(FAULT_CHILD), 'descendant', str(pidfile)])
    try:
        assert isinstance(scorer.score(entries(), 0.5)[0], Failure)
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
            if not status or status.startswith('Z'):
                break
            time.sleep(0.05)
        else:
            os.kill(pid, 9)
            pytest.fail('Inference descendant survived group shutdown')
    finally:
        scorer.close()
