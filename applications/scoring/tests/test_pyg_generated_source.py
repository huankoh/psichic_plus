"""Real PyG generation on CPU, without trained checkpoints or inference."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = r'''
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

from applications.scorer.psichic_plus.predictor import load_helpers
from applications.scoring import runtime
from applications.scoring.executor import _profile_assets
from applications.scoring.pyg_source import expected_source

runtime.offline_only()
load_helpers()
import psichic_model
import torch
import torch_geometric

directory = Path(sys.argv[1])
mode = sys.argv[2]
classes = (psichic_model.PNAConv, psichic_model.DrugProteinConv, psichic_model.GCNConv)
names = [f'{cls.__module__}_{cls.__name__}_propagate' for cls in classes]
profile_path = directory / 'profile.json'

def rejected(profile, message):
    try:
        runtime.verify_loaded(profile)
    except runtime.RuntimeMismatch as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError('Unexpected source accepted: ' + message)

if mode == 'preflight':
    with patch.object(torch.nn.Module, '__init__', side_effect=AssertionError('model constructed')), \
         patch.object(torch, 'load', side_effect=AssertionError('checkpoint deserialized')), \
         patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA initialized')):
        profile = runtime.inspect_psichic({'model_dir': str(directory / 'weights'), 'device': 'cpu',
                                          'esm_checkpoint': str(directory / 'weights/esm.pt')})
    assert all(name not in sys.modules for name in names)
    assert not torch.cuda.is_initialized()
    assert {item['module'] for item in profile['generated_source']} == set(names)
    templates = {item['name'] for item in profile['source'] if item['path'].endswith('.jinja')}
    assert {'torch_geometric/nn/conv/propagate.jinja', 'torch_geometric/nn/conv/collect.jinja'} <= templates
    profile_path.write_text(json.dumps(profile))
    for change in ('sha256', 'module', 'generator', 'fuse'):
        changed = json.loads(json.dumps(profile))
        changed['generated_source'][0][change] = 'changed'
        assert runtime.identity(changed) != runtime.identity(profile)
        assert _profile_assets(changed, 'source') != _profile_assets(profile, 'source')
    with patch.object(torch_geometric, '__version__', 'unsupported'):
        try:
            expected_source()
        except runtime.RuntimeMismatch as error:
            assert 'qualified PyG' in str(error)
        else:
            raise AssertionError('Unsupported generator version accepted')
    for method in ('message_and_aggregate', 'edge_update'):
        with patch.object(classes[0], method, lambda self, x: x):
            try:
                expected_source()
            except runtime.RuntimeMismatch as error:
                assert 'Unsupported generated source contract' in str(error)
            else:
                raise AssertionError('Unsupported generation configuration accepted')
    print(json.dumps({'validated': True, 'generated_source': profile['generated_source']}))
else:
    profile = json.loads(profile_path.read_text())
    width = int(mode)
    runtime.verify_loaded(profile)
    model = psichic_model.PSICHICPlusNet(
        torch.tensor([0, 1]), torch.tensor([0, 1]), hidden_channels=width,
        total_layer=1, K=[2], heads=1, device='cpu')
    native_layers = [layer for layer in model.modules()
                     if isinstance(layer, torch_geometric.nn.MessagePassing)]
    assert {type(layer) for layer in native_layers} == set(classes)
    assert all(layer.fuse == (type(layer) is psichic_model.GCNConv) for layer in native_layers)
    loaded = runtime.verify_loaded(profile)
    observed = {name: loaded[name] for name in names}
    for name in names:
        original = sys.modules[name]
        actual = Path(original.__file__).read_bytes()
        (directory / (name + '.py')).write_bytes(actual)
        altered = directory / ('altered_' + name + '.py')
        altered.write_bytes(actual + b'\nchanged = True\n')
        spec = importlib.util.spec_from_file_location(name, altered)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(sys.modules[name])
            rejected(profile, 'Loaded source changed')
        finally:
            sys.modules[name] = original
        unexpected = name + '_unexpected'
        spec = importlib.util.spec_from_file_location(unexpected, original.__file__)
        sys.modules[unexpected] = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(sys.modules[unexpected])
            rejected(profile, unexpected)
        finally:
            del sys.modules[unexpected]
    missing = {**profile, 'generated_source': []}
    rejected(missing, 'Loaded source was not fingerprinted')
    unexpected_file = directory / 'unexpected.py'
    unexpected_file.write_text('unexpected = True\n')
    spec = importlib.util.spec_from_file_location('psichic_model_Unexpected_propagate', unexpected_file)
    sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(sys.modules[spec.name])
        rejected(profile, spec.name)
    finally:
        del sys.modules[spec.name]
    assert not torch.cuda.is_initialized()
    print(json.dumps({'loaded': observed, 'width': width, 'tampering_rejected': True}))
'''


def test_native_generated_source_matches_frozen_preflight_in_separate_processes(tmp_path):
    weights = tmp_path / 'weights'
    weights.mkdir()
    for name in ('config.json', 'degree.pt', 'model.pt', 'esm.pt', 'esm-contact-regression.pt'):
        (weights / name).write_text('Must be hashed, never deserialized during validation')
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1',
           'PSICHIC_SCREENING_DIR': str(ROOT / 'psichic_plus_screening'), 'CUDA_VISIBLE_DEVICES': '-1'}
    results = []
    for mode in ('preflight', '4', '8'):
        result = subprocess.run([sys.executable, '-B', '-c', SCRIPT, str(tmp_path), mode],
                                env=env, cwd=ROOT, capture_output=True, text=True, timeout=120)
        (tmp_path / f'{mode}.stdout').write_text(result.stdout)
        (tmp_path / f'{mode}.stderr').write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        results.append(json.loads(result.stdout))
    first, second = results[1]['loaded'], results[2]['loaded']
    assert set(first) == set(second)
    for name in first:
        assert first[name]['sha256'] == second[name]['sha256']
        assert first[name]['path'] != second[name]['path']
