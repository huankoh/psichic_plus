"""Installed console and standalone worker imports outside the checkout."""

import ast
import json
import os
import shutil
import subprocess
import sys
import zipfile

from applications.screener.tests.test_integration import ROOT, arguments, setup_runtime

pytest_plugins = ['applications.scoring.tests.test_executor_cli']


def test_installed_screen_commands_and_python310_worker_package(runtime):
    root = runtime['root']
    native = setup_runtime(runtime)
    source = root / 'build-source'
    shutil.copytree(ROOT / 'applications', source / 'applications', ignore=shutil.ignore_patterns('build', '*.egg-info', '__pycache__'))
    shutil.copytree(ROOT / 'psichic_plus_screening', source / 'psichic_plus_screening', ignore=shutil.ignore_patterns('pretrained_weights', '__pycache__', 'tests'))
    shutil.copyfile(ROOT / 'LICENSE', source / 'LICENSE')
    wheels, install = root / 'wheels', root / 'installed'
    result = subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                             '--wheel-dir', str(wheels), str(source / 'applications'), str(source / 'applications/screener')],
                            capture_output=True, text=True, timeout=120)
    (root / 'build.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    paths = list(wheels.glob('*.whl'))
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-compile', '--target', str(install), *map(str, paths)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    worker_wheel = next(p for p in paths if p.name.startswith('psichic_adaptiveflow'))
    with zipfile.ZipFile(worker_wheel) as archive:
        for name in archive.namelist():
            assert not name.endswith(('.pt', '.pth', '.ckpt'))
            if name.endswith('.py'):
                ast.parse(archive.read(name), filename=name, feature_version=(3, 10))
        assert 'psichic_adaptiveflow/_outcomes/__init__.py' in archive.namelist()
    outside = root / 'outside'
    outside.mkdir()
    env = dict(runtime['env'], PYTHONPATH=str(install))
    commands = ([str(install / 'bin/psichic-screener')], [sys.executable, '-m', 'applications.screener'])
    for number, command in enumerate(commands):
        args = arguments(runtime, native, extra=['--validate-only'])
        process = subprocess.run([*command, *args], env=env, cwd=outside, capture_output=True, text=True, timeout=180)
        (root / f'installed-{number}.json').write_text(process.stdout)
        (root / f'installed-{number}.stderr').write_text(process.stderr)
        assert process.returncode == 0 and json.loads(process.stdout)['status'] == 'validated', (process.stdout, process.stderr)
        human = subprocess.run([*command, *[a for a in args if a != '--json']], env=env, cwd=outside, capture_output=True, text=True, timeout=180)
        assert human.returncode == 0 and 'VALIDATED' in human.stdout
        bad = subprocess.run([*command, 'prepare', '--bogus', '--json'], env=env, cwd=outside, capture_output=True, text=True, timeout=20)
        assert bad.returncode == 2 and json.loads(bad.stdout)['error']['code'] == 'ARGUMENTS'
    origin = subprocess.run([sys.executable, '-c', 'import applications.screener,psichic_adaptiveflow.protocol; print(applications.screener.__file__); print(psichic_adaptiveflow.protocol.__file__)'],
                            env=env, cwd=outside, capture_output=True, text=True, timeout=20)
    assert origin.returncode == 0 and str(ROOT) not in origin.stdout and str(install) in origin.stdout
    (root / 'installed-imports.log').write_text(origin.stdout)
    assert not runtime['run'].exists()
    native_python = os.environ.get('AFVS_WORKER_PYTHON')
    if native_python:
        result = subprocess.run([native_python, '-c', 'import sys,psichic_adaptiveflow.protocol,psichic_adaptiveflow.external; assert sys.version_info[:2] == (3,10); print(sys.version)'],
                                env=env, cwd=outside, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
