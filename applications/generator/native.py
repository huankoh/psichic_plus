"""Optional native parser and guarded generation child, with no scoring coordinator."""

from functools import wraps
import importlib
import os
from pathlib import Path
import sys

from applications.scoring.config import ConfigurationError
from applications.scoring.contracts import canonical, loads
from applications.scoring.guardian import start_guardian
from applications.scoring.processes import parent_guard
from applications.scoring.runtime import checksum, offline_only
from . import UPSTREAM_REVISION

INSTALL = 'Install the documented SyntheMol dependencies in this interpreter, then run python -m applications.generator.install --source /path/to/pinned/SyntheMol. No package is installed automatically.'


def native_launcher():
    try:
        return importlib.import_module('synthemol.generate.generate')
    except ImportError as error:
        raise ConfigurationError('SYNTHEMOL_UNAVAILABLE', f'Native SyntheMol is unavailable or incompatible ({error}).', INSTALL) from error


def preflight(argv):
    module = native_launcher()
    from tap import tapify

    @wraps(module.generate)
    def capture(**options):
        return options

    options = tapify(capture, command_line_args=argv)
    if 'psichic_plus' not in options['score_types']:
        raise ConfigurationError('SEARCH_OBJECTIVE', 'Automatic final scoring requires the native psichic_plus scalar search scorer')
    target = options.get('psichic_target_fasta')
    if target is None:
        raise ConfigurationError('SEARCH_TARGET', 'Supply native --psichic_target_fasta for PSICHIC search')
    from applications.scorer.boltz2.target import Target

    sequence = Target.load(fasta=target, msa_mode='single').protein_sequence()[1]
    sources = []
    for name in ('synthemol.generate.generate', 'synthemol.generate.scorer', 'synthemol.constants'):
        path = Path(importlib.import_module(name).__file__).resolve()
        sources.append({'module': name, 'path': str(path), 'sha256': checksum(path)})
    return {'schema_version': 1, 'source_run': str(options['save_dir'].resolve()),
            'target_sequence': ''.join(sequence.split()).upper(), 'score_types': options['score_types'],
            'expected_revision': UPSTREAM_REVISION, 'loaded_source': sources}


def main():
    action = sys.argv[1]
    if action == 'generate':
        directory = Path(sys.argv[2])
        parent_guard()
        start_guardian(directory)
        offline_only()
        request = loads((directory / 'request.json').read_bytes())
        for item in request['preflight']['loaded_source']:
            if checksum(item['path']) != item['sha256']:
                raise ConfigurationError('NATIVE_SOURCE_CHANGED', 'SyntheMol source changed after preflight')
        sys.argv = ['synthemol', *request['native_args']]
        native_launcher().generate_command_line()
        return 0
    response = None
    saved = os.dup(1)
    try:
        sys.stdout.flush()
        os.dup2(2, 1)
        offline_only()
        response = {'preflight': preflight(loads(sys.stdin.read()))}
    except (ImportError, ValueError, OSError) as error:
        response = {'error': {'code': getattr(error, 'code', 'NATIVE_ARGUMENTS'), 'message': str(error),
                              'next_action': getattr(error, 'next_action', INSTALL)}}
    except SystemExit:
        response = {'error': {'code': 'NATIVE_ARGUMENTS', 'message': 'Native argument validation did not complete.',
                              'next_action': 'Read the native parser diagnostics on stderr and correct the arguments.'}}
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
    print(canonical(response))
    return 2 if 'error' in response else 0


if __name__ == '__main__':
    sys.exit(main())
