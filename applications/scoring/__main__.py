"""Human-first durable ranking, with one bounded JSON response for automation."""

import argparse
import csv
from pathlib import Path
import sys
import subprocess

import yaml

from applications.scorer.boltz2.models import EnsembleMethod, Metric, Mode
from .config import ConfigurationError, resolve
from .contracts import Engine, RunBusy, StateError, canonical
from .executor import create_run, execute
from .reports import Waiting
from .run import ScoringRun


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ConfigurationError('ARGUMENTS', message, f'Read {self.prog} --help and correct the arguments.')


def add_rank_arguments(rank, *, candidates=True):
    """Share the resolved scoring contract with application adapters."""
    if candidates:
        rank.add_argument('--candidates', type=Path, required=True)
    rank.add_argument('--id-column', default='ID')
    rank.add_argument('--smiles-column', default='SMILES')
    target = rank.add_mutually_exclusive_group(required=True)
    target.add_argument('--fasta', type=Path)
    target.add_argument('--target-yaml', type=Path)
    rank.add_argument('--ligand-id')
    rank.add_argument('--psichic-chain')
    rank.add_argument('--msa-mode', required=True, choices=['single'])
    rank.add_argument('--mode', required=True, choices=['psichic-only', *[m.value for m in Mode]])
    rank.add_argument('--shortlist-size', type=int, help='Required global K for pyramid modes; no default.')
    rank.add_argument('--ensemble', choices=[m.value for m in EnsembleMethod])
    rank.add_argument('--rank-by', choices=[m.value for m in Metric])
    rank.add_argument('--psichic-weight', type=float)
    rank.add_argument('--boltz-weight', type=float)
    rank.add_argument('--psichic-output', choices=['affinity', 'distribution'])
    rank.add_argument('--model-dir', type=Path, help='PSICHIC weights, or PSICHIC_MODEL_DIR.')
    rank.add_argument('--psichic-device', help='cpu or cuda:0 within one assigned GPU; PSICHIC_DEVICE defaults to cpu.')
    rank.add_argument('--psichic-cache', type=Path, help='Cache parent, or PSICHIC_CACHE. Each invocation owns its subdirectory.')
    rank.add_argument('--esm-checkpoint', type=Path, help='Cached ESM model beside its contact-regression file; PSICHIC_ESM_CHECKPOINT.')
    rank.add_argument('--batch-size', type=int, help='PSICHIC loader batch size, default 32.')
    rank.add_argument('--chunk-size', type=int, default=64, help='Entries per exact invocation, 1..4096, default 64.')
    rank.add_argument('--boltz-python', help='Isolated Python 3.12 with unmodified Boltz 2.2.1; BOLTZ_PYTHON.')
    rank.add_argument('--boltz-cache', type=Path, help='Complete offline Boltz cache; BOLTZ_CACHE.')
    rank.add_argument('--boltz-timeout', type=float, help='Seconds per Boltz invocation. Omitted means no timeout.')
    rank.add_argument('--seed', type=int, default=42, help='Recorded seed for each exact batch, default 42.')
    rank.add_argument('--preset', choices=['normal', 'quick-check'], default='normal')
    rank.add_argument('--output', type=Path, required=True)
    rank.add_argument('--validate-only', action='store_true', help='Inspect inputs, cached assets, dependencies and defaults without inference or run creation.')
    rank.add_argument('--json', action='store_true', default=argparse.SUPPRESS)
    rank.add_argument('boltz_options', nargs=argparse.REMAINDER, help='Explicit native prediction options after --. No MSA, authentication, or owned-path overrides.')


def parser():
    result = Parser(prog='psichic', description='Rank candidates with durable PSICHIC+ and optional isolated Boltz predictions.',
                    allow_abbrev=False)
    result.add_argument('--json', action='store_true', help='One versioned JSON envelope on stdout.')
    sub = result.add_subparsers(dest='action', required=True, parser_class=Parser)
    rank = sub.add_parser('rank', help='Validate and start a new frozen scoring run.', allow_abbrev=False)
    add_rank_arguments(rank)
    for name, help_text in [('resume', 'Continue pending or confirmed stopped work.'),
                            ('retry', 'Explicitly retry recorded failures, preserving old attempts.'),
                            ('status', 'Read the latest stored report and current completion counts.')]:
        command = sub.add_parser(name, help=help_text, allow_abbrev=False)
        command.add_argument('--run', required=True, type=Path)
        command.add_argument('--json', action='store_true', default=argparse.SUPPRESS)
        if name == 'retry':
            command.add_argument('--failed', action='store_true', required=True)
            command.add_argument('--engine', choices=[e.value for e in Engine])
    return result


def envelope(run=None, outcome=None):
    value = {'schema_version': 1, 'status': 'waiting', 'run_id': run.run_id if run else None,
             'population_digest': run.population_digest if run else None, 'report_version': None,
             'counts': {}, 'reports': None, 'run_directory': str(run.path) if run else None,
             'invocations': str(run.path / 'local/invocations') if run else None, 'next_action': None}
    if run is None:
        return value
    counts = {'entries': run.row_count}
    for diagnostic in run.diagnostics():
        name = f'{diagnostic.engine.value}_{diagnostic.state.value}'
        counts[name] = counts.get(name, 0) + 1
    view = run.report_view()
    value['counts'] = counts
    unresolved = any(counts.get(f'{e.value}_{state}', 0) for e in Engine
                     for state in ('pending', 'submitted', 'interrupted-or-unknown'))
    if view is not None:
        report = run.latest()
        value.update(report_version=report.version, report_id=report.report_id,
                     reports={name: str(report.directory / name) for name in ('summary.md', 'results.csv', 'results.json')},
                     ranking=view['ranking'], warnings=view['warnings'], shortage=view['shortage'],
                     top=view['top'], settings=view['settings'])
        if not unresolved:
            value.update(status=report.status.value, counts=view['counts'])
    if unresolved or isinstance(outcome, Waiting):
        value.update(status='waiting', next_action=f'Inspect {run.path / "local/invocations"} for live or interrupted work; then psichic resume --run {run.path}')
    elif value['status'] == 'partial':
        value['next_action'] = f'psichic retry --run {run.path} --failed'
    elif value['status'] == 'shortage':
        value['next_action'] = 'Retry recorded PSICHIC failures, or create a new run with a smaller explicit K.'
    return value


def _cell(value):
    if value is None:
        return '-'
    return str(value).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')


def human_result(value):
    print(value['status'].upper())
    if value.get('counts'):
        print('Counts: ' + ', '.join(f'{name}={count}' for name, count in value['counts'].items()))
    if value.get('ranking'):
        print(f"Ranking {value['ranking']['method']}: {value['ranking']['meaning']}")
        distribution = value['settings']['psichic_output'] == 'distribution'
        fields = ['Position', 'ID', 'SMILES', 'PSICHIC affinity', 'Boltz affinity', 'Binder probability']
        if distribution:
            fields += ['Alpha', 'Beta', 'Concentration']
        fields += ['Score', 'Status']
        print('\t'.join(fields))
        for row in value.get('top', []):
            p, b = row['psichic']['raw'] or {}, row['boltz']['raw'] or {}
            values = [row['position'], row['entry']['id'], row['entry']['smiles'], p.get('raw_affinity'),
                      b.get('affinity_pred_value'), b.get('affinity_probability_binary')]
            if distribution:
                d = p.get('distribution', {})
                values += [d.get(k) for k in ('alpha', 'beta', 'concentration')]
            print('\t'.join(map(_cell, [*values, row['score'], row['status']])))
    for warning in value.get('warnings', []):
        print(warning)
    for name, path in (value.get('reports') or {}).items():
        print(f'{name}: {path}')
    if value.get('run_directory') and not value.get('reports'):
        print('Run: ' + value['run_directory'])
    if value.get('error'):
        print(f"{value['error']['code']}: {value['error']['message']}")
    if value.get('next_action'):
        print('Next: ' + value['next_action'])


def human_configuration(settings, target, entries):
    print('Resolved settings')
    print(f'Mode {settings.mode}; K={settings.shortlist_size if settings.pyramid else "all"}; MSA single')
    print(f'Target: {len(target.sequence)} residues; identity {target.identity}')
    if settings.ensemble:
        print(f'Ensemble: {settings.ensemble.value}; metric: {settings.metric.value}')
        if settings.ensemble is EnsembleMethod.RANK:
            print(f'Rank weights: PSICHIC={settings.psichic_weight}, Boltz={settings.boltz_weight}')
    elif settings.metric:
        print(f'Boltz metric: {settings.metric.value}')
    if Engine.PSICHIC in settings.engines:
        print(f'PSICHIC display: {settings.psichic_output.value}; required evidence: {settings.required_output.value}')
    counts = {engine.value: settings.shortlist_size if engine is Engine.BOLTZ and settings.pyramid else entries
              for engine in settings.engines}
    print('Evaluations: ' + ', '.join(f'{engine}={count if count is not None else "unknown until generation completes"}'
                                    for engine, count in counts.items()))
    if settings.pyramid:
        print('Boltz starts only after K eligible PSICHIC entries; a shortage starts no Boltz jobs.')
    for engine in settings.engines:
        print(f'{engine.value} settings: ' + getattr(settings, engine.value + '_options'))
    print('Evaluation counts describe work, not a runtime estimate.', flush=True)


def reject_msa_authentication(argv):
    if any(token.split('=', 1)[0] in ('--msa_server_password', '--msa_server_username', '--api_key_value', '--api_key_header') for token in argv):
        raise ConfigurationError('NATIVE_OPTION', 'MSA authentication options are not accepted in single-sequence mode')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    json_mode = '--json' in argv
    run = None
    try:
        reject_msa_authentication(argv)
        args = parser().parse_args(argv)
        json_mode = args.json
        if args.action == 'rank':
            plan = resolve(args)
            summary = plan.summary()
            if not json_mode:
                human_configuration(plan.settings, plan.target, plan.row_count)
                print(summary['quality'])
            if args.validate_only:
                value = {**envelope(), 'status': 'validated', 'resolved': summary, 'next_action': 'Remove --validate-only to execute this configuration.'}
                code = 0
            else:
                run = create_run(plan)
                outcome = execute(run, verified_runtime=plan.runtime, progress=lambda message: print(message, file=sys.stderr, flush=True))
                value = envelope(run, outcome)
                value['resolved'] = summary
                code = 0 if value['status'] == 'complete' else 1
        else:
            run = ScoringRun.open(args.run)
            if not json_mode:
                human_configuration(run.settings, run.target, run.row_count)
            if args.action == 'status':
                outcome = None
            else:
                outcome = execute(run, retry_failed=args.action == 'retry', engine=getattr(args, 'engine', None),
                                  progress=lambda message: print(message, file=sys.stderr, flush=True))
            value = envelope(run, outcome)
            code = 0 if value['status'] == 'complete' else 1
    except RunBusy as error:
        value = {**envelope(), 'status': 'waiting', 'error': {'code': 'RUN_BUSY', 'message': str(error)},
                 'next_action': 'Wait for the current local executor, then resume.'}
        code = 1
    except (ValueError, OSError, csv.Error, yaml.YAMLError, StateError, subprocess.SubprocessError) as error:
        value = {**envelope(), 'status': 'configuration-error',
                 'error': {'code': getattr(error, 'code', 'INVALID_CONFIGURATION'), 'message': str(error)},
                 'next_action': getattr(error, 'next_action', 'Correct the error. Use a new run when changing frozen settings.')}
        code = 2
    if json_mode:
        print(canonical(value))
    else:
        human_result(value)
    return code

if __name__ == '__main__':
    sys.exit(main())
