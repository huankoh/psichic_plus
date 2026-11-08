"""Rank native products explicitly or after an observed successful generation."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import yaml

from applications.scoring.__main__ import Parser, add_rank_arguments, envelope, human_configuration, human_result, reject_msa_authentication
from applications.scoring.config import ConfigurationError, resolve
from applications.scoring.contracts import Engine, RunBusy, StateError, canonical
from applications.scoring.executor import execute
from applications.scoring.run import ScoringRun
from .final_scoring import generation_context, import_products, scoring_contract
from . import workflow


def parser():
    result = Parser(prog='psichic-generator', description=__doc__, allow_abbrev=False)
    result.add_argument('--json', action='store_true')
    sub = result.add_subparsers(dest='action', required=True, parser_class=Parser)
    for name in ('rank', 'generate-and-rank'):
        command = sub.add_parser(name, allow_abbrev=False,
                                 help='Rank caller-confirmed completed products.' if name == 'rank' else
                                      'Run native generation in a child process, then rank completed products.')
        add_rank_arguments(command, candidates=False)
        command.set_defaults(id_column='node_id', smiles_column='smiles')
        if name == 'rank':
            command.add_argument('--source-run', type=Path, required=True,
                                 help='Native run directory. You assert generation has completed; a CSV alone cannot prove it.')
        else:
            command.add_argument('--native-args', nargs=argparse.REMAINDER,
                                 help='Native SyntheMol arguments. Put this delimiter last, after any -- Boltz options.')
    for name in ('resume', 'retry', 'status'):
        command = sub.add_parser(name, allow_abbrev=False)
        command.add_argument('--run', type=Path, required=True, help='Durable scoring directory or automatic workflow directory.')
        command.add_argument('--json', action='store_true', default=argparse.SUPPRESS)
        if name == 'retry':
            command.add_argument('--failed', action='store_true', required=True)
            command.add_argument('--engine', choices=[e.value for e in Engine])
    return result


def progress(message):
    print(message, file=sys.stderr, flush=True)


def score(run, *, plan=None, action='resume', engine=None, json_mode=False):
    generation = generation_context(run)
    if not json_mode:
        human_configuration(run.settings, run.target, run.row_count)
    outcome = None if action == 'status' else execute(
        run, retry_failed=action == 'retry', engine=engine, progress=progress,
        verified_runtime=plan.runtime if plan is not None else None)
    value = envelope(run, outcome)
    value['generation'] = generation
    return value, 0 if value['status'] == 'complete' else 1


def rank(args, *, completion, expected_contract=None):
    run, plan, generation = import_products(args, completion=completion, expected_contract=expected_contract)
    if args.validate_only:
        if not args.json:
            human_configuration(plan.settings, plan.target, plan.row_count)
            print(plan.summary()['quality'])
        return {**envelope(), 'status': 'validated', 'resolved': plan.summary(), 'generation': generation,
                'population_state': 'existing-products',
                'next_action': 'Remove --validate-only to import and score these product rows.'}, 0
    value, code = score(run, plan=plan, json_mode=args.json)
    value['resolved'] = plan.summary()
    value['population_state'] = 'existing-products'
    return value, code


def automatic(args, native_args):
    if not native_args:
        raise ConfigurationError('NATIVE_ARGUMENTS', 'Supply the native launcher arguments after --native-args')
    native = workflow.preflight(native_args)
    source = Path(native['source_run'])
    if source.exists():
        raise ConfigurationError('NATIVE_OUTPUT_EXISTS', 'Automatic generation requires a fresh native --save_dir.',
                                 'Choose a new native save directory, or use rank --source-run for completed outputs.')
    directory = args.output.expanduser().resolve()
    owned_paths = (directory / 'scoring', directory / 'generation')
    if (directory == source or source in directory.parents
            or any(source == path or source in path.parents or path in source.parents for path in owned_paths)):
        raise ConfigurationError('OUTPUT_OVERLAP', 'Keep native output, workflow evidence, and final scoring directories distinct')
    if directory.exists():
        raise ConfigurationError('OUTPUT_EXISTS', f'Workflow already exists: {directory}', 'Use resume --run for that workflow, or choose a fresh output.')
    args.source_run, args.candidates, args.output = source, source / 'molecules.csv', directory / 'scoring'
    plan = resolve(args, future_population=True)
    if native['target_sequence'] != plan.target.sequence:
        raise ConfigurationError('TARGET_MISMATCH', 'Generation and final scoring must use the same selected protein sequence')
    if not args.json:
        human_configuration(plan.settings, plan.target, None)
        print('Generation preflight only. Completed product counts are not yet known.', flush=True)
    if args.validate_only:
        return {**envelope(), 'status': 'validated', 'resolved': plan.summary(), 'native': native,
                'population_state': 'future-generation', 'counts': {},
                'scoring_directory': str(args.output),
                'next_action': 'Remove --validate-only to generate, then import and score the completed products.'}, 0
    frozen = {name: str(value.expanduser().resolve()) if isinstance(value, Path) else value for name, value in vars(args).items()}
    generation_env = {'PSICHIC_ESM_CHECKPOINT': plan.runtime['controls']['PSICHIC_ESM_CHECKPOINT']} if plan.settings.mode != 'boltz-only' else {}
    request = workflow.create(directory, frozen, native_args, native, scoring_contract(plan), generation_env)
    progress(f'Native generation logs: {directory / "generation"}. Recover final scoring with psichic-generator resume --run {directory}')
    completion = workflow.generate(directory, request)
    if not workflow.completed(completion):
        return {**envelope(), 'status': 'generation-failed', 'workflow_directory': str(directory),
                'scoring_directory': str(args.output), 'generation': completion,
                'next_action': f'Inspect {directory / "generation"}. Dependent scoring was not started.'}, 1
    value, code = rank(args, completion={**completion, 'workflow_directory': str(directory)},
                       expected_contract=request['scoring_contract'])
    value['workflow_directory'] = str(directory)
    return value, code


def saved(args):
    directory = args.run.expanduser().resolve()
    if not (directory / 'workflow.json').exists():
        return score(ScoringRun.open(directory), action=args.action, engine=getattr(args, 'engine', None), json_mode=args.json)
    request = workflow.read(directory)
    completion = workflow.completion(directory)
    scoring = Path(request['scoring_directory'])
    if scoring.exists():
        value, code = score(ScoringRun.open(scoring), action=args.action, engine=getattr(args, 'engine', None), json_mode=args.json)
    elif args.action == 'status':
        value, code = {**envelope(), 'status': 'generation-complete', 'scoring_directory': str(scoring),
                       'next_action': f'psichic-generator resume --run {directory}'}, 1
    else:
        frozen = argparse.Namespace(**request['scoring_args'])
        frozen.json = args.json
        value, code = rank(frozen, completion={**completion, 'workflow_directory': str(directory)},
                           expected_contract=request['scoring_contract'])
    value['workflow_directory'] = str(directory)
    return value, code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    json_mode = '--json' in argv
    native_args = []
    workflow_directory = None
    if '--native-args' in argv:
        index = argv.index('--native-args')
        argv, native_args = argv[:index], argv[index + 1:]
    try:
        reject_msa_authentication(argv)
        args = parser().parse_args(argv)
        json_mode = args.json
        if args.action == 'rank':
            if native_args:
                raise ConfigurationError('NATIVE_ARGUMENTS', '--native-args applies only to generate-and-rank')
            value, code = rank(args, completion={'kind': 'caller-declared-completed-run', 'process_completion_verified': False})
        elif args.action == 'generate-and-rank':
            workflow_directory = args.output.expanduser().resolve()
            value, code = automatic(args, native_args)
        else:
            if native_args:
                raise ConfigurationError('NATIVE_ARGUMENTS', '--native-args applies only to generate-and-rank')
            value, code = saved(args)
    except RunBusy as error:
        value, code = {**envelope(), 'status': 'waiting', 'error': {'code': 'RUN_BUSY', 'message': str(error)},
                       'next_action': 'Wait for the current owner, then resume this directory.'}, 1
    except KeyboardInterrupt:
        value, code = {**envelope(), 'status': 'interrupted',
                       'next_action': 'Inspect the saved workflow and scoring directories; resume never reruns generation.'}, 1
    except (ValueError, OSError, csv.Error, yaml.YAMLError, StateError, subprocess.SubprocessError) as error:
        value, code = {**envelope(), 'status': 'configuration-error',
                       'error': {'code': getattr(error, 'code', 'INVALID_CONFIGURATION'), 'message': str(error)},
                       'next_action': getattr(error, 'next_action', 'Correct the error. Preserve existing outputs and use a new run for changed settings.')}, 2
    if workflow_directory is not None:
        value['workflow_directory'] = str(workflow_directory)
        value['scoring_directory'] = str(workflow_directory / 'scoring')
    if json_mode:
        print(canonical(value))
    else:
        if (value.get('generation') or {}).get('meaning'):
            print(value['generation']['meaning'])
            print('Native source: ' + value['generation']['source_file'])
        if value.get('population_state') == 'existing-products' and value.get('resolved') and value['status'] == 'validated':
            print(f"Validated {value['resolved']['entries']} existing product rows.")
        human_result(value)
    return code
