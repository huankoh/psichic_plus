"""Prepare or extend a native screen with durable final scoring."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import shlex
import tarfile

import yaml

from applications.scoring.__main__ import Parser, add_rank_arguments, envelope, human_configuration, human_result, reject_msa_authentication
from applications.scoring.contracts import Engine, Resolution, RunBusy, StateError, canonical
from applications.scoring.run import ScoringRun
from . import workflow


def parser():
    result = Parser(prog='psichic-screener', description=__doc__, allow_abbrev=False)
    result.add_argument('--json', action='store_true')
    sub = result.add_subparsers(dest='action', required=True, parser_class=Parser)
    for name in ('prepare', 'rank'):
        command = sub.add_parser(name, allow_abbrev=False)
        add_rank_arguments(command, candidates=False)
        command.add_argument('--afvs', required=True, type=Path, help='Pinned configured AFVS checkout. Used read-only as the native template.')
        command.add_argument('--scenario', required=True, help='One named native target/scenario, with one replica.')
        command.add_argument('--afvs-python', help='Explicit native preparation Python, independent of the coordinator inference Python.')
        command.add_argument('--stage-size', type=int, default=4096, help='Bounded native preparation window, 1..4096 entries.')
        if name == 'rank':
            command.add_argument('--source-run', required=True, type=Path, help='Caller-confirmed completed native run. Legacy scores are metadata; candidates are rescored.')
    for name in ('resume', 'retry', 'status', 'reconcile'):
        command = sub.add_parser(name, allow_abbrev=False)
        command.add_argument('--run', type=Path, required=True)
        command.add_argument('--json', action='store_true', default=argparse.SUPPRESS)
        if name == 'retry':
            command.add_argument('--failed', action='store_true', required=True)
            command.add_argument('--engine', choices=[e.value for e in Engine])
        if name == 'reconcile':
            command.add_argument('--attempt', required=True)
            command.add_argument('--resolution', choices=[r.value for r in Resolution], required=True)
            command.add_argument('--job-id', required=True)
            command.add_argument('--evidence', required=True, help='Observed native scheduler state. Unknown state never authorizes redispatch.')
    return result


def response(run, outcome=None, new_stage=None):
    value = envelope(run, outcome)
    value['screening'] = workflow.read(run)['source']
    value['native_stages'] = list(workflow.stages(run))
    value['prepared_stage'] = str(new_stage) if new_stage else None
    value['stage_directory'] = str(run.path / 'screening/stages')
    if value['status'] == 'waiting':
        action = f'Submit the newly prepared stage {new_stage} explicitly.' if new_stage else 'Inspect AFVS scheduler status. Do not resubmit live or unresolved workunits.'
        value['next_action'] = f'{action} Then psichic-screener resume --run {run.path}'
    elif value['status'] == 'partial':
        value['next_action'] = f'psichic-screener retry --run {run.path} --failed'
    return value


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    json_mode = '--json' in argv
    try:
        reject_msa_authentication(argv)
        args = parser().parse_args(argv)
        json_mode = args.json
        if args.action in ('prepare', 'rank'):
            run, plan, screen = workflow.create(args)
            if not json_mode:
                human_configuration(plan.settings, plan.target, plan.row_count)
            if args.validate_only:
                value = {**envelope(), 'status': 'validated', 'resolved': plan.summary(), 'screening': screen['source'],
                         'next_action': 'Remove --validate-only to prepare native workunits. Submission remains explicit.'}
            else:
                outcome, transfers, stage = workflow.advance(run, verified_runtime=plan.runtime)
                value = {**response(run, outcome, stage), 'resolved': plan.summary(), 'transfers': transfers}
        else:
            run = ScoringRun.open(args.run)
            workflow.read(run)
            if args.action == 'reconcile':
                workflow.reconcile(run, args.attempt, args.resolution, args.evidence, args.job_id)
            outcome, stage = None, None
            if args.action in ('resume', 'retry'):
                outcome, transfers, stage = workflow.advance(run, retry=args.action == 'retry', engine=getattr(args, 'engine', None))
            value = response(run, outcome, stage)
        code = 0 if value['status'] in ('validated', 'complete') else 1
    except RunBusy as error:
        value, code = {**envelope(), 'status': 'waiting', 'error': {'code': 'RUN_BUSY', 'message': str(error)}}, 1
    except (ValueError, OSError, csv.Error, tarfile.TarError, yaml.YAMLError, StateError, subprocess.SubprocessError) as error:
        value, code = {**envelope(), 'status': 'configuration-error',
                       'error': {'code': getattr(error, 'code', 'INVALID_CONFIGURATION'), 'message': str(error)},
                       'next_action': getattr(error, 'next_action', 'Inspect the error and preserve existing evidence. Changed settings require a new run.')}, 2
    if json_mode:
        print(canonical(value))
    else:
        human_result(value)
        for stage in value.get('native_stages', []):
            print('Native stage: ' + stage['directory'])
            if stage['directory'] == value.get('prepared_stage'):
                print('Submit explicitly from ' + stage['submission_directory'] + ': ' + shlex.join(stage['submission']))
    return code
