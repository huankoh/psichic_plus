"""Import completed native products into the one durable scoring executor."""

from dataclasses import asdict, replace
import os

from applications.scoring.config import ConfigurationError, resolve
from applications.scoring.contracts import canonical, digest, loads
from applications.scoring.evidence import fsync_directory, write_file
from applications.scoring.executor import create_run
from applications.scoring.runtime import identity
from .products import snapshot


def import_products(args, *, completion, expected_contract=None):
    with snapshot(args.source_run, args.id_column, args.smiles_column) as products:
        if completion['kind'] == 'observed-native-exit':
            if completion.get('source_sha256') != products.sha256:
                raise ConfigurationError('SOURCE_CHANGED', 'Native products differ from the snapshot bound to successful process completion.',
                                         'Preserve the workflow evidence. Explicitly import a separately confirmed completed run into a new scoring directory.')
        args.candidates = products.path
        plan = resolve(args)
        generation = {'schema_version': 1, 'source_run': str(products.source.parent),
                      'source_file': str(products.source), 'sha256': products.sha256,
                      'columns': list(products.columns), 'rows': products.row_count,
                      'completion': completion, 'search_unchanged': True,
                      'search_scores_reused': False,
                      'meaning': 'Ranks the imported completed products. Native search and output scores are unchanged.'}
        runtime = {**plan.runtime, 'source': str(products.source)}
        plan = replace(plan, runtime=runtime)
        products.verify()
        if expected_contract is not None and scoring_contract(plan) != expected_contract:
            raise ConfigurationError('SCORING_CHANGED', 'The scoring runtime or settings changed after generation preflight.',
                                     'Preserve the native outputs. Validate and explicitly import them into a new scoring run.')
        if args.validate_only:
            return None, plan, generation
        run = create_run(plan, entries=products.entries(args.id_column, args.smiles_column, generation))
        write_file(run.path / 'generation.json', canonical(generation) + '\n')
        write_file(run.path / 'generation.sha256', digest(generation) + '\n')
        os.link(run.path / 'inputs/candidates.csv', run.path / 'inputs/molecules.csv')
        write_file(run.path / 'generation.md', '\n'.join([
            '# Final ranking of generated products', '', generation['meaning'],
            f'Native source is `{products.source}`.', f'The native CSV SHA-256 is `{products.sha256}`.',
            f'The completion boundary is `{completion["kind"]}`.',
            'An explicit existing-run import is the caller\'s completion claim, not proof of native process completion.',
            'Search scores are retained as metadata and final inference uses the shared receipt contract.',
            'The retained native file is [molecules.csv](inputs/molecules.csv).',
            'The versioned ranked reports are identified by [latest.json](latest.json).', '',
        ]))
        fsync_directory(run.path / 'inputs')
        fsync_directory(run.path)
        products.verify()
        return run, plan, generation


def scoring_contract(plan):
    return digest({'target': asdict(plan.target), 'settings': asdict(plan.settings),
                   'profiles': {name: identity(profile) for name, profile in plan.runtime['profiles'].items()}})


def generation_context(run):
    path = run.path / 'generation.json'
    if not path.exists():
        return None
    value = loads(path.read_bytes())
    if digest(value) != (run.path / 'generation.sha256').read_text().strip():
        raise ValueError('The frozen generation source record changed')
    return {**value, 'retained_source': str(run.path / 'inputs/molecules.csv'),
            'explanation': str(run.path / 'generation.md')}
