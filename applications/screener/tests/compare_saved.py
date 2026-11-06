"""Replay saved native observations through the standalone mathematical coordinator."""

import json
import math
from pathlib import Path
import sys

from applications.scorer import BetaDistribution, DistributionFailure
from applications.scorer.boltz2.ensemble import average_ranks
from applications.scorer.boltz2.models import BoltzScore, Candidate, EngineRun, EnsembleMethod, Metric, Mode, PSICHICScore, Plan
from applications.scorer.boltz2.workflow import run_workflow


def compare(path):
    report = json.loads(Path(path).read_text())
    saved, settings = report['rows'], report['settings']
    if settings['mode'] == 'psichic-only':
        assert [r['entry']['ordinal'] for r in saved] == list(range(1, len(saved) + 1))
        values = {r['entry']['key']: r['psichic']['raw']['raw_affinity'] for r in saved if r['psichic']['state'] == 'succeeded'}
        ranks = average_ranks(values, higher_is_better=True)
        for row in saved:
            key = row['entry']['key']
            assert row['score'] == values.get(key) and row['final_rank'] == ranks.get(key)
        return {'mode': 'psichic-only', 'entries': len(saved), 'same_saved_observations': True}
    candidates = [Candidate(i, row['entry']['id'], row['entry']['smiles']) for i, row in enumerate(saved, 1)]
    psichic, boltz = {}, {}
    for candidate, row in zip(candidates, saved, strict=True):
        raw = row['psichic']['raw']
        if raw and raw['status'] == 'succeeded':
            d = raw['distribution']
            options = {}
            if d['state'] == 'valid':
                options['distribution'] = BetaDistribution(d['alpha'], d['beta'])
            elif d['state'] == 'distribution-failure':
                options['distribution'] = DistributionFailure(d['reason'], d['raw_alpha'], d['raw_beta'])
            psichic[candidate.key] = PSICHICScore(raw['raw_affinity'], raw['adaptive_score'], candidate.smiles, **options)
        raw = row['boltz']['raw']
        if raw and raw['status'] == 'succeeded':
            boltz[candidate.key] = BoltzScore(raw['affinity_probability_binary'], raw['affinity_pred_value'], raw['native'], '', candidate.smiles)
    options = {'ensemble': EnsembleMethod(settings['ensemble'])} if settings['ensemble'] else {}
    if settings['ensemble'] == 'rank-v1':
        options.update(psichic_weight=settings['psichic_weight'], boltz_weight=settings['boltz_weight'])
    plan = Plan(Mode(settings['mode']), shortlist_size=settings['shortlist_size'], metric=Metric(settings['metric']), **options)
    reference = run_workflow(candidates, plan,
                             psichic=lambda rows: EngineRun({r.key: psichic[r.key] for r in rows if r.key in psichic}),
                             boltz=lambda rows: EngineRun({r.key: boltz[r.key] for r in rows if r.key in boltz}))
    for actual, expected in zip(saved, reference.rows, strict=True):
        assert actual['selected'] == (expected.selection == 'selected')
        score = (expected.normalized_ensemble_score if settings['ensemble'] == 'concentration-v1' else
                 expected.experimental_combined_value if settings['ensemble'] == 'rank-v1' else
                 getattr(expected.boltz, settings['metric'], None))
        assert actual['final_rank'] == expected.final_rank
        assert actual['score'] is None if score is None else math.isclose(actual['score'], score, rel_tol=1e-12, abs_tol=1e-12)
    return {'mode': settings['mode'], 'entries': len(saved), 'method': settings['ensemble'], 'same_saved_observations': True}


if __name__ == '__main__':
    print(json.dumps(compare(sys.argv[1]), sort_keys=True))
