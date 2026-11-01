"""Run the real two-tranche workflow and compare every scored item with the existing CLI."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from psichic_plus_adaptive import UPSTREAM_REVISION
from psichic_plus_adaptive.workflow import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    example = Path(__file__).resolve().parent
    output = args.output.resolve()
    run(example / 'library.csv', example / 'target.fasta', args.model_dir, args.upstream,
        output, 2, 2, args.device, 1)
    manifest = pd.read_csv(output / 'library/manifest.csv', dtype=str, keep_default_na=False)
    manifest.rename(columns={'token': 'ID'})[['ID', 'smiles']].to_csv(output / 'direct-input.csv', index=False)
    screening = Path(os.environ.get('PSICHIC_SCREENING_DIR', example.parents[1] / 'psichic_plus_screening')).resolve()
    command = [sys.executable, str(screening / 'inference.py'),
               '--input_file', str(output / 'direct-input.csv'), '--protein_fasta', str(example / 'target.fasta'),
               '--output_folder', str(output / 'direct'), '--model_dir', str(args.model_dir.resolve()),
               '--ligand_column', 'smiles', '--device', args.device,
               '--batch_size', '1', '--num_workers', '0', '--no-pin-memory']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (output / 'direct.log').open('w') as handle:
        handle.write(json.dumps(command) + '\n')
        handle.flush()
        subprocess.run(command, cwd=output, check=True, env=env, stdout=handle, stderr=subprocess.STDOUT)
    direct = pd.read_csv(output / 'direct/inference_results.csv', dtype={'ID': str}).set_index('ID')
    maximum_delta = 0.
    compared = 0
    for stage in ('prescreen', 'primaryscreen'):
        frame = pd.read_csv(output / stage / 'results.csv', dtype={'collection_id': str, 'molecule_id': str})
        for row in frame.itertuples(index=False):
            reference = direct.loc[row.token]
            assert row.status == reference.inference_status, (row.token, row.status, reference.inference_status)
            if row.status == 'success':
                delta = abs(row.raw_affinity - reference.predicted_binding_affinity)
                maximum_delta = max(maximum_delta, delta)
                assert delta <= 1e-5, (row.token, delta)
                assert row.adaptive_score == -row.raw_affinity
                compared += 1
            else:
                assert row.failure_stage == reference.failure_stage
                assert row.failure_reason == reference.failure_reason
        invalid = frame[frame.smiles == 'invalid-smiles']
        assert len(invalid) == 1 and invalid.status.tolist() == ['failed']
    repeated_ids = manifest[manifest.molecule_id == '0001']
    assert repeated_ids.collection_id.tolist() == ['001', '002']
    assert repeated_ids.token.nunique() == 2
    todo = (output / 'primary.todo').read_bytes()
    assert todo == (output / 'primaryscreen/tools/templates/todo.all').read_bytes()
    selection = json.loads((output / 'selection.json').read_text())
    assert hashlib.sha256(todo).hexdigest() == selection['todo_sha256']
    assert selection['primary_target'] == 2 and selection['selected_molecules'] == 3
    prescreen = pd.read_csv(output / 'prescreen/results.csv')
    means = prescreen[prescreen.status == 'success'].groupby('tranche').adaptive_score.mean()
    selected_key = todo.decode().split()[0]
    selected_tranche = manifest[manifest.af_collection == selected_key].tranche.unique().tolist()
    assert selected_tranche == [means.idxmin()]
    ranked = pd.read_csv(output / 'ranked.csv')
    assert len(ranked) == 2 and ranked.raw_affinity.is_monotonic_decreasing
    summary = {'upstream_revision': UPSTREAM_REVISION, 'device': args.device,
               'compared_predictions': compared, 'max_absolute_affinity_delta': maximum_delta,
               'direct_successes': int((direct.inference_status == 'success').sum()),
               'direct_failures': int((direct.inference_status == 'failed').sum()),
               'selected_tranche': selected_tranche[0], 'primary_target': 2, 'primary_molecules': 3,
               'ranked': len(ranked), 'todo_preserved': True}
    (output / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
