"""Run prescreen, upstream adaptive selection, and primary screening locally."""

import argparse
from pathlib import Path

from .workflow import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--target-fasta', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True, help='Clean pinned AdaptiveFlow-VS Git checkout')
    parser.add_argument('--output', type=Path, required=True, help='New directory for this run')
    parser.add_argument('--samples-per-tranche', type=int, required=True)
    parser.add_argument('--primary-target', type=int, required=True, help='Target molecule count, rounded up to whole collections')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    try:
        run(args.library, args.target_fasta, args.model_dir, args.upstream, args.output,
            args.samples_per_tranche, args.primary_target, args.device, args.batch_size)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')
    print(f'Results saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
