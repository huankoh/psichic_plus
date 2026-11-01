import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate molecules with SyntheMol MCTS and PSICHIC+ affinity.")
    parser.add_argument("--target-fasta", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--building-blocks", type=Path, required=True)
    parser.add_argument("--reaction-map", type=Path, required=True, help="Trusted SyntheMol pickle only")
    parser.add_argument("--chemical-space", choices=("real", "wuxi"), default="real")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--max-reactions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.rollouts < 1 or args.max_reactions < 1:
        parser.error("--rollouts and --max-reactions must be positive")
    from .generate import run
    path = run(**vars(args))
    print(f"Saved generated molecules to {path}")


if __name__ == "__main__":
    main()
