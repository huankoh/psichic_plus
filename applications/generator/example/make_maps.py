import pickle
from pathlib import Path

from synthemol.reactions import CHEMICAL_SPACE_TO_REACTIONS


def main() -> None:
    for space, enabled in [("real", "22"), ("wuxi", "1")]:
        mapping = {
            reaction.id: {index: set() for index in range(reaction.num_reactants)}
            for reaction in CHEMICAL_SPACE_TO_REACTIONS[space]
        }
        mapping[enabled][0] = {"CNc1ccccc1"}
        mapping[enabled][1] = {"O=C(O)c1ccccc1"}
        path = Path(__file__).resolve().parent / f"{space}.pkl"
        path.write_bytes(pickle.dumps(mapping, protocol=4))


if __name__ == "__main__":
    main()
