import copy
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rdkit import Chem
from synthemol.generate.generator import Generator
from synthemol.generate.utils import save_generated_molecules
from synthemol.reactions import CHEMICAL_SPACE_TO_REACTIONS, set_all_building_blocks

from psichic_plus_adaptive.predictor import Predictor
from . import UPSTREAM_REVISION
from .scorer import PSICHICScorer


@dataclass(frozen=True)
class Catalogue:
    chemical_space: Literal["real", "wuxi"]
    smiles_to_id: dict[str, str]
    reaction_map: dict[str, dict[int, set[str]]]


def read_catalogue(building_blocks: Path, reaction_map: Path, chemical_space: str) -> Catalogue:
    if chemical_space not in ("real", "wuxi"):
        raise ValueError("chemical_space must be real or wuxi")
    smiles_to_id = {}
    identifiers = set()
    with building_blocks.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"reagent_id", "smiles"}.issubset(reader.fieldnames or []):
            raise ValueError("Building-block CSV requires reagent_id and smiles columns")
        for row in reader:
            identifier, smiles = row["reagent_id"], row["smiles"]
            if not identifier or not smiles or not identifier.strip() or not smiles.strip():
                raise ValueError("Building-block IDs and SMILES must not be empty")
            if identifier == "-1":
                raise ValueError("Building-block ID -1 is reserved for upstream intermediate reactants")
            if identifier in identifiers or smiles in smiles_to_id:
                raise ValueError(f"Duplicate building-block ID or SMILES at {identifier!r}, {smiles!r}")
            if Chem.MolFromSmiles(smiles) is None:
                raise ValueError(f"Invalid building-block SMILES {smiles!r}")
            identifiers.add(identifier)
            smiles_to_id[smiles] = identifier
    if not smiles_to_id:
        raise ValueError("Building-block CSV is empty")
    with reaction_map.open("rb") as handle:
        mapping = pickle.load(handle)
    reactions = CHEMICAL_SPACE_TO_REACTIONS[chemical_space]
    if not isinstance(mapping, dict) or mapping.keys() != {reaction.id for reaction in reactions}:
        raise ValueError("Reaction map must contain exactly the upstream reaction IDs as strings")
    normalized = {}
    for reaction in reactions:
        slots = mapping[reaction.id]
        if (not isinstance(slots, dict) or any(type(key) is not int for key in slots)
                or slots.keys() != set(range(reaction.num_reactants))):
            raise ValueError(f"Reaction {reaction.id} requires exactly its upstream integer reactant slots")
        normalized[reaction.id] = {}
        for index, reactant in enumerate(reaction.reactants):
            blocks = slots[index]
            if not isinstance(blocks, (list, set)) or any(not isinstance(s, str) for s in blocks):
                raise ValueError(f"Reaction {reaction.id} slot {index} requires a list or set of SMILES")
            blocks = set(blocks)
            if not blocks.issubset(smiles_to_id):
                raise ValueError(f"Reaction {reaction.id} slot {index} contains SMILES absent from the CSV")
            for smiles in blocks:
                if not reactant.has_substruct_match(smiles):
                    raise ValueError(f"Reaction {reaction.id} slot {index} SMARTS rejects {smiles!r}")
            normalized[reaction.id][index] = blocks
    if not any(all(slots.values()) for slots in normalized.values()):
        raise ValueError("Reaction map has no usable route with a building block in every slot")
    return Catalogue(chemical_space, smiles_to_id, normalized)


def make_generator(catalogue: Catalogue, scorer: PSICHICScorer, max_reactions: int, seed: int) -> Generator:
    reactions = copy.deepcopy(tuple(
        reaction for reaction in CHEMICAL_SPACE_TO_REACTIONS[catalogue.chemical_space]
        if all(catalogue.reaction_map[reaction.id].values())
    ))
    set_all_building_blocks(reactions, set(catalogue.smiles_to_id))
    for reaction in reactions:
        for index, reactant in enumerate(reaction.reactants):
            reactant.allowed_building_blocks = catalogue.reaction_map[reaction.id][index]
    return Generator(
        search_type="mcts",
        chemical_space_to_building_block_smiles_to_id={catalogue.chemical_space: catalogue.smiles_to_id},
        max_reactions=max_reactions, scorer=scorer, score_weights=scorer.score_weights,
        success_comparators=None, explore_weight=10.0, num_expand_nodes=None,
        rl_base_temperature=0.1, rl_temperature_similarity_target=None, rl_train_frequency=10,
        reactions=reactions, rng_seed=seed, no_building_block_diversity=False,
        store_nodes=False, verbose=False,
    )


def run(target_fasta: Path, model_dir: Path, building_blocks: Path, reaction_map: Path,
        chemical_space: str, output: Path, rollouts: int = 100, max_reactions: int = 1,
        seed: int = 0, device: str = "cpu") -> Path:
    if rollouts < 1 or max_reactions < 1:
        raise ValueError("rollouts and max_reactions must be positive")
    catalogue = read_catalogue(building_blocks, reaction_map, chemical_space)
    output.mkdir(parents=True, exist_ok=False)
    predictor = Predictor(str(target_fasta), str(model_dir), device, str(output / "cache"), batch_size=1)
    scorer = PSICHICScorer(predictor)
    generator = make_generator(catalogue, scorer, max_reactions, seed)
    try:
        generator.generate(n_rollout=rollouts)
    except ValueError as error:
        if not any(node.num_molecules == 1 and node.num_reactions > 0 for node in generator.node_map):
            raise RuntimeError(f"No reaction products generated. Upstream search stopped: {error}") from error
        raise
    if not any(node.num_molecules == 1 and node.num_reactions > 0 for node in generator.node_map):
        raise RuntimeError("No reaction products generated. Check the catalogue and increase the rollout budget.")
    nodes = generator.get_full_molecule_nodes()
    path = output / "molecules.csv"
    save_generated_molecules(
        nodes, {chemical_space: {identifier: smiles for smiles, identifier in catalogue.smiles_to_id.items()}},
        scorer.score_weights.score_names, path,
    )
    (output / "run.json").write_text(json.dumps({
        "synthemol_revision": UPSTREAM_REVISION, "chemical_space": chemical_space,
        "rollouts": rollouts, "max_reactions": max_reactions, "seed": seed, "device": device,
        "products": len(nodes), "scored_smiles": len(scorer.scores), "batch_size": 1,
    }, indent=2) + "\n")
    return path
