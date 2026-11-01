import csv
import pickle
from pathlib import Path

import pytest

pytest.importorskip("synthemol")

from psichic_plus_adaptive import Failure, Success
from psichic_plus_synthemol.generate import make_generator, read_catalogue, run
from psichic_plus_synthemol.scorer import PSICHICScorer

EXAMPLE = Path(__file__).resolve().parents[1] / "psichic_plus_synthemol/example"
AMINE = "CNc1ccccc1"
ACID = "O=C(O)c1ccccc1"


class StubPredictor:
    def __init__(self, affinity=7.0):
        self.affinity = affinity
        self.calls = []

    def score(self, inputs):
        self.calls.append(inputs)
        return [Success(token, self.affinity, -self.affinity) for token, _ in inputs]


def test_scores_are_raw_positive_and_cached_by_exact_smiles():
    predictor = StubPredictor()
    scorer = PSICHICScorer(predictor)
    assert scorer(AMINE) == 7.0
    individual = scorer.compute_individual_scores(AMINE)
    individual[0] = -50
    assert scorer.compute_individual_scores(AMINE) == [7.0]
    assert scorer("c1ccccc1NC") == 7.0
    assert predictor.calls == [[(AMINE, AMINE)], [("c1ccccc1NC", "c1ccccc1NC")]]
    assert scorer.num_scores == 1
    assert scorer.score_weights.score_names == ("psichic_affinity",)
    assert scorer.score_weights.signed_weights == (1.0,)
    with pytest.raises(ValueError, match="immutable"):
        scorer.score_weights.weights = [2.0]


@pytest.mark.parametrize("results, reason", [
    ([], "one matching"),
    ([Success("other", 7., -7.)], "one matching"),
    ([Success(AMINE, 7., -7.), Success(AMINE, 7., -7.)], "one matching"),
    ([Failure(AMINE, "ligand_normalization", "too small")], "ligand_normalization.*too small"),
    ([Success(AMINE, float("nan"), 0.)], "non-finite"),
    ([Success(AMINE, float("inf"), 0.)], "non-finite"),
])
def test_failed_predictions_never_become_affinities(results, reason):
    class Predictor:
        def score(self, inputs):
            return results
    scorer = PSICHICScorer(Predictor())
    with pytest.raises(RuntimeError, match=reason) as error:
        scorer(AMINE)
    assert AMINE in str(error.value)
    assert scorer.scores == {}


def test_prediction_exception_identifies_smiles_and_stage():
    class Predictor:
        def score(self, inputs):
            raise ValueError("broken graph")
    with pytest.raises(RuntimeError, match=f"{AMINE!r}, stage prediction, reason broken graph"):
        PSICHICScorer(Predictor())(AMINE)


@pytest.mark.parametrize("space, reaction", [("real", "22"), ("wuxi", "1")])
def test_open_fixture_generates_and_preserves_upstream_routes(tmp_path, space, reaction):
    from synthemol.generate.utils import save_generated_molecules
    catalogue = read_catalogue(EXAMPLE / "building_blocks.csv", EXAMPLE / f"{space}.pkl", space)
    assert catalogue.smiles_to_id == {AMINE: "0001", ACID: "0002"}
    for _ in range(2):
        scorer = PSICHICScorer(StubPredictor())
        generator = make_generator(catalogue, scorer, 1, 0)
        nodes = generator.generate(n_rollout=2)
        assert nodes and all(node.num_reactions == 1 for node in nodes)
        path = tmp_path / "molecules.csv"
        save_generated_molecules(nodes, {space: {"0001": AMINE, "0002": ACID}},
                                 scorer.score_weights.score_names, path)
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        assert all(row["reaction_1_id"] == reaction for row in rows)
        assert all(row["reaction_1_chemical_space"] == space for row in rows)
        assert {rows[0][f"building_block_1_{i}_id"] for i in (1, 2)} == {"0001", "0002"}
        assert float(rows[0]["psichic_affinity"]) == 7.0


def test_scores_change_the_actual_search_path():
    catalogue = read_catalogue(EXAMPLE / "building_blocks.csv", EXAMPLE / "real.pkl", "real")
    other = "CCNc1ccccc1"
    catalogue.smiles_to_id[other] = "0003"
    catalogue.reaction_map["22"][0].add(other)
    visited = []
    for favorite in (AMINE, other):
        class Predictor:
            def score(self, inputs):
                return [Success(token, 20. if smiles == favorite else 1., 0.) for token, smiles in inputs]
        generator = make_generator(catalogue, PSICHICScorer(Predictor()), 1, 0)
        generator.generate(n_rollout=1)
        visited.append({node.molecules[0] for node in generator.node_map
                        if node.num_reactions == 0 and node.num_molecules == 1 and node.num_visits})
    assert visited == [{AMINE}, {other}]


def write_blocks(tmp_path, rows):
    path = tmp_path / "blocks.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reagent_id", "smiles"])
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("rows, reason", [
    ([], "empty"),
    ([("1", AMINE), ("1", ACID)], "Duplicate"),
    ([("1", AMINE), ("2", AMINE)], "Duplicate"),
    ([("", AMINE)], "empty"),
    ([("1", "not-smiles")], "Invalid"),
])
def test_bad_catalogue_csv_is_rejected(tmp_path, rows, reason):
    with pytest.raises(ValueError, match=reason):
        read_catalogue(write_blocks(tmp_path, rows), EXAMPLE / "real.pkl", "real")


@pytest.mark.parametrize("fault, reason", [
    ("missing_reaction", "exactly the upstream reaction"),
    ("extra_reaction", "exactly the upstream reaction"),
    ("missing_slot", "integer reactant slots"),
    ("string_slot", "integer reactant slots"),
    ("unknown_smiles", "absent from the CSV"),
    ("wrong_smarts", "SMARTS rejects"),
    ("string_blocks", "list or set"),
    ("disabled", "no usable route"),
])
def test_bad_reaction_maps_are_rejected(tmp_path, fault, reason):
    mapping = pickle.loads((EXAMPLE / "real.pkl").read_bytes())
    if fault == "missing_reaction":
        del mapping["22"]
    elif fault == "extra_reaction":
        mapping["unknown"] = {}
    elif fault == "missing_slot":
        del mapping["22"][0]
    elif fault == "string_slot":
        mapping["22"]["0"] = mapping["22"].pop(0)
    elif fault == "unknown_smiles":
        mapping["22"][0] = {"CCCC"}
    elif fault == "wrong_smarts":
        mapping["22"][0] = {ACID}
    elif fault == "string_blocks":
        mapping["22"][0] = AMINE
    elif fault == "disabled":
        mapping["22"][0] = set()
    path = tmp_path / "map.pkl"
    path.write_bytes(pickle.dumps(mapping))
    with pytest.raises(ValueError, match=reason):
        read_catalogue(EXAMPLE / "building_blocks.csv", path, "real")


def test_upstream_list_maps_normalize_to_sets(tmp_path):
    mapping = pickle.loads((EXAMPLE / "wuxi.pkl").read_bytes())
    path = tmp_path / "map.pkl"
    path.write_bytes(pickle.dumps({key: {slot: list(values) for slot, values in slots.items()}
                                  for key, slots in mapping.items()}))
    catalogue = read_catalogue(EXAMPLE / "building_blocks.csv", path, "wuxi")
    assert catalogue.reaction_map["1"][0] == {AMINE}


@pytest.mark.parametrize("upstream_error", [False, True])
def test_no_products_reports_explicit_failure(tmp_path, monkeypatch, upstream_error):
    from psichic_plus_synthemol import generate
    class EmptyGenerator:
        node_map = {}
        def generate(self, n_rollout):
            if upstream_error:
                raise ValueError("min() iterable argument is empty")
            return []
    monkeypatch.setattr(generate, "Predictor", lambda *args, **kwargs: StubPredictor())
    monkeypatch.setattr(generate, "make_generator", lambda *args: EmptyGenerator())
    with pytest.raises(RuntimeError, match="No reaction products generated"):
        run(EXAMPLE / "target.fasta", tmp_path, EXAMPLE / "building_blocks.csv",
            EXAMPLE / "real.pkl", "real", tmp_path / "output")
    assert not (tmp_path / "output/molecules.csv").exists()


def test_partial_routes_are_disabled_before_search(tmp_path):
    mapping = pickle.loads((EXAMPLE / "real.pkl").read_bytes())
    bromide = "Brc1ccccc1"
    mapping["2230"][1] = {bromide}
    path = tmp_path / "map.pkl"
    path.write_bytes(pickle.dumps(mapping))
    catalogue = read_catalogue(write_blocks(tmp_path, [("0001", AMINE), ("0002", ACID), ("0003", bromide)]),
                               path, "real")
    class Predictor:
        def score(self, inputs):
            return [Success(token, 100. if smiles == bromide else 1., 0.) for token, smiles in inputs]
    for max_reactions in (1, 2):
        generator = make_generator(catalogue, PSICHICScorer(Predictor()), max_reactions, 0)
        nodes = generator.generate(n_rollout=3)
        assert nodes
        assert {reaction.id for reaction in generator.reactions} == {"22"}
        assert bromide not in generator.all_building_blocks


def test_intermediate_sentinel_is_not_a_catalogue_id(tmp_path):
    with pytest.raises(ValueError, match="reserved.*intermediate"):
        read_catalogue(write_blocks(tmp_path, [("-1", AMINE), ("0002", ACID)]),
                       EXAMPLE / "real.pkl", "real")
