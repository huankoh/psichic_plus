import ast
import csv
import inspect
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("synthemol")

from applications.scorer import Failure, Success
from applications.generator import UPSTREAM_REVISION
from applications.generator.prepare import score_building_blocks
from synthemol.generate.scorer import MoleculeScorer, create_scorer
from synthemol.generate.score_weights import ScoreWeights
from synthemol.reactions import CHEMICAL_SPACE_TO_REACTIONS

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "applications/generator/example"
AMINE = "CNc1ccccc1"
ACID = "O=C(O)c1ccccc1"


@pytest.fixture
def predictor(monkeypatch):
    class Predictor:
        instances = []
        results = None

        def __init__(self, *args, **kwargs):
            self.calls = []
            self.settings = args, kwargs
            self.instances.append(self)

        def score(self, inputs):
            self.calls.append(inputs)
            return self.results if self.results is not None else [Success(token, 7., -7.) for token, _ in inputs]

    monkeypatch.setattr("applications.scorer.predictor.Predictor", Predictor)
    return Predictor


def molecule_scorer(tmp_path, **kwargs):
    return MoleculeScorer(
        score_types=["psichic_plus"], score_weights=ScoreWeights([1.], immutable=True),
        model_paths=[tmp_path / "weights"], psichic_target_fasta=EXAMPLE / "target.fasta",
        psichic_cache_dir=tmp_path / "cache", **kwargs,
    )


def test_native_cache_precedes_prediction_and_predictor_initializes_once(tmp_path, predictor):
    scorer = molecule_scorer(tmp_path, smiles_to_scores={AMINE: [123.]})
    assert scorer(AMINE) == 123.
    assert not predictor.instances
    assert scorer(ACID) == 7.
    assert scorer.compute_individual_scores(ACID) == [7.]
    assert scorer(ACID) == 7.
    assert len(predictor.instances) == 1
    assert predictor.instances[0].calls == [[(ACID, ACID)]]
    scorer("c1ccccc1NC")
    assert predictor.instances[0].calls[-1] == [("c1ccccc1NC", "c1ccccc1NC")]


@pytest.mark.parametrize("results, reason", [
    ([], "one matching"),
    ([Success("other", 7., -7.)], "one matching"),
    ([Success(AMINE, 7., -7.), Success(AMINE, 7., -7.)], "one matching"),
    ([Failure(AMINE, "ligand_normalization", "too small")], "ligand_normalization.*too small"),
    ([Success(AMINE, float("nan"), 0.)], "non-finite"),
    ([Success(AMINE, float("inf"), 0.)], "non-finite"),
])
def test_prediction_failure_has_smiles_stage_reason_and_is_not_cached(tmp_path, predictor, results, reason):
    predictor.results = results
    scorer = molecule_scorer(tmp_path, smiles_to_scores={})
    with pytest.raises(RuntimeError, match=reason) as error:
        scorer(AMINE)
    assert AMINE in str(error.value)
    predictor.results = None
    assert scorer(AMINE) == 7.


def test_predictor_exception_is_visible(tmp_path, predictor):
    scorer = molecule_scorer(tmp_path, smiles_to_scores={})
    def fail(self, inputs):
        raise ValueError("broken graph")
    predictor.score = fail
    with pytest.raises(RuntimeError, match="stage prediction, reason broken graph"):
        scorer(AMINE)


def test_native_composition_signs_weights_and_dynamic_reweighting(tmp_path, predictor):
    from rdkit import Chem
    from rdkit.Chem.Crippen import MolLogP
    weights = ScoreWeights([.25, .75], immutable=False, score_signs=[-1, 1])
    scorer = MoleculeScorer(
        ["psichic_plus", "clogp"], weights, model_paths=[tmp_path, None],
        psichic_target_fasta=EXAMPLE / "target.fasta", psichic_cache_dir=tmp_path,
        smiles_to_scores={AMINE: [50., 3.]},
    )
    assert scorer(AMINE) == -10.25
    assert not predictor.instances
    assert scorer(ACID) == pytest.approx(-.25 * 7. + .75 * MolLogP(Chem.MolFromSmiles(ACID)))
    weights.weights = [.5, .5]
    from synthemol.generate.node import Node
    assert Node(10., scorer, molecules=(AMINE,)).property_score == -23.5


def test_native_device_and_explicit_override_reach_predictor(tmp_path, predictor):
    import torch
    scorer = molecule_scorer(tmp_path, smiles_to_scores={}, device=torch.device("cuda"))
    assert not predictor.instances
    scorer(AMINE)
    assert predictor.instances[0].settings[0][2] == "cuda"
    scorer = molecule_scorer(tmp_path, smiles_to_scores={}, psichic_device="cpu", device=torch.device("cuda"))
    scorer(AMINE)
    assert predictor.instances[-1].settings[0][2] == "cpu"


def test_native_psichic_requires_model_target_and_no_fingerprint(tmp_path):
    with pytest.raises(ValueError, match="requires a model"):
        create_scorer("psichic_plus")
    with pytest.raises(ValueError, match="fingerprints"):
        create_scorer("psichic_plus", model_path=tmp_path, psichic_target_fasta=tmp_path,
                      psichic_cache_dir=tmp_path, fingerprint_type="morgan")


def prepare(tmp_path, **kwargs):
    return score_building_blocks(
        tmp_path / "input.csv", tmp_path / "output.csv", EXAMPLE / "target.fasta",
        tmp_path / "weights", tmp_path / "cache", **kwargs,
    )


def test_preparation_preserves_ids_columns_and_other_scores(tmp_path, predictor):
    (tmp_path / "input.csv").write_text(f'reagent_id,smiles,other_score,note\n0001,{AMINE},9.50,"one,two"\n')
    prepare(tmp_path)
    with (tmp_path / "output.csv").open() as handle:
        assert list(csv.DictReader(handle)) == [{
            "reagent_id": "0001", "smiles": AMINE, "other_score": "9.50", "note": "one,two", "score": "7.0",
        }]
    with pytest.raises(FileExistsError):
        prepare(tmp_path)
    assert len(predictor.instances) == 1


def test_preparation_score_replacement_requires_explicit_flag(tmp_path, predictor):
    (tmp_path / "input.csv").write_text(f'reagent_id,smiles,score\n0001,{AMINE},88\n')
    with pytest.raises(ValueError, match="already exists"):
        prepare(tmp_path)
    assert not predictor.instances
    prepare(tmp_path, overwrite_score_column=True)
    assert (tmp_path / "output.csv").read_text().endswith(f'0001,{AMINE},7.0\n')
    (tmp_path / "output.csv").write_text("old output")
    prepare(tmp_path, overwrite_score_column=True, overwrite_output=True)
    assert "old output" not in (tmp_path / "output.csv").read_text()


def test_preparation_failure_writes_no_placeholder(tmp_path, predictor):
    (tmp_path / "input.csv").write_text(f'reagent_id,smiles\n0001,{AMINE}\n')
    predictor.results = [Failure(AMINE, "standardization", "rejected")]
    with pytest.raises(RuntimeError, match="rejected"):
        prepare(tmp_path)
    assert not (tmp_path / "output.csv").exists()


def run_cli(output, args=(), source=None, descriptor_predictor=False):
    command = [sys.executable, "-m", "applications.generator"] if source is None else [
        sys.executable, "-c", "from synthemol.generate.generate import generate_command_line; generate_command_line()",
    ]
    if descriptor_predictor:
        command = [sys.executable, "-c", """
import runpy
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from applications.scorer import Success
import applications.scorer.predictor as predictor
class DescriptorPredictor:
    def __init__(self, *args, **kwargs):
        pass
    def score(self, inputs):
        return [Success(token, MolLogP(Chem.MolFromSmiles(smiles)), 0.) for token, smiles in inputs]
predictor.Predictor = DescriptorPredictor
runpy.run_module('applications.generator', run_name='__main__')
"""]
    env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if source is not None:
        env["PYTHONPATH"] = str(source) + os.pathsep + str(ROOT)
    result = subprocess.run(command + list(args), cwd=ROOT, env=env, text=True, capture_output=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "command.txt").write_text(repr(command + list(args)) + "\n")
    (output / "cli.log").write_text(result.stdout + result.stderr)
    return result


def cli_args(tmp_path, spaces=("real",), **options):
    blocks = tmp_path / "blocks.csv"
    if not blocks.exists():
        blocks.write_text(f"reagent_id,smiles,score\n0001,{AMINE},20\n0002,{ACID},19\n")
    args = ["--search_type", "mcts", "--save_dir", str(tmp_path / "result"), "--score_types", "clogp",
            "--chemical_spaces", *spaces, "--building_blocks_paths", *[str(blocks)] * len(spaces),
            "--reaction_to_building_blocks_paths", *[str(EXAMPLE / f"{space}.pkl") for space in spaces],
            "--n_rollout", "2", "--save_frequency", "1", "--store_nodes", "--rng_seed", "0"]
    for name, value in options.items():
        args += [f"--{name}"] + ([] if value is True else [str(value)])
    return args


@pytest.mark.parametrize("spaces", [("real",), ("wuxi",), ("real", "wuxi")])
def test_native_cli_spaces_chunked_saves_logs_stats_and_ids(tmp_path, spaces):
    result = run_cli(tmp_path, cli_args(tmp_path, spaces))
    assert result.returncode == 0, result.stderr
    assert "Running rollouts 0 through 0" in result.stdout
    assert "Running rollouts 1 through 1" in result.stdout
    rows = list(csv.DictReader((tmp_path / "result/molecules.csv").open()))
    assert rows and {row["reaction_1_chemical_space"] for row in rows} <= set(spaces)
    assert all(f"Loaded 2 {space} building blocks" in result.stdout for space in spaces)
    assert all({row["building_block_1_1_id"], row["building_block_1_2_id"]} == {"0001", "0002"} for row in rows)
    stats = list(csv.DictReader((tmp_path / "result/generation_stats.csv").open()))[0]
    assert int(stats["num_nodes_searched"]) > 0
    assert (tmp_path / "result/logs.pkl").stat().st_size > 0


def test_native_cli_custom_columns_and_single_score_sign_weight(tmp_path):
    (tmp_path / "blocks.csv").write_text(f"identifier,molecule,affinity\n0001,{AMINE},20\n0002,{ACID},19\n")
    args = cli_args(tmp_path) + [
        "--building_blocks_id_column", "identifier", "--building_blocks_smiles_column", "molecule",
        "--building_blocks_score_columns", "affinity", "--score_names", "logp",
        "--base_score_weights", "0.25", "--score_signs", "-1", "--explore_weight", "5",
        "--num_expand_nodes", "2", "--no_building_block_diversity", "--verbose",
    ]
    result = run_cli(tmp_path, args)
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader((tmp_path / "result/molecules.csv").open()))
    assert rows and all(float(row["score"]) == pytest.approx(-.25 * float(row["logp"])) for row in rows)


def two_step_args(tmp_path):
    hydroxy_acid = "O=C(O)c1ccc(O)cc1"
    bromide = "Brc1ccccc1"
    (tmp_path / "blocks.csv").write_text(
        f"reagent_id,smiles,score\n0001,{AMINE},100\n0002,{hydroxy_acid},99\n0003,{bromide},-100\n"
    )
    mapping = {reaction.id: {index: set() for index in range(reaction.num_reactants)}
               for reaction in CHEMICAL_SPACE_TO_REACTIONS["real"]}
    mapping["22"] = {0: {AMINE}, 1: {hydroxy_acid}}
    mapping["7_2"] = {0: set(), 1: {bromide}}
    path = tmp_path / "partial.pkl"
    path.write_bytes(pickle.dumps(mapping))
    args = cli_args(tmp_path, max_reactions=2)
    index = args.index("--reaction_to_building_blocks_paths")
    args[index + 1] = str(path)
    return args


def test_native_cli_keeps_empty_slot_for_two_step_intermediate(tmp_path):
    result = run_cli(tmp_path, two_step_args(tmp_path))
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader((tmp_path / "result/molecules.csv").open()))
    routes = [row for row in rows if row["num_reactions"] == "2"]
    assert routes, rows
    assert any(row["reaction_1_id"] == "22" and float(row["reaction_2_id"]) == 7 and
               row["building_block_2_1_id"] == "-1" and row["building_block_2_2_id"] == "0003" for row in routes)


@pytest.fixture(scope="session")
def baseline(tmp_path_factory):
    source = os.environ.get("SYNTHEMOL_SOURCE")
    if not source:
        pytest.skip("Set SYNTHEMOL_SOURCE to the read-only pinned upstream Git checkout for parity comparisons")
    root = tmp_path_factory.mktemp("synthemol-baseline")
    subprocess.run(["git", "clone", "--no-hardlinks", source, str(root / "source")], check=True, capture_output=True)
    subprocess.run(["git", "checkout", "--detach", UPSTREAM_REVISION], cwd=root / "source", check=True, capture_output=True)
    return root / "source"


def test_native_signature_and_cli_flags_preserve_every_original_default(tmp_path, baseline):
    from importlib import import_module
    module = import_module("synthemol.generate.generate")
    old_tree = ast.parse((baseline / "synthemol/generate/generate.py").read_text())
    new_tree = ast.parse(inspect.getsource(module))
    old = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate").args
    new = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate").args
    assert [ast.dump(a) for a in old.args] == [ast.dump(a) for a in new.args[:len(old.args)]]
    assert [ast.dump(d) for d in old.defaults] == [ast.dump(d) for d in new.defaults[:-3]]
    old_help = run_cli(tmp_path / "old", ["--help"], baseline)
    new_help = run_cli(tmp_path / "new", ["--help"])
    assert old_help.returncode == new_help.returncode == 0
    def flags(text):
        return set(re.findall(r"--[a-z_]+", text))
    assert flags(new_help.stdout) - flags(old_help.stdout) == {
        "--psichic_target_fasta", "--psichic_device", "--psichic_cache_dir",
    }
    assert flags(old_help.stdout) <= flags(new_help.stdout)


@pytest.mark.parametrize("case", ["mcts", "two-step", "rl", "dead-end", "qed", "rl-default", "replicate", "mixed", "only-partial"])
def test_seeded_native_results_and_failures_match_pinned_upstream(tmp_path, baseline, case):
    args = two_step_args(tmp_path) if case == "two-step" else cli_args(tmp_path)
    if case == "rl":
        args[args.index("mcts")] = "rl"
        args += ["--rl_model_type", "mlp", "--rl_model_fingerprint_type", "rdkit",
                 "--rl_prediction_types", "regression", "--rl_train_frequency", "1", "--rl_train_epochs", "1",
                 "--rl_temperature_similarity_target", "-1"]
    elif case in ("dead-end", "only-partial"):
        mapping = pickle.loads((EXAMPLE / "real.pkl").read_bytes())
        mapping["2230"][1] = {"Brc1ccccc1"}
        if case == "only-partial":
            mapping["22"] = {0: set(), 1: set()}
        path = tmp_path / "dead-end.pkl"
        path.write_bytes(pickle.dumps(mapping))
        (tmp_path / "blocks.csv").write_text(f"reagent_id,smiles,score\n0001,{AMINE},1\n0002,{ACID},1\n0003,Brc1ccccc1,100\n")
        args[args.index("--reaction_to_building_blocks_paths") + 1] = str(path)
    elif case == "mixed":
        (tmp_path / "blocks.csv").write_text(f"reagent_id,smiles,score,other\n0001,{AMINE},20,3\n0002,{ACID},19,3\n")
        args[args.index("clogp") + 1:args.index("clogp") + 1] = ["sp2_network"]
        args += ["--building_blocks_score_columns", "score", "other", "--base_score_weights", ".25", ".75",
                 "--score_signs", "-1", "1"]
    elif case == "qed":
        args[args.index("clogp")] = "qed"
    elif case == "rl-default":
        args[args.index("mcts")] = "rl"
    elif case == "replicate":
        args += ["--replicate_mcts"]
    old_args = args.copy()
    old_args[old_args.index("--save_dir") + 1] = str(tmp_path / "baseline-result")
    old = run_cli(tmp_path / "baseline-log", old_args, baseline)
    new = run_cli(tmp_path / "patched-log", args)
    assert old.returncode == new.returncode
    if case in ("dead-end", "only-partial", "qed", "rl-default", "replicate"):
        assert new.returncode != 0
        assert old.stderr.splitlines()[-1] == new.stderr.splitlines()[-1]
    else:
        assert new.returncode == 0, new.stderr
        assert (tmp_path / "result/molecules.csv").read_bytes() == (tmp_path / "baseline-result/molecules.csv").read_bytes()
        old_stats = list(csv.DictReader((tmp_path / "baseline-result/generation_stats.csv").open()))[0]
        new_stats = list(csv.DictReader((tmp_path / "result/generation_stats.csv").open()))[0]
        assert {k: v for k, v in new_stats.items() if k != "generation_time"} == {
            k: v for k, v in old_stats.items() if k != "generation_time"
        }


def test_psichic_factory_cli_matches_native_baseline_with_explicit_descriptor_fixture(tmp_path, baseline):
    args = two_step_args(tmp_path)
    old_args = args.copy()
    old_args[old_args.index("--save_dir") + 1] = str(tmp_path / "baseline-result")
    old = run_cli(tmp_path / "baseline-log", old_args, baseline)
    args[args.index("clogp")] = "psichic_plus"
    args += ["--score_model_paths", str(tmp_path / "unused-model"),
             "--psichic_target_fasta", str(EXAMPLE / "target.fasta")]
    new = run_cli(tmp_path / "patched-log", args, descriptor_predictor=True)
    assert old.returncode == new.returncode == 0, new.stderr
    assert (tmp_path / "result/molecules.csv").read_bytes() == (tmp_path / "baseline-result/molecules.csv").read_bytes()
