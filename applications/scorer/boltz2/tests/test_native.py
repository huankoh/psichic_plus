import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from applications.scorer import Failure, Success
from applications.scorer.boltz2.__main__ import main, read_candidates
from applications.scorer.boltz2.adapter import BoltzRunner, parse_affinity, recorded_command
from applications.scorer.boltz2.models import Candidate, Mode
from applications.scorer.boltz2.target import Target


def test_simple_target_adds_no_constraints_or_templates(fasta, tmp_path):
    target = Target.load(fasta=fasta, msa_mode="single")
    prepared, _ = target.prepare(tmp_path)
    path = tmp_path / "candidate.yaml"
    target.render(prepared, Candidate(1, "../unsafe", "CCO"), path)
    actual = yaml.safe_load(path.read_text())
    assert actual["sequences"][0]["protein"]["msa"] == "empty"
    assert actual["sequences"][1]["ligand"]["smiles"] == "CCO"
    assert "constraints" not in actual and "templates" not in actual


def native_target(tmp_path):
    (tmp_path / "chain.a3m").write_text(">query\nACDE\n")
    (tmp_path / "protein.cif").write_text("data_test\n")
    doc = {"version": 1, "sequences": [
        {"protein": {"id": ["A", "C"], "sequence": "ACDE", "msa": "chain.a3m"}},
        {"ligand": {"id": "B", "ccd": "ATP"}}, {"ligand": {"id": "D", "ccd": "MG"}},
    ], "templates": [{"cif": "protein.cif", "chain_id": "A", "force": True, "threshold": 2}],
        "constraints": [{"pocket": {"binder": "B", "contacts": [["A", 2]], "max_distance": 6}}],
        "properties": [{"affinity": {"binder": "B"}}]}
    path = tmp_path / "target.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path, doc


def test_native_settings_assets_and_other_chains_survive_replacement(tmp_path):
    path, original = native_target(tmp_path)
    target = Target.load(native=path, ligand_id="B", msa_mode="local")
    run = tmp_path / "run"
    run.mkdir()
    prepared, assets = target.prepare(run)
    output = run / "row.yaml"
    target.render(prepared, Candidate(1, "same", "CN"), output)
    doc = yaml.safe_load(output.read_text())
    assert target.document == original
    assert doc["sequences"][1] == {"ligand": {"id": "B", "smiles": "CN"}}
    assert doc["sequences"][2] == original["sequences"][2]
    assert doc["constraints"] == original["constraints"]
    assert doc["properties"] == original["properties"]
    assert Path(doc["sequences"][0]["protein"]["msa"]).read_text() == ">query\nACDE\n"
    assert Path(doc["templates"][0]["cif"]).read_text() == "data_test\n"
    assert doc["templates"][0]["force"] is True
    assert len(assets) == 2
    with pytest.raises(ValueError, match="explicitly"):
        target.protein_sequence()
    assert target.protein_sequence("C") == ("C", "ACDE")


def test_single_sequence_preserves_multichain_targets(tmp_path):
    path, doc = native_target(tmp_path)
    del doc["sequences"][0]["protein"]["msa"]
    path.write_text(yaml.safe_dump(doc))
    target = Target.load(native=path, ligand_id="B", msa_mode="single")
    prepared, _ = target.prepare(tmp_path)
    assert prepared["sequences"][0]["protein"]["id"] == ["A", "C"]
    assert prepared["sequences"][0]["protein"]["msa"] == "empty"


@pytest.mark.parametrize("change,reason", [
    (lambda d: d.update(version=2), "version"),
    (lambda d: d["properties"][0]["affinity"].update(binder="D"), "Affinity"),
    (lambda d: d["sequences"][1]["ligand"].update(id=["B", "E"]), "single ligand"),
    (lambda d: d["sequences"][2]["ligand"].update(id="A"), "Duplicate"),
    (lambda d: d.update(constraints=[{"bond": {"atom1": ["B", 1, "C1"], "atom2": ["A", 1, "CA"]}}]), "bond"),
])
def test_incompatible_native_inputs_are_rejected(tmp_path, change, reason):
    path, doc = native_target(tmp_path)
    change(doc)
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match=reason):
        Target.load(native=path, ligand_id="B", msa_mode="local")


def test_msa_choice_never_silently_replaces_local_input(tmp_path):
    path, _ = native_target(tmp_path)
    with pytest.raises(ValueError, match="discard"):
        Target.load(native=path, ligand_id="B", msa_mode="single")


def test_cli_rejects_mixed_custom_and_automatic_msas_before_execution(tmp_path, capsys):
    path, doc = native_target(tmp_path)
    doc["sequences"].append({"protein": {"id": "E", "sequence": "ACDE"}})
    path.write_text(yaml.safe_dump(doc))
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("ID,SMILES\nligand,CCO\n")
    output = tmp_path / "run"
    assert main(["--mode", "boltz-only", "--candidates", str(candidates),
                 "--target-yaml", str(path), "--ligand-id", "B", "--msa-mode", "server",
                 "--boltz-python", "/unused/python", "--output", str(output)]) == 2
    assert "cannot mix custom and auto-generated MSAs" in capsys.readouterr().err
    assert not output.exists()


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "target.yaml"
    path.write_text("version: 1\nversion: 1\n")
    with pytest.raises(ValueError, match="Duplicate YAML"):
        Target.load(native=path, ligand_id="B", msa_mode="single")


def test_native_yaml_merge_overrides_match_upstream(tmp_path):
    path = tmp_path / "target.yaml"
    path.write_text("""version: 1
sequences:
  - protein: &protein
      id: A
      sequence: ACDE
  - protein:
      <<: *protein
      id: C
  - protein:
      <<: [*protein, {sequence: XXXX}]
      id: D
  - ligand:
      id: B
      smiles: CCO
""")
    target = Target.load(native=path, ligand_id="B", msa_mode="single")
    assert target.document == yaml.safe_load(path.read_text())
    prepared, _ = target.prepare(tmp_path)
    output = tmp_path / "row.yaml"
    target.render(prepared, Candidate(1, "candidate", "CN"), output)
    proteins = [entry["protein"] for entry in yaml.safe_load(output.read_text())["sequences"][:3]]
    assert [protein["id"] for protein in proteins] == ["A", "C", "D"]
    assert all(protein["sequence"] == "ACDE" and protein["msa"] == "empty" for protein in proteins)


def test_duplicate_explicit_keys_inside_yaml_merge_are_rejected(tmp_path):
    path = tmp_path / "target.yaml"
    path.write_text("version: 1\nsequences:\n  - protein:\n      <<: {id: A, id: C, sequence: ACDE}\n")
    with pytest.raises(ValueError, match="Duplicate YAML"):
        Target.load(native=path, ligand_id="B", msa_mode="single")


@pytest.mark.parametrize("value", [None, True, "0.5", float("nan"), float("inf"), 1.1, -.1])
def test_native_probability_validation(tmp_path, value):
    path = tmp_path / "affinity.json"
    path.write_text(json.dumps({"affinity_probability_binary": value, "affinity_pred_value": 2}))
    with pytest.raises(ValueError):
        parse_affinity(path, Candidate(1, "ID", "CCO"))


def test_adapter_uses_native_paths_and_keeps_partial_outputs(fake_boltz, fasta, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BOLTZ_EXIT", "7")
    directory = tmp_path / "boltz"
    directory.mkdir()
    target = Target.load(fasta=fasta, msa_mode="single")
    runner = BoltzRunner(fake_boltz, target, directory, options=("--accelerator", "cpu", "--sampling_steps=2"))
    runner.check()
    rows = [Candidate(1, "same", "CCO"), Candidate(2, "same", "invalid")]
    run = runner(rows)
    assert run.error and run.metadata["returncode"] == 7
    first = run.outcomes[rows[0].key]
    assert first.affinity_probability_binary == .1
    assert first.native["affinity_pred_value1"] == -.1
    assert (Path(first.output_directory) / f"{rows[0].key}_model_0.cif").exists()
    assert isinstance(run.outcomes[rows[1].key], Failure)
    args = json.loads((directory / "native/boltz_results_inputs/argv.json").read_text())
    assert "--sampling_steps=2" in args and "--use_msa_server" not in args
    assert not list(directory.rglob("same*"))


def test_timeout_returns_failures_and_preserves_logs(fake_boltz, fasta, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BOLTZ_TIMEOUT", "1")
    directory = tmp_path / "boltz"
    directory.mkdir()
    runner = BoltzRunner(fake_boltz, Target.load(fasta=fasta, msa_mode="single"), directory, timeout=.1)
    run = runner([Candidate(1, "a", "CCO")])
    assert run.error and isinstance(run.outcomes["row_00000001"], Failure)
    assert (directory / "stderr.log").is_file()


@pytest.mark.parametrize("option", ["--out_dir=/tmp/out", "--model", "--use_msa_server", "--override", "--help", "--"])
def test_adapter_owned_options_cannot_be_overridden(option, fake_boltz, fasta, tmp_path):
    with pytest.raises(ValueError, match="owns"):
        BoltzRunner(fake_boltz, Target.load(fasta=fasta, msa_mode="single"), tmp_path, options=[option])


def test_secret_values_are_not_written_in_command_metadata():
    assert recorded_command(["--msa_server_password", "secret", "--api_key_value=other"]) == [
        "--msa_server_password", "<redacted>", "--api_key_value=<redacted>",
    ]


@pytest.mark.parametrize("mode", list(Mode))
def test_cli_four_modes_with_controlled_engines(mode, fake_boltz, fasta, tmp_path, monkeypatch):
    from applications.scorer.psichic_plus import predictor

    class FakePredictor:
        def __init__(self, *args):
            self.helpers = SimpleNamespace(__file__=str(fasta.parent / "inference.py"))

        def score(self, rows):
            return [Success(key, float(index), -float(index)) for index, (key, _) in enumerate(rows)]

    monkeypatch.setattr(predictor, "Predictor", FakePredictor)
    (fasta.parent / "inference.py").write_text("controlled_scorer = True\n")
    candidates = tmp_path / "candidates.csv"
    candidates.write_text('ID,SMILES\nsame,CCO\nsame,CCN\n,CCC\n')
    output = tmp_path / mode.value
    args = ["--mode", mode.value, "--candidates", str(candidates), "--fasta", str(fasta), "--msa-mode", "single",
            "--boltz-python", str(fake_boltz), "--output", str(output)]
    if mode is not Mode.BOLTZ:
        args += ["--model-dir", str(tmp_path)]
    if mode.pyramid:
        args += ["--shortlist-size", "2"]
    assert main([*args, "--", "--accelerator", "cpu"]) == 0
    records = json.loads((output / "results.json").read_text())
    assert [r["candidate"]["id"] for r in records] == ["same", "same", ""]
    assert len({r["candidate"]["key"] for r in records}) == 3
    assert sum(r["boltz"]["status"] == "success" for r in records) == (2 if mode.pyramid else 3)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    if mode.ensemble:
        assert manifest["ensemble"]["method"] == "rank-v1"
        assert manifest["ensemble"]["direction"] == "ascending"
        assert manifest["ranking"]["ensemble"] == (
            "experimental weighted mean of ranks within the both-successful cohort; lower is better"
        )
        assert all(record["normalized_ensemble_score"] is None for record in records)
        assert any(record["experimental_combined_value"] is not None for record in records)
    if mode is not Mode.BOLTZ:
        assert manifest["psichic"]["source"]["screening_directory"] == str(fasta.parent)
        assert manifest["engines"]["psichic"]["source"]["source_sha256"]
    before = (output / "results.json").read_bytes()
    assert main(args) == 2
    assert (output / "results.json").read_bytes() == before


def test_csv_ids_are_strings_even_when_they_look_missing(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text('ID,SMILES\nNone,CCO\nnan,CCN\n,CCC\n001,CCO\n')
    assert [row.id for row in read_candidates(path)] == ["None", "nan", "", "001"]


@pytest.mark.parametrize("mode,options", [
    ("full-ensemble", ["--ensemble", "concentration-v1", "--rank-by", "affinity_probability_binary"]),
    ("full-ensemble", ["--ensemble", "concentration-v1", "--psichic-weight", "1.0"]),
    ("full-ensemble", ["--ensemble", "concentration-v1", "--boltz-weight", "1.0"]),
    ("pyramid-refinement", ["--shortlist-size", "1", "--ensemble", "rank-v1"]),
    ("boltz-only", ["--ensemble", "concentration-v1"]),
    ("boltz-only", ["--psichic-output", "affinity"]),
    ("boltz-only", ["--psichic-output", "distribution"]),
    ("boltz-only", ["--psichic-weight", "0.5"]),
])
def test_cli_rejects_incompatible_explicit_options_before_engine_or_output(mode, options, fasta, tmp_path, capsys):
    output = tmp_path / "rejected"
    assert main(["--mode", mode, "--candidates", "/unused.csv", "--fasta", str(fasta), "--msa-mode", "single",
                 "--output", str(output), *options]) == 2
    assert "Configuration error" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("mode,distribution_failure", [("full-ensemble", False), ("pyramid-ensemble", False),
                                                       ("full-ensemble", True), ("pyramid-ensemble", True)])
def test_module_cli_concentration_with_bounded_controlled_engines(mode, distribution_failure, fake_boltz, fasta, tmp_path):
    import csv
    import subprocess
    import sys

    candidates = tmp_path / "candidates.csv"
    candidates.write_text("ID,SMILES\nsame,CCO\nsame,CCO\n,CCN\n")
    (tmp_path / "inference.py").write_text("controlled_scorer = True\n")
    output = tmp_path / "module-output"
    bootstrap = '''
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from types import SimpleNamespace
from applications.scorer import BetaDistribution, DistributionFailure, Output, Success
from applications.scorer.psichic_plus import predictor
source = Path(sys.argv.pop(1))
fault = sys.argv.pop(1) == "True"
class ControlledPredictor:
    def __init__(self, *args):
        self.helpers = SimpleNamespace(__file__=str(source))
    def score(self, pairs, *, output):
        assert output is Output.DISTRIBUTION
        outcomes = []
        for index, (key, _) in enumerate(pairs):
            alpha, beta = [(1, 3), (4, 4), (12, 4)][index]
            distribution = (DistributionFailure("controlled invalid distribution", "bad", "4")
                            if fault and index == 2 else BetaDistribution(alpha, beta))
            affinity = 12 * alpha / (alpha + beta)
            outcomes.append(Success(key, affinity, -affinity, distribution))
        return outcomes
predictor.Predictor = ControlledPredictor
runpy.run_module("applications.scorer.boltz2", run_name="__main__")
'''
    bootstrap_path = tmp_path / "controlled_cli.py"
    bootstrap_path.write_text(bootstrap)
    command = [sys.executable, str(bootstrap_path), str(tmp_path / "inference.py"), str(distribution_failure),
               "--mode", mode, "--ensemble", "concentration-v1", "--candidates", str(candidates),
               "--fasta", str(fasta), "--msa-mode", "single", "--model-dir", str(tmp_path),
               "--boltz-python", str(fake_boltz), "--boltz-timeout", "5", "--output", str(output)]
    if mode == "pyramid-ensemble":
        command += ["--shortlist-size", "3"]
    command += ["--", "--accelerator", "cpu"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
    (tmp_path / "cli-stdout.log").write_text(completed.stdout)
    (tmp_path / "cli-stderr.log").write_text(completed.stderr)
    assert completed.returncode == int(distribution_failure), completed.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    records = json.loads((output / "results.json").read_text())
    csv_records = list(csv.DictReader((output / "results.csv").open()))
    assert [r["candidate"]["id"] for r in records] == ["same", "same", ""]
    assert manifest["plan"]["ensemble"] == "concentration-v1"
    assert manifest["plan"]["metric"] == "affinity_pred_value"
    assert manifest["psichic"]["requested_output"] == "affinity"
    assert manifest["psichic"]["required_output"] == "distribution"
    assert manifest["ranking"]["ensemble_method"]["meaning"] == "relative normalized score, not a calibrated affinity"
    assert manifest["ensemble"]["reference_commit"] == "00a44d9a99cac5dc35913b0f9e40ed9d82169896"
    assert manifest["ensemble"]["direction"] == "descending"
    assert all(record["experimental_combined_value"] is None for record in records)
    assert csv_records[0]["ensemble_method"] == "concentration-v1"
    assert csv_records[0]["psichic_alpha"] == "1"
    assert csv_records[0]["psichic_concentration"] == "4"
    if distribution_failure:
        assert manifest["status"] == "partial"
        assert records[2]["psichic"]["raw_affinity"] == 9
        assert records[2]["psichic"]["distribution"]["state"] == "distribution-failure"
        assert csv_records[2]["psichic_distribution_failure"] == "controlled invalid distribution"
        assert csv_records[2]["psichic_raw_alpha"] == "bad"
        assert records[2]["unranked_reason"] == "psichic_distribution_failed"
    if distribution_failure and mode == "pyramid-ensemble":
        assert "boltz" not in manifest["engines"]
        assert not (output / "boltz/inputs").exists()
        assert all(record["boltz"]["status"] == "not_run" for record in records)
        assert not manifest["ensemble_cohort"]
    else:
        assert all(record["boltz"]["status"] == "success" for record in records)
        assert len(manifest["ensemble_cohort"]) == (2 if distribution_failure else 3)
        assert records[0]["normalized_ensemble_score"] == 0
        assert records[1]["normalized_ensemble_score"] == (1.0 if distribution_failure else .5)
        prepared = yaml.safe_load((output / "boltz/inputs/row_00000001.yaml").read_text())
        assert prepared["sequences"][0]["protein"]["msa"] == "empty"


@pytest.mark.parametrize("weights", [[], ["--psichic-weight", "1.0", "--boltz-weight", "1.0"]])
def test_module_cli_boltz_only_subprocess(weights, fake_boltz, fasta, tmp_path):
    import subprocess
    import sys

    candidates = tmp_path / "candidates.csv"
    candidates.write_text("ID,SMILES\none,CCO\n")
    output = tmp_path / "boltz-module"
    completed = subprocess.run([sys.executable, "-m", "applications.scorer.boltz2", "--mode", "boltz-only",
                                "--candidates", str(candidates), "--fasta", str(fasta), "--msa-mode", "single",
                                "--output", str(output), "--boltz-python", str(fake_boltz), "--boltz-timeout", "5",
                                *weights, "--", "--accelerator", "cpu"], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert json.loads((output / "results.json").read_text())[0]["final_rank"] == 1


@pytest.mark.parametrize("mode", [Mode.ENSEMBLE, Mode.REFINEMENT])
def test_cli_explicit_distribution_and_compatible_rank_options(mode, fake_boltz, fasta, tmp_path, monkeypatch):
    from applications.scorer import BetaDistribution, Output
    from applications.scorer.psichic_plus import predictor

    class RichPredictor:
        def __init__(self, *args):
            self.helpers = SimpleNamespace(__file__=str(tmp_path / "inference.py"))

        def score(self, pairs, *, output):
            assert output is Output.DISTRIBUTION
            return [Success(key, 6, -6, BetaDistribution(4, 4)) for key, _ in pairs]

    (tmp_path / "inference.py").write_text("controlled_scorer = True\n")
    monkeypatch.setattr(predictor, "Predictor", RichPredictor)
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("ID,SMILES\nfirst,CCO\nsecond,CCN\n")
    output = tmp_path / "rank-distribution"
    args = ["--mode", mode.value, "--candidates", str(candidates), "--fasta", str(fasta), "--msa-mode", "single",
            "--psichic-output", "distribution", "--model-dir", str(tmp_path), "--boltz-python", str(fake_boltz),
            "--output", str(output)]
    if mode.ensemble:
        args += ["--ensemble", "rank-v1", "--rank-by", "affinity_probability_binary",
                 "--psichic-weight", "1.0", "--boltz-weight", "1.0"]
    else:
        args += ["--shortlist-size", "2"]
    assert main(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    records = json.loads((output / "results.json").read_text())
    assert manifest["psichic"]["required_output"] == "distribution"
    assert all(record["psichic"]["distribution"]["state"] == "valid" for record in records)
    assert all(record["normalized_ensemble_score"] is None for record in records)
    assert [record["final_rank"] for record in records] == [2, 1]
    if mode.ensemble:
        assert [record["experimental_combined_value"] for record in records] == [1.75, 1.25]
