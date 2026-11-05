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
