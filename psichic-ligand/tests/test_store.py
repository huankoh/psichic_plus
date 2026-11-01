import hashlib
import multiprocessing
import pickle

import pytest
import torch

from helpers import graph_digest
from psichic_ligand import (
    FEATURIZATION_VERSION,
    LigandFailure,
    LigandFeaturizationError,
    LigandStore,
    featurize_ligand,
)
from psichic_ligand import _ligands

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def unexpected(*args, **kwargs):
    raise AssertionError("A cache hit recomputed the molecule")


def test_versioned_cache_hit_and_legacy_isolation(tmp_path, monkeypatch, caplog):
    name = hashlib.sha256(ASPIRIN.encode()).hexdigest()[:32] + ".pt"
    legacy = tmp_path / name
    legacy.write_bytes(b"legacy graph must not be opened")
    store = LigandStore(tmp_path)
    graph = store.get(ASPIRIN)
    assert store.directory == tmp_path / FEATURIZATION_VERSION
    assert list(store.directory.iterdir()) == [store.directory / name]
    assert "Ignoring legacy" in caplog.text
    monkeypatch.setattr(_ligands, "featurize_ligand", unexpected)
    assert graph_digest(LigandStore(tmp_path).get(ASPIRIN)) == graph_digest(graph)
    assert legacy.read_bytes() == b"legacy graph must not be opened"


@pytest.mark.parametrize("corrupt", [b"truncated", None, {"smiles": "wrong format"}])
def test_corrupt_file_is_recomputed(tmp_path, corrupt, caplog):
    store = LigandStore(tmp_path)
    graph = store.get(ASPIRIN)
    path = store._path(ASPIRIN)
    if isinstance(corrupt, bytes):
        path.write_bytes(corrupt)
    else:
        torch.save(corrupt, path)
    assert graph_digest(store.get(ASPIRIN)) == graph_digest(graph)
    assert "Unreadable ligand cache" in caplog.text
    assert graph_digest(torch.load(path, weights_only=True)) == graph_digest(graph)


@pytest.mark.parametrize(
    "failure",
    [
        LigandFailure(
            "ligand_standardization", "INVALID_SMILES", "Could not parse SMILES"
        ),
        LigandFailure("ligand_graph", "GRAPH_CONSTRUCTION_ERROR", "detail"),
    ],
)
def test_failure_memo_and_exact_log(tmp_path, monkeypatch, failure):
    log = tmp_path / "errors.log"
    store = LigandStore(tmp_path / "cache", error_log=log)
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        return failure

    monkeypatch.setattr(_ligands, "featurize_ligand", fail)
    for _ in range(2):
        with pytest.raises(LigandFeaturizationError) as exc:
            store.get("bad")
        assert exc.value.failure == failure
        assert str(exc.value) == failure.reason
    assert len(calls) == 1
    status = (
        "failed to normalize because [INVALID_SMILES]"
        if failure.stage == "ligand_standardization"
        else "successful normalization but fail at graph construction"
    )
    message = (
        "Could not parse SMILES" if failure.stage == "ligand_standardization" else None
    )
    assert (
        log.read_text()
        == f"||||| bad ||||| with status ({status}); and error message - {message}\n"
    )
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("operation", ["serialize", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_atomic_failure_cleans_temporary_files_and_preserves_destination(
    tmp_path, monkeypatch, operation, existing
):
    store = LigandStore(tmp_path)
    graph = featurize_ligand(ASPIRIN)
    path = store._path(ASPIRIN)
    if existing:
        store._publish(path, graph)
    before = path.read_bytes() if path.exists() else None

    def fail(*args, **kwargs):
        raise OSError("injected storage failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            _ligands.torch, "save", fail
        ) if operation == "serialize" else patch.setattr(_ligands.os, "replace", fail)
        with pytest.raises(OSError, match="injected"):
            store._publish(path, graph)
    assert not list(tmp_path.rglob("*.tmp"))
    assert (path.read_bytes() if path.exists() else None) == before
    store._publish(path, graph)
    assert graph_digest(torch.load(path, weights_only=True)) == graph_digest(graph)


def test_save_failure_does_not_memoize_a_success(tmp_path, monkeypatch):
    store = LigandStore(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(
            store, "_publish", lambda *args: (_ for _ in ()).throw(OSError("full"))
        )
        with pytest.raises(OSError, match="full"):
            store.get(ASPIRIN)
    assert store.get(ASPIRIN)["atom_num_nodes"] == 13


def test_no_disk_cache_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = LigandStore(None)
    assert store.directory is None
    assert store.get(ASPIRIN)["atom_num_nodes"] == 13
    assert list(tmp_path.iterdir()) == []


def spawn_get(store, queue):
    try:
        queue.put(graph_digest(pickle.loads(pickle.dumps(store)).get(ASPIRIN)))
    except Exception as exc:
        queue.put(repr(exc))


def test_two_spawned_writers_publish_one_valid_graph(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    store = LigandStore(tmp_path)
    processes = [ctx.Process(target=spawn_get, args=(store, queue)) for _ in range(2)]
    for p in processes:
        p.start()
    results = [queue.get(timeout=60) for _ in processes]
    for p in processes:
        p.join(60)
        assert p.exitcode == 0
    assert results == [graph_digest(featurize_ligand(ASPIRIN))] * 2
    assert len(list(store.directory.iterdir())) == 1
    assert not list(tmp_path.rglob("*.tmp"))
