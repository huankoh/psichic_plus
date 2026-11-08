from applications.scoring import Acceptance, Engine, ScoringRun, Waiting
from .conftest import prepare, transfer


def test_large_population_creation_and_reading_use_a_stream(make_run):
    run = make_run(n=100003)
    assert run.row_count == 100003
    for ordinal, row in enumerate(ScoringRun.open(run.path).population(), 1):
        assert row.ordinal == ordinal and row.entry.source == f"fixture.csv:{ordinal - 1}"
    assert ordinal == 100003
    assert len(run.pending(Engine.PSICHIC, limit=37)) == 37
    assert len(run.pending(Engine.PSICHIC, after=100000, limit=37)) == 3
    assert run.finalize() == Waiting(100003, 0, 0)


def test_bounded_batches_and_disk_ranking_correctness(make_run, tmp_path):
    count = 2003
    run = make_run(n=count)
    while tasks := prepare(run, limit=31):
        for task in reversed(tasks):
            ordinal = int(task.entry_key[4:])
            directory = transfer(tmp_path / task.attempt_id, task, affinity=float(ordinal % 7))
            assert run.accept(directory) is Acceptance.ACCEPTED
    report = run.finalize(top_n=5)
    assert report.version == 1
    frequencies = [sum(1 for i in range(1, count + 1) if i % 7 == value) for value in range(7)]
    for ordinal, record in enumerate(run.results(), 1):
        value = ordinal % 7
        better = sum(frequencies[value + 1:])
        assert record["entry"]["ordinal"] == ordinal
        assert record["score"] == value
        assert record["final_rank"] == better + (frequencies[value] + 1) / 2
    assert ordinal == count
    assert not run.pending(Engine.PSICHIC)
