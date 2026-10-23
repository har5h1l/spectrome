"""Queue checks use fake runners and never launch queued cohort experiments."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
from spectrome.training import run


PARAMETERS = dict(hidden_sizes=[8], activation="tanh", omega_scale=282.7433388230814,
                  epochs=100, learning_rate=0.001, seed=7)


def queue(tmp_path, *names):
    path = tmp_path / "queue.json"
    path.write_text(json.dumps([{"results_dir": name, "parameters": PARAMETERS} for name in names]))
    return path


def completed(directory, *, seed=7):
    directory.mkdir(parents=True)
    record = {"status": "complete", "results_path": "results.pt",
              "model_config": {key: PARAMETERS[key] for key in ("hidden_sizes", "omega_scale")},
              "training_config": {"epochs": 100, "learning_rate": 0.001, "seed": seed}}
    (directory / "config.json").write_text(json.dumps(record))
    (directory / "results.pt").write_bytes(b"fake result for queue bookkeeping")


def test_shipped_queue_and_completed_baseline_skip(tmp_path, monkeypatch):
    entries = json.loads(Path(run.__file__).with_name("experiments.json").read_text())
    assert [entry["results_dir"] for entry in entries] == [f"basic_mlp_{i}" for i in range(16)]
    expected = [(widths, activation, epochs)
                for widths in ([8], [32], [32, 32])
                for epochs in (100, 500) for activation in ("tanh", "sigmoid")]
    expected.append(([128] * 5, "tanh", 5000))
    expected.extend((widths, "tanh", 5000) for widths in ([128] * 5, [32], [32, 32]))
    assert [(e["parameters"]["hidden_sizes"], e["parameters"]["activation"],
             e["parameters"]["epochs"]) for e in entries] == expected
    assert all(e["parameters"][key] == PARAMETERS[key] for e in entries
               for key in ("omega_scale", "seed"))
    assert [e["parameters"]["learning_rate"] for e in entries] == [0.001] * 13 + [0.0003] * 3
    baseline, next_run = [entry["parameters"] for entry in entries[:2]]
    assert {**baseline, "activation": "sigmoid"} == next_run
    path = tmp_path / "baseline_only.json"
    path.write_text(json.dumps(entries[:1]))
    root = tmp_path / "results"
    completed(root / entries[0]["results_dir"])
    monkeypatch.setattr(run, "run_experiment", lambda *a, **k: pytest.fail("Completed baseline must not retrain"))
    directories = run.run_experiments(path, results_root=root)
    assert directories[0].name == entries[0]["results_dir"]


def test_shipped_queue_skips_both_completed_runs_and_selects_remaining(tmp_path, monkeypatch):
    entries = json.loads(Path(run.__file__).with_name("experiments.json").read_text())
    root = tmp_path / "results"
    for entry in entries[:2]:
        directory = root / entry["results_dir"]
        completed(directory)
        record = json.loads((directory / "config.json").read_text())
        record["model_config"]["activation"] = entry["parameters"]["activation"]
        (directory / "config.json").write_text(json.dumps(record))
    calls = []
    monkeypatch.setattr(run, "run_experiment", lambda path, **settings: calls.append((path.name, settings)))
    run.run_experiments(results_root=root)
    assert calls == [(entry["results_dir"], entry["parameters"]) for entry in entries[2:]]


def test_skips_complete_and_runs_missing_in_order(tmp_path, monkeypatch):
    root = tmp_path / "results"
    completed(root / "existing")  # Older records omit activation; default is Tanh.
    calls = []

    def execute(directory, **parameters):
        calls.append((directory.name, parameters))
        completed(directory)

    monkeypatch.setattr(run, "run_experiment", execute)
    path = queue(tmp_path, "existing", "next", "last")
    directories = run.run_experiments(path, results_root=root)
    assert [p.name for p in directories] == ["existing", "next", "last"]
    assert calls == [("next", PARAMETERS), ("last", PARAMETERS)]
    run.run_experiments(path, results_root=root)
    assert len(calls) == 2


@pytest.mark.parametrize("status", [None, "failed", "training", "complete"])
def test_incomplete_directory_stops_without_training(tmp_path, monkeypatch, status):
    root = tmp_path / "results"
    directory = root / "existing"
    directory.mkdir(parents=True)
    if status is not None:
        (directory / "config.json").write_text(json.dumps({"status": status}))
    monkeypatch.setattr(run, "run_experiment", lambda *a, **k: pytest.fail("Do not resume or overwrite"))
    with pytest.raises(RuntimeError, match="incomplete"):
        run.run_experiments(queue(tmp_path, "existing", "later"), results_root=root)


def test_mismatched_completed_settings_stop(tmp_path, monkeypatch):
    root = tmp_path / "results"
    completed(root / "existing", seed=11)
    monkeypatch.setattr(run, "run_experiment", lambda *a, **k: pytest.fail("Do not overwrite"))
    with pytest.raises(ValueError, match="settings differ"):
        run.run_experiments(queue(tmp_path, "existing"), results_root=root)


def test_training_failure_stops_before_later_experiments(tmp_path, monkeypatch):
    calls = []

    def fail(directory, **parameters):
        calls.append(directory.name)
        raise RuntimeError("training failed")

    monkeypatch.setattr(run, "run_experiment", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        run.run_experiments(queue(tmp_path, "first", "later"), results_root=tmp_path / "results")
    assert calls == ["first"]


def test_output_directory_must_stay_inside_results(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        run.run_experiments(queue(tmp_path, "../outside"), results_root=tmp_path / "results")
