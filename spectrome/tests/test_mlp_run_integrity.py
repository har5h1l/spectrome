"""Reject damaged inputs and saved evidence without starting cohort experiments."""

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

torch = pytest.importorskip("torch")

from spectrome.mlp.analysis import analyze_run
from spectrome.mlp.data.helpers import validate_training_cache
from spectrome.mlp.training.run import run_experiment, run_experiments

FIXTURE = Path(__file__).with_name("fixtures") / "mlp_reference_cache.pt"


@pytest.mark.parametrize("damage, message", [
    ("omega", "omega must match"),
    ("graph_frequency", "graph frequency differs"),
    ("duplicate_id", "unique nonempty"),
    ("observed_regions", "outside the connectome"),
    ("nonfinite_graph", "nonfinite"),
])
def test_cache_rejects_inconsistent_inputs(damage, message):
    cache = torch.load(FIXTURE, weights_only=True)
    if damage == "omega":
        cache["omega"] *= 2
    elif damage == "graph_frequency":
        cache["subjects"][0]["graphs"][0]["w"] += 1
    elif damage == "duplicate_id":
        cache["subjects"][1]["subject_id"] = cache["subjects"][0]["subject_id"]
    elif damage == "observed_regions":
        cache["subjects"][0]["observed_regions"][0] = 999
    else:
        cache["subjects"][0]["graphs"][0]["eigenvectors"][0, 0] = float("nan")
    with pytest.raises(ValueError, match=message):
        validate_training_cache(cache)


@pytest.mark.parametrize("field, value", [
    ("activation", "invalid"), ("epochs", 0), ("learning_rate", -0.1),
    ("hidden_sizes", [True]),
])
def test_bad_later_candidate_fails_before_creating_run(tmp_path, field, value):
    candidate = {
        "candidate_id": "valid",
        "model_config": {"hidden_sizes": [3], "activation": "tanh", "omega_scale": 100},
        "training_config": {"epochs": 1, "learning_rate": 0.001},
    }
    invalid = copy.deepcopy(candidate)
    invalid["candidate_id"] = "invalid"
    group = "model_config" if field in candidate["model_config"] else "training_config"
    invalid[group][field] = value
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match=field):
        run_experiment(destination, cache_source=FIXTURE, protocol={
            "training_seed": 7, "candidates": [candidate, invalid],
        })
    assert not destination.exists()


def test_impossible_inner_folds_fail_before_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr("spectrome.mlp.training.nested_cv.train_filter",
                        lambda *args, **kwargs: pytest.fail("Invalid splits must not fit a model"))
    with pytest.raises(ValueError, match="no greater than the subject count"):
        run_experiment(tmp_path / "invalid-splits", cache_source=FIXTURE, protocol={
            "outer_strategy": "leave_one_subject_out", "inner_folds": 3,
            "split_seed": 19, "training_seed": 7,
            "candidates": [{
                "candidate_id": "small",
                "model_config": {"hidden_sizes": [3], "omega_scale": 100, "activation": "tanh"},
                "training_config": {"epochs": 1, "learning_rate": 0.001},
            }],
        })


@pytest.fixture(scope="module")
def completed_loocv(tmp_path_factory):
    root = tmp_path_factory.mktemp("loocv-integrity")
    directory = run_experiment(root / "reference", cache_source=FIXTURE,
                               hidden_sizes=(3,), omega_scale=100, epochs=2)
    return directory


@pytest.mark.parametrize("damage", ["missing_checkpoint", "weights", "prediction", "score", "summary"])
def test_fixed_run_damage_blocks_reporting_and_queue_skip(completed_loocv, tmp_path, damage):
    directory = tmp_path / "run"
    shutil.copytree(completed_loocv, directory)
    result_path = directory / "results.pt"
    results = torch.load(result_path, weights_only=True)
    fold = results["folds"][0]
    checkpoint_path = directory / fold["checkpoint_path"]
    if damage == "missing_checkpoint":
        checkpoint_path.unlink()
    elif damage == "weights":
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        checkpoint["state_dict"]["network.0.weight"] += 0.1
        torch.save(checkpoint, checkpoint_path)
    else:
        if damage == "prediction":
            fold["learned_prediction"] += 0.1
        elif damage == "score":
            fold["learned_correlation"] += 0.1
        else:
            results["mean_learned_correlation"] += 0.1
        torch.save(results, result_path)
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps([{
        "results_dir": "run",
        "parameters": {"hidden_sizes": [3], "omega_scale": 100, "epochs": 2,
                       "learning_rate": 0.001, "seed": 7, "activation": "tanh"},
    }]))
    summary = directory / "summary.md"
    before = hashlib.sha256(summary.read_bytes()).hexdigest()
    for operation in (lambda: analyze_run(directory),
                      lambda: run_experiments(queue, results_root=tmp_path)):
        with pytest.raises(ValueError, match="artifact|replay|summary"):
            operation()
        assert hashlib.sha256(summary.read_bytes()).hexdigest() == before


def test_fixed_run_metadata_and_verified_skip(completed_loocv, tmp_path):
    config = json.loads((completed_loocv / "config.json").read_text())
    assert {"git_commit", "git_dirty", "torch_version", "torch_threads", "device", "dtype"} <= config.keys()
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps([{
        "results_dir": completed_loocv.name,
        "parameters": {**config["model_config"], **{
            key: config["training_config"][key] for key in ("epochs", "learning_rate", "seed")}},
    }]))
    assert run_experiments(queue, results_root=completed_loocv.parent) == [completed_loocv]
