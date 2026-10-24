"""A completed nested run must remain auditable before reporting or skipping."""

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

torch = pytest.importorskip("torch")

from spectrome.mlp.analysis import analyze_run
from spectrome.mlp.analysis.helpers import load_nested_run
from spectrome.mlp.training.run import run_experiments, run_experiment
from spectrome.mlp.analysis.helpers import validate_nested_results


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    """Fit only a three-subject, five-frequency, one-epoch test case."""
    root = tmp_path_factory.mktemp("nested-artifact-reference")
    fixture = Path(__file__).with_name("fixtures") / "mlp_reference_cache.pt"
    cache_path = root / "cache.pt"
    shutil.copyfile(fixture, cache_path)
    protocol = {
        "outer_strategy": "leave_one_subject_out", "outer_folds": None,
        "inner_folds": 2, "split_seed": 19, "training_seed": 7,
        "candidates": [{
            "candidate_id": "small",
            "model_config": {"hidden_sizes": [3], "omega_scale": 100.0, "activation": "tanh"},
            "training_config": {"epochs": 1, "learning_rate": 0.001},
        }],
        "declared_comparisons": ["selected_mlp_vs_analytical"],
    }
    run_dir = run_experiment(root / "nested", cache_source=cache_path, protocol=protocol)
    return run_dir, cache_path, protocol


@pytest.fixture
def saved_run(completed_run, tmp_path):
    template, cache_path, protocol = completed_run
    run_dir = tmp_path / "nested"
    shutil.copytree(template, run_dir)
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps([{
        "evaluation": "nested_cross_validation", "results_dir": "nested",
        "cache_source": str(cache_path), "protocol": protocol,
    }]))
    return run_dir, queue


def _outputs(run_dir):
    paths = [run_dir / "summary.md", *sorted((run_dir / "tables").glob("*.csv"))]
    return {str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _assert_rejected_without_rewriting(saved_run, match):
    run_dir, queue = saved_run
    before = _outputs(run_dir)
    for operation in (
        lambda: analyze_run(run_dir),
        lambda: run_experiments(queue, results_root=run_dir.parent),
    ):
        with pytest.raises((ValueError, RuntimeError), match=match):
            operation()
        assert _outputs(run_dir) == before


@pytest.mark.parametrize("artifact", ["checkpoint_path", "prediction_path"])
def test_missing_fold_artifact_cannot_be_reported_or_skipped(saved_run, artifact):
    run_dir, _ = saved_run
    result = json.loads((run_dir / "nested_results.json").read_text())
    (run_dir / result["outer_folds"][0][artifact]).unlink()
    _assert_rejected_without_rewriting(saved_run, "Missing or invalid nested artifact")


@pytest.mark.parametrize("metric", [
    "mean_analytical_r", "mean_selected_mlp_r", "mean_raw_r_difference",
    "n_subject_predictions", "selected_candidate_counts",
])
def test_false_outer_summary_is_rejected(saved_run, metric):
    run_dir, _ = saved_run
    path = run_dir / "nested_results.json"
    result = json.loads(path.read_text())
    result["outer_summary"][metric] = (
        {"small": 999} if metric == "selected_candidate_counts" else 999
    )
    path.write_text(json.dumps(result))
    _assert_rejected_without_rewriting(saved_run, "Outer summary differs")


@pytest.mark.parametrize("setting", ["architecture", "learning_rate", "epochs", "seed"])
def test_refit_must_match_selected_settings(saved_run, setting):
    run_dir, _ = saved_run
    path = run_dir / "nested_results.json"
    result = json.loads(path.read_text())
    refit = result["outer_folds"][0]["outer_refit"]
    if setting == "architecture":
        refit["model_config"]["hidden_sizes"] = [4]
    elif setting == "epochs":
        refit["training_config"]["epochs"] = 2
        refit["trained_epochs"] = 2
        refit["training_history"].append({"epoch": 2, "train_loss_before_step": 0.5})
    else:
        refit["training_config"][setting] = 999
    path.write_text(json.dumps(result))
    _assert_rejected_without_rewriting(saved_run, "Outer-refit settings differ")


@pytest.mark.parametrize("damage", ["count", "empty_training", "duplicate", "seed"])
def test_inner_memberships_must_match_declared_protocol(saved_run, damage):
    run_dir, _ = saved_run
    path = run_dir / "nested_results.json"
    result = json.loads(path.read_text())
    outer = result["outer_folds"][0]
    if damage == "count":
        outer["inner_splits"].pop()
    elif damage == "empty_training":
        outer["inner_splits"][0]["training_subject_ids"] = []
    elif damage == "duplicate":
        values = outer["inner_splits"][0]["training_subject_ids"]
        values.append(values[0])
    else:
        outer["inner_split_seed"] += 1
    path.write_text(json.dumps(result))
    _assert_rejected_without_rewriting(saved_run, "Inner splits do not match")


@pytest.mark.parametrize("damage", ["architecture", "seed", "weights"])
def test_checkpoint_is_bound_to_selected_refit_and_predictions(saved_run, damage):
    run_dir, _ = saved_run
    result = json.loads((run_dir / "nested_results.json").read_text())
    path = run_dir / result["outer_folds"][0]["checkpoint_path"]
    checkpoint = torch.load(path, weights_only=True)
    if damage == "architecture":
        checkpoint["model_config"]["hidden_sizes"] = [4]
    elif damage == "seed":
        checkpoint["training_config"]["seed"] = 999
    else:
        checkpoint["state_dict"]["network.0.weight"] += 0.1
    torch.save(checkpoint, path)
    _assert_rejected_without_rewriting(saved_run, "checkpoint settings|checkpoint replay")


def test_false_statistics_are_rejected(saved_run):
    run_dir, _ = saved_run
    path = run_dir / "nested_results.json"
    result = json.loads(path.read_text())
    result["statistics"][0]["p_value_unadjusted"] = 0.5
    path.write_text(json.dumps(result))
    _assert_rejected_without_rewriting(saved_run, "Saved nested statistics differ")


def test_statistical_roundoff_does_not_invalidate_an_equivalent_run(saved_run):
    run_dir, queue = saved_run
    path = run_dir / "nested_results.json"
    result = json.loads(path.read_text())
    result["statistics"][0]["p_value_unadjusted"] += 1e-16
    path.write_text(json.dumps(result))
    load_nested_run(run_dir, require_complete=True)
    assert run_experiments(queue, results_root=run_dir.parent) == [run_dir.resolve()]


def test_missing_outer_fold_is_not_complete(saved_run):
    run_dir, _ = saved_run
    path = run_dir / "nested_results.json"
    result = json.loads(path.read_text())
    result["outer_folds"].pop()
    path.write_text(json.dumps(result))
    _assert_rejected_without_rewriting(saved_run, "Outer memberships do not match")


def test_subject_row_must_identify_its_containing_fold(saved_run):
    run_dir, _ = saved_run
    path = run_dir / "nested_results.json"
    result = json.loads(path.read_text())
    result["outer_folds"][0]["outer_subject_records"][0]["outer_fold"] = 999
    path.write_text(json.dumps(result))
    _assert_rejected_without_rewriting(saved_run, "Outer subject score or artifact reference")


def test_valid_run_reports_and_skips_after_read_only_verification(saved_run):
    run_dir, queue = saved_run
    result, rows = load_nested_run(run_dir, require_complete=True)
    assert len(rows) == result["outer_summary"]["n_subject_predictions"] == 3
    before = _outputs(run_dir)
    assert run_experiments(queue, results_root=run_dir.parent) == [run_dir.resolve()]
    assert _outputs(run_dir) == before
    analyze_run(run_dir)
    assert _outputs(run_dir) == before


@pytest.mark.parametrize("invalid", ["missing_cache", "conflicting_settings", "invalid_protocol"])
def test_single_experiment_api_rejects_ambiguous_nested_inputs(completed_run, tmp_path, invalid):
    _, cache_path, protocol = completed_run
    arguments = {"cache_source": cache_path, "protocol": protocol}
    if invalid == "missing_cache":
        arguments.pop("cache_source")
        message = "protocol dictionary and cache_source"
    elif invalid == "conflicting_settings":
        arguments["epochs"] = 2
        message = "settings belong in protocol candidates"
    else:
        arguments["protocol"] = []
        message = "protocol dictionary and cache_source"
    destination = tmp_path / invalid
    with pytest.raises(ValueError, match=message):
        run_experiment(destination, **arguments)
    assert not destination.exists()


def test_nested_experiment_retains_shared_run_log(completed_run):
    run_dir, _, _ = completed_run
    log = (run_dir / "training.log").read_text()
    assert "starting nested cross-validation" in log
    assert "Complete:" in log


def test_referenced_candidates_match_explicit_run_and_analysis_needs_no_catalog(completed_run, tmp_path):
    reference_dir, cache_path, protocol = completed_run
    candidate = protocol["candidates"][0]
    catalog = tmp_path / "training_runs.json"
    catalog.write_text(json.dumps([{
        "results_dir": candidate["candidate_id"],
        "parameters": {**candidate["model_config"], **candidate["training_config"], "seed": 7},
    }]))
    requested = {key: value for key, value in protocol.items() if key != "candidates"}
    requested["candidate_run_ids"] = [candidate["candidate_id"]]
    directory = run_experiment(tmp_path / "referenced", cache_source=cache_path,
                               protocol=requested, training_runs_path=catalog)
    config = json.loads((directory / "config.json").read_text())
    assert config["protocol"] == protocol
    expected = json.loads((reference_dir / "nested_results.json").read_text())
    actual = json.loads((directory / "nested_results.json").read_text())
    assert actual["candidates"] == expected["candidates"]
    assert actual["outer_folds"] == expected["outer_folds"]
    catalog.unlink()
    before = _outputs(directory)
    analyze_run(directory)
    assert _outputs(directory) == before


def test_grouped_run_is_verified_with_unequal_outer_fold_sizes(completed_run, tmp_path):
    _, cache_path, protocol = completed_run
    cache = torch.load(cache_path, weights_only=True)
    for index, subject in enumerate(list(cache["subjects"])[:2]):
        duplicate = copy.deepcopy(subject)
        duplicate["subject_id"] = f"synthetic-grouped-{index}"
        cache["subjects"].append(duplicate)
    source = tmp_path / "grouped-cache.pt"
    torch.save(cache, source)
    grouped = {**protocol, "outer_strategy": "grouped_kfold", "outer_folds": 2}
    run_dir = run_experiment(tmp_path / "grouped", cache_source=source, protocol=grouped)
    result, rows = load_nested_run(run_dir, require_complete=True)
    validate_nested_results(result)
    assert sorted(len(fold["test_subject_ids"]) for fold in result["outer_folds"]) == [2, 3]
    assert len(rows) == 5
    assert result["outer_summary"]["mean_selected_mlp_r"] == pytest.approx(
        sum(row["learned_r"] for row in rows) / 5
    )
