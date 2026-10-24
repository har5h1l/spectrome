"""Preservation and bounded acceptance checks for the canonical MLP package."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

torch = pytest.importorskip("torch")

from spectrome.mlp import model as mlp_model
from spectrome.mlp.analysis import analyze_run
from spectrome.mlp.data import prepare_spectral_subject
from spectrome.mlp.training import nested_cv as cross_validation
from spectrome.mlp.training.nested_cv import nested_cross_validate, select_candidate
from spectrome.mlp.spectra import evaluate_filter
from spectrome.mlp.training.helpers import make_inner_splits, make_outer_splits
from spectrome.mlp.analysis.statistics import paired_fisher_z_test
from spectrome.mlp.training.run import run_experiments, run_experiment
from spectrome.mlp.training.training import train_filter
from spectrome.mlp.analysis.helpers import validate_nested_results
ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures")


@pytest.fixture
def reference_cache():
    return torch.load(FIXTURES / "mlp_reference_cache.pt", map_location="cpu", weights_only=True)


def _json_tensor(value):
    if isinstance(value, list):
        return torch.tensor(value, dtype=torch.float64)
    return torch.tensor(value, dtype=torch.float64)


def _state_as_lists(state):
    return {name: value.detach().cpu().tolist() for name, value in state.items()}


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
def test_seeded_fit_matches_pre_reorganization_trace(reference_cache, monkeypatch, activation):
    """Match initialization, gradients, Adam state, updates, and final scores."""
    trace_path = FIXTURES / "mlp_reference_trace.json"
    reference = json.loads(trace_path.read_text())["activations"][activation]
    subjects = {row["subject_id"]: prepare_spectral_subject(row)
                for row in reference_cache["subjects"]}
    fixture = json.loads(trace_path.read_text())["fixture"]
    training = [subjects[value] for value in fixture["training_subject_ids"]]
    held_out = subjects[fixture["held_out_subject_id"]]

    models = []
    initial_states = []
    optimizer_steps = []
    original_model = mlp_model.FeGraphMLP
    original_adam = torch.optim.Adam

    def capture_model(*args, **kwargs):
        model = original_model(*args, **kwargs)
        models.append(model)
        initial_states.append(_state_as_lists(model.state_dict()))
        return model

    class RecordingAdam(original_adam):
        def __init__(self, parameters, **kwargs):
            super().__init__(parameters, **kwargs)
            assert not self.state

        def step(self, *args, **kwargs):
            model = models[-1]
            names = {parameter: name for name, parameter in model.named_parameters()}
            gradients = {
                names[parameter]: parameter.grad.detach().cpu().tolist()
                for group in self.param_groups for parameter in group["params"]
            }
            result = super().step(*args, **kwargs)
            state = {}
            for group in self.param_groups:
                for parameter in group["params"]:
                    name = names[parameter]
                    state[name] = {
                        key: (value.detach().cpu().tolist() if torch.is_tensor(value) else value)
                        for key, value in self.state[parameter].items()
                    }
            optimizer_steps.append({
                "gradients": gradients,
                "optimizer_state": state,
                "model_state": _state_as_lists(model.state_dict()),
            })
            return result

    monkeypatch.setattr("spectrome.mlp.training.training.FeGraphMLP", capture_model)
    monkeypatch.setattr(torch.optim, "Adam", RecordingAdam)
    model, history = train_filter(
        training, reference_cache["omega"],
        {**fixture["model"], "activation": activation},
        {
            "epochs": fixture["training"]["steps"],
            "learning_rate": fixture["training"]["learning_rate"],
            "seed": fixture["training"]["seed"],
        },
    )

    assert initial_states == [reference["model_state_before_training"]]
    assert len(optimizer_steps) == fixture["training"]["steps"]
    for index, actual in enumerate(optimizer_steps):
        expected = reference["steps"][index]
        assert actual["gradients"] == expected["gradients_before_step"]
        assert actual["optimizer_state"] == expected["optimizer_state_after_step"]
        assert actual["model_state"] == expected["model_state_after_step"]
        assert history[index] == {
            "epoch": index + 1,
            "train_loss_before_step": expected["loss_before_step"],
        }

    evaluation = evaluate_filter(model, [held_out], reference_cache["omega"])[0]
    actual_filter = evaluation["Fe_graph"]
    final = reference["final"]
    np.testing.assert_array_equal(actual_filter.real.numpy(), final["learned_filter"]["real"])
    np.testing.assert_array_equal(actual_filter.imag.numpy(), final["learned_filter"]["imaginary"])
    np.testing.assert_array_equal(evaluation["learned_prediction"].numpy(),
                                  np.asarray(final["learned_prediction"]["real"]) +
                                  1j * np.asarray(final["learned_prediction"]["imaginary"]))
    assert evaluation["learned_correlation"] == final["learned_score"]
    assert evaluation["analytical_correlation"] == final["analytical_score"]


def test_old_training_package_is_removed():
    assert not (ROOT / "spectrome" / "training").exists()


def test_split_memberships_are_deterministic_and_partition_subjects():
    ids = [f"s{index}" for index in range(9)]
    first = make_outer_splits(ids, strategy="grouped_kfold", n_splits=4, seed=11)
    second = make_outer_splits(ids, strategy="grouped_kfold", n_splits=4, seed=11)
    assert first == second
    assert {value for fold in first for value in fold["test_subject_ids"]} == set(ids)
    assert sum(len(fold["test_subject_ids"]) for fold in first) == len(ids)
    for fold in first:
        assert not set(fold["training_subject_ids"]) & set(fold["test_subject_ids"])
        inner = make_inner_splits(fold["training_subject_ids"], n_splits=3, seed=19)
        assert {value for item in inner for value in item["validation_subject_ids"]} == set(
            fold["training_subject_ids"]
        )
        for item in inner:
            assert not set(item["training_subject_ids"]) & set(item["validation_subject_ids"])
            assert set(item["training_subject_ids"]) | set(item["validation_subject_ids"]) == set(
                fold["training_subject_ids"]
            )
    with pytest.raises(ValueError, match="no greater"):
        make_inner_splits(["a", "b", "c"], n_splits=5, seed=7)


def test_candidate_selection_uses_pooled_subject_scores_and_declared_tie_rule():
    candidates = [
        {"candidate_id": "first"},
        {"candidate_id": "second"},
    ]
    # A fold-mean comparison with unequal fold sizes would favor "second":
    # first=(0.9+0.2)/2, second=(0.6+0.6)/2. Pooled subject means favor first.
    scores = {
        "first": {"a": 0.9, "b": 0.9, "c": 0.2},
        "second": {"a": 0.6, "b": 0.6, "c": 0.6},
    }
    selected, means = select_candidate(candidates, scores, ["a", "b", "c"])
    assert selected == "first"
    assert means["first"] == pytest.approx(2.0 / 3.0)
    assert means["second"] == pytest.approx(0.6)
    selected, _ = select_candidate(candidates, {"first": scores["second"],
                                                "second": scores["second"]}, ["a", "b", "c"])
    assert selected == "first"


def test_paired_fisher_z_test_matches_scipy_and_aligns_by_id():
    left = [{"subject_id": "c", "r": 0.3}, {"subject_id": "a", "r": 0.1},
            {"subject_id": "b", "r": 0.2}]
    right = [{"subject_id": "b", "r": 0.0}, {"subject_id": "c", "r": 0.0},
             {"subject_id": "a", "r": 0.0}]
    actual = paired_fisher_z_test(left, right, left_label="model", right_label="control",
                                  contrast="model_vs_control")
    expected = stats.ttest_rel(np.arctanh([0.1, 0.2, 0.3]), np.zeros(3))
    assert actual["n_subject_pairs"] == 3
    assert actual["degrees_of_freedom"] == 2
    assert actual["t_statistic"] == pytest.approx(expected.statistic)
    assert actual["p_value_unadjusted"] == pytest.approx(expected.pvalue)
    assert [row["subject_id"] for row in actual["subject_rows"]] == ["a", "b", "c"]
    reversed_result = paired_fisher_z_test(list(reversed(left)), list(reversed(right)),
                                           left_label="model", right_label="control",
                                           contrast="model_vs_control")
    assert reversed_result["t_statistic"] == actual["t_statistic"]
    assert reversed_result["subject_rows"] == actual["subject_rows"]
    with pytest.raises(ValueError, match="exactly the expected"):
        paired_fisher_z_test(left, right[:-1], left_label="model", right_label="control",
                             contrast="bad")
    with pytest.raises(ValueError, match="strictly between"):
        paired_fisher_z_test({"a": 1.0, "b": 0.2}, {"a": 0.0, "b": 0.0},
                             left_label="model", right_label="control", contrast="bad")
    degenerate = paired_fisher_z_test({"a": 0.2, "b": 0.2}, {"a": 0.1, "b": 0.1},
                                      left_label="model", right_label="control", contrast="same")
    assert degenerate["status"] == "degenerate_zero_variance"
    assert degenerate["p_value_unadjusted"] is None
    with pytest.raises(ValueError, match="distinct nonempty"):
        paired_fisher_z_test({"a": 0.2, "b": 0.3}, {"a": 0.1, "b": 0.2},
                             left_label="run", right_label="run", contrast="ambiguous")


def _candidate(candidate_id, hidden_sizes):
    return {
        "candidate_id": candidate_id,
        "model_config": {"hidden_sizes": hidden_sizes, "omega_scale": 100.0,
                         "activation": "tanh"},
        "training_config": {"epochs": 1, "learning_rate": 0.001},
    }


def _protocol():
    return {
        "outer_strategy": "leave_one_subject_out",
        "outer_folds": None,
        "inner_folds": 2,
        "split_seed": 19,
        "training_seed": 7,
        "candidates": [_candidate("small", [3]), _candidate("wide", [4])],
        "declared_comparisons": ["selected_mlp_vs_analytical"],
    }


def _load_result(path):
    return json.loads((Path(path) / "nested_results.json").read_text())


def test_nested_run_analysis_queue_and_candidate_order(reference_cache, tmp_path):
    cache_path = tmp_path / "cache.pt"
    torch.save(reference_cache, cache_path)
    run_dir = tmp_path / "nested"
    run_experiment(run_dir, cache_source=cache_path, protocol=_protocol())
    result = _load_result(run_dir)
    assert result["status"] == "complete"
    assert result["outer_summary"]["n_subject_predictions"] == 3
    assert len(result["statistics"]) == 1
    assert len(list((run_dir / "tables").glob("*.csv"))) == 4
    assert (run_dir / "summary.md").is_file()
    analyze_run(run_dir)

    reversed_dir = tmp_path / "nested-reversed"
    reversed_dir.mkdir()
    cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    nested_cross_validate(reference_cache, reversed_dir, _protocol(),
                          source_sha256=result["source_sha256"],
                          cache_sha256=cache_hash,
                          candidate_execution_order=["wide", "small"])
    reversed_result = _load_result(reversed_dir)
    for expected, actual in zip(result["outer_folds"], reversed_result["outer_folds"]):
        assert actual["candidate_mean_raw_r"] == expected["candidate_mean_raw_r"]
        assert actual["candidate_scores"] == expected["candidate_scores"]
        assert actual["selected_candidate_id"] == expected["selected_candidate_id"]
        assert actual["outer_refit"]["training_history"] == expected["outer_refit"]["training_history"]
        expected_checkpoint = torch.load(run_dir / expected["checkpoint_path"], weights_only=True)
        actual_checkpoint = torch.load(reversed_dir / actual["checkpoint_path"], weights_only=True)
        assert expected_checkpoint["selected_candidate_id"] == actual_checkpoint["selected_candidate_id"]
        for name, value in expected_checkpoint["state_dict"].items():
            torch.testing.assert_close(value, actual_checkpoint["state_dict"][name], rtol=0, atol=0)

    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps([{
        "evaluation": "nested_cross_validation",
        "results_dir": "nested",
        "cache_source": str(cache_path),
        "protocol": _protocol(),
    }]))
    assert run_experiments(queue_path, results_root=tmp_path) == [run_dir.resolve()]
    mismatched_queue = json.loads(queue_path.read_text())
    mismatched_queue[0]["protocol"]["split_seed"] += 1
    queue_path.write_text(json.dumps(mismatched_queue))
    with pytest.raises(ValueError, match="protocol, source, or cache differs"):
        run_experiments(queue_path, results_root=tmp_path)
    with pytest.raises(FileExistsError):
        run_experiment(run_dir, cache_source=cache_path, protocol=_protocol())

    malformed = copy.deepcopy(result)
    malformed["outer_folds"][0]["outer_subject_records"].pop()
    with pytest.raises(ValueError, match="Outer subject records"):
        validate_nested_results(malformed)

    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    incomplete_queue = [{
        "evaluation": "nested_cross_validation",
        "results_dir": "incomplete",
        "cache_source": str(cache_path),
        "protocol": _protocol(),
    }]
    queue_path.write_text(json.dumps(incomplete_queue))
    with pytest.raises(RuntimeError, match="Existing nested run is incomplete"):
        run_experiments(queue_path, results_root=tmp_path)


def test_nested_runner_rejects_incompatible_cache(reference_cache, tmp_path):
    bad_cache = copy.deepcopy(reference_cache)
    bad_cache["forward_sha256"] = "stale-forward-source"
    cache_path = tmp_path / "stale-cache.pt"
    torch.save(bad_cache, cache_path)
    run_dir = tmp_path / "bad-run"
    with pytest.raises(ValueError, match="Forward source changed"):
        run_experiment(run_dir, cache_source=cache_path, protocol=_protocol())
    config = json.loads((run_dir / "config.json").read_text())
    assert config["status"] == "failed"
    assert "Forward source changed" in config["error"]


def test_outer_test_target_isolation_and_failure_evidence(reference_cache, tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.pt"
    torch.save(reference_cache, cache_path)
    baseline_dir = tmp_path / "baseline"
    run_experiment(baseline_dir, cache_source=cache_path, protocol=_protocol())
    baseline = _load_result(baseline_dir)
    held_out_id = baseline["outer_folds"][0]["test_subject_ids"][0]

    changed = copy.deepcopy(reference_cache)
    target = next(row for row in changed["subjects"] if row["subject_id"] == held_out_id)
    target["empirical_meg"] = target["empirical_meg"].flip(-1).contiguous()
    changed_path = tmp_path / "changed-cache.pt"
    torch.save(changed, changed_path)
    changed_dir = tmp_path / "changed"
    run_experiment(changed_dir, cache_source=changed_path, protocol=_protocol())
    changed_result = _load_result(changed_dir)
    before = baseline["outer_folds"][0]
    after = changed_result["outer_folds"][0]
    assert before["selected_candidate_id"] == after["selected_candidate_id"]
    assert before["candidate_scores"] == after["candidate_scores"]
    assert before["candidate_mean_raw_r"] == after["candidate_mean_raw_r"]
    assert before["outer_refit"] == after["outer_refit"]
    before_checkpoint = torch.load(baseline_dir / before["checkpoint_path"], weights_only=True)
    after_checkpoint = torch.load(changed_dir / after["checkpoint_path"], weights_only=True)
    for name, value in before_checkpoint["state_dict"].items():
        torch.testing.assert_close(value, after_checkpoint["state_dict"][name], rtol=0, atol=0)
    assert before["outer_subject_records"][0]["learned_r"] != after["outer_subject_records"][0]["learned_r"]

    failure_dir = tmp_path / "partial"
    failure_dir.mkdir()
    original_fit = cross_validation.train_filter
    calls = 0

    def fail_after_one_outer_fold(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("synthetic bounded fit failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(cross_validation, "train_filter", fail_after_one_outer_fold)
    cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="synthetic bounded"):
        nested_cross_validate(reference_cache, failure_dir, {
            **_protocol(), "candidates": [_candidate("only", [3])],
            "declared_comparisons": [],
        }, source_sha256={"test": "source"}, cache_sha256=cache_hash)
    partial = _load_result(failure_dir)
    assert partial["status"] == "failed"
    assert len(partial["outer_folds"]) == 1
    assert (failure_dir / partial["outer_folds"][0]["checkpoint_path"]).is_file()
    assert (failure_dir / partial["outer_folds"][0]["prediction_path"]).is_file()


def test_grouped_nested_outer_folds_save_one_checkpoint_per_fold(reference_cache, tmp_path):
    cache = copy.deepcopy(reference_cache)
    original_subjects = list(cache["subjects"])
    for index, original in enumerate(original_subjects):
        duplicate = copy.deepcopy(original)
        duplicate["subject_id"] = f"synthetic-{index}"
        duplicate["provenance"] = {**duplicate["provenance"],
                                   "bounded_grouped_split_fixture": True}
        cache["subjects"].append(duplicate)
    cache_path = tmp_path / "six-subject-cache.pt"
    torch.save(cache, cache_path)
    output_dir = tmp_path / "grouped"
    output_dir.mkdir()
    cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    result = nested_cross_validate(cache, output_dir, {
        "outer_strategy": "grouped_kfold", "outer_folds": 2, "inner_folds": 2,
        "split_seed": 23, "training_seed": 7,
        "candidates": [_candidate("only", [3])], "declared_comparisons": [],
    }, source_sha256={"test": "source"}, cache_sha256=cache_hash)
    assert result["status"] == "complete"
    assert len(result["outer_folds"]) == 2
    assert sum(len(fold["outer_subject_records"]) for fold in result["outer_folds"]) == 6
    assert len(list((output_dir / "checkpoints").glob("*.pt"))) == 2
    assert len(list((output_dir / "predictions").glob("*.pt"))) == 2
    assert result["outer_summary"]["n_subject_predictions"] == 6
