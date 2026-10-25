"""Load, check, and replay saved results; share analysis utilities."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..data.data import prepare_spectral_subject
from ..data.helpers import validate_training_cache
from ..model import FeGraphMLP
from ..spectra import SCORING_ID, evaluate_filter, transform_spectra
from ..training.helpers import (
    inner_split_seed, make_inner_splits, make_outer_splits, load_checkpoint,
)


# Load saved runs and scores


def load_run_config(run_dir):
    """Read the saved settings used to choose the analysis report."""
    return json.loads((Path(run_dir) / "config.json").read_text())


def load_saved_object(value):
    """Load a weights-only Torch artifact, or return an already loaded object."""
    return value if isinstance(value, dict) else torch.load(
        value, map_location="cpu", weights_only=True
    )


def load_loocv_run(run_dir, config):
    """Verify the cache and replay every saved fixed-configuration checkpoint."""
    run_dir = Path(run_dir)
    cache_path = run_dir / config["cache_path"]
    if hashlib.sha256(cache_path.read_bytes()).hexdigest() != config["cache_sha256"]:
        raise ValueError("Run cache hash differs from config.json")
    results, cache, _ = validate_loocv_results(run_dir / config["results_path"], cache_path)
    validate_training_cache(cache)
    verify_loocv_checkpoints(run_dir, results, cache, config)
    return results, cache


def load_loocv_scores(run_dir):
    """Load verified fixed-run subject scores and their comparison provenance."""
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text())
    results, cache = load_loocv_run(run_dir, config)
    subjects = {subject["subject_id"]: subject for subject in cache["subjects"]}
    if results.get("loss_preprocessing") != "magnitude_conv5_sqrt_floor1e-12_demean":
        raise ValueError(f"{run_dir.name}: unsupported saved scoring convention")
    if results.get("scoring_id", SCORING_ID) != SCORING_ID:
        raise ValueError(f"{run_dir.name}: scoring IDs differ")
    ids = list(subjects)
    region_maps = {
        subject_id: subjects[subject_id]["observed_regions"].tolist()
        for subject_id in ids
    }
    provenance = {
        "run_dir": str(run_dir),
        "cache_sha256": config["cache_sha256"],
        "preprocessing_id": cache["preprocessing_id"],
        "forward_sha256": cache["forward_sha256"],
        "frequencies_hz": cache["frequencies_hz"].tolist(),
        "subject_ids": ids,
        "observed_region_maps": region_maps,
        "scoring_id": SCORING_ID,
    }
    subject_table_path = run_dir / "tables/subject_scores.csv"
    table = {}
    if subject_table_path.is_file():
        with subject_table_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                subject_id = row["subject_id"]
                if subject_id in table:
                    raise ValueError(f"{run_dir.name}: duplicate subject score table row")
                table[subject_id] = row

    learned = {}
    analytical = {}
    for fold in results["folds"]:
        subject_id = fold["held_out_subject_id"]
        learned[subject_id] = float(fold["learned_correlation"])
        analytical[subject_id] = float(fold["analytical_correlation"])
        if table:
            if subject_id not in table:
                raise ValueError(f"{run_dir.name}: subject score table is missing {subject_id}")
            if not math.isclose(float(table[subject_id]["learned_correlation"]), learned[subject_id],
                                abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"{run_dir.name}: subject score table differs for {subject_id}")
            if not math.isclose(float(table[subject_id]["analytical_correlation"]), analytical[subject_id],
                                abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"{run_dir.name}: analytical score table differs for {subject_id}")
    return {"learned": learned, "analytical": analytical, "provenance": provenance}


@torch.no_grad()
def load_nested_run(run_dir, config=None, *, require_complete=False):
    """Verify protocol, files, checkpoints, predictions, and reported metrics.

    Queue completion and analysis use this same read-only gate. Analysis may
    run while config status is ``analyzing``; results must already be complete.
    """
    run_dir = Path(run_dir).resolve()
    if config is None:
        config = json.loads((run_dir / "config.json").read_text())
    if (config.get("evaluation") != "nested_cross_validation"
            or config.get("format_version") != 1):
        raise ValueError("Unsupported nested run configuration")
    allowed_statuses = {"complete"} if require_complete else {"analyzing", "complete"}
    if config.get("status") not in allowed_statuses:
        raise ValueError("Nested run configuration is not complete or ready for analysis")
    cache_path = _artifact_path(run_dir, config["cache_path"])
    results_path = _artifact_path(run_dir, config.get("results_path", "nested_results.json"))
    if hashlib.sha256(cache_path.read_bytes()).hexdigest() != config.get("cache_sha256"):
        raise ValueError("Nested run cache hash differs from config.json")
    cache = load_checkpoint(cache_path)
    validate_training_cache(cache)
    results = json.loads(results_path.read_text())
    if results.get("status") != "complete":
        raise ValueError("Nested results are not complete")
    validate_nested_results(results)
    validate_nested_protocol(results, config["protocol"])
    if (results.get("cache_sha256") != config.get("cache_sha256")
            or results.get("source_sha256") != config.get("source_sha256")
            or results.get("preprocessing_id") != cache.get("preprocessing_id")
            or results.get("forward_sha256") != cache.get("forward_sha256")
            or config.get("forward_sha256") != cache.get("forward_sha256")
            or results.get("scoring_id") != SCORING_ID
            or results.get("frequencies_hz") != cache["frequencies_hz"].tolist()
            or results.get("subject_ids") != [s["subject_id"] for s in cache["subjects"]]
            or config.get("subject_ids") != results["subject_ids"]):
        raise ValueError("Nested results, config, and cache provenance do not match")
    prepared = {
        subject["subject_id"]: prepare_spectral_subject(subject)
        for subject in cache["subjects"]
    }
    verified_rows = []
    for outer in results["outer_folds"]:
        checkpoint = load_checkpoint(_artifact_path(run_dir, outer["checkpoint_path"]))
        predictions = load_checkpoint(_artifact_path(run_dir, outer["prediction_path"]))
        expected_checkpoint = {
            "format_version": 1,
            "evaluation": "nested_cross_validation",
            "outer_fold": outer["outer_fold"],
            "selected_candidate_id": outer["selected_candidate_id"],
            "model_config": outer["outer_refit"]["model_config"],
            "training_config": outer["outer_refit"]["training_config"],
            "training_subject_ids": outer["training_subject_ids"],
            "test_subject_ids": outer["test_subject_ids"],
            "inner_splits": outer["inner_splits"],
            "cache_sha256": results["cache_sha256"],
            "forward_sha256": results["forward_sha256"],
            "source_sha256": results["source_sha256"],
        }
        if any(checkpoint.get(key) != value for key, value in expected_checkpoint.items()):
            raise ValueError("Nested checkpoint settings or provenance differ from the selected refit")
        if (predictions.get("outer_fold") != outer["outer_fold"]
                or predictions.get("selected_candidate_id") != outer["selected_candidate_id"]
                or set(predictions.get("subject_predictions", {})) != set(outer["test_subject_ids"])):
            raise ValueError("Saved outer predictions do not match the test fold")
        model = FeGraphMLP(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        replayed = evaluate_filter(
            model, [prepared[value] for value in outer["test_subject_ids"]], cache["omega"]
        )
        replay_by_id = {row["subject_id"]: row for row in replayed}
        for row in outer["outer_subject_records"]:
            subject_id = row["subject_id"]
            replay = replay_by_id[subject_id]
            saved = predictions["subject_predictions"][subject_id]
            for name in ("learned_prediction", "analytical_prediction", "Fe_graph"):
                expected = replay[name]
                actual = saved.get(name)
                if (not torch.is_tensor(actual) or actual.shape != expected.shape
                        or not torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)):
                    raise ValueError(f"{subject_id}: checkpoint replay differs for {name}")
            scores = {
                "analytical_r": replay["analytical_correlation"],
                "learned_r": replay["learned_correlation"],
                "raw_r_difference": replay["correlation_difference"],
            }
            if any(not math.isclose(row[key], value, abs_tol=1e-12, rel_tol=1e-12)
                   for key, value in scores.items()):
                raise ValueError(f"{subject_id}: recomputed nested scores differ from saved results")
            verified_rows.append({
                **row, **scores,
                "n_regions": int(prepared[subject_id]["empirical_meg"].shape[0]),
            })

    summary = summarize_outer_scores(
        verified_rows, [candidate["candidate_id"] for candidate in results["candidates"]]
    )
    validate_outer_summary(results["outer_summary"], summary)
    # Use the validated stored scores for exact statistical reconstruction;
    # checkpoint replay can introduce harmless floating-point rounding.
    saved_rows = [row for outer in results["outer_folds"] for row in outer["outer_subject_records"]]
    from .statistics import paired_fisher_z_test

    expected_statistics = []
    if "selected_mlp_vs_analytical" in results["protocol"]["declared_comparisons"]:
        expected_statistics.append(paired_fisher_z_test(
            {row["subject_id"]: row["learned_r"] for row in saved_rows},
            {row["subject_id"]: row["analytical_r"] for row in saved_rows},
            left_label="nested_selected_mlp", right_label="analytical",
            contrast="selected_mlp_vs_analytical", expected_subject_ids=results["subject_ids"],
            provenance={
                "cache_sha256": results["cache_sha256"],
                "preprocessing_id": results["preprocessing_id"],
                "forward_sha256": results["forward_sha256"],
                "scoring_id": SCORING_ID,
            },
        ))
    if not _matching_statistics(results.get("statistics"), expected_statistics):
        raise ValueError("Saved nested statistics differ from the validated subject scores")
    return results, verified_rows


# Fixed-configuration verification


def validate_loocv_results(results, cache):
    """Match every fixed-configuration held-out prediction to its cached subject."""
    results = load_saved_object(results)
    cache = load_saved_object(cache)
    if results.get("evaluation") != "leave_one_subject_out" or cache.get("format_version") != 2:
        raise ValueError("Analysis requires LOOCV results and a version 2 cohort cache")
    if (results["preprocessing_id"] != cache["preprocessing_id"]
            or results["forward_sha256"] != cache["forward_sha256"]
            or not torch.equal(results["frequencies_hz"], cache["frequencies_hz"])):
        raise ValueError("Results and cache have different preprocessing, forward source, or frequencies")
    subjects = {subject["subject_id"]: subject for subject in cache["subjects"]}
    ids = [fold["held_out_subject_id"] for fold in results["folds"]]
    if (len(subjects) != len(cache["subjects"]) or len(ids) != len(set(ids))
            or set(ids) != set(subjects)):
        raise ValueError("Results must contain exactly one held-out fold per cached subject")
    for fold in results["folds"]:
        subject = subjects[fold["held_out_subject_id"]]
        training_ids = fold["training_subject_ids"]
        if (len(training_ids) != len(subjects) - 1
                or set(training_ids) != set(subjects) - {subject["subject_id"]}):
            raise ValueError("Fold training IDs do not match leave-one-subject-out membership")
        shape = subject["empirical_meg"].shape
        if (fold["analytical_prediction"].shape != shape
                or fold["learned_prediction"].shape != shape
                or fold["Fe_graph"].shape != cache["frequencies_hz"].shape):
            raise ValueError("Saved prediction/filter dimensions do not match the cache")
    return results, cache, subjects


@torch.no_grad()
def verify_loocv_checkpoints(run_dir, results, cache, config):
    """Reject missing weights or predictions that do not match their saved model."""
    model_config = {"activation": "tanh", **results["model_config"]}
    saved_model_config = {"activation": "tanh", **config["model_config"]}
    if model_config != saved_model_config:
        raise ValueError("LOOCV model settings differ from config.json")
    for key in ("epochs", "learning_rate", "seed"):
        if results["training_config"][key] != config["training_config"][key]:
            raise ValueError("LOOCV training settings differ from config.json")
    subjects = {subject["subject_id"]: prepare_spectral_subject(subject)
                for subject in cache["subjects"]}
    scores = []
    for fold in results["folds"]:
        checkpoint = load_checkpoint(_artifact_path(run_dir.resolve(), fold["checkpoint_path"]))
        expected = {
            "held_out_subject_id": fold["held_out_subject_id"],
            "training_subject_ids": fold["training_subject_ids"],
            "training_config": results["training_config"],
            "forward_sha256": results["forward_sha256"],
            "preprocessing_id": results["preprocessing_id"],
        }
        checkpoint_model = {"activation": "tanh", **checkpoint["model_config"]}
        if checkpoint_model != model_config or any(checkpoint.get(k) != v for k, v in expected.items()):
            raise ValueError("LOOCV checkpoint settings differ from the saved fold")
        model = FeGraphMLP(**model_config)
        model.load_state_dict(checkpoint["state_dict"])
        replay = evaluate_filter(model, [subjects[fold["held_out_subject_id"]]], cache["omega"])[0]
        for key in ("analytical_prediction", "learned_prediction", "Fe_graph"):
            if (fold[key].shape != replay[key].shape
                    or not torch.allclose(fold[key], replay[key], atol=1e-12, rtol=1e-12)):
                raise ValueError(f"LOOCV checkpoint replay differs for {key}")
        for key in ("analytical_correlation", "learned_correlation", "correlation_difference"):
            if not math.isclose(fold[key], replay[key], abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"LOOCV checkpoint replay differs for {key}")
        scores.append(replay)
    for field, key in (("mean_analytical_correlation", "analytical_correlation"),
                       ("mean_learned_correlation", "learned_correlation"),
                       ("mean_correlation_difference", "correlation_difference")):
        expected_mean = math.fsum(row[key] for row in scores) / len(scores)
        if not math.isclose(results[field], expected_mean, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError(f"LOOCV summary differs from replay: {field}")


# Nested protocol, selection, and record verification


def validate_nested_protocol(results, requested):
    """Bind realized evaluation metadata to the requested run configuration."""
    recorded = results["protocol"]
    expected_splits = make_outer_splits(
        results["subject_ids"], strategy=requested["outer_strategy"],
        n_splits=requested.get("outer_folds"), seed=requested["split_seed"],
    )
    expected = {
        "outer_strategy": expected_splits[0]["strategy"],
        "outer_folds": len(expected_splits),
        "outer_membership": expected_splits,
        "inner_folds": requested["inner_folds"],
        "split_seed": requested["split_seed"],
        "training_seed": requested["training_seed"],
        "declared_comparisons": requested.get("declared_comparisons", []),
        "candidate_execution_order": requested.get("candidate_execution_order") or [
            candidate["candidate_id"] for candidate in requested["candidates"]
        ],
    }
    if (results["candidates"] != requested["candidates"]
            or any(recorded.get(key) != value for key, value in expected.items())):
        raise ValueError("Nested result protocol differs from the requested configuration")


def validate_nested_results(results):
    """Validate complete memberships, selection, refit settings, and metrics."""
    if results.get("format_version") != 1 or results.get("evaluation") != "nested_cross_validation":
        raise ValueError("Unsupported nested-result format")
    subject_ids = results.get("subject_ids")
    if not isinstance(subject_ids, list) or len(subject_ids) < 3:
        raise ValueError("Nested results require at least three subject IDs")
    if any(not isinstance(value, str) or not value for value in subject_ids):
        raise ValueError("Nested subject IDs must be nonempty strings")
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Nested subject IDs must be unique")
    all_ids = set(subject_ids)
    outer_folds = results.get("outer_folds")
    if not isinstance(outer_folds, list) or not outer_folds:
        raise ValueError("Nested results require outer-fold records")

    protocol = results.get("protocol", {})
    for name in ("training_seed", "split_seed"):
        if isinstance(protocol.get(name), bool) or not isinstance(protocol.get(name), int):
            raise ValueError(f"Nested protocol requires an integer {name}")
    constants = {
        "inner_strategy": "shuffled_balanced_kfold",
        "selection_metric": "mean_raw_subject_spectral_correlation",
        "subject_weighting": "equal_subject_weight; pooled subject scores across inner folds",
        "tie_rule": "first candidate in declared order",
    }
    if any(protocol.get(key) != value for key, value in constants.items()):
        raise ValueError("Nested selection rules do not match the supported protocol")
    expected_outer = make_outer_splits(
        subject_ids, strategy=protocol.get("outer_strategy"),
        n_splits=protocol.get("outer_folds"), seed=protocol["split_seed"],
    )
    if protocol.get("outer_membership") != expected_outer or len(outer_folds) != len(expected_outer):
        raise ValueError("Outer memberships do not match the declared strategy and seed")
    candidates = results.get("candidates", [])
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Nested candidates must have unique IDs")
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    execution_order = protocol.get("candidate_execution_order", [])
    if len(execution_order) != len(candidate_ids) or set(execution_order) != set(candidate_ids):
        raise ValueError("Candidate execution order must cover declared candidates once")

    outer_seen = set()
    prediction_seen = []
    seen_fold_numbers = set()
    for outer in outer_folds:
        fold_number = outer.get("outer_fold")
        if isinstance(fold_number, bool) or not isinstance(fold_number, int) or fold_number < 1:
            raise ValueError("Outer fold numbers must be positive integers")
        if fold_number in seen_fold_numbers:
            raise ValueError("Outer fold numbers must be unique")
        seen_fold_numbers.add(fold_number)
        train_ids = outer.get("training_subject_ids", [])
        test_ids = outer.get("test_subject_ids", [])
        if (not train_ids or not test_ids or len(set(train_ids)) != len(train_ids)
                or len(set(test_ids)) != len(test_ids)):
            raise ValueError("Outer split IDs must be nonempty and unique")
        if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != all_ids:
            raise ValueError("Outer training/test IDs must be disjoint and cover the cohort")
        if outer_seen.intersection(test_ids):
            raise ValueError("Outer test subjects must occur in exactly one fold")
        outer_seen.update(test_ids)

        inner_splits = outer.get("inner_splits", [])
        expected_seed = inner_split_seed(protocol["split_seed"], fold_number)
        expected_inner = make_inner_splits(
            train_ids, n_splits=protocol.get("inner_folds"), seed=expected_seed,
        )
        if outer.get("inner_split_seed") != expected_seed or inner_splits != expected_inner:
            raise ValueError("Inner splits do not match the declared count, memberships, and seed")
        candidate_scores = outer.get("candidate_scores", {})
        if set(candidate_scores) != set(candidate_ids):
            raise ValueError("Each outer fold must record scores for every declared candidate")
        for candidate_id in candidate_ids:
            rows = candidate_scores[candidate_id]
            row_ids = [row["subject_id"] for row in rows]
            if (len(row_ids) != len(set(row_ids)) or set(row_ids) != set(train_ids)
                    or any(not math.isfinite(float(row["r"])) or not -1 <= float(row["r"]) <= 1
                           for row in rows)):
                raise ValueError("Candidate inner scores must cover outer training subjects once")
        if set(outer.get("candidate_mean_raw_r", {})) != set(candidate_ids):
            raise ValueError("Nested result must save one mean inner score per candidate")
        means = {}
        for candidate_id in candidate_ids:
            scores = {row["subject_id"]: float(row["r"]) for row in candidate_scores[candidate_id]}
            means[candidate_id] = math.fsum(scores[value] for value in train_ids) / len(train_ids)
            if not math.isclose(means[candidate_id], float(outer["candidate_mean_raw_r"][candidate_id]),
                                abs_tol=1e-15, rel_tol=1e-15):
                raise ValueError("Saved candidate mean does not match its per-subject inner scores")
        expected_selected = candidate_ids[0]
        for candidate_id in candidate_ids[1:]:
            if means[candidate_id] > means[expected_selected]:
                expected_selected = candidate_id
        if outer.get("selected_candidate_id") not in candidate_ids:
            raise ValueError("Selected candidate must be in the declared candidate list")
        if outer["selected_candidate_id"] != expected_selected:
            raise ValueError("Selected candidate does not follow the declared metric and tie rule")
        checkpoint_path = Path(outer.get("checkpoint_path", ""))
        prediction_path = Path(outer.get("prediction_path", ""))
        if (not checkpoint_path.parts or not prediction_path.parts
                or checkpoint_path.is_absolute() or prediction_path.is_absolute()
                or ".." in checkpoint_path.parts or ".." in prediction_path.parts):
            raise ValueError("Nested artifact paths must be relative and remain inside the run folder")
        refit = outer.get("outer_refit", {})
        selected = candidates_by_id[outer["selected_candidate_id"]]
        expected_training = {**selected["training_config"], "seed": protocol["training_seed"]}
        if (refit.get("model_config") != selected["model_config"]
                or refit.get("training_config") != expected_training
                or refit.get("trained_epochs") != expected_training["epochs"]):
            raise ValueError("Outer-refit settings differ from the selected candidate or training seed")
        history = refit.get("training_history", [])
        trained_epochs = refit.get("trained_epochs")
        if (isinstance(trained_epochs, bool) or not isinstance(trained_epochs, int)
                or trained_epochs < 1 or len(history) != trained_epochs
                or [entry.get("epoch") for entry in history] != list(range(1, trained_epochs + 1))
                or any(not math.isfinite(float(entry["train_loss_before_step"])) for entry in history)):
            raise ValueError("Outer-refit history must contain one finite loss per fixed epoch")
        prediction_rows = outer.get("outer_subject_records", [])
        prediction_ids = [row["subject_id"] for row in prediction_rows]
        if len(prediction_ids) != len(set(prediction_ids)) or set(prediction_ids) != set(test_ids):
            raise ValueError("Outer subject records must cover each test subject exactly once")
        for row in prediction_rows:
            values = (float(row["analytical_r"]), float(row["learned_r"]),
                      float(row["raw_r_difference"]))
            if (any(not math.isfinite(value) for value in values)
                    or not -1 <= values[0] <= 1 or not -1 <= values[1] <= 1
                    or not math.isclose(values[2], values[1] - values[0], abs_tol=1e-15, rel_tol=1e-15)
                    or row.get("outer_fold") != fold_number
                    or row.get("selected_candidate_id") != outer["selected_candidate_id"]
                    or row.get("prediction_path") != prediction_path.as_posix()):
                raise ValueError("Outer subject score or artifact reference is inconsistent")
        prediction_seen.extend(prediction_ids)

    if outer_seen != all_ids or len(prediction_seen) != len(set(prediction_seen)):
        raise ValueError("Outer evaluation must produce exactly one prediction per subject")
    memberships = results.get("protocol", {}).get("outer_membership", [])
    if len(memberships) != len(outer_folds):
        raise ValueError("Protocol must save every realized outer membership")
    for membership, outer in zip(memberships, outer_folds):
        if (membership.get("outer_fold") != outer["outer_fold"]
                or membership.get("training_subject_ids") != outer["training_subject_ids"]
                or membership.get("test_subject_ids") != outer["test_subject_ids"]):
            raise ValueError("Realized outer membership differs from its fold record")
    rows = [row for outer in outer_folds for row in outer["outer_subject_records"]]
    validate_outer_summary(results.get("outer_summary"), summarize_outer_scores(rows, candidate_ids))
    return results


def summarize_outer_scores(rows, candidate_ids):
    """Calculate equally weighted outer metrics from subject records."""
    if not rows:
        raise ValueError("Outer summary requires subject records")
    return {
        "n_subject_predictions": len(rows),
        "mean_analytical_r": math.fsum(row["analytical_r"] for row in rows) / len(rows),
        "mean_selected_mlp_r": math.fsum(row["learned_r"] for row in rows) / len(rows),
        "mean_raw_r_difference": math.fsum(row["raw_r_difference"] for row in rows) / len(rows),
        "selected_candidate_counts": {
            candidate_id: sum(row["selected_candidate_id"] == candidate_id for row in rows)
            for candidate_id in candidate_ids
        },
    }


def validate_outer_summary(saved, expected):
    """Reject aggregate values that contradict validated subject records."""
    if not isinstance(saved, dict) or set(saved) != set(expected):
        raise ValueError("Outer summary must contain all calculated metrics")
    for key, value in expected.items():
        if isinstance(value, float):
            matches = math.isclose(float(saved[key]), value, abs_tol=1e-12, rel_tol=1e-12)
        else:
            matches = saved[key] == value
        if not matches:
            raise ValueError(f"Outer summary differs from subject records: {key}")


def _artifact_path(run_dir, relative_path):
    path = Path(relative_path)
    resolved = (run_dir / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(run_dir) or not resolved.is_file():
        raise ValueError(f"Missing or invalid nested artifact: {relative_path}")
    return resolved


def _matching_statistics(saved, expected):
    """Compare saved tests while allowing numerical-library roundoff."""
    if isinstance(expected, dict):
        return (isinstance(saved, dict) and set(saved) == set(expected)
                and all(_matching_statistics(saved[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(saved, list) and len(saved) == len(expected)
                and all(_matching_statistics(left, right) for left, right in zip(saved, expected)))
    if isinstance(expected, float):
        return (isinstance(saved, (int, float)) and not isinstance(saved, bool)
                and math.isclose(saved, expected, abs_tol=1e-12, rel_tol=1e-12))
    return type(saved) is type(expected) and saved == expected


# Spectrum and plot-input utilities


def normalize_spectra(values):
    transformed = transform_spectra(values)
    norms = transformed.norm(dim=-1, keepdim=True)
    if not torch.isfinite(transformed).all() or (norms <= 1e-12).any():
        raise ValueError("Analysis requires finite, nonconstant regional spectra")
    return transformed / norms


def regional_scores(prediction, target):
    return (normalize_spectra(prediction) * normalize_spectra(target)).sum(dim=-1)


def analytical_graph_filter(subject, frequencies):
    tau_e = subject["parameters"]["tau_e"]
    omega = 2 * torch.pi * frequencies
    return (1 / tau_e**2) / (1j * omega + 1 / tau_e)**2


def prepare_cohort_plot_series(subject_records, subjects, series_specs):
    """Validate common region order, then normalize each region before averaging.

    Return node indices and arrays with subject, region, and frequency axes.
    The subject axis remains the sampling unit for standard-deviation bands.
    """
    nodes = subjects[subject_records[0]["held_out_subject_id"]]["observed_regions"]
    if any(not torch.equal(subjects[f["held_out_subject_id"]]["observed_regions"], nodes)
           for f in subject_records):
        raise ValueError("Across-subject spectra require identical observed region order")
    if len(subject_records) < 2:
        raise ValueError("Across-subject sample SD requires at least two subjects")
    cohort_series = []
    for label, key, color, linestyle in series_specs:
        values = torch.stack([
            normalize_spectra(subjects[f["held_out_subject_id"]][key]
                                if label == "MEG" else f[key]) for f in subject_records
        ]).numpy()  # subjects x regions x frequencies
        cohort_series.append((label, values, color, linestyle))

    return nodes, cohort_series


def subject_plot_rows(subject_id, n_regions, region_rows=None):
    """Validate a plot directory name and choose explicit observed-region rows."""
    if Path(subject_id).name != subject_id or subject_id in (".", "..") or "\\" in subject_id:
        raise ValueError("Subject ID must be a safe directory name")
    rows = list(region_rows) if region_rows is not None else sorted({0, n_regions // 2, n_regions - 1})
    if not rows or any(not isinstance(row, int) or row < 0 or row >= n_regions for row in rows):
        raise ValueError("region_rows must be nonempty valid zero-based observed rows")
    return rows


# Report rows, files, and summary text


@torch.no_grad()
def loocv_report_rows(results, cache):
    """Build subject, region, history, and filter rows from checked LOOCV data."""
    results, cache, subjects = validate_loocv_results(results, cache)
    subject_rows, region_rows, history_rows, filter_rows = [], [], [], []
    for fold in results["folds"]:
        subject = subjects[fold["held_out_subject_id"]]
        base = {"fold": fold["fold"], "subject_id": subject["subject_id"]}
        analytical = regional_scores(fold["analytical_prediction"], subject["empirical_meg"])
        learned = regional_scores(fold["learned_prediction"], subject["empirical_meg"])
        if (not np.isclose(analytical.mean().item(), fold["analytical_correlation"], atol=1e-10, rtol=1e-8)
                or not np.isclose(learned.mean().item(), fold["learned_correlation"], atol=1e-10, rtol=1e-8)):
            raise ValueError("Recomputed scores differ from saved results; check the cache")
        subject_rows.append({**base, "n_training_subjects": len(fold["training_subject_ids"]),
                             "n_regions": analytical.numel(),
                             "analytical_training_correlation": fold.get("analytical_training_correlation", ""),
                             "learned_training_correlation": fold.get("learned_training_correlation", ""),
                             "analytical_correlation": analytical.mean().item(),
                             "learned_correlation": learned.mean().item(),
                             "correlation_difference": (learned.mean() - analytical.mean()).item()})
        for row, node in enumerate(subject["observed_regions"].tolist()):
            region_rows.append({**base, "meg_region_row": row, "model_node_index": node,
                                "analytical_correlation": analytical[row].item(),
                                "learned_correlation": learned[row].item(),
                                "correlation_difference": (learned[row] - analytical[row]).item()})
        history_rows.extend({**base, **entry} for entry in fold["history"])
        control = analytical_graph_filter(subject, cache["frequencies_hz"])
        for frequency, baseline, value in zip(cache["frequencies_hz"], control, fold["Fe_graph"]):
            filter_rows.append({**base, "frequency_hz": frequency.item(),
                                "analytical_real": baseline.real.item(),
                                "analytical_imaginary": baseline.imag.item(),
                                "analytical_magnitude": baseline.abs().item(),
                                "analytical_phase_rad": baseline.angle().item(),
                                "learned_real": value.real.item(),
                                "learned_imaginary": value.imag.item(),
                                "learned_magnitude": value.abs().item(),
                                "learned_phase_rad": value.angle().item()})
    return {"subject_scores": subject_rows, "region_scores": region_rows,
            "training_history": history_rows, "filter_values": filter_rows}


def nested_report_rows(results, verified_rows):
    """Build outer scores, candidate choices, inner scores, and refit histories."""
    outer_rows = [{
        "outer_fold": row["outer_fold"],
        "subject_id": row["subject_id"],
        "selected_candidate_id": row["selected_candidate_id"],
        "n_regions": row["n_regions"],
        "analytical_r": row["analytical_r"],
        "selected_mlp_r": row["learned_r"],
        "raw_r_difference": row["raw_r_difference"],
        "prediction_path": row["prediction_path"],
    } for row in verified_rows]
    candidate_rows, inner_rows, history_rows = [], [], []
    for outer in results["outer_folds"]:
        inner_fold_for_subject = {
            subject_id: inner["inner_fold"]
            for inner in outer["inner_splits"]
            for subject_id in inner["validation_subject_ids"]
        }
        for candidate in results["candidates"]:
            candidate_id = candidate["candidate_id"]
            selected = candidate_id == outer["selected_candidate_id"]
            candidate_rows.append({
                "outer_fold": outer["outer_fold"],
                "candidate_id": candidate_id,
                "selected": selected,
                "mean_inner_raw_r": outer["candidate_mean_raw_r"][candidate_id],
                "model_config": json.dumps(candidate["model_config"], sort_keys=True),
                "training_config": json.dumps(candidate["training_config"], sort_keys=True),
            })
            inner_rows.extend({
                "outer_fold": outer["outer_fold"],
                "inner_fold": inner_fold_for_subject[row["subject_id"]],
                "candidate_id": candidate_id,
                "subject_id": row["subject_id"],
                "validation_r": row["r"],
                "selected": selected,
            } for row in outer["candidate_scores"][candidate_id])
        history_rows.extend({
            "outer_fold": outer["outer_fold"], **entry
        } for entry in outer["outer_refit"]["training_history"])

    return {"outer_subject_scores": outer_rows, "candidate_selection": candidate_rows,
            "inner_subject_scores": inner_rows, "outer_refit_history": history_rows}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_report_tables(output_dir, tables):
    """Write each named row collection to its corresponding CSV file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {name: write_csv(output_dir / f"{name}.csv", rows) for name, rows in tables.items()}


def loocv_summary_lines(run_dir, config, results):
    """Describe one fixed-configuration evaluation and its diagnostic figures."""
    analytical = np.array([fold["analytical_correlation"] for fold in results["folds"]])
    learned = np.array([fold["learned_correlation"] for fold in results["folds"]])
    difference = learned - analytical
    training = config["training_config"]
    lines = ["# Run summary", "",
             f"Run: `{run_dir.name}`.", "",
             f"LOOCV: {len(difference)} subjects; {training['epochs']} fixed epochs; seed {training['seed']}.",
             f"Model settings: `{json.dumps(config['model_config'], sort_keys=True)}`. "
             f"Adam learning rate: {training['learning_rate']}.", "",
             "| Held-out measure | Value |", "|---|---:|",
             f"| Mean analytical r | {analytical.mean():.6f} |",
             f"| Mean MLP r | {learned.mean():.6f} |",
             f"| Mean difference (MLP − analytical) | {difference.mean():.6f} |",
             f"| Median difference | {np.median(difference):.6f} |",
             f"| Subjects improved | {int((difference > 0).sum())}/{len(difference)} |", "",
             "Each score is the mean spectral correlation over the subject's observed regions. "
             "The seven fitted SGM parameters and local analytical filter stay fixed. "
             "This evaluates filter transfer conditional on those supplied fitted parameters.", "",
             "## Figures", "",
             "- [Held-out scores](figures/summary/held_out_scores.png): "
             "A, paired scores; B, identity comparison with equal axes; C, subject differences.",
             "- [Training curves](figures/summary/training_curves.png): "
             "gray fold curves and black mean; loss is measured before each optimizer step.",
             "- [Across-subject spectra](figures/summary/spectra_across_subjects.png): "
             "normalize each regional spectrum, average regions within each subject, then show "
             "the mean ± one sample SD across subjects (equal subject weights).",
             "- [Regional spectra](figures/summary/spectra_by_region.png): "
             "each observed region separately, with mean ± one sample SD across subjects. "
             "Both summaries show MEG, analytical and held-out MLP shapes, not absolute power; "
             "bands describe subject variation, not confidence intervals.",
             "- [Subject figures](figures/subjects/): spectra and filter magnitude/phase for every subject. "
             "Spectra use the first, middle and last observed rows (zero-based). "
             "Magnitude, smoothing, square root, demeaning and unit norm produce the displayed shape; "
             "these are not absolute power spectra. Filter phase is unwrapped, and the analytical "
             "filter uses the held-out subject's fixed tau_e.", "",
             "## Saved data", "",
             "- [Subject scores](tables/subject_scores.csv)",
             "- [Region scores and model-node indices](tables/region_scores.csv)",
             "- [Training history](tables/training_history.csv)",
             "- [Complex filter values](tables/filter_values.csv)",
             "- [Configuration and original training-source hashes](config.json)", ""]
    return lines


def nested_summary_lines(run_dir, results, verified_rows):
    """Describe final held-out performance and link its saved outputs."""
    summary_metrics = summarize_outer_scores(
        verified_rows, [candidate["candidate_id"] for candidate in results["candidates"]]
    )
    selected_counts = summary_metrics["selected_candidate_counts"]
    lines = [
        "# Nested evaluation summary", "",
        f"Run: `{run_dir.name}`.", "",
        f"Outer evaluation: {len(verified_rows)} subject predictions across {len(results['outer_folds'])} folds.",
        f"Inner selection: {results['protocol']['inner_folds']} balanced subject folds per outer fold.",
        "Candidates are ranked by the pooled mean raw subject spectral correlation. "
        "Each subject has equal weight. The first declared candidate wins an exact tie.", "",
        "| Outer measure | Value |", "|---|---:|",
        f"| Mean analytical r | {summary_metrics['mean_analytical_r']:.6f} |",
        f"| Mean selected MLP r | {summary_metrics['mean_selected_mlp_r']:.6f} |",
        f"| Mean raw-r difference | {summary_metrics['mean_raw_r_difference']:.6f} |", "",
        "## Selected configurations", "",
        "| Candidate | Outer subjects |", "|---|---:|",
        *[f"| `{candidate_id}` | {count} |" for candidate_id, count in selected_counts.items()], "",
        "Outer predictions are evaluated once after selection and fresh refitting. "
        "Inner scores are selection evidence; outer scores are evaluation evidence.",
        "The cohort has already informed prior model choices, and fitted SGM parameters "
        "come from MEG, so nested evaluation does not create an untouched cohort or "
        "establish structure-only prediction.", "",
        "Across-subject means weight subjects equally. Correlations describe spectral shape, "
        "not calibrated absolute power.", "",
    ]
    if results.get("statistics"):
        lines.extend(["## Declared comparisons", ""])
        for item in results["statistics"]:
            pvalue = item.get("p_value_unadjusted")
            value = "undefined" if pvalue is None else f"{pvalue:.6g}"
            lines.append(
                f"- `{item['contrast']}`: n={item['n_subject_pairs']}, "
                f"t({item['degrees_of_freedom']})={item['t_statistic']}, "
                f"unadjusted p={value}; exploratory because outer training sets overlap."
            )
        lines.append("")
    lines.extend([
        "## Saved tables", "",
        "- [Outer subject scores](tables/outer_subject_scores.csv)",
        "- [Candidate selection and inner means](tables/candidate_selection.csv)",
        "- [Inner subject scores](tables/inner_subject_scores.csv)",
        "- [Outer-refit training histories](tables/outer_refit_history.csv)",
        "- [Versioned result record](nested_results.json)",
        "- [Configuration and provenance](config.json)", "",
    ])
    lines.extend([
        "## Figures", "",
        "- [Outer-refit loss curves](figures/summary/training_curves.png): one curve per outer model; "
        "a mean is shown only when epoch grids match.",
        "- [Held-out scores](figures/summary/held_out_scores.png): paired scores and subject differences.",
        "- [Across-subject spectra](figures/summary/spectra_across_subjects.png): "
        "normalize each region, average regions within each subject, then show mean ± one sample SD "
        "across subjects. The bands describe subject variation, not confidence intervals.",
        "- [Regional spectra](figures/summary/spectra_by_region.png): the same summary for each region.",
        "- [Subject figures](figures/subjects/): spectra and filter magnitude/phase for every subject.", "",
    ])
    return lines
