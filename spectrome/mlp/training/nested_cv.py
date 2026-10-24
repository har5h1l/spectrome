"""Select settings inside each outer training set, then refit and evaluate."""

from datetime import datetime, timezone
import math
from pathlib import Path

import torch

from ..data.data import prepare_spectral_subject
from ..data.helpers import validate_training_cache
from ..spectra import SCORING_ID, evaluate_filter
from ..analysis.helpers import summarize_outer_scores, validate_nested_results
from ..training.helpers import save_json_atomic
from .helpers import inner_split_seed, make_inner_splits, make_outer_splits, save_checkpoint
from .training import train_filter


def _normalize_candidates(candidates, training_seed):
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a nonempty list")
    normalized = []
    seen = set()
    for candidate in candidates:
        if set(candidate) != {"candidate_id", "model_config", "training_config"}:
            raise ValueError("Each candidate must define candidate_id, model_config, and training_config")
        candidate_id = candidate["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ValueError("Candidate IDs must be unique nonempty strings")
        seen.add(candidate_id)
        model_config = candidate["model_config"]
        training_config = candidate["training_config"]
        if set(model_config) != {"hidden_sizes", "omega_scale", "activation"}:
            raise ValueError(f"{candidate_id}: complete model_config is required")
        if set(training_config) != {"epochs", "learning_rate"}:
            raise ValueError(f"{candidate_id}: training_config must specify epochs and learning_rate only")
        normalized.append({
            "candidate_id": candidate_id,
            "model_config": {
                "hidden_sizes": list(model_config["hidden_sizes"]),
                "omega_scale": float(model_config["omega_scale"]),
                "activation": model_config["activation"],
            },
            "training_config": {
                "epochs": training_config["epochs"],
                "learning_rate": float(training_config["learning_rate"]),
            },
        })
    return normalized


def select_candidate(candidates, candidate_scores, subject_ids):
    """Select by pooled raw subject r, with declared candidate order as tie rule."""
    ids = list(subject_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Selection requires unique subject IDs")
    declared_ids = [candidate["candidate_id"] for candidate in candidates]
    if len(declared_ids) != len(set(declared_ids)) or set(candidate_scores) != set(declared_ids):
        raise ValueError("Every declared candidate must have exactly one score mapping")
    means = {}
    for candidate_id in declared_ids:
        scores = candidate_scores[candidate_id]
        if set(scores) != set(ids):
            raise ValueError(f"{candidate_id}: inner scores must cover every training subject once")
        values = [float(scores[subject_id]) for subject_id in ids]
        if any(not math.isfinite(value) or not -1 <= value <= 1 for value in values):
            raise ValueError(f"{candidate_id}: inner scores must be finite correlations")
        means[candidate_id] = math.fsum(values) / len(values)
    selected = declared_ids[0]
    for candidate_id in declared_ids[1:]:
        if means[candidate_id] > means[selected]:
            selected = candidate_id
    return selected, means


def nested_cross_validate(cache, output_dir, protocol, *, source_sha256,
                          cache_sha256, candidate_execution_order=None):
    """Run nested subject-level selection and save auditable fold records.

    ``output_dir`` must already exist and belong to a fresh run. The caller
    owns cache acquisition and run-status metadata.
    """
    validate_training_cache(cache)
    output_dir = Path(output_dir)
    subject_ids = [subject["subject_id"] for subject in cache["subjects"]]
    if len(subject_ids) < 3 or len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Nested evaluation requires at least three subjects with unique IDs")
    candidates = _normalize_candidates(protocol["candidates"], protocol["training_seed"])
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    execution_order = candidate_execution_order or candidate_ids
    if len(execution_order) != len(set(execution_order)) or set(execution_order) != set(candidate_ids):
        raise ValueError("candidate_execution_order must contain every candidate exactly once")
    split_seed = protocol["split_seed"]
    outer_strategy = protocol["outer_strategy"]
    outer_folds = protocol.get("outer_folds")
    inner_folds = protocol["inner_folds"]
    training_seed = protocol["training_seed"]
    declared_comparisons = protocol.get("declared_comparisons", [])
    if any(item != "selected_mlp_vs_analytical" for item in declared_comparisons):
        raise ValueError("Supported nested contrast: selected_mlp_vs_analytical")

    splits = make_outer_splits(
        subject_ids, strategy=outer_strategy, n_splits=outer_folds, seed=split_seed
    )
    prepared = [prepare_spectral_subject(subject) for subject in cache["subjects"]]
    by_id = {subject["subject_id"]: subject for subject in prepared}
    protocol_record = {
        "outer_strategy": "leave_one_subject_out" if outer_strategy in {
            "leave_one_out", "leave_one_subject_out"
        } else "grouped_kfold",
        "outer_folds": len(splits),
        "outer_membership": splits,
        "inner_strategy": "shuffled_balanced_kfold",
        "inner_folds": inner_folds,
        "split_seed": split_seed,
        "training_seed": training_seed,
        "selection_metric": "mean_raw_subject_spectral_correlation",
        "subject_weighting": "equal_subject_weight; pooled subject scores across inner folds",
        "tie_rule": "first candidate in declared order",
        "candidate_execution_order": list(execution_order),
        "declared_comparisons": list(declared_comparisons),
    }
    result = {
        "format_version": 1,
        "evaluation": "nested_cross_validation",
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scoring_id": SCORING_ID,
        "loss_preprocessing": "magnitude_conv5_sqrt_floor1e-12_demean",
        "subject_ids": subject_ids,
        "subject_metadata": [{
            "subject_id": subject["subject_id"],
            "provenance": subject["provenance"],
            "observed_regions": subject["observed_regions"].tolist(),
            "parameters": {name: value.item() for name, value in subject["parameters"].items()},
        } for subject in cache["subjects"]],
        "preprocessing_id": cache["preprocessing_id"],
        "frequencies_hz": cache["frequencies_hz"].tolist(),
        "forward_sha256": cache["forward_sha256"],
        "cache_sha256": cache_sha256,
        "source_sha256": source_sha256,
        "protocol": protocol_record,
        "candidates": candidates,
        "outer_folds": [],
    }
    result_path = output_dir / "nested_results.json"
    save_json_atomic(result_path, result)
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}

    try:
        for outer in splits:
            outer_fold = outer["outer_fold"]
            train_ids = outer["training_subject_ids"]
            test_ids = outer["test_subject_ids"]
            inner_seed = inner_split_seed(split_seed, outer_fold)
            inner_splits = make_inner_splits(
                train_ids, n_splits=inner_folds, seed=inner_seed
            )
            scores_by_candidate = {candidate_id: {} for candidate_id in candidate_ids}
            inner_memberships = []
            for inner in inner_splits:
                inner_memberships.append({
                    "inner_fold": inner["inner_fold"],
                    "training_subject_ids": inner["training_subject_ids"],
                    "validation_subject_ids": inner["validation_subject_ids"],
                })
            for candidate_id in execution_order:
                candidate = candidates_by_id[candidate_id]
                model_config = candidate["model_config"]
                train_config = {
                    **candidate["training_config"],
                    "seed": training_seed,
                }
                for inner in inner_splits:
                    model, _history = train_filter(
                        [by_id[value] for value in inner["training_subject_ids"]],
                        cache["omega"], model_config, train_config,
                    )
                    evaluated = evaluate_filter(
                        model,
                        [by_id[value] for value in inner["validation_subject_ids"]],
                        cache["omega"],
                    )
                    for row in evaluated:
                        scores_by_candidate[candidate_id][row["subject_id"]] = row["learned_correlation"]
            selected_id, candidate_means = select_candidate(
                candidates, scores_by_candidate, train_ids
            )
            selected = candidates_by_id[selected_id]
            selected_train_config = {
                **selected["training_config"],
                "seed": training_seed,
            }
            outer_model, outer_history = train_filter(
                [by_id[value] for value in train_ids], cache["omega"],
                selected["model_config"], selected_train_config,
            )
            outer_evaluation = evaluate_filter(
                outer_model, [by_id[value] for value in test_ids], cache["omega"]
            )

            checkpoint_path = Path("checkpoints") / f"outer-fold-{outer_fold:03d}.pt"
            checkpoint = {
                "format_version": 1,
                "evaluation": "nested_cross_validation",
                "outer_fold": outer_fold,
                "model_config": selected["model_config"],
                "training_config": selected_train_config,
                "selected_candidate_id": selected_id,
                "training_subject_ids": train_ids,
                "test_subject_ids": test_ids,
                "inner_splits": inner_memberships,
                "cache_sha256": cache_sha256,
                "forward_sha256": cache["forward_sha256"],
                "source_sha256": source_sha256,
                "state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in outer_model.state_dict().items()
                },
            }
            save_checkpoint(output_dir / checkpoint_path, checkpoint)
            prediction_path = Path("predictions") / f"outer-fold-{outer_fold:03d}.pt"
            save_checkpoint(output_dir / prediction_path, {
                "outer_fold": outer_fold,
                "selected_candidate_id": selected_id,
                "subject_predictions": {
                    row["subject_id"]: {
                        key: row[key] for key in (
                            "analytical_prediction", "learned_prediction", "Fe_graph"
                        )
                    }
                    for row in outer_evaluation
                },
            })
            outer_records = [{
                "outer_fold": outer_fold,
                "subject_id": row["subject_id"],
                "selected_candidate_id": selected_id,
                "analytical_r": row["analytical_correlation"],
                "learned_r": row["learned_correlation"],
                "raw_r_difference": row["correlation_difference"],
                "prediction_path": prediction_path.as_posix(),
            } for row in outer_evaluation]
            candidate_records = {}
            for candidate_id in candidate_ids:
                candidate_records[candidate_id] = [{
                    "subject_id": subject_id,
                    "r": float(scores_by_candidate[candidate_id][subject_id]),
                } for subject_id in train_ids]
            outer_record = {
                "outer_fold": outer_fold,
                "training_subject_ids": train_ids,
                "test_subject_ids": test_ids,
                "inner_split_seed": inner_seed,
                "inner_splits": inner_memberships,
                "candidate_scores": candidate_records,
                "candidate_mean_raw_r": candidate_means,
                "selected_candidate_id": selected_id,
                "outer_refit": {
                    "model_config": selected["model_config"],
                    "training_config": selected_train_config,
                    "trained_epochs": selected_train_config["epochs"],
                    "training_history": outer_history,
                },
                "checkpoint_path": checkpoint_path.as_posix(),
                "prediction_path": prediction_path.as_posix(),
                "outer_subject_records": outer_records,
            }
            result["outer_folds"].append(outer_record)
            save_json_atomic(result_path, result)

        subject_records = [
            row for outer in result["outer_folds"]
            for row in outer["outer_subject_records"]
        ]
        result["outer_summary"] = summarize_outer_scores(subject_records, candidate_ids)
        result["statistics"] = []
        if "selected_mlp_vs_analytical" in declared_comparisons:
            from ..analysis.statistics import paired_fisher_z_test
            result["statistics"].append(paired_fisher_z_test(
                {row["subject_id"]: row["learned_r"] for row in subject_records},
                {row["subject_id"]: row["analytical_r"] for row in subject_records},
                left_label="nested_selected_mlp",
                right_label="analytical",
                contrast="selected_mlp_vs_analytical",
                expected_subject_ids=subject_ids,
                provenance={
                    "cache_sha256": cache_sha256,
                    "preprocessing_id": cache["preprocessing_id"],
                    "forward_sha256": cache["forward_sha256"],
                    "scoring_id": SCORING_ID,
                },
            ))
        result["status"] = "complete"
        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        validate_nested_results(result)
        save_json_atomic(result_path, result)
        return result
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = f"{type(error).__name__}: {error}"
        save_json_atomic(result_path, result)
        raise
