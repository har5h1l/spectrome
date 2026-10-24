"""Select settings inside each outer training set, then refit and evaluate."""

import math
from pathlib import Path

from ..data.data import prepare_spectral_subject
from ..data.helpers import validate_training_cache
from ..spectra import evaluate_filter
from .helpers import (
    inner_split_seed, make_inner_splits, make_outer_splits, normalize_candidates,
    save_json_atomic, create_nested_result, save_nested_fold, complete_nested_result,
    report_nested_epoch, validate_nested_settings,
)
from .training import train_filter


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
    validate_nested_settings(protocol)
    validate_training_cache(cache)
    output_dir = Path(output_dir)
    subject_ids = [subject["subject_id"] for subject in cache["subjects"]]
    if len(subject_ids) < 3 or len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Nested evaluation requires at least three subjects with unique IDs")
    candidates = normalize_candidates(protocol["candidates"], protocol["training_seed"])
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    execution_order = candidate_execution_order or candidate_ids
    if len(execution_order) != len(set(execution_order)) or set(execution_order) != set(candidate_ids):
        raise ValueError("candidate_execution_order must contain every candidate exactly once")
    split_seed = protocol["split_seed"]
    outer_strategy = protocol["outer_strategy"]
    outer_folds = protocol.get("outer_folds")
    inner_folds = protocol["inner_folds"]
    training_seed = protocol["training_seed"]
    splits = make_outer_splits(
        subject_ids, strategy=outer_strategy, n_splits=outer_folds, seed=split_seed
    )
    inner_splits_by_outer = {
        outer["outer_fold"]: make_inner_splits(
            outer["training_subject_ids"], n_splits=inner_folds,
            seed=inner_split_seed(split_seed, outer["outer_fold"]),
        ) for outer in splits
    }
    prepared = [prepare_spectral_subject(subject) for subject in cache["subjects"]]
    by_id = {subject["subject_id"]: subject for subject in prepared}
    result = create_nested_result(
        cache, protocol, splits, candidates, execution_order,
        source_sha256=source_sha256, cache_sha256=cache_sha256,
    )
    result_path = output_dir / "nested_results.json"
    save_json_atomic(result_path, result)
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}

    try:
        for outer in splits:
            outer_fold = outer["outer_fold"]
            train_ids = outer["training_subject_ids"]
            test_ids = outer["test_subject_ids"]
            inner_splits = inner_splits_by_outer[outer_fold]
            scores_by_candidate = {candidate_id: {} for candidate_id in candidate_ids}
            for candidate_id in execution_order:
                candidate = candidates_by_id[candidate_id]
                model_config = candidate["model_config"]
                train_config = {
                    **candidate["training_config"],
                    "seed": training_seed,
                }
                for inner in inner_splits:
                    label = (f"outer={outer_fold}/{len(splits)} candidate={candidate_id} "
                             f"inner={inner['inner_fold']}/{len(inner_splits)}")
                    print(f"Starting {label}", flush=True)
                    model, _history = train_filter(
                        [by_id[value] for value in inner["training_subject_ids"]],
                        cache["omega"], model_config, train_config,
                        on_epoch=report_nested_epoch(label, train_config["epochs"]),
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
            label = f"outer={outer_fold}/{len(splits)} refit={selected_id}"
            print(f"Starting {label}", flush=True)
            outer_model, outer_history = train_filter(
                [by_id[value] for value in train_ids], cache["omega"],
                selected["model_config"], selected_train_config,
                on_epoch=report_nested_epoch(label, selected_train_config["epochs"]),
            )
            outer_evaluation = evaluate_filter(
                outer_model, [by_id[value] for value in test_ids], cache["omega"]
            )

            outer_record = save_nested_fold(
                output_dir, result, outer, inner_splits, selected,
                scores_by_candidate, candidate_means, outer_model,
                outer_history, outer_evaluation,
            )
            result["outer_folds"].append(outer_record)
            save_json_atomic(result_path, result)

        complete_nested_result(result, result_path)
        return result
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = f"{type(error).__name__}: {error}"
        save_json_atomic(result_path, result)
        raise
