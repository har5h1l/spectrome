"""Select settings inside each outer training set, then refit and evaluate."""

import math
from pathlib import Path

from ..data.data import prepare_spectral_subject
from ..spectra import evaluate_filter
from .helpers import (
    prepare_nested_evaluation, validate_candidate_scores, nested_fit_progress,
    save_json_atomic, create_nested_result, save_nested_fold, complete_nested_result,
)
from .training import train_filter


def select_candidate(candidates, candidate_scores, subject_ids):
    """Select by pooled raw subject r, with declared candidate order as tie rule."""
    validate_candidate_scores(candidates, candidate_scores, subject_ids)
    ids = list(subject_ids)
    declared_ids = [candidate["candidate_id"] for candidate in candidates]
    # Pool subject scores so every subject has equal weight.
    means = {}
    for candidate_id in declared_ids:
        scores = candidate_scores[candidate_id]
        values = [float(scores[subject_id]) for subject_id in ids]
        means[candidate_id] = math.fsum(values) / len(values)
    # Keep the first declared candidate when mean scores tie.
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
    # 1. Validate inputs and prepare all subject splits before fitting.
    candidates, execution_order, outer_splits, inner_splits_by_outer = prepare_nested_evaluation(
        cache, protocol, candidate_execution_order,
    )
    output_dir = Path(output_dir)
    training_seed = protocol["training_seed"]
    prepared_subjects = [prepare_spectral_subject(subject) for subject in cache["subjects"]]
    subjects_by_id = {subject["subject_id"]: subject for subject in prepared_subjects}
    result = create_nested_result(
        cache, protocol, outer_splits, candidates, execution_order,
        source_sha256=source_sha256, cache_sha256=cache_sha256,
    )
    result_path = output_dir / "nested_results.json"
    save_json_atomic(result_path, result)
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}

    try:
        for outer_split in outer_splits:
            outer_fold = outer_split["outer_fold"]
            train_ids = outer_split["training_subject_ids"]
            test_ids = outer_split["test_subject_ids"]
            inner_splits = inner_splits_by_outer[outer_fold]
            # 2. Compare configurations using only inner training/validation subjects.
            scores_by_candidate = {candidate_id: {} for candidate_id in candidates_by_id}
            for candidate_id in execution_order:
                candidate = candidates_by_id[candidate_id]
                model_config = candidate["model_config"]
                train_config = {
                    **candidate["training_config"],
                    "seed": training_seed,
                }
                for inner_split in inner_splits:
                    progress = nested_fit_progress(
                        outer_fold, len(outer_splits), candidate, inner_split, len(inner_splits),
                    )
                    model, _history = train_filter(
                        [subjects_by_id[value] for value in inner_split["training_subject_ids"]],
                        cache["omega"], model_config, train_config,
                        on_epoch=progress,
                    )
                    validation_results = evaluate_filter(
                        model,
                        [subjects_by_id[value] for value in inner_split["validation_subject_ids"]],
                        cache["omega"],
                    )
                    for row in validation_results:
                        scores_by_candidate[candidate_id][row["subject_id"]] = row["learned_correlation"]
            # 3. Select settings from the inner validation scores.
            selected_id, candidate_means = select_candidate(
                candidates, scores_by_candidate, train_ids
            )
            selected_candidate = candidates_by_id[selected_id]
            selected_train_config = {
                **selected_candidate["training_config"],
                "seed": training_seed,
            }
            # 4. Start fresh and fit the selected settings on all outer-training subjects.
            progress = nested_fit_progress(outer_fold, len(outer_splits), selected_candidate)
            outer_model, outer_history = train_filter(
                [subjects_by_id[value] for value in train_ids], cache["omega"],
                selected_candidate["model_config"], selected_train_config,
                on_epoch=progress,
            )
            # 5. Evaluate the reserved outer-test subjects once after selection and refitting.
            outer_evaluation = evaluate_filter(
                outer_model, [subjects_by_id[value] for value in test_ids], cache["omega"]
            )

            # 6. Save the fitted weights, predictions, and selection evidence.
            outer_record = save_nested_fold(
                output_dir, result, outer_split, inner_splits, selected_candidate,
                scores_by_candidate, candidate_means, outer_model,
                outer_history, outer_evaluation,
            )
            result["outer_folds"].append(outer_record)
            save_json_atomic(result_path, result)

        # Summarize only the outer held-out predictions for final reporting.
        complete_nested_result(result, result_path)
        return result
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = f"{type(error).__name__}: {error}"
        save_json_atomic(result_path, result)
        raise
